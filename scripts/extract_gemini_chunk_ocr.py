#!/usr/bin/env python3
"""Chunkwise Gemini OCR sidecar extraction.

This path is intentionally separate from the main transcript call. It extracts
meaningful visible on-screen text from each saved chunk span and writes a
structured chunk-scoped OCR artifact that can be reused for review, glossary,
and translation support.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chigyusubs.audio import extract_inline_video_chunk, get_duration
from chigyusubs.chunking import chunk_coverage_issues
from chigyusubs.gemini_presets import preset_names, resolve_settings
from chigyusubs.metadata import finish_run, metadata_path, start_run, write_metadata
from chigyusubs.paths import find_episode_dir_from_path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transcribe_gemini import (
    _backoff_delay,
    _build_request_parts,
    _build_thinking_config,
    _countdown,
    _make_client,
    _usage_metadata_to_dict,
)
from transcribe_gemini_video import (
    _cost_for_tokens,
    _default_chunk_bounds,
    _load_chunk_bounds,
    _pricing_for_model,
    _round_cost,
    _usage_cost_summary,
)


KIND_VALUES = {"title_card", "name_card", "info_card", "label", "other"}
IMPORTANCE_VALUES = {"high", "medium", "low"}


def log(msg: str = "") -> None:
    print(msg, flush=True)


def build_ocr_prompt() -> str:
    return "\n".join(
        [
            "You are extracting meaningful visible on-screen text from a Japanese variety/comedy show video chunk.",
            "",
            "Output ONLY JSON.",
            "",
            "Return exactly one JSON object with this shape:",
            "{",
            '  "items": [',
            "    {",
            '      "text": "visible text exactly as shown",',
            '      "kind_guess": "title_card | name_card | info_card | label | other",',
            '      "importance": "high | medium | low"',
            "    }",
            "  ]",
            "}",
            "",
            "Rules:",
            "1. Extract only text that is visibly present on screen.",
            "2. Do NOT transcribe spoken dialogue unless the same words are visibly shown on screen.",
            "3. Preserve visible wording when readable, including kanji, kana, katakana, digits, and Latin text.",
            "4. Prefer exact visible text over paraphrase or summary.",
            "5. If text is only partially readable, include only the confidently readable portion. Do NOT guess missing characters.",
            "6. Keep duplicates minimal. If the same text appears repeatedly in this chunk, include it once unless the wording changes.",
            "7. Use `title_card` for opening/title cards or big featured titles.",
            "8. Use `name_card` for cast/member/person name cards.",
            "9. Use `info_card` for mission prompts, instructions, challenge text, or other large information-bearing cards.",
            "10. Use `label` for smaller labels, counters, signs, maps, prices, or short UI-like text.",
            "11. Use `other` only when the visible text is meaningful but does not fit the other kinds.",
            "12. Mark `importance` as `high` for text that would materially help transcription or translation, `medium` for useful context, and `low` for minor but readable labels.",
            "13. If there is no meaningful readable on-screen text, return {\"items\": []}.",
            "",
            "Do not add commentary, markdown, timestamps, or extra fields.",
        ]
    )


def _normalize_kind(value: Any) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "title": "title_card",
        "titlecard": "title_card",
        "name": "name_card",
        "namecard": "name_card",
        "rule": "info_card",
        "rulecard": "info_card",
        "rule_text": "info_card",
        "infocard": "info_card",
    }
    text = aliases.get(text, text)
    return text if text in KIND_VALUES else "other"


def _normalize_importance(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text in IMPORTANCE_VALUES else "medium"


def _clean_text(value: Any) -> str:
    text = str(value or "").strip()
    if text.startswith("[画面:") and text.endswith("]"):
        text = text[4:-1].strip(" :")
    return text.strip()


def _parse_ocr_json(raw: str) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    text = raw.strip()
    if not text:
        return [], warnings

    payload: Any
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        warnings.append("invalid_json_response")
        items = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("[画面:") and line.endswith("]"):
                items.append(
                    {
                        "text": _clean_text(line),
                        "kind_guess": "other",
                        "importance": "medium",
                    }
                )
        return items, warnings

    if isinstance(payload, dict):
        items = payload.get("items", [])
    elif isinstance(payload, list):
        items = payload
    else:
        warnings.append("unexpected_json_root")
        return [], warnings

    if not isinstance(items, list):
        warnings.append("items_not_list")
        return [], warnings

    seen: set[tuple[str, str, str]] = set()
    normalized: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            warnings.append("non_object_item")
            continue
        cleaned = {
            "text": _clean_text(item.get("text")),
            "kind_guess": _normalize_kind(item.get("kind_guess")),
            "importance": _normalize_importance(item.get("importance")),
        }
        if not cleaned["text"]:
            warnings.append("empty_text_item")
            continue
        key = (cleaned["text"], cleaned["kind_guess"], cleaned["importance"])
        if key in seen:
            continue
        seen.add(key)
        normalized.append(cleaned)
    return normalized, warnings


def extract_ocr_chunk_result(
    *,
    video_bytes: bytes,
    prompt: str,
    model: str,
    location: str,
    temperature: float,
    preset_name: str | None,
    media_resolution: str,
    thinking_level: str,
    thinking_budget: int | None,
    api_key: str = "",
    vertex: bool = False,
    max_retries: int = 10,
) -> dict[str, Any]:
    from google.genai import types

    client = _make_client(location, api_key=api_key, vertex=vertex)
    parts = _build_request_parts(video_bytes, prompt, "video/mp4")

    media_map = {
        "unspecified": types.MediaResolution.MEDIA_RESOLUTION_UNSPECIFIED,
        "low": types.MediaResolution.MEDIA_RESOLUTION_LOW,
        "medium": types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
        "high": types.MediaResolution.MEDIA_RESOLUTION_HIGH,
    }
    if media_resolution not in media_map:
        raise ValueError(f"Unsupported media_resolution: {media_resolution}")

    config = types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="application/json",
        max_output_tokens=8192,
        httpOptions=types.HttpOptions(timeout=180_000),
        media_resolution=media_map[media_resolution],
        thinking_config=_build_thinking_config(
            thinking_level=thinking_level,
            thinking_budget=thinking_budget,
        ),
    )

    for attempt in range(max_retries):
        attempt_label = f"[attempt {attempt + 1}/{max_retries}]"
        try:
            print(f"  {attempt_label} Requesting...", end="", flush=True)
            t0 = time.time()
            chunks: list[str] = []
            char_count = 0
            first_chunk = True
            usage_metadata = None
            response_id = None
            model_version = None
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=parts,
                config=config,
            ):
                text = chunk.text or ""
                if getattr(chunk, "usage_metadata", None) is not None:
                    usage_metadata = _usage_metadata_to_dict(chunk.usage_metadata)
                if getattr(chunk, "response_id", None):
                    response_id = chunk.response_id
                if getattr(chunk, "model_version", None):
                    model_version = chunk.model_version
                if first_chunk and text:
                    print(f" first token in {time.time() - t0:.1f}s, streaming", end="", flush=True)
                    first_chunk = False
                chunks.append(text)
                char_count += len(text)
                if char_count % 1000 < len(text):
                    print(".", end="", flush=True)

            elapsed = time.time() - t0
            raw_text = "".join(chunks).strip()
            items, parse_warnings = _parse_ocr_json(raw_text)
            print(f" {len(raw_text)} chars in {elapsed:.1f}s", flush=True)
            return {
                "raw_response_text": raw_text,
                "items": items,
                "parse_warnings": parse_warnings,
                "usage_metadata": usage_metadata,
                "response_id": response_id,
                "model_version": model_version,
                "elapsed_seconds": round(elapsed, 3),
            }
        except Exception as e:
            elapsed = time.time() - t0
            print(flush=True)
            if attempt < max_retries - 1:
                msg = str(e).strip().splitlines()[0] if str(e).strip() else repr(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    err_type = "RATE LIMITED"
                elif "499" in msg or "CANCELLED" in msg:
                    err_type = "CANCELLED"
                elif "500" in msg or "INTERNAL" in msg:
                    err_type = "SERVER ERROR"
                else:
                    err_type = "ERROR"
                print(f"  {attempt_label} {err_type} after {elapsed:.0f}s: {msg}", flush=True)
                _countdown(_backoff_delay(attempt + 1))
                continue
            raise

    raise RuntimeError("OCR request failed with no response.")


def _default_output_path(video_path: str) -> str:
    video = Path(video_path)
    episode_dir = find_episode_dir_from_path(video)
    stem = video.stem
    if episode_dir is not None:
        return str(episode_dir / "ocr" / f"{stem}_flash_lite_chunk_ocr.json")
    return str(video.with_name(f"{stem}_flash_lite_chunk_ocr.json"))


def run_chunk_ocr(
    *,
    video_path: str,
    output_path: str,
    model: str,
    location: str,
    chunk_seconds: float,
    chunk_json: str,
    fps: float,
    width: int,
    audio_bitrate: str,
    crf: int,
    max_inline_mb: float,
    temperature: float,
    media_resolution: str,
    thinking_level: str,
    thinking_budget: int | None,
    input_price_per_million: float | None,
    output_price_per_million: float | None,
    api_key: str = "",
    vertex: bool = False,
) -> None:
    run = start_run("extract_gemini_chunk_ocr")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    duration = get_duration(video_path)
    chunk_bounds = _load_chunk_bounds(chunk_json) if chunk_json else _default_chunk_bounds(duration, chunk_seconds)
    if chunk_json:
        issues = chunk_coverage_issues(chunk_bounds, duration)
        if issues:
            details = "; ".join(issues[:5])
            if len(issues) > 5:
                details += f"; ... {len(issues) - 5} more"
            raise ValueError(
                f"Chunk JSON is not full-coverage: {details}. Rebuild it with scripts/build_vad_chunks.py."
            )
    pricing = _pricing_for_model(model, input_price_per_million, output_price_per_million)
    prompt = build_ocr_prompt()

    log(f"Video: {video_path}")
    log(f"Output: {output_path}")
    log(f"Duration: {duration:.1f}s")
    if chunk_json:
        log(f"Chunks: {len(chunk_bounds)} from {chunk_json}")
    else:
        log(f"Chunks: {len(chunk_bounds)} x {chunk_seconds:.0f}s")

    all_chunks: list[dict[str, Any]] = []
    start_chunk = 0
    if output.exists():
        all_chunks = json.loads(output.read_text(encoding="utf-8"))
        done_chunks = set(c.get("chunk", 0) for c in all_chunks)
        start_chunk = max(done_chunks) + 1 if done_chunks else 0
        if start_chunk > 0:
            log(f"Resuming from chunk {start_chunk + 1} ({len(all_chunks)} chunks already saved)")

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, (c_start, c_end) in enumerate(chunk_bounds):
            if i < start_chunk:
                continue
            chunk_path = os.path.join(tmpdir, f"chunk_{i}.mp4")
            log()
            log(f"--- Chunk {i + 1}/{len(chunk_bounds)} ---")
            log(f"Time: {c_start / 60:.1f} - {c_end / 60:.1f} min ({c_end - c_start:.0f}s)")
            log("Encoding inline-safe video chunk...")
            extract_inline_video_chunk(
                video_path,
                chunk_path,
                start_s=c_start,
                duration_s=c_end - c_start,
                fps=fps,
                width=width,
                audio_bitrate=audio_bitrate,
                crf=crf,
            )
            chunk_size_mb = Path(chunk_path).stat().st_size / (1024 * 1024)
            log(f"Chunk size: {chunk_size_mb:.2f} MB")
            if chunk_size_mb > max_inline_mb:
                raise RuntimeError(
                    f"Chunk {i} encoded to {chunk_size_mb:.2f} MB, above inline target {max_inline_mb:.2f} MB"
                )

            log("Sending video chunk to Gemini...")
            result = extract_ocr_chunk_result(
                video_bytes=Path(chunk_path).read_bytes(),
                prompt=prompt,
                model=model,
                location=location,
                temperature=temperature,
                media_resolution=media_resolution,
                thinking_level=thinking_level,
                thinking_budget=thinking_budget,
                api_key=api_key,
                vertex=vertex,
            )

            items = [
                {
                    "text": item["text"],
                    "kind_guess": item["kind_guess"],
                    "importance": item["importance"],
                    "start_s": c_start,
                    "end_s": c_end,
                    "timing_basis": "chunk_span",
                }
                for item in result["items"]
            ]
            log(f"Items: {len(items)}")
            chunk_record: dict[str, Any] = {
                "chunk": i,
                "chunk_start_s": c_start,
                "chunk_end_s": c_end,
                "chunk_size_mb": round(chunk_size_mb, 3),
                "media_resolution": media_resolution,
                "thinking_level": thinking_level,
                "thinking_budget": thinking_budget,
                "items": items,
                "raw_response_text": result["raw_response_text"],
                "parse_warnings": result["parse_warnings"],
            }
            if result.get("usage_metadata") is not None:
                chunk_record["usage_metadata"] = result["usage_metadata"]
            if result.get("response_id"):
                chunk_record["response_id"] = result["response_id"]
            if result.get("model_version"):
                chunk_record["model_version"] = result["model_version"]
            all_chunks.append(chunk_record)
            output.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    usage_records = [c["usage_metadata"] for c in all_chunks if isinstance(c.get("usage_metadata"), dict)]
    items_total = sum(len(c.get("items", [])) for c in all_chunks)
    kind_counts: dict[str, int] = {}
    for chunk in all_chunks:
        for item in chunk.get("items", []):
            kind = item.get("kind_guess", "other")
            kind_counts[kind] = kind_counts.get(kind, 0) + 1

    stats = {
        "chunks": len(all_chunks),
        "items": items_total,
        "kind_counts": kind_counts,
        "chunks_with_parse_warnings": sum(1 for c in all_chunks if c.get("parse_warnings")),
        **_usage_cost_summary(usage_records, pricing),
    }
    metadata = finish_run(
        run,
        inputs={"video": video_path},
        outputs={"chunk_ocr_json": str(output)},
        settings={
            "model": model,
            "location": location,
            "chunk_seconds": chunk_seconds,
            "chunk_json": chunk_json,
            "fps": fps,
            "width": width,
            "audio_bitrate": audio_bitrate,
            "crf": crf,
            "max_inline_mb": max_inline_mb,
            "temperature": temperature,
            "preset": preset_name,
            "media_resolution": media_resolution,
            "thinking_level": thinking_level,
            "thinking_budget": thinking_budget,
            "pricing": pricing,
        },
        stats=stats,
    )
    write_metadata(output, metadata)
    log(f"Saved: {output}")
    log(f"Metadata written: {metadata_path(output)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract chunkwise structured OCR sidecar text with Gemini.")
    parser.add_argument("--video", required=True, help="Input video file.")
    parser.add_argument("--output", default="", help="Output JSON path. Defaults to episode ocr dir.")
    parser.add_argument(
        "--preset",
        choices=preset_names("extract_gemini_chunk_ocr"),
        default=None,
        help="Optional named Gemini OCR settings preset.",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--location", default=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"))
    parser.add_argument("--chunk-seconds", type=float, default=240.0)
    parser.add_argument("--chunk-json", default="", help="Optional saved chunk boundaries JSON (e.g. vad_chunks.json).")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--audio-bitrate", default="24k")
    parser.add_argument("--crf", type=int, default=36)
    parser.add_argument("--max-inline-mb", type=float, default=19.5)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--media-resolution", choices=["unspecified", "low", "medium", "high"], default=None)
    parser.add_argument("--thinking-level", choices=["unspecified", "minimal", "low", "medium", "high"], default=None)
    parser.add_argument("--thinking-budget", type=int, default=None)
    parser.add_argument("--input-price-per-1m", type=float, default=None)
    parser.add_argument("--output-price-per-1m", type=float, default=None)
    parser.add_argument("--vertex", action="store_true", help="Use Vertex AI instead of Gemini API.")
    args = parser.parse_args()

    model_override = args.model
    if model_override is None and not args.preset:
        model_override = os.environ.get("GEMINI_OCR_MODEL")

    resolved, chosen_preset = resolve_settings(
        "extract_gemini_chunk_ocr",
        args.preset,
        {
            "model": model_override,
            "temperature": args.temperature,
            "media_resolution": args.media_resolution,
            "thinking_level": args.thinking_level,
            "thinking_budget": args.thinking_budget,
        },
    )

    output = args.output or _default_output_path(args.video)
    print("Using Vertex AI" if args.vertex else "Using Gemini API")
    if chosen_preset:
        print(f"Using preset: {chosen_preset}")
    run_chunk_ocr(
        video_path=args.video,
        output_path=output,
        model=resolved["model"],
        location=args.location,
        chunk_seconds=args.chunk_seconds,
        chunk_json=args.chunk_json,
        fps=args.fps,
        width=args.width,
        audio_bitrate=args.audio_bitrate,
        crf=args.crf,
        max_inline_mb=args.max_inline_mb,
        temperature=resolved["temperature"],
        preset_name=chosen_preset,
        media_resolution=resolved["media_resolution"],
        thinking_level=resolved["thinking_level"],
        thinking_budget=resolved["thinking_budget"],
        input_price_per_million=args.input_price_per_1m,
        output_price_per_million=args.output_price_per_1m,
        vertex=args.vertex,
    )


if __name__ == "__main__":
    main()
