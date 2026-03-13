#!/usr/bin/env python3
"""Video-only Gemini transcription with inline-safe compressed chunks.

This path avoids OCR entirely and sends low-bitrate video chunks inline to Gemini.
Output is plain Japanese text with `-- ` speaker-turn markers.
"""

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chigyusubs.audio import extract_inline_video_chunk, get_duration
from chigyusubs.chunking import chunk_coverage_issues
from chigyusubs.metadata import finish_run, metadata_path, start_run, write_metadata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transcribe_gemini import build_prompt, count_request_tokens, transcribe_chunk_result


def log(msg: str = ""):
    print(msg, flush=True)


def _round_cost(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 6)


def _cost_for_tokens(tokens: int | None, price_per_million: float | None) -> float | None:
    if tokens is None or price_per_million is None:
        return None
    return tokens / 1_000_000.0 * price_per_million


def _pricing_for_model(
    model: str,
    input_price_per_million: float | None,
    output_price_per_million: float | None,
) -> dict[str, Any]:
    lower = model.lower()
    inferred = False
    pricing_note = None
    if input_price_per_million is None and output_price_per_million is None:
        inferred = True
        if "gemini-3.1-flash-lite-preview" in lower:
            input_price_per_million = 0.25
            output_price_per_million = 1.50
            pricing_note = "Inferred from the official Gemini 3.1 Flash-Lite Preview pricing page."
        elif "gemini-3-flash-preview" in lower:
            input_price_per_million = 0.50
            output_price_per_million = 3.00
            pricing_note = "Inferred from the official Gemini 3 Flash Preview pricing page."
        elif "gemini-3.1-pro-preview" in lower:
            input_price_per_million = 2.00
            output_price_per_million = 12.00
            pricing_note = (
                "Inferred from the official Gemini 3.1 Pro Preview pricing page. "
                "This assumes each chunk stays at or below the 200k-token pricing tier."
            )
        elif "gemini-2.5-pro" in lower:
            input_price_per_million = 1.25
            output_price_per_million = 10.00
            pricing_note = (
                "Inferred from the official Gemini 2.5 Pro pricing page. "
                "This assumes each chunk stays at or below the 200k-token pricing tier."
            )
        elif "gemini-2.5-flash-lite" in lower:
            input_price_per_million = 0.10
            output_price_per_million = 0.40
            pricing_note = "Inferred from the official Gemini 2.5 Flash-Lite pricing page."
        else:
            inferred = False
    return {
        "input_price_per_million_usd": input_price_per_million,
        "output_price_per_million_usd": output_price_per_million,
        "pricing_inference": pricing_note if inferred else None,
    }


def _usage_cost_summary(usage_records: list[dict], pricing: dict[str, Any]) -> dict[str, Any]:
    prompt_tokens = sum(int(r.get("prompt_token_count") or 0) for r in usage_records)
    output_tokens = sum(int(r.get("candidates_token_count") or 0) for r in usage_records)
    total_tokens = sum(int(r.get("total_token_count") or 0) for r in usage_records)
    input_cost = _cost_for_tokens(prompt_tokens or None, pricing.get("input_price_per_million_usd"))
    output_cost = _cost_for_tokens(output_tokens or None, pricing.get("output_price_per_million_usd"))
    total_cost = None
    if input_cost is not None or output_cost is not None:
        total_cost = (input_cost or 0.0) + (output_cost or 0.0)
    return {
        "chunks_with_usage_metadata": len(usage_records),
        "prompt_token_count": prompt_tokens,
        "candidates_token_count": output_tokens,
        "total_token_count": total_tokens,
        "input_cost_usd": _round_cost(input_cost),
        "output_cost_usd": _round_cost(output_cost),
        "estimated_total_cost_usd": _round_cost(total_cost),
    }


def _rolling_prev_context(all_chunks: list[dict], upto_chunk: int, keep_chunks: int) -> str | None:
    if keep_chunks <= 0 or not all_chunks:
        return None
    start_chunk = max(0, upto_chunk - keep_chunks)
    kept = [c.get("text", "").strip() for c in all_chunks if start_chunk <= c.get("chunk", -1) < upto_chunk]
    kept = [c for c in kept if c]
    if not kept:
        return None
    return "\n".join(kept)


def _default_chunk_bounds(duration: float, chunk_seconds: float) -> list[tuple[float, float]]:
    bounds = []
    start = 0.0
    while start < duration:
        end = min(duration, start + chunk_seconds)
        bounds.append((start, end))
        start = end
    return bounds


def _load_chunk_bounds(chunk_json_path: str) -> list[tuple[float, float]]:
    with open(chunk_json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return [(float(c["start_sec"]), float(c["end_sec"])) for c in chunks]


def _max_line_length(text: str) -> int:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return max((len(ln) for ln in lines), default=0)


def _detect_loop_issue(text: str, max_line_length: int) -> dict | None:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for idx, line in enumerate(lines):
        body = line
        if body.startswith("-- "):
            body = body[3:]
        elif body.startswith("[画面: "):
            body = body[5:-1] if body.endswith("]") else body[5:]

        if len(line) > max_line_length:
            return {
                "reason": "line_too_long",
                "line_index": idx,
                "line_length": len(line),
                "line_prefix": line[:200],
            }

        clauses = [c.strip() for c in re.split(r"[、,。！？!\?\s]+", body) if c.strip()]
        if not clauses:
            continue
        run_term = clauses[0]
        run_len = 1
        for clause in clauses[1:]:
            if clause == run_term:
                run_len += 1
                if run_len >= 8 and len(run_term) <= 24:
                    return {
                        "reason": "repeated_clause_loop",
                        "line_index": idx,
                        "line_length": len(line),
                        "repeated_clause": run_term,
                        "repeat_count": run_len,
                        "line_prefix": line[:200],
                    }
            else:
                run_term = clause
                run_len = 1
    return None


def _retry_prompt() -> str:
    return (
        "RETRY INSTRUCTION:\n"
        "Your previous output got stuck repeating a word, phrase, or visual caption.\n"
        "Do not repeat any short word or phrase more than 3 times.\n"
        "If a visual caption is important, write one short `[画面: ...]` line only.\n"
        "Transcribe the scene again from the media.\n"
    )


def run_video_transcription(
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
    rolling_context_chunks: int,
    temperature: float,
    retry_temperature: float,
    loop_max_line_length: int,
    spoken_only: bool,
    media_resolution: str,
    thinking_level: str,
    thinking_budget: int | None,
    preview_cost: bool,
    count_tokens: bool,
    input_price_per_million: float | None,
    output_price_per_million: float | None,
    api_key: str = "",
    vertex: bool = False,
):
    run = start_run("transcribe_gemini_video")
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(output_path).stem
    raw_json_path = str(out_dir / f"{stem}_gemini_raw.json")
    preview_json_path = str(out_dir / f"{stem}_gemini_cost_preview.json")
    pricing = _pricing_for_model(model, input_price_per_million, output_price_per_million)
    if (
        (preview_cost or count_tokens)
        and (media_resolution != "unspecified" or thinking_level != "unspecified" or thinking_budget is not None)
        and not vertex
    ):
        raise ValueError(
            "Gemini API count_tokens does not support generation-config overrides such as media_resolution "
            "or thinking settings. Use --vertex for exact preflight counts, or run a real generation request "
            "and inspect usage_metadata."
        )

    duration = get_duration(video_path)
    if chunk_json:
        chunk_bounds = _load_chunk_bounds(chunk_json)
        issues = chunk_coverage_issues(chunk_bounds, duration)
        if issues:
            details = "; ".join(issues[:5])
            if len(issues) > 5:
                details += f"; ... {len(issues) - 5} more"
            raise ValueError(
                f"Chunk JSON is not full-coverage: {details}. Rebuild it with scripts/build_vad_chunks.py."
            )
    else:
        chunk_bounds = _default_chunk_bounds(duration, chunk_seconds)

    log(f"Video: {video_path}")
    log(f"Output: {output_path}")
    log(f"Duration: {duration:.1f}s")
    if chunk_json:
        log(f"Chunks: {len(chunk_bounds)} from {chunk_json}")
    else:
        log(f"Chunks: {len(chunk_bounds)} x {chunk_seconds:.0f}s")
    if pricing.get("input_price_per_million_usd") is not None or pricing.get("output_price_per_million_usd") is not None:
        log(
            "Pricing: "
            f"input=${pricing.get('input_price_per_million_usd')}/1M, "
            f"output=${pricing.get('output_price_per_million_usd')}/1M"
        )
        if pricing.get("pricing_inference"):
            log(pricing["pricing_inference"])
    if preview_cost:
        log("Preview mode: counting input tokens only; no transcription requests will be sent.")

    all_chunks: list[dict] = []
    preview_chunks: list[dict] = []
    start_chunk = 0
    if not preview_cost and os.path.exists(raw_json_path):
        with open(raw_json_path, "r", encoding="utf-8") as f:
            all_chunks = json.load(f)
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

            prompt = build_prompt(
                [],
                prev_context=_rolling_prev_context(all_chunks, i, rolling_context_chunks),
                include_visual_brackets=not spoken_only,
            )
            video_bytes = Path(chunk_path).read_bytes()
            token_preview = None
            if count_tokens or preview_cost:
                log("Counting prompt tokens...")
                token_preview = count_request_tokens(
                    video_bytes,
                    prompt,
                    model,
                    location,
                    mime_type="video/mp4",
                    media_resolution=media_resolution,
                    thinking_level=thinking_level,
                    thinking_budget=thinking_budget,
                    api_key=api_key,
                    vertex=vertex,
                )
                token_total = token_preview.get("total_tokens")
                token_cost = _cost_for_tokens(token_total, pricing.get("input_price_per_million_usd"))
                log(
                    "Prompt tokens: "
                    f"{token_total if token_total is not None else 'unknown'}"
                    + (
                        f" (~${_round_cost(token_cost):.6f})"
                        if token_cost is not None
                        else ""
                    )
                )
            if preview_cost:
                preview_chunks.append(
                    {
                        "chunk": i,
                        "chunk_start_s": c_start,
                        "chunk_end_s": c_end,
                        "chunk_size_mb": round(chunk_size_mb, 3),
                        "media_resolution": media_resolution,
                        "prompt_token_preview": token_preview,
                        "estimated_input_cost_usd": _round_cost(
                            _cost_for_tokens(
                                token_preview.get("total_tokens") if token_preview else None,
                                pricing.get("input_price_per_million_usd"),
                            )
                        ),
                    }
                )
                with open(preview_json_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "video": video_path,
                            "model": model,
                            "media_resolution": media_resolution,
                            "pricing": pricing,
                            "chunks": preview_chunks,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                continue

            log("Sending video chunk to Gemini...")
            result = transcribe_chunk_result(
                video_bytes,
                prompt,
                model,
                location,
                max_retries=20,
                temperature=temperature,
                mime_type="video/mp4",
                media_resolution=media_resolution,
                thinking_level=thinking_level,
                thinking_budget=thinking_budget,
                api_key=api_key,
                vertex=vertex,
            )
            text = result["text"]
            retry_info = None
            issue = _detect_loop_issue(text, loop_max_line_length)
            if issue:
                log(f"Detected loop-like chunk output ({issue['reason']}); retrying with temp={retry_temperature:.2f} and no rolling context...")
                retry_prompt = build_prompt([], prev_context=None, include_visual_brackets=not spoken_only) + "\n\n" + _retry_prompt()
                retried_result = transcribe_chunk_result(
                    video_bytes,
                    retry_prompt,
                    model,
                    location,
                    max_retries=20,
                    temperature=retry_temperature,
                    mime_type="video/mp4",
                    media_resolution=media_resolution,
                    thinking_level=thinking_level,
                    thinking_budget=thinking_budget,
                    api_key=api_key,
                    vertex=vertex,
                )
                retried = retried_result["text"]
                retry_issue = _detect_loop_issue(retried, loop_max_line_length)
                retry_info = {
                    "trigger": issue,
                    "retry_temperature": retry_temperature,
                    "retry_issue": retry_issue,
                    "retry_used": retry_issue is None or _max_line_length(retried) < _max_line_length(text),
                    "retry_usage_metadata": retried_result.get("usage_metadata"),
                    "retry_response_id": retried_result.get("response_id"),
                    "retry_model_version": retried_result.get("model_version"),
                }
                if retry_info["retry_used"]:
                    text = retried
                    result = retried_result

            line_count = len([ln for ln in text.splitlines() if ln.strip()])
            log(f"Result: {line_count} lines, {len(text)} chars")
            if result.get("usage_metadata"):
                usage = result["usage_metadata"]
                log(
                    "Usage: "
                    f"prompt={usage.get('prompt_token_count')} "
                    f"output={usage.get('candidates_token_count')} "
                    f"total={usage.get('total_token_count')}"
                )

            chunk_record = {
                "chunk": i,
                "chunk_start_s": c_start,
                "chunk_end_s": c_end,
                "text": text,
                "chunk_size_mb": round(chunk_size_mb, 3),
                "media_resolution": media_resolution,
                "thinking_level": thinking_level,
                "thinking_budget": thinking_budget,
            }
            if token_preview:
                chunk_record["prompt_token_preview"] = token_preview
            if result.get("usage_metadata") is not None:
                chunk_record["usage_metadata"] = result["usage_metadata"]
            if result.get("response_id"):
                chunk_record["response_id"] = result["response_id"]
            if result.get("model_version"):
                chunk_record["model_version"] = result["model_version"]
            if retry_info:
                chunk_record["retry"] = retry_info
            all_chunks.append(chunk_record)

            with open(raw_json_path, "w", encoding="utf-8") as f:
                json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    if preview_cost:
        preview_prompt_tokens = sum(
            int((c.get("prompt_token_preview") or {}).get("total_tokens") or 0) for c in preview_chunks
        )
        preview_input_cost = _cost_for_tokens(preview_prompt_tokens or None, pricing.get("input_price_per_million_usd"))
        preview_payload = {
            "video": video_path,
            "model": model,
            "media_resolution": media_resolution,
            "thinking_level": thinking_level,
            "thinking_budget": thinking_budget,
            "pricing": pricing,
            "summary": {
                "chunks": len(preview_chunks),
                "prompt_token_count": preview_prompt_tokens,
                "estimated_input_cost_usd": _round_cost(preview_input_cost),
                "note": "Preflight token counting is exact for the request payload. Output tokens cannot be known before generation.",
            },
            "chunks": preview_chunks,
        }
        with open(preview_json_path, "w", encoding="utf-8") as f:
            json.dump(preview_payload, f, ensure_ascii=False, indent=2)
        metadata = finish_run(
            run,
            inputs={"video": video_path},
            outputs={"cost_preview_json": preview_json_path},
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
                "rolling_context_chunks": rolling_context_chunks,
                "spoken_only": spoken_only,
                "media_resolution": media_resolution,
                "thinking_level": thinking_level,
                "thinking_budget": thinking_budget,
                "preview_cost": preview_cost,
                "count_tokens": count_tokens,
                "pricing": pricing,
            },
            stats=preview_payload["summary"],
        )
        write_metadata(preview_json_path, metadata)
        log(f"Saved cost preview: {preview_json_path}")
        log(f"Metadata written: {metadata_path(preview_json_path)}")
        return

    total_lines = sum(len([ln for ln in c.get("text", "").splitlines() if ln.strip()]) for c in all_chunks)
    total_chars = sum(len(c.get("text", "")) for c in all_chunks)
    usage_records = [c["usage_metadata"] for c in all_chunks if isinstance(c.get("usage_metadata"), dict)]
    prompt_preview_tokens = sum(int((c.get("prompt_token_preview") or {}).get("total_tokens") or 0) for c in all_chunks)
    prompt_preview_cost = _cost_for_tokens(prompt_preview_tokens or None, pricing.get("input_price_per_million_usd"))

    metadata = finish_run(
        run,
        inputs={"video": video_path},
        outputs={"video_gemini_json": raw_json_path},
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
            "rolling_context_chunks": rolling_context_chunks,
            "temperature": temperature,
            "retry_temperature": retry_temperature,
            "loop_max_line_length": loop_max_line_length,
            "spoken_only": spoken_only,
            "media_resolution": media_resolution,
            "thinking_level": thinking_level,
            "thinking_budget": thinking_budget,
            "preview_cost": preview_cost,
            "count_tokens": count_tokens,
            "pricing": pricing,
        },
        stats={
            "duration_seconds": round(duration, 3),
            "chunks": len(all_chunks),
            "lines": total_lines,
            "characters": total_chars,
            "prompt_token_preview_count": prompt_preview_tokens if count_tokens else None,
            "prompt_token_preview_cost_usd": _round_cost(prompt_preview_cost) if count_tokens else None,
            "usage": _usage_cost_summary(usage_records, pricing),
        },
    )
    write_metadata(raw_json_path, metadata)
    log(f"Saved: {raw_json_path}")
    log(f"Metadata written: {metadata_path(raw_json_path)}")


def main():
    parser = argparse.ArgumentParser(description="Video-only Gemini transcription with inline video chunks.")
    parser.add_argument("--video", required=True, help="Input video file.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-2.5-pro"))
    parser.add_argument("--location", default=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"))
    parser.add_argument("--chunk-seconds", type=float, default=600.0)
    parser.add_argument("--chunk-json", default="", help="Optional saved chunk boundaries JSON (e.g. vad_chunks.json).")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--audio-bitrate", default="24k")
    parser.add_argument("--crf", type=int, default=36)
    parser.add_argument("--max-inline-mb", type=float, default=14.0)
    parser.add_argument("--rolling-context-chunks", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--retry-temperature", type=float, default=0.3)
    parser.add_argument("--loop-max-line-length", type=int, default=300)
    parser.add_argument(
        "--spoken-only",
        action="store_true",
        help="Ask Gemini for spoken dialogue only and do not request [画面: ...] visual cue lines.",
    )
    parser.add_argument(
        "--media-resolution",
        choices=["unspecified", "low", "medium", "high"],
        default="unspecified",
        help="Gemini media resolution hint for multimodal processing.",
    )
    parser.add_argument(
        "--thinking-level",
        choices=["unspecified", "minimal", "low", "medium", "high"],
        default="unspecified",
        help="Optional Gemini thinking level override.",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=None,
        help="Optional Gemini thinking budget override.",
    )
    parser.add_argument(
        "--preview-cost",
        action="store_true",
        help="Encode chunks and call Gemini count_tokens for an exact input-token preview without generating a transcript.",
    )
    parser.add_argument(
        "--count-tokens",
        action="store_true",
        help="Call Gemini count_tokens for each chunk before generation and store prompt-token previews in the output JSON.",
    )
    parser.add_argument(
        "--input-price-per-1m",
        type=float,
        default=None,
        help="Optional USD price per 1M input tokens for cost summaries.",
    )
    parser.add_argument(
        "--output-price-per-1m",
        type=float,
        default=None,
        help="Optional USD price per 1M output tokens for cost summaries.",
    )
    parser.add_argument("--vertex", action="store_true", help="Use Vertex AI instead of Gemini API. Default uses GEMINI_API_KEY.")
    args = parser.parse_args()

    if args.vertex:
        print("Using Vertex AI", flush=True)
    else:
        print("Using Gemini API", flush=True)

    run_video_transcription(
        video_path=args.video,
        output_path=args.output,
        model=args.model,
        location=args.location,
        chunk_seconds=args.chunk_seconds,
        chunk_json=args.chunk_json,
        fps=args.fps,
        width=args.width,
        audio_bitrate=args.audio_bitrate,
        crf=args.crf,
        max_inline_mb=args.max_inline_mb,
        rolling_context_chunks=args.rolling_context_chunks,
        temperature=args.temperature,
        retry_temperature=args.retry_temperature,
        loop_max_line_length=args.loop_max_line_length,
        spoken_only=args.spoken_only,
        media_resolution=args.media_resolution,
        thinking_level=args.thinking_level,
        thinking_budget=args.thinking_budget,
        preview_cost=args.preview_cost,
        count_tokens=args.count_tokens,
        input_price_per_million=args.input_price_per_1m,
        output_price_per_million=args.output_price_per_1m,
        vertex=args.vertex,
    )


if __name__ == "__main__":
    main()
