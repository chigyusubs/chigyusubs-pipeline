#!/usr/bin/env python3
"""Chunkwise OCR sidecar extraction.

This path is intentionally separate from the main transcript call. It extracts
meaningful visible on-screen text from each saved chunk span and writes a
structured chunk-scoped OCR artifact that can be reused for review, glossary,
and translation support.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chigyusubs.audio import extract_inline_video_chunk, get_duration
from chigyusubs.chunking import chunk_coverage_issues, describe_chunk_plan
from chigyusubs.env import load_repo_env
from chigyusubs.gemini_presets import preset_names, resolve_settings
from chigyusubs.metadata import finish_run, metadata_path, start_run, write_metadata
from chigyusubs.paths import find_episode_dir_from_path

load_repo_env()

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
DEFAULT_LLAMA_CPP_BASE_URL = "http://127.0.0.1:8000/v1/chat/completions"
DEFAULT_LLAMA_CPP_OCR_MODEL = "gemma-4-26B-A4B-it-IQ4_XS"

LOCAL_OCR_SYSTEM = (
    "You are reading on-screen Japanese text from frames of a Japanese "
    "variety show. Extract only text physically visible in the images. "
    "Do not transcribe spoken dialogue unless it is printed on screen."
)


def log(msg: str = "") -> None:
    print(msg, flush=True)


class _ThreadRateLimiter:
    def __init__(self, rpm: int):
        self.interval = 60.0 / max(1, rpm)
        self._lock = threading.Lock()
        self._last_request = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._last_request + self.interval - now
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()


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
    text = text.strip()
    while text[:1] in {"-", "*", "・", "•"}:
        text = text[1:].strip()
    if len(text) >= 3 and text[0].isdigit() and text[1] in {".", ")", "、"}:
        text = text[2:].strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()
    return text.strip()


def _parse_ocr_lines(raw: str) -> tuple[list[dict[str, Any]], list[str]]:
    warnings = ["invalid_json_response", "line_fallback_used"]
    seen: set[str] = set()
    items: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = _clean_text(line)
        if not line or line.lower() in {"(none)", "none", "なし", "ない"}:
            continue
        if line.startswith("```") or line in {"{", "}", "[", "]"}:
            continue
        if line.startswith(("Output", "JSON", "items", '"items"', '"kind_guess"', '"importance"')):
            continue
        if line in {"},", "},"}:
            continue
        if line.startswith('"text"'):
            _, _, value = line.partition(":")
            line = _clean_text(value.rstrip(","))
            if not line:
                continue
        if len(line) > 160:
            warnings.append("overlong_line_skipped")
            continue
        if line in seen:
            continue
        seen.add(line)
        items.append(
            {
                "text": line,
                "kind_guess": "other",
                "importance": "medium",
            }
        )
    return items, warnings


def _parse_ocr_json(raw: str) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    text = raw.strip()
    if not text:
        return [], warnings
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    payload: Any
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return _parse_ocr_lines(text)

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


def _image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _post_json(base_url: str, payload: dict[str, Any], timeout_s: int) -> tuple[dict[str, Any], float]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read()), time.monotonic() - t0
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:1000]
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def _select_evenly(paths: list[Path], max_items: int) -> list[Path]:
    if max_items <= 0 or len(paths) <= max_items:
        return paths
    if max_items == 1:
        return [paths[len(paths) // 2]]
    last = len(paths) - 1
    indexes = [round(i * last / (max_items - 1)) for i in range(max_items)]
    out: list[Path] = []
    seen: set[int] = set()
    for idx in indexes:
        if idx in seen:
            continue
        seen.add(idx)
        out.append(paths[idx])
    return out


def extract_chunk_frames(
    *,
    video_path: str,
    output_dir: Path,
    start_s: float,
    duration_s: float,
    fps: float,
    height: int,
    max_frames: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "f_%05d.jpg"
    vf = f"fps={fps},scale=-2:{height}"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-t",
        f"{duration_s:.3f}",
        "-i",
        video_path,
        "-vf",
        vf,
        "-q:v",
        "3",
        str(pattern),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    frames = sorted(output_dir.glob("f_*.jpg"))
    return _select_evenly(frames, max_frames)


def extract_llama_cpp_ocr_chunk_result(
    *,
    frame_paths: list[Path],
    prompt: str,
    model: str,
    base_url: str,
    temperature: float,
    top_p: float,
    top_k: int,
    frequency_penalty: float,
    max_output_tokens: int,
    request_timeout_s: int,
    max_retries: int = 2,
    rate_limiter_acquire=None,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": _image_data_url(path)}}
        for path in frame_paths
    ]
    content.append({"type": "text", "text": prompt})
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": LOCAL_OCR_SYSTEM},
            {"role": "user", "content": content},
        ],
        "max_tokens": max_output_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "frequency_penalty": frequency_penalty,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    for attempt in range(max_retries):
        attempt_label = f"[attempt {attempt + 1}/{max_retries}]"
        try:
            if rate_limiter_acquire is not None:
                rate_limiter_acquire()
            print(f"  {attempt_label} Requesting llama.cpp...", end="", flush=True)
            body, elapsed = _post_json(base_url, payload, request_timeout_s)
            if "error" in body:
                raise RuntimeError(str(body["error"]))
            raw_text = str(body["choices"][0]["message"]["content"]).strip()
            items, parse_warnings = _parse_ocr_json(raw_text)
            print(f" {len(raw_text)} chars in {elapsed:.1f}s", flush=True)
            return {
                "raw_response_text": raw_text,
                "items": items,
                "parse_warnings": parse_warnings,
                "usage_metadata": body.get("usage"),
                "elapsed_seconds": round(elapsed, 3),
            }
        except Exception as exc:
            print(flush=True)
            if attempt < max_retries - 1:
                msg = str(exc).strip().splitlines()[0] if str(exc).strip() else repr(exc)
                print(f"  {attempt_label} ERROR: {msg}", flush=True)
                _countdown(_backoff_delay(attempt + 1))
                continue
            raise

    raise RuntimeError("llama.cpp OCR request failed with no response.")


def _parse_chunk_filter(value: str) -> set[int]:
    selected: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left)
            end = int(right)
            if end < start:
                raise ValueError(f"Invalid chunk range: {part}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    return selected


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
    rate_limiter_acquire=None,
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
            if rate_limiter_acquire is not None:
                rate_limiter_acquire()
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
    backend: str,
    model: str,
    location: str,
    base_url: str,
    chunk_seconds: float,
    chunk_json: str,
    fps: float,
    width: int | None,
    audio_bitrate: str,
    crf: int,
    max_inline_mb: float,
    temperature: float,
    preset_name: str | None,
    media_resolution: str,
    thinking_level: str,
    thinking_budget: int | None,
    local_frame_fps: float,
    local_frame_height: int,
    local_max_frames: int,
    max_output_tokens: int,
    top_p: float,
    top_k: int,
    frequency_penalty: float,
    request_timeout_s: int,
    input_price_per_million: float | None,
    output_price_per_million: float | None,
    api_key: str = "",
    vertex: bool = False,
    concurrency: int = 1,
    rpm: int = 5,
    max_request_retries: int = 10,
    only_chunks: set[int] | None = None,
    force_chunks: set[int] | None = None,
    stop_after_chunks: int = 0,
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
    chunk_plan = describe_chunk_plan(chunk_json, chunk_bounds) if chunk_json else None
    prompt = build_ocr_prompt()

    log(f"Video: {video_path}")
    log(f"Output: {output_path}")
    log(f"Backend: {backend}")
    if backend == "llama-cpp":
        log(f"llama.cpp endpoint: {base_url}")
    log(f"Duration: {duration:.1f}s")
    if chunk_json:
        log(f"Chunks: {len(chunk_bounds)} from {chunk_json}")
        if chunk_plan is not None:
            log(
                "Chunk plan: "
                f"{chunk_plan['label']} "
                f"(min={chunk_plan['min_chunk_s']:.1f}s avg={chunk_plan['avg_chunk_s']:.1f}s max={chunk_plan['max_chunk_s']:.1f}s)"
            )
    else:
        log(f"Chunks: {len(chunk_bounds)} x {chunk_seconds:.0f}s")
    log(f"Concurrency: {max(1, concurrency)} workers, {rpm} RPM")
    if backend == "llama-cpp":
        log(
            "Local frames: "
            f"{local_frame_fps:g} fps, height={local_frame_height}, max={local_max_frames} frames/chunk"
        )

    all_chunks: list[dict[str, Any]] = []
    if output.exists():
        all_chunks = json.loads(output.read_text(encoding="utf-8"))
        all_chunks = [c for c in all_chunks if not c.get("error")]
        done_chunks = set(c.get("chunk", 0) for c in all_chunks)
        if done_chunks:
            log(f"Resuming: {len(done_chunks)} chunks already saved, {len(chunk_bounds) - len(done_chunks)} remaining")
    else:
        done_chunks = set()

    pending_indices = [i for i in range(len(chunk_bounds)) if i not in done_chunks]
    if force_chunks:
        pending_indices.extend(i for i in sorted(force_chunks) if 0 <= i < len(chunk_bounds))
        pending_indices = sorted(set(pending_indices))
        log(f"Force chunks: {sorted(force_chunks)}")
    if only_chunks:
        pending_indices = [i for i in pending_indices if i in only_chunks]
        log(f"Chunk filter: {len(pending_indices)} pending chunks selected")
    if stop_after_chunks > 0:
        pending_indices = pending_indices[:stop_after_chunks]
        log(f"Stop-after limit: attempting first {len(pending_indices)} pending chunks")
    if not pending_indices:
        log("All OCR chunks already completed.")

    write_lock = threading.Lock()

    def save_record(record: dict[str, Any]) -> None:
        nonlocal all_chunks
        with write_lock:
            all_chunks = [c for c in all_chunks if c.get("chunk") != record.get("chunk")]
            all_chunks.append(record)
            all_chunks.sort(key=lambda c: c.get("chunk", 0))
            output.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rate_limiter = _ThreadRateLimiter(rpm)

    with tempfile.TemporaryDirectory() as tmpdir:
        def process_chunk(i: int) -> dict[str, Any]:
            c_start, c_end = chunk_bounds[i]
            log()
            log(f"--- Chunk {i + 1}/{len(chunk_bounds)} ---")
            log(f"Time: {c_start / 60:.1f} - {c_end / 60:.1f} min ({c_end - c_start:.0f}s)")
            chunk_size_mb = None
            frame_count = None

            if backend == "gemini":
                chunk_path = os.path.join(tmpdir, f"chunk_{i}.mp4")
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
                    preset_name=preset_name,
                    media_resolution=media_resolution,
                    thinking_level=thinking_level,
                    thinking_budget=thinking_budget,
                    api_key=api_key,
                    vertex=vertex,
                    max_retries=max_request_retries,
                    rate_limiter_acquire=rate_limiter.acquire,
                )
            elif backend == "llama-cpp":
                frame_dir = Path(tmpdir) / f"chunk_{i}_frames"
                log("Extracting sampled frames for llama.cpp...")
                frame_paths = extract_chunk_frames(
                    video_path=video_path,
                    output_dir=frame_dir,
                    start_s=c_start,
                    duration_s=c_end - c_start,
                    fps=local_frame_fps,
                    height=local_frame_height,
                    max_frames=local_max_frames,
                )
                frame_count = len(frame_paths)
                log(f"Frames: {frame_count}")
                if not frame_paths:
                    raise RuntimeError(f"Chunk {i} produced no frames for local OCR")

                log("Sending frames to llama.cpp...")
                result = extract_llama_cpp_ocr_chunk_result(
                    frame_paths=frame_paths,
                    prompt=prompt,
                    model=model,
                    base_url=base_url,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    frequency_penalty=frequency_penalty,
                    max_output_tokens=max_output_tokens,
                    request_timeout_s=request_timeout_s,
                    max_retries=max_request_retries,
                    rate_limiter_acquire=rate_limiter.acquire if rpm > 0 else None,
                )
            else:
                raise ValueError(f"Unsupported OCR backend: {backend}")

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
                "backend": backend,
                "media_resolution": media_resolution,
                "thinking_level": thinking_level,
                "thinking_budget": thinking_budget,
                "items": items,
                "raw_response_text": result["raw_response_text"],
                "parse_warnings": result["parse_warnings"],
            }
            if chunk_size_mb is not None:
                chunk_record["chunk_size_mb"] = round(chunk_size_mb, 3)
            if frame_count is not None:
                chunk_record["frame_count"] = frame_count
            if result.get("usage_metadata") is not None:
                chunk_record["usage_metadata"] = result["usage_metadata"]
            if result.get("response_id"):
                chunk_record["response_id"] = result["response_id"]
            if result.get("model_version"):
                chunk_record["model_version"] = result["model_version"]
            save_record(chunk_record)
            return chunk_record

        if pending_indices:
            workers = max(1, concurrency)
            completed = 0
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(process_chunk, i): i for i in pending_indices}
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        c_start, c_end = chunk_bounds[idx]
                        error_record = {
                            "chunk": idx,
                            "chunk_start_s": c_start,
                            "chunk_end_s": c_end,
                            "items": [],
                            "error": {
                                "type": "request_failed",
                                "reason": str(exc).strip().splitlines()[0] if str(exc).strip() else repr(exc),
                            },
                        }
                        save_record(error_record)
                        log(f"Chunk {idx + 1}/{len(chunk_bounds)} failed after retries: {error_record['error']['reason']}")
                    completed += 1
                    log(f"OCR progress: {completed}/{len(pending_indices)} newly attempted")

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
            "backend": backend,
            "location": location,
            "base_url": base_url if backend == "llama-cpp" else "",
            "chunk_seconds": chunk_seconds,
            "chunk_json": chunk_json,
            "chunk_plan": chunk_plan,
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
            "local_frame_fps": local_frame_fps,
            "local_frame_height": local_frame_height,
            "local_max_frames": local_max_frames,
            "max_output_tokens": max_output_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "frequency_penalty": frequency_penalty,
            "request_timeout_s": request_timeout_s,
            "concurrency": concurrency,
            "rpm": rpm,
            "pricing": pricing,
        },
        stats=stats,
    )
    write_metadata(output, metadata)
    log(f"Saved: {output}")
    log(f"Metadata written: {metadata_path(output)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract chunkwise structured OCR sidecar text.")
    parser.add_argument("--video", required=True, help="Input video file.")
    parser.add_argument("--output", default="", help="Output JSON path. Defaults to episode ocr dir.")
    parser.add_argument(
        "--backend",
        choices=["gemini", "llama-cpp"],
        default="gemini",
        help="OCR backend. llama-cpp uses an OpenAI-compatible local /v1/chat/completions endpoint.",
    )
    parser.add_argument(
        "--preset",
        choices=preset_names("extract_gemini_chunk_ocr"),
        default=None,
        help="Optional named Gemini OCR settings preset.",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--location", default=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"))
    parser.add_argument("--base-url", default=DEFAULT_LLAMA_CPP_BASE_URL, help="llama.cpp chat completions URL.")
    parser.add_argument("--chunk-seconds", type=float, default=240.0)
    parser.add_argument(
        "--chunk-json",
        default="",
        help=(
            "Optional saved chunk boundaries JSON, such as vad_chunks.json "
            "(default VAD plan), vad_chunks_semantic_180.json (reviewed semantic plan), "
            "or a *_repair*.json repair plan."
        ),
    )
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Optional output width for inline video chunks. Default keeps source width.",
    )
    parser.add_argument("--audio-bitrate", default="24k")
    parser.add_argument("--crf", type=int, default=36)
    parser.add_argument("--max-inline-mb", type=float, default=19.5)
    parser.add_argument("--concurrency", type=int, default=1, help="Max concurrent OCR chunk workers.")
    parser.add_argument("--rpm", type=int, default=5, help="Max request attempts per minute across workers.")
    parser.add_argument("--max-request-retries", type=int, default=10, help="Max request attempts per chunk.")
    parser.add_argument("--only-chunks", default="", help="Comma-separated zero-based chunk indexes or ranges to attempt.")
    parser.add_argument("--force-chunks", default="", help="Comma-separated zero-based chunk indexes or ranges to re-run even if saved.")
    parser.add_argument("--stop-after-chunks", type=int, default=0, help="Stop after attempting N pending chunks.")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--local-frame-fps", type=float, default=0.5)
    parser.add_argument("--local-frame-height", type=int, default=720)
    parser.add_argument("--local-max-frames", type=int, default=15)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
    parser.add_argument("--request-timeout-s", type=int, default=600)
    parser.add_argument("--media-resolution", choices=["unspecified", "low", "medium", "high"], default=None)
    parser.add_argument("--thinking-level", choices=["unspecified", "minimal", "low", "medium", "high"], default=None)
    parser.add_argument("--thinking-budget", type=int, default=None)
    parser.add_argument("--input-price-per-1m", type=float, default=None)
    parser.add_argument("--output-price-per-1m", type=float, default=None)
    parser.add_argument("--vertex", action="store_true", help="Use Vertex AI instead of Gemini API.")
    args = parser.parse_args()

    model_override = args.model
    if model_override is None and args.backend == "llama-cpp":
        model_override = os.environ.get("LLAMA_CPP_OCR_MODEL", DEFAULT_LLAMA_CPP_OCR_MODEL)
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
    if args.backend == "llama-cpp" and args.temperature is None:
        resolved["temperature"] = 1.0

    output = args.output or _default_output_path(args.video)
    only_chunks = _parse_chunk_filter(args.only_chunks) if args.only_chunks else None
    force_chunks = _parse_chunk_filter(args.force_chunks) if args.force_chunks else None
    if args.backend == "llama-cpp":
        print("Using llama.cpp OCR backend")
    else:
        print("Using Vertex AI" if args.vertex else "Using Gemini API")
    if chosen_preset:
        print(f"Using preset: {chosen_preset}")
    run_chunk_ocr(
        video_path=args.video,
        output_path=output,
        backend=args.backend,
        model=resolved["model"],
        location=args.location,
        base_url=args.base_url,
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
        local_frame_fps=args.local_frame_fps,
        local_frame_height=args.local_frame_height,
        local_max_frames=args.local_max_frames,
        max_output_tokens=args.max_output_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        frequency_penalty=args.frequency_penalty,
        request_timeout_s=args.request_timeout_s,
        input_price_per_million=args.input_price_per_1m,
        output_price_per_million=args.output_price_per_1m,
        vertex=args.vertex,
        concurrency=args.concurrency,
        rpm=args.rpm,
        max_request_retries=args.max_request_retries,
        only_chunks=only_chunks,
        force_chunks=force_chunks,
        stop_after_chunks=args.stop_after_chunks,
    )


if __name__ == "__main__":
    main()
