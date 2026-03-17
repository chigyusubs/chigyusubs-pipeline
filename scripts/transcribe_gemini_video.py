#!/usr/bin/env python3
"""Video-only Gemini transcription with inline-safe compressed chunks.

This path avoids OCR entirely and sends low-bitrate video chunks inline to Gemini.
Output is plain Japanese text with `-- ` speaker-turn markers.

Supports bounded concurrency with RPM-aware rate limiting, quota-aware model
fallback, per-chunk file storage for safe concurrent writes, and a tight retry
policy that completes a full first pass before spending more effort on hard chunks.
"""

import argparse
import asyncio
import json
import os
import re
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chigyusubs.audio import extract_inline_video_chunk, get_duration
from chigyusubs.chunking import chunk_coverage_issues, describe_chunk_plan
from chigyusubs.env import load_repo_env
from chigyusubs.gemini_presets import preset_names, resolve_settings
from chigyusubs.metadata import (
    finish_run,
    lineage_output_path,
    metadata_path,
    preferred_manifest_path,
    read_artifact_metadata,
    start_run,
    update_preferred_manifest,
    write_metadata,
)
from chigyusubs.raw_chunk_sanity import inspect_chunk

load_repo_env()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transcribe_gemini import build_prompt, count_request_tokens, transcribe_chunk_result

# Thread-safe print lock
_print_lock = threading.Lock()


def log(msg: str = ""):
    with _print_lock:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# RPM-aware rate limiter
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Async rate limiter that enforces a maximum number of requests per minute.

    Uses a simple interval-based approach: ensures at least (60/rpm) seconds
    between consecutive requests.  Thread-safe via asyncio lock.
    """

    def __init__(self, rpm: int):
        self.interval = 60.0 / max(1, rpm)
        self._lock = asyncio.Lock()
        self._last_request = 0.0

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            wait = self._last_request + self.interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request = time.monotonic()


# ---------------------------------------------------------------------------
# Per-chunk file I/O
# ---------------------------------------------------------------------------

def _chunks_dir_for_raw(raw_json_path: str) -> Path:
    """Derive the per-chunk storage directory from the assembled JSON path.

    E.g. transcription/r3a7f01b_gemini_raw.json
      -> transcription/r3a7f01b_gemini_raw_chunks/
    """
    raw = Path(raw_json_path)
    return raw.parent / f"{raw.stem}_chunks"


def _chunk_file_path(chunks_dir: Path, chunk_index: int) -> Path:
    return chunks_dir / f"chunk_{chunk_index:03d}.json"


def _save_chunk_file(chunks_dir: Path, record: dict) -> Path:
    """Save a single chunk record to its own file."""
    chunks_dir.mkdir(parents=True, exist_ok=True)
    path = _chunk_file_path(chunks_dir, record["chunk"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return path


def _load_chunk_files(chunks_dir: Path) -> list[dict]:
    """Load all chunk records from the chunks directory."""
    if not chunks_dir.exists():
        return []
    records = []
    for path in sorted(chunks_dir.glob("chunk_*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                records.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue
    return records


def _assemble_raw_json(chunks_dir: Path, raw_json_path: str) -> list[dict]:
    """Assemble individual chunk files into the combined raw JSON artifact."""
    records = _load_chunk_files(chunks_dir)
    records.sort(key=lambda c: c.get("chunk", 0))
    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return records


# ---------------------------------------------------------------------------
# Pricing / cost helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Chunk boundary helpers
# ---------------------------------------------------------------------------

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


def _requested_output_anchor_from_meta(meta: dict | None) -> str | None:
    if not isinstance(meta, dict):
        return None
    anchor = meta.get("requested_output_anchor")
    if anchor is not None:
        return str(anchor)
    settings = meta.get("settings")
    if isinstance(settings, dict) and settings.get("requested_output_anchor") is not None:
        return str(settings["requested_output_anchor"])
    return None


def _candidate_matches_requested_output(candidate: Path, requested_output: Path) -> bool:
    meta = read_artifact_metadata(candidate)
    if not meta:
        return requested_output.parent == candidate.parent
    anchor = _requested_output_anchor_from_meta(meta)
    if anchor and anchor != str(requested_output):
        return False
    return True


def _successful_chunk_count_for_raw(candidate: Path) -> int:
    chunks_dir = _chunks_dir_for_raw(str(candidate))
    chunk_records = _load_chunk_files(chunks_dir)
    if chunk_records:
        return sum(1 for c in chunk_records if c.get("text") and not c.get("error"))
    if not candidate.exists():
        return 0
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    if not isinstance(payload, list):
        return 0
    return sum(1 for c in payload if isinstance(c, dict) and c.get("text") and not c.get("error"))


def _reuse_lineage_output_from_preferred(area_dir: Path, key: str, requested_output: Path) -> Path | None:
    manifest_path = preferred_manifest_path(area_dir)
    if not manifest_path.exists():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    name = payload.get(key)
    if not name:
        return None
    candidate = area_dir / str(name)
    if not candidate.exists():
        return None
    return candidate if _candidate_matches_requested_output(candidate, requested_output) else None


def _artifact_glob_for_preferred_key(key: str) -> str:
    if key == "gemini_raw":
        return "*_gemini_raw.json"
    if key == "gemini_cost_preview":
        return "*_gemini_cost_preview.json"
    return "*.json"


def _reuse_lineage_output_from_scan(area_dir: Path, key: str, requested_output: Path) -> Path | None:
    candidates = []
    for candidate in area_dir.glob(_artifact_glob_for_preferred_key(key)):
        if not candidate.is_file():
            continue
        if not _candidate_matches_requested_output(candidate, requested_output):
            continue
        success_count = _successful_chunk_count_for_raw(candidate)
        try:
            mtime = candidate.stat().st_mtime
        except OSError:
            mtime = 0.0
        candidates.append((success_count, mtime, candidate))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2].name))
    return candidates[-1][2]


# ---------------------------------------------------------------------------
# Quality detection helpers
# ---------------------------------------------------------------------------

def _max_line_length(text: str) -> int:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return max((len(ln) for ln in lines), default=0)


def _detect_loop_issue(text: str, max_line_length: int) -> dict | None:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(text) > 4000:
        return {
            "reason": "chunk_output_too_long",
            "char_count": len(text),
            "line_count": len(lines),
            "line_prefix": text[:200],
        }
    if len(lines) > 300:
        return {
            "reason": "too_many_lines",
            "char_count": len(text),
            "line_count": len(lines),
            "line_prefix": lines[0][:200] if lines else "",
        }
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


def _qa_retry_prompt(*, spoken_only: bool) -> str:
    lines = [
        "RETRY INSTRUCTION:",
        "Your previous output had transcript quality problems.",
        "Output only the transcript.",
        "Do not include analysis, headings, or commentary.",
        "Keep spoken lines in the `-- ...` format.",
        "Do not get stuck repeating a short reaction or phrase.",
    ]
    if spoken_only:
        lines.append("Do not include visual-only text or `[画面: ...]` lines.")
    return "\n".join(lines) + "\n"


def _red_chunk_report(chunk_record: dict) -> dict | None:
    report = inspect_chunk(chunk_record)
    return report if report["status"] == "red" else None


# ---------------------------------------------------------------------------
# Error classification for retry policy decisions
# ---------------------------------------------------------------------------

def _classify_api_error(e: Exception) -> str:
    """Classify a Gemini API error for retry policy decisions.

    Returns one of:
        'quota'     — 429 / RESOURCE_EXHAUSTED (model-level quota hit)
        'transient' — 500, INTERNAL, CANCELLED, network-like errors
        'timeout'   — DEADLINE_EXCEEDED, read timeout
        'other'     — unrecognized errors
    """
    msg = str(e)
    upper = msg.upper()
    lower = msg.lower()
    if "429" in msg or "RESOURCE_EXHAUSTED" in upper:
        return "quota"
    if "NO_PROGRESS_TIMEOUT" in upper:
        return "no_progress"
    if (
        "DEADLINE_EXCEEDED" in upper
        or "TIMED OUT" in upper
        or "read operation timed out" in lower
        or "timeout" in lower
    ):
        return "timeout"
    if "499" in msg or "CANCELLED" in upper:
        return "transient"
    if "500" in msg or "INTERNAL" in upper:
        return "transient"
    if any(kw in lower for kw in ("connectionreset", "connectionerror", "brokenpipe", "eof", "ssl")):
        return "transient"
    return "other"


# ---------------------------------------------------------------------------
# Single-chunk transcription with tight retry policy
# ---------------------------------------------------------------------------

def _transcribe_single_chunk(
    *,
    chunk_index: int,
    total_chunks: int,
    video_path: str,
    c_start: float,
    c_end: float,
    tmpdir: str,
    model: str,
    location: str,
    temperature: float,
    retry_temperature: float,
    loop_max_line_length: int,
    spoken_only: bool,
    media_resolution: str,
    thinking_level: str,
    thinking_budget: int | None,
    fps: float,
    width: int | None,
    audio_bitrate: str,
    crf: int,
    max_inline_mb: float,
    api_key: str,
    vertex: bool,
    count_tokens: bool,
    input_price_per_million: float | None,
    rate_limiter_acquire,
    first_token_timeout_s: float,
) -> dict:
    """Transcribe a single chunk with the maintained retry policy.

    Retry policy (per chunk):
    - One normal attempt.
    - If transient (network/500): one retry, same settings.
    - If quota (429/RESOURCE_EXHAUSTED): no retry, return quota error immediately.
    - If loop/red QA/timeout: one retry with retry_temperature, no rolling context.
    - If still bad after retry: return the failure record without raising.

    Returns a chunk record dict.  On unrecoverable failure the record includes
    an 'error' key with structured failure details.
    """
    i = chunk_index
    label = f"[{i + 1}/{total_chunks}]"
    chunk_path = os.path.join(tmpdir, f"chunk_{i}.mp4")

    log(f"{label} Encoding chunk {c_start / 60:.1f}-{c_end / 60:.1f}min ({c_end - c_start:.0f}s)...")
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
    if chunk_size_mb > max_inline_mb:
        return {
            "chunk": i,
            "chunk_start_s": c_start,
            "chunk_end_s": c_end,
            "chunk_size_mb": round(chunk_size_mb, 3),
            "error": {
                "type": "encoding",
                "reason": f"Chunk encoded to {chunk_size_mb:.2f} MB, above inline target {max_inline_mb:.2f} MB",
            },
        }

    prompt = build_prompt([], prev_context=None, include_visual_brackets=not spoken_only)
    video_bytes = Path(chunk_path).read_bytes()

    token_preview = None
    if count_tokens:
        try:
            token_preview = count_request_tokens(
                video_bytes, prompt, model, location,
                mime_type="video/mp4",
                media_resolution=media_resolution,
                thinking_level=thinking_level,
                thinking_budget=thinking_budget,
                api_key=api_key,
                vertex=vertex,
            )
        except Exception:
            pass  # token counting is best-effort

    attempt_history: list[dict] = []

    def _do_attempt(
        *,
        attempt_model: str,
        attempt_temp: float,
        attempt_prompt: str,
        attempt_label: str,
        reason: str,
    ) -> dict:
        """Make one API call and return an attempt record."""
        # Wait for rate limiter before sending the request
        rate_limiter_acquire()

        t0 = time.time()
        try:
            result = transcribe_chunk_result(
                video_bytes,
                attempt_prompt,
                attempt_model,
                location,
                max_retries=1,  # no internal retries — we handle retries here
                temperature=attempt_temp,
                mime_type="video/mp4",
                media_resolution=media_resolution,
                thinking_level=thinking_level,
                thinking_budget=thinking_budget,
                api_key=api_key,
                vertex=vertex,
                max_timeout_errors=1,
                max_rate_limit_errors=1,
                first_token_timeout_s=first_token_timeout_s,
            )
            elapsed = time.time() - t0
            text = result["text"]
            line_count = len([ln for ln in text.splitlines() if ln.strip()])
            log(f"{label} {attempt_label}: {line_count} lines, {len(text)} chars in {elapsed:.1f}s (model={attempt_model})")
            return {
                "status": "ok",
                "reason": reason,
                "model": attempt_model,
                "temperature": attempt_temp,
                "text": text,
                "elapsed_s": round(elapsed, 3),
                "usage_metadata": result.get("usage_metadata"),
                "response_id": result.get("response_id"),
                "model_version": result.get("model_version"),
            }
        except Exception as e:
            elapsed = time.time() - t0
            err_class = _classify_api_error(e)
            msg = str(e).strip().splitlines()[0] if str(e).strip() else repr(e)
            log(f"{label} {attempt_label}: {err_class} after {elapsed:.1f}s — {msg}")
            return {
                "status": "error",
                "reason": reason,
                "model": attempt_model,
                "temperature": attempt_temp,
                "error_class": err_class,
                "error_message": msg,
                "elapsed_s": round(elapsed, 3),
            }

    # --- Attempt 1: normal ---
    a1 = _do_attempt(
        attempt_model=model,
        attempt_temp=temperature,
        attempt_prompt=prompt,
        attempt_label="attempt 1",
        reason="initial",
    )
    attempt_history.append(a1)

    if a1["status"] == "error":
        err_class = a1["error_class"]

        if err_class == "quota":
            return _build_chunk_record(
                i, c_start, c_end, chunk_size_mb, media_resolution,
                thinking_level, thinking_budget, token_preview,
                attempt_history=attempt_history,
                error={"type": "quota", "reason": a1["error_message"], "model": model},
            )

        if err_class == "no_progress":
            return _build_chunk_record(
                i, c_start, c_end, chunk_size_mb, media_resolution,
                thinking_level, thinking_budget, token_preview,
                attempt_history=attempt_history,
                error={"type": "no_progress", "reason": a1["error_message"], "model": model},
            )

        if err_class == "transient":
            log(f"{label} Retrying (transient error)...")
            a2 = _do_attempt(
                attempt_model=model,
                attempt_temp=temperature,
                attempt_prompt=prompt,
                attempt_label="retry (transient)",
                reason="transient_retry",
            )
            attempt_history.append(a2)
            if a2["status"] == "ok":
                a1 = a2
            else:
                if a2.get("error_class") == "quota":
                    return _build_chunk_record(
                        i, c_start, c_end, chunk_size_mb, media_resolution,
                        thinking_level, thinking_budget, token_preview,
                        attempt_history=attempt_history,
                        error={"type": "quota", "reason": a2["error_message"], "model": model},
                    )
                return _build_chunk_record(
                    i, c_start, c_end, chunk_size_mb, media_resolution,
                    thinking_level, thinking_budget, token_preview,
                    attempt_history=attempt_history,
                    error={"type": a2["error_class"], "reason": a2["error_message"], "model": model},
                )

        elif err_class in ("timeout", "other"):
            log(f"{label} Retrying (temp={retry_temperature})...")
            retry_prompt = prompt + "\n\n" + _retry_prompt()
            a2 = _do_attempt(
                attempt_model=model,
                attempt_temp=retry_temperature,
                attempt_prompt=retry_prompt,
                attempt_label="retry (temp bump)",
                reason="timeout_retry",
            )
            attempt_history.append(a2)
            if a2["status"] == "ok":
                a1 = a2
            else:
                if a2.get("error_class") == "quota":
                    return _build_chunk_record(
                        i, c_start, c_end, chunk_size_mb, media_resolution,
                        thinking_level, thinking_budget, token_preview,
                        attempt_history=attempt_history,
                        error={"type": "quota", "reason": a2["error_message"], "model": model},
                    )
                return _build_chunk_record(
                    i, c_start, c_end, chunk_size_mb, media_resolution,
                    thinking_level, thinking_budget, token_preview,
                    attempt_history=attempt_history,
                    error={"type": a2["error_class"], "reason": a2["error_message"], "model": model},
                )

    if a1["status"] == "error":
        return _build_chunk_record(
            i, c_start, c_end, chunk_size_mb, media_resolution,
            thinking_level, thinking_budget, token_preview,
            attempt_history=attempt_history,
            error={"type": a1["error_class"], "reason": a1["error_message"], "model": model},
        )

    # --- We got text.  Check quality. ---
    text = a1["text"]

    # Loop detection
    loop_issue = _detect_loop_issue(text, loop_max_line_length)
    if loop_issue:
        log(f"{label} Loop detected ({loop_issue['reason']}); retrying with temp={retry_temperature}...")
        retry_prompt = build_prompt([], prev_context=None, include_visual_brackets=not spoken_only) + "\n\n" + _retry_prompt()
        a2 = _do_attempt(
            attempt_model=model,
            attempt_temp=retry_temperature,
            attempt_prompt=retry_prompt,
            attempt_label="retry (loop)",
            reason="loop_retry",
        )
        attempt_history.append(a2)
        if a2["status"] == "ok":
            retried_loop = _detect_loop_issue(a2["text"], loop_max_line_length)
            if retried_loop is None or _max_line_length(a2["text"]) < _max_line_length(text):
                text = a2["text"]
                a1 = a2
                loop_issue = retried_loop
        if loop_issue is not None:
            return _build_chunk_record(
                i, c_start, c_end, chunk_size_mb, media_resolution,
                thinking_level, thinking_budget, token_preview,
                attempt_history=attempt_history,
                error={"type": "loop", "reason": loop_issue["reason"], "model": model, "details": loop_issue},
            )

    # Red-chunk QA
    temp_record = {
        "chunk": i, "chunk_start_s": c_start, "chunk_end_s": c_end, "text": text,
        "chunk_size_mb": round(chunk_size_mb, 3),
        "media_resolution": media_resolution,
        "thinking_level": thinking_level,
        "thinking_budget": thinking_budget,
    }
    qa_red = _red_chunk_report(temp_record)
    if qa_red is not None:
        log(f"{label} Red QA ({', '.join(qa_red['issue_codes'])}); retrying with temp={retry_temperature}...")
        retry_prompt = build_prompt([], prev_context=None, include_visual_brackets=not spoken_only) + "\n\n" + _qa_retry_prompt(spoken_only=spoken_only)
        a2 = _do_attempt(
            attempt_model=model,
            attempt_temp=retry_temperature,
            attempt_prompt=retry_prompt,
            attempt_label="retry (QA red)",
            reason="qa_retry",
        )
        attempt_history.append(a2)
        if a2["status"] == "ok":
            retry_loop = _detect_loop_issue(a2["text"], loop_max_line_length)
            retry_temp_record = dict(temp_record)
            retry_temp_record["text"] = a2["text"]
            retry_qa = _red_chunk_report(retry_temp_record)
            if retry_loop is None and retry_qa is None:
                text = a2["text"]
                a1 = a2
                qa_red = None
        if qa_red is not None:
            return _build_chunk_record(
                i, c_start, c_end, chunk_size_mb, media_resolution,
                thinking_level, thinking_budget, token_preview,
                attempt_history=attempt_history,
                error={"type": "qa_red", "reason": ", ".join(qa_red["issue_codes"]), "model": model, "details": qa_red},
            )

    # --- Success ---
    return _build_chunk_record(
        i, c_start, c_end, chunk_size_mb, media_resolution,
        thinking_level, thinking_budget, token_preview,
        text=text,
        attempt_history=attempt_history,
        model_used=a1.get("model", model),
        usage_metadata=a1.get("usage_metadata"),
        response_id=a1.get("response_id"),
        model_version=a1.get("model_version"),
    )


def _build_chunk_record(
    chunk_index: int,
    c_start: float,
    c_end: float,
    chunk_size_mb: float,
    media_resolution: str,
    thinking_level: str,
    thinking_budget: int | None,
    token_preview: dict | None,
    *,
    text: str | None = None,
    attempt_history: list[dict] | None = None,
    error: dict | None = None,
    model_used: str | None = None,
    usage_metadata: dict | None = None,
    response_id: str | None = None,
    model_version: str | None = None,
) -> dict:
    record: dict[str, Any] = {
        "chunk": chunk_index,
        "chunk_start_s": c_start,
        "chunk_end_s": c_end,
        "chunk_size_mb": round(chunk_size_mb, 3),
        "media_resolution": media_resolution,
        "thinking_level": thinking_level,
        "thinking_budget": thinking_budget,
    }
    if text is not None:
        record["text"] = text
    if model_used:
        record["model_used"] = model_used
    if token_preview:
        record["prompt_token_preview"] = token_preview
    if usage_metadata is not None:
        record["usage_metadata"] = usage_metadata
    if response_id:
        record["response_id"] = response_id
    if model_version:
        record["model_version"] = model_version
    if attempt_history:
        # Strip text from the attempt whose output was chosen as the top-level
        # text — it's already there and duplicating it is just noise.  Keep text
        # on failed/superseded attempts so the audit trail shows what changed.
        cleaned = []
        for a in attempt_history:
            if text is not None and a.get("text") == text:
                a = {k: v for k, v in a.items() if k != "text"}
            cleaned.append(a)
        record["attempt_history"] = cleaned
    if error:
        record["error"] = error
    return record


# ---------------------------------------------------------------------------
# Concurrent orchestrator
# ---------------------------------------------------------------------------

def run_video_transcription(
    *,
    video_path: str,
    output_path: str,
    model: str,
    location: str,
    chunk_seconds: float,
    chunk_json: str,
    fps: float,
    width: int | None,
    audio_bitrate: str,
    crf: int,
    max_inline_mb: float,
    rolling_context_chunks: int,
    temperature: float,
    retry_temperature: float,
    loop_max_line_length: int,
    max_request_retries: int,
    max_timeout_errors: int,
    max_rate_limit_errors: int,
    stop_after_chunks: int,
    spoken_only: bool,
    preset_name: str | None,
    media_resolution: str,
    thinking_level: str,
    thinking_budget: int | None,
    preview_cost: bool,
    count_tokens: bool,
    input_price_per_million: float | None,
    output_price_per_million: float | None,
    api_key: str = "",
    vertex: bool = False,
    concurrency: int = 5,
    fallback_models: list[str] | None = None,
    rpm: int = 5,
    first_token_timeout_s: float = 60.0,
):
    run = start_run("transcribe_gemini_video")
    requested_output = Path(output_path)
    out_dir = requested_output.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_dir.name == "transcription" and not requested_output.name.startswith(f"{run['run_id']}_"):
        reused_raw = _reuse_lineage_output_from_preferred(out_dir, "gemini_raw", requested_output)
        if reused_raw is None:
            reused_raw = _reuse_lineage_output_from_scan(out_dir, "gemini_raw", requested_output)
        if reused_raw is not None:
            run["run_id"] = reused_raw.name.split("_", 1)[0]
            raw_json_path = str(reused_raw)
            preview_json_path = str(out_dir / f"{run['run_id']}_gemini_cost_preview.json")
        else:
            raw_json_path = str(lineage_output_path(out_dir, artifact_type="gemini_raw", run=run, suffix=".json"))
            preview_json_path = str(lineage_output_path(out_dir, artifact_type="gemini_cost_preview", run=run, suffix=".json"))
    else:
        stem = requested_output.stem
        raw_json_path = str(out_dir / f"{stem}_gemini_raw.json")
        preview_json_path = str(out_dir / f"{stem}_gemini_cost_preview.json")

    chunks_dir = _chunks_dir_for_raw(raw_json_path)

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
    chunk_plan = describe_chunk_plan(chunk_json, chunk_bounds) if chunk_json else None

    # Rolling context is incompatible with concurrency
    effective_concurrency = max(1, concurrency)
    if effective_concurrency > 1 and rolling_context_chunks > 0:
        log(f"Note: rolling_context_chunks={rolling_context_chunks} is ignored in concurrent mode (concurrency={effective_concurrency}).")
        rolling_context_chunks = 0

    log(f"Video: {video_path}")
    log(f"Output: {output_path}")
    log(f"Chunks dir: {chunks_dir}")
    log(f"Duration: {duration:.1f}s")
    log(f"Concurrency: {effective_concurrency} workers, {rpm} RPM")
    if fallback_models:
        log(f"Model chain: {model} -> {' -> '.join(fallback_models)}")
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

    # --- Cost preview mode (sequential, like before) ---
    if preview_cost:
        _run_cost_preview(
            video_path=video_path,
            chunk_bounds=chunk_bounds,
            model=model,
            location=location,
            fps=fps,
            width=width,
            audio_bitrate=audio_bitrate,
            crf=crf,
            max_inline_mb=max_inline_mb,
            media_resolution=media_resolution,
            thinking_level=thinking_level,
            thinking_budget=thinking_budget,
            api_key=api_key,
            vertex=vertex,
            pricing=pricing,
            preview_json_path=preview_json_path,
            raw_json_path=raw_json_path,
            requested_output=requested_output,
            run=run,
            chunk_json=chunk_json,
            chunk_plan=chunk_plan,
            chunk_seconds=chunk_seconds,
            rolling_context_chunks=rolling_context_chunks,
            spoken_only=spoken_only,
            preset_name=preset_name,
            count_tokens=count_tokens,
        )
        return

    # --- Resume from existing chunk files ---
    existing_chunks = _load_chunk_files(chunks_dir)
    done_indices: set[int] = set()

    # Also check for an existing assembled JSON (backward compat with old runs)
    if not existing_chunks and os.path.exists(raw_json_path):
        with open(raw_json_path, "r", encoding="utf-8") as f:
            old_chunks = json.load(f)
        # Migrate old assembled format to per-chunk files
        for c in old_chunks:
            if c.get("text") and not c.get("error"):
                _save_chunk_file(chunks_dir, c)
        existing_chunks = _load_chunk_files(chunks_dir)
        log(f"Migrated {len(existing_chunks)} chunks from existing assembled JSON to per-chunk files.")

    # Check for red chunks in existing completed data
    red_chunks = []
    for saved_chunk in existing_chunks:
        if saved_chunk.get("error"):
            continue  # failed chunks from a previous run are retryable
        report = _red_chunk_report(saved_chunk)
        if report is not None:
            red_chunks.append({"chunk": report["chunk"], "issue_codes": report["issue_codes"], "reasons": report["reasons"]})
    if red_chunks:
        raise RuntimeError(
            "Existing raw lineage contains red chunks; repair before resuming: "
            + ", ".join(
                f"chunk {item['chunk']} ({'/'.join(item['issue_codes'])})"
                for item in red_chunks
            )
        )

    # Find successfully completed chunks (those with text and no error)
    for c in existing_chunks:
        if c.get("text") and not c.get("error"):
            done_indices.add(c["chunk"])
    if done_indices:
        log(f"Resuming: {len(done_indices)} chunks already completed, {len(chunk_bounds) - len(done_indices)} remaining")

    # Figure out which chunks still need work
    pending_indices = [i for i in range(len(chunk_bounds)) if i not in done_indices]
    if stop_after_chunks > 0:
        pending_indices = pending_indices[:stop_after_chunks]

    if not pending_indices:
        log("All chunks already completed.")
    else:
        # --- Run transcription passes ---
        model_chain = [model] + (fallback_models or [])
        remaining_indices = list(pending_indices)

        for model_idx, current_model in enumerate(model_chain):
            if not remaining_indices:
                break

            is_fallback = model_idx > 0
            if is_fallback:
                log()
                log(f"=== Falling back to {current_model} for {len(remaining_indices)} remaining chunks ===")

            results = _run_concurrent_pass(
                indices=remaining_indices,
                chunk_bounds=chunk_bounds,
                video_path=video_path,
                model=current_model,
                location=location,
                temperature=temperature,
                retry_temperature=retry_temperature,
                loop_max_line_length=loop_max_line_length,
                spoken_only=spoken_only,
                media_resolution=media_resolution,
                thinking_level=thinking_level,
                thinking_budget=thinking_budget,
                fps=fps,
                width=width,
                audio_bitrate=audio_bitrate,
                crf=crf,
                max_inline_mb=max_inline_mb,
                api_key=api_key,
                vertex=vertex,
                count_tokens=count_tokens,
                input_price_per_million=input_price_per_million,
                concurrency=effective_concurrency,
                rpm=rpm,
                chunks_dir=chunks_dir,
                first_token_timeout_s=first_token_timeout_s,
            )

            succeeded = []
            failed_quota = []
            failed_other = []

            for r in results:
                idx = r["chunk"]
                if r.get("error"):
                    err_type = r["error"].get("type", "")
                    if err_type == "quota":
                        failed_quota.append(idx)
                    else:
                        failed_other.append(idx)
                else:
                    succeeded.append(idx)

            log()
            log(f"Pass on {current_model}: {len(succeeded)} succeeded, {len(failed_quota)} quota-blocked, {len(failed_other)} failed")

            # Remaining = only quota-blocked chunks advance to next model
            remaining_indices = failed_quota

            if failed_other:
                log(f"Hard failures on chunks: {failed_other}")
                log("These chunks need manual repair/splitting before the next run.")

        if remaining_indices:
            log(f"{len(remaining_indices)} chunks still quota-blocked with no more fallback models.")

    # --- Assemble final JSON from chunk files ---
    all_chunks = _assemble_raw_json(chunks_dir, raw_json_path)
    if Path(raw_json_path).parent.name == "transcription" and stop_after_chunks <= 0:
        update_preferred_manifest(Path(raw_json_path).parent, gemini_raw=Path(raw_json_path).name)

    # --- Final summary ---
    success_chunks = [c for c in all_chunks if c.get("text") and not c.get("error")]
    error_chunks = [c for c in all_chunks if c.get("error")]

    total_lines = sum(len([ln for ln in c.get("text", "").splitlines() if ln.strip()]) for c in success_chunks)
    total_chars = sum(len(c.get("text", "")) for c in success_chunks)
    usage_records = [c["usage_metadata"] for c in success_chunks if isinstance(c.get("usage_metadata"), dict)]
    prompt_preview_tokens = sum(int((c.get("prompt_token_preview") or {}).get("total_tokens") or 0) for c in success_chunks)
    prompt_preview_cost = _cost_for_tokens(prompt_preview_tokens or None, pricing.get("input_price_per_million_usd"))

    models_used = {}
    for c in success_chunks:
        m = c.get("model_used", model)
        models_used[m] = models_used.get(m, 0) + 1

    metadata = finish_run(
        run,
        inputs={"video": video_path},
        outputs={"video_gemini_json": raw_json_path, "chunks_dir": str(chunks_dir)},
        settings={
            "model": model,
            "fallback_models": fallback_models or [],
            "location": location,
            "chunk_seconds": chunk_seconds,
            "chunk_json": chunk_json,
            "chunk_plan": chunk_plan,
            "fps": fps,
            "width": width,
            "audio_bitrate": audio_bitrate,
            "crf": crf,
            "max_inline_mb": max_inline_mb,
            "rolling_context_chunks": rolling_context_chunks,
            "temperature": temperature,
            "retry_temperature": retry_temperature,
            "loop_max_line_length": loop_max_line_length,
            "concurrency": effective_concurrency,
            "rpm": rpm,
            "first_token_timeout_s": first_token_timeout_s,
            "stop_after_chunks": stop_after_chunks,
            "spoken_only": spoken_only,
            "preset": preset_name,
            "media_resolution": media_resolution,
            "thinking_level": thinking_level,
            "thinking_budget": thinking_budget,
            "preview_cost": preview_cost,
            "count_tokens": count_tokens,
            "pricing": pricing,
            "requested_output_anchor": str(requested_output),
        },
        stats={
            "duration_seconds": round(duration, 3),
            "chunks_total": len(chunk_bounds),
            "chunks_succeeded": len(success_chunks),
            "chunks_failed": len(error_chunks),
            "models_used": models_used,
            "lines": total_lines,
            "characters": total_chars,
            "prompt_token_preview_count": prompt_preview_tokens if count_tokens else None,
            "prompt_token_preview_cost_usd": _round_cost(prompt_preview_cost) if count_tokens else None,
            "usage": _usage_cost_summary(usage_records, pricing),
        },
    )
    write_metadata(raw_json_path, metadata)
    if Path(raw_json_path).parent.name == "transcription" and stop_after_chunks <= 0:
        update_preferred_manifest(Path(raw_json_path).parent, gemini_raw=Path(raw_json_path).name)

    log()
    log(f"Saved: {raw_json_path}")
    log(f"Chunks dir: {chunks_dir}")
    log(f"Metadata written: {metadata_path(raw_json_path)}")
    log(f"Chunks: {len(success_chunks)}/{len(chunk_bounds)} succeeded")
    if models_used:
        log(f"Models used: {models_used}")
    if error_chunks:
        log(f"Failed chunks: {[c['chunk'] for c in error_chunks]}")
        log("Run again to retry, or repair/split failed chunks first.")


def _run_concurrent_pass(
    *,
    indices: list[int],
    chunk_bounds: list[tuple[float, float]],
    video_path: str,
    model: str,
    location: str,
    temperature: float,
    retry_temperature: float,
    loop_max_line_length: int,
    spoken_only: bool,
    media_resolution: str,
    thinking_level: str,
    thinking_budget: int | None,
    fps: float,
    width: int | None,
    audio_bitrate: str,
    crf: int,
    max_inline_mb: float,
    api_key: str,
    vertex: bool,
    count_tokens: bool,
    input_price_per_million: float | None,
    concurrency: int,
    rpm: int,
    chunks_dir: Path,
    first_token_timeout_s: float,
) -> list[dict]:
    """Run a concurrent transcription pass over the given chunk indices.

    Returns a list of chunk records (one per index).  Each record is either
    a success (has 'text', no 'error') or a failure (has 'error').

    Each completed chunk is saved to its own file immediately.
    """
    total = len(chunk_bounds)

    # Detect quota exhaustion: if we see N consecutive quota errors across workers,
    # stop sending new chunks on this model.
    quota_exhausted = threading.Event()
    systemic_no_progress = threading.Event()
    consecutive_quota = [0]
    quota_lock = threading.Lock()
    QUOTA_THRESHOLD = 3  # stop after 3 consecutive quota errors

    with tempfile.TemporaryDirectory() as tmpdir:
        rate_limiter = _RateLimiter(rpm)

        # Bridge: synchronous callable that blocks until rate limiter allows.
        # Called from worker threads; internally runs the async acquire.
        _rl_loop = asyncio.new_event_loop()
        _rl_thread = threading.Thread(target=_rl_loop.run_forever, daemon=True)
        _rl_thread.start()

        def sync_acquire():
            future = asyncio.run_coroutine_threadsafe(rate_limiter.acquire(), _rl_loop)
            future.result()

        def process_chunk(idx: int) -> dict:
            if systemic_no_progress.is_set():
                raise RuntimeError("NO_PROGRESS_TIMEOUT: aborting queued work after systemic no-progress failure.")
            if quota_exhausted.is_set():
                c_start, c_end = chunk_bounds[idx]
                record = _build_chunk_record(
                    idx, c_start, c_end, 0.0, media_resolution,
                    thinking_level, thinking_budget, None,
                    error={"type": "quota", "reason": "skipped — model quota exhausted", "model": model},
                )
                _save_chunk_file(chunks_dir, record)
                return record

            c_start, c_end = chunk_bounds[idx]
            record = _transcribe_single_chunk(
                chunk_index=idx,
                total_chunks=total,
                video_path=video_path,
                c_start=c_start,
                c_end=c_end,
                tmpdir=tmpdir,
                model=model,
                location=location,
                temperature=temperature,
                retry_temperature=retry_temperature,
                loop_max_line_length=loop_max_line_length,
                spoken_only=spoken_only,
                media_resolution=media_resolution,
                thinking_level=thinking_level,
                thinking_budget=thinking_budget,
                fps=fps,
                width=width,
                audio_bitrate=audio_bitrate,
                crf=crf,
                max_inline_mb=max_inline_mb,
                api_key=api_key,
                vertex=vertex,
                count_tokens=count_tokens,
                input_price_per_million=input_price_per_million,
                rate_limiter_acquire=sync_acquire,
                first_token_timeout_s=first_token_timeout_s,
            )

            # Save immediately to per-chunk file
            _save_chunk_file(chunks_dir, record)

            # Track quota exhaustion
            err = record.get("error")
            if err and err.get("type") == "quota":
                with quota_lock:
                    consecutive_quota[0] += 1
                    if consecutive_quota[0] >= QUOTA_THRESHOLD:
                        quota_exhausted.set()
                        log(f"Model {model} quota appears exhausted after {consecutive_quota[0]} consecutive quota errors.")
            else:
                with quota_lock:
                    consecutive_quota[0] = 0

            return record

        executor = ThreadPoolExecutor(max_workers=concurrency)
        try:
            futures = {executor.submit(process_chunk, idx): idx for idx in indices}
            results = []
            initial_indices = set(indices[: min(concurrency, len(indices))])
            initial_finished: set[int] = set()
            initial_no_progress: set[int] = set()
            saw_real_response = False

            for future in as_completed(futures):
                idx = futures[future]
                record = future.result()
                results.append(record)

                err = record.get("error")
                if not (err and err.get("type") == "no_progress"):
                    saw_real_response = True

                if idx in initial_indices:
                    initial_finished.add(idx)
                    if err and err.get("type") == "no_progress":
                        initial_no_progress.add(idx)
                    if (
                        not saw_real_response
                        and initial_finished == initial_indices
                        and initial_no_progress == initial_indices
                    ):
                        systemic_no_progress.set()
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise RuntimeError(
                            "No chunk received a first token within the watchdog window. "
                            "This looks like a blocked API/network path, often from sandboxed execution. "
                            "Abort and rerun with network access."
                        )
        finally:
            executor.shutdown(wait=False)
            _rl_loop.call_soon_threadsafe(_rl_loop.stop)

    return results


# ---------------------------------------------------------------------------
# Cost preview (unchanged, still sequential)
# ---------------------------------------------------------------------------

def _run_cost_preview(
    *,
    video_path: str,
    chunk_bounds: list[tuple[float, float]],
    model: str,
    location: str,
    fps: float,
    width: int | None,
    audio_bitrate: str,
    crf: int,
    max_inline_mb: float,
    media_resolution: str,
    thinking_level: str,
    thinking_budget: int | None,
    api_key: str,
    vertex: bool,
    pricing: dict,
    preview_json_path: str,
    raw_json_path: str,
    requested_output: Path,
    run: dict,
    chunk_json: str,
    chunk_plan: dict | None,
    chunk_seconds: float,
    rolling_context_chunks: int,
    spoken_only: bool,
    preset_name: str | None,
    count_tokens: bool,
):
    preview_chunks: list[dict] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, (c_start, c_end) in enumerate(chunk_bounds):
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

            prompt = build_prompt([], prev_context=None, include_visual_brackets=not spoken_only)
            video_bytes = Path(chunk_path).read_bytes()
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
            "chunk_plan": chunk_plan,
            "fps": fps,
            "width": None,
            "audio_bitrate": "24k",
            "crf": 36,
            "max_inline_mb": 14.0,
            "rolling_context_chunks": rolling_context_chunks,
            "spoken_only": spoken_only,
            "preset": preset_name,
            "media_resolution": media_resolution,
            "thinking_level": thinking_level,
            "thinking_budget": thinking_budget,
            "preview_cost": True,
            "count_tokens": count_tokens,
            "pricing": pricing,
            "requested_output_anchor": str(requested_output),
        },
        stats=preview_payload["summary"],
    )
    write_metadata(preview_json_path, metadata)
    if Path(preview_json_path).parent.name == "transcription":
        update_preferred_manifest(Path(preview_json_path).parent, gemini_cost_preview=Path(preview_json_path).name)
    log(f"Saved cost preview: {preview_json_path}")
    log(f"Metadata written: {metadata_path(preview_json_path)}")


def main():
    parser = argparse.ArgumentParser(description="Video-only Gemini transcription with inline video chunks.")
    parser.add_argument("--video", required=True, help="Input video file.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument(
        "--preset",
        choices=preset_names("transcribe_gemini_video"),
        default=None,
        help="Optional named Gemini settings preset.",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--location", default=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"))
    parser.add_argument("--chunk-seconds", type=float, default=600.0)
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
    parser.add_argument("--max-inline-mb", type=float, default=14.0)
    parser.add_argument("--rolling-context-chunks", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--retry-temperature", type=float, default=None)
    parser.add_argument("--loop-max-line-length", type=int, default=300)
    parser.add_argument("--max-request-retries", type=int, default=None)
    parser.add_argument("--max-timeout-errors", type=int, default=None)
    parser.add_argument("--max-rate-limit-errors", type=int, default=None)
    parser.add_argument(
        "--stop-after-chunks",
        type=int,
        default=0,
        help="Stop cleanly after writing this many newly completed chunks. Useful for one-chunk smoke tests.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Max concurrent chunk workers (default: 5). Use 1 for sequential mode.",
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=None,
        help="Max API requests per minute per model (default: 5, matching free tier).",
    )
    parser.add_argument(
        "--first-token-timeout-s",
        type=float,
        default=None,
        help="Abort a request attempt if no first token arrives within this many seconds (default: 60).",
    )
    parser.add_argument(
        "--fallback-models",
        default=None,
        help="Comma-separated fallback model chain for quota exhaustion (e.g. 'gemini-3-flash-preview,gemini-2.5-flash').",
    )
    prompt_mode = parser.add_mutually_exclusive_group()
    prompt_mode.add_argument(
        "--spoken-only",
        dest="spoken_only",
        action="store_const",
        const=True,
        default=None,
        help="Ask Gemini for spoken dialogue only and do not request [画面: ...] visual cue lines.",
    )
    prompt_mode.add_argument(
        "--spoken-plus-visual",
        dest="spoken_only",
        action="store_const",
        const=False,
        help="Ask Gemini for spoken dialogue plus selective [画面: ...] visual cue lines.",
    )
    parser.add_argument(
        "--media-resolution",
        choices=["unspecified", "low", "medium", "high"],
        default=None,
        help="Gemini media resolution hint for multimodal processing.",
    )
    parser.add_argument(
        "--thinking-level",
        choices=["unspecified", "minimal", "low", "medium", "high"],
        default=None,
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

    model_override = args.model
    if model_override is None and not args.preset:
        model_override = os.environ.get("GEMINI_MODEL")

    fallback_models_override = None
    if args.fallback_models is not None:
        fallback_models_override = [m.strip() for m in args.fallback_models.split(",") if m.strip()]

    resolved, chosen_preset = resolve_settings(
        "transcribe_gemini_video",
        args.preset,
        {
            "model": model_override,
            "temperature": args.temperature,
            "retry_temperature": args.retry_temperature,
            "spoken_only": args.spoken_only,
            "media_resolution": args.media_resolution,
            "thinking_level": args.thinking_level,
            "thinking_budget": args.thinking_budget,
            "rolling_context_chunks": args.rolling_context_chunks,
            "max_request_retries": args.max_request_retries,
            "max_timeout_errors": args.max_timeout_errors,
            "max_rate_limit_errors": args.max_rate_limit_errors,
            "concurrency": args.concurrency,
            "fallback_models": fallback_models_override,
            "rpm": args.rpm,
            "first_token_timeout_s": args.first_token_timeout_s,
        },
    )

    if args.vertex:
        print("Using Vertex AI", flush=True)
    else:
        print("Using Gemini API", flush=True)
    if chosen_preset:
        print(f"Using preset: {chosen_preset}", flush=True)

    run_video_transcription(
        video_path=args.video,
        output_path=args.output,
        model=resolved["model"],
        location=args.location,
        chunk_seconds=args.chunk_seconds,
        chunk_json=args.chunk_json,
        fps=args.fps,
        width=args.width,
        audio_bitrate=args.audio_bitrate,
        crf=args.crf,
        max_inline_mb=args.max_inline_mb,
        rolling_context_chunks=resolved["rolling_context_chunks"],
        temperature=resolved["temperature"],
        retry_temperature=resolved["retry_temperature"],
        loop_max_line_length=args.loop_max_line_length,
        max_request_retries=resolved["max_request_retries"],
        max_timeout_errors=resolved["max_timeout_errors"],
        max_rate_limit_errors=resolved["max_rate_limit_errors"],
        stop_after_chunks=args.stop_after_chunks,
        spoken_only=resolved["spoken_only"],
        preset_name=chosen_preset,
        media_resolution=resolved["media_resolution"],
        thinking_level=resolved["thinking_level"],
        thinking_budget=resolved["thinking_budget"],
        preview_cost=args.preview_cost,
        count_tokens=args.count_tokens,
        input_price_per_million=args.input_price_per_1m,
        output_price_per_million=args.output_price_per_1m,
        vertex=args.vertex,
        concurrency=resolved["concurrency"],
        fallback_models=resolved["fallback_models"],
        rpm=resolved["rpm"],
        first_token_timeout_s=resolved["first_token_timeout_s"],
    )


if __name__ == "__main__":
    main()
