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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chigyusubs.audio import extract_inline_video_chunk, get_duration
from chigyusubs.metadata import finish_run, metadata_path, start_run, write_metadata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transcribe_gemini import build_prompt, transcribe_chunk


def log(msg: str = ""):
    print(msg, flush=True)


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
):
    run = start_run("transcribe_gemini_video")
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(output_path).stem
    raw_json_path = str(out_dir / f"{stem}_gemini_raw.json")

    duration = get_duration(video_path)
    if chunk_json:
        chunk_bounds = _load_chunk_bounds(chunk_json)
    else:
        chunk_bounds = _default_chunk_bounds(duration, chunk_seconds)

    log(f"Video: {video_path}")
    log(f"Output: {output_path}")
    log(f"Duration: {duration:.1f}s")
    if chunk_json:
        log(f"Chunks: {len(chunk_bounds)} from {chunk_json}")
    else:
        log(f"Chunks: {len(chunk_bounds)} x {chunk_seconds:.0f}s")

    all_chunks: list[dict] = []
    start_chunk = 0
    if os.path.exists(raw_json_path):
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
                include_visual_brackets=True,
            )
            log("Sending video chunk to Gemini...")
            video_bytes = Path(chunk_path).read_bytes()
            text = transcribe_chunk(
                video_bytes,
                prompt,
                model,
                location,
                max_retries=20,
                temperature=temperature,
                mime_type="video/mp4",
            )
            retry_info = None
            issue = _detect_loop_issue(text, loop_max_line_length)
            if issue:
                log(f"Detected loop-like chunk output ({issue['reason']}); retrying with temp={retry_temperature:.2f} and no rolling context...")
                retry_prompt = build_prompt([], prev_context=None, include_visual_brackets=True) + "\n\n" + _retry_prompt()
                retried = transcribe_chunk(
                    video_bytes,
                    retry_prompt,
                    model,
                    location,
                    max_retries=20,
                    temperature=retry_temperature,
                    mime_type="video/mp4",
                )
                retry_issue = _detect_loop_issue(retried, loop_max_line_length)
                retry_info = {
                    "trigger": issue,
                    "retry_temperature": retry_temperature,
                    "retry_issue": retry_issue,
                    "retry_used": retry_issue is None or _max_line_length(retried) < _max_line_length(text),
                }
                if retry_info["retry_used"]:
                    text = retried

            line_count = len([ln for ln in text.splitlines() if ln.strip()])
            log(f"Result: {line_count} lines, {len(text)} chars")

            chunk_record = {
                "chunk": i,
                "chunk_start_s": c_start,
                "chunk_end_s": c_end,
                "text": text,
                "chunk_size_mb": round(chunk_size_mb, 3),
            }
            if retry_info:
                chunk_record["retry"] = retry_info
            all_chunks.append(chunk_record)

            with open(raw_json_path, "w", encoding="utf-8") as f:
                json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    total_lines = sum(len([ln for ln in c.get("text", "").splitlines() if ln.strip()]) for c in all_chunks)
    total_chars = sum(len(c.get("text", "")) for c in all_chunks)

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
        },
        stats={
            "duration_seconds": round(duration, 3),
            "chunks": len(all_chunks),
            "lines": total_lines,
            "characters": total_chars,
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
    args = parser.parse_args()

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
    )


if __name__ == "__main__":
    main()
