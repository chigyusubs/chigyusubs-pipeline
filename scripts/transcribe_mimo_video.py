#!/usr/bin/env python3
"""Transcribe video using Xiaomi MiMo v2 Omni via OpenAI-compatible API.

Sends compressed video chunks with base64 encoding to the MiMo API.
Outputs Gemini-compatible raw JSON for apples-to-apples comparison.

Requires: XIAOMI_API_KEY in .env or environment.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chigyusubs.audio import extract_inline_video_chunk, get_duration
from chigyusubs.chunking import find_chunk_boundaries
from chigyusubs.env import load_repo_env
from chigyusubs.metadata import finish_run, metadata_path, start_run, write_metadata
from chigyusubs.vad import run_silero_vad

load_repo_env()


def log(msg: str = "", end="\n"):
    print(msg, end=end, flush=True)


# ---------------------------------------------------------------------------
# VAD chunk loading
# ---------------------------------------------------------------------------

def _load_vad_chunk_bounds(chunk_json_path: str) -> list[tuple[float, float]]:
    with open(chunk_json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return [(float(c["start_sec"]), float(c["end_sec"])) for c in chunks]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a professional Japanese transcription assistant. "
    "You transcribe spoken dialogue from Japanese variety/comedy shows."
)

def build_transcription_prompt(context: str = "") -> str:
    lines = [
        "Transcribe ALL spoken Japanese dialogue in this video clip faithfully.",
        "",
        "Instructions:",
        "1. Output ONLY plain text. Do NOT use JSON or markdown code blocks.",
        "2. Do NOT add timestamps or speaker names/labels.",
        "3. Indicate speaker turns by starting the line with '-- '.",
        "4. INCLUDE standard Japanese punctuation (、。！？) to reflect natural flow.",
        "5. Do NOT translate — keep everything in Japanese.",
        "6. Do NOT skip or summarize — transcribe every utterance verbatim.",
        "7. Output each utterance on a new line.",
        "8. If you see on-screen text (テロップ), mark it as [画面: テキスト内容].",
        "9. If the audio is silent or background music only, output nothing.",
        "10. Do NOT hallucinate or loop. If speech repeats, transcribe it naturally.",
    ]
    if context:
        lines.append("")
        lines.append("### CONTEXT ###")
        lines.append(context[:500])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Video encoding
# ---------------------------------------------------------------------------

def _encode_video_base64(video_path: str) -> str:
    """Read a video file and return its base64 data URL."""
    with open(video_path, "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:video/mp4;base64,{b64}"


# ---------------------------------------------------------------------------
# Repetition loop detection
# ---------------------------------------------------------------------------

def _truncate_repetition(text: str, max_repeats: int = 5) -> str:
    """Detect and truncate repetition loops in transcription output."""
    lines = text.splitlines()
    if len(lines) < max_repeats * 2:
        return text

    # Sliding window: if the same line appears max_repeats times in a row, truncate
    result: list[str] = []
    repeat_count = 0
    prev_line = None
    for line in lines:
        stripped = line.strip()
        if stripped == prev_line and stripped:
            repeat_count += 1
            if repeat_count >= max_repeats:
                continue  # skip further repeats
        else:
            repeat_count = 1
            prev_line = stripped
        result.append(line)

    return "\n".join(result)


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def transcribe_chunk_mimo(
    client,
    video_path: str,
    prompt: str,
    *,
    model: str = "mimo-v2-omni",
    fps: float = 2.0,
    media_resolution: str = "default",
    max_tokens: int = 4096,
    enable_thinking: bool | None = None,
) -> dict:
    """Call MiMo API with a base64-encoded video chunk."""
    video_url = _encode_video_base64(video_path)

    extra_body: dict = {}
    if enable_thinking is not None:
        extra_body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": video_url},
                        "fps": fps,
                        "media_resolution": media_resolution,
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
        ],
        max_completion_tokens=max_tokens,
        frequency_penalty=1.0,
        **({"extra_body": extra_body} if extra_body else {}),
    )

    choice = response.choices[0]
    text = (choice.message.content or "").strip()
    usage = response.usage

    return {
        "text": text,
        "usage": {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
        },
        "finish_reason": choice.finish_reason,
        "model": response.model,
    }


# ---------------------------------------------------------------------------
# Main transcription loop
# ---------------------------------------------------------------------------

def transcribe_chunks(
    client,
    video_path: str,
    chunk_bounds: list[tuple[float, float]],
    *,
    prompt: str,
    model: str = "mimo-v2-omni",
    fps: float = 2.0,
    media_resolution: str = "default",
    max_tokens: int = 4096,
    enable_thinking: bool | None = None,
    video_fps: float = 1.0,
    video_width: int | None = None,
    video_crf: int = 36,
    audio_bitrate: str = "24k",
    max_chunks: int = 0,
) -> list[dict]:
    results: list[dict] = []
    total = len(chunk_bounds)
    if max_chunks > 0:
        chunk_bounds = chunk_bounds[:max_chunks]
        log(f"Limiting to {max_chunks}/{total} chunks")

    with tempfile.TemporaryDirectory() as workdir:
        for idx, (start_s, end_s) in enumerate(chunk_bounds):
            duration_s = end_s - start_s
            log(f"Chunk {idx + 1}/{len(chunk_bounds)}: {start_s:.1f}-{end_s:.1f}s ({duration_s:.1f}s)")

            # Compress video chunk
            chunk_path = os.path.join(workdir, f"mimo_{idx}.mp4")
            extract_inline_video_chunk(
                video_path, chunk_path,
                start_s=start_s, duration_s=duration_s,
                fps=video_fps, width=video_width,
                crf=video_crf, audio_bitrate=audio_bitrate,
            )
            chunk_size_mb = os.path.getsize(chunk_path) / (1024 * 1024)
            # base64 inflates by ~33%
            b64_size_mb = chunk_size_mb * 4 / 3
            log(f"  Compressed: {chunk_size_mb:.1f}MB (b64: {b64_size_mb:.1f}MB)")

            if b64_size_mb > 10.0:
                log(f"  WARNING: base64 size {b64_size_mb:.1f}MB exceeds 10MB limit, skipping")
                results.append({
                    "chunk": idx,
                    "chunk_start_s": start_s,
                    "chunk_end_s": end_s,
                    "text": "",
                    "chunk_size_mb": round(chunk_size_mb, 3),
                    "error": f"base64 size {b64_size_mb:.1f}MB exceeds 10MB limit",
                })
                continue

            t0 = time.monotonic()
            try:
                result = transcribe_chunk_mimo(
                    client, chunk_path, prompt,
                    model=model, fps=fps,
                    media_resolution=media_resolution,
                    max_tokens=max_tokens,
                    enable_thinking=enable_thinking,
                )
                elapsed = time.monotonic() - t0
                text = _truncate_repetition(result["text"])

                log(f"  {len(text)} chars, {elapsed:.1f}s, "
                    f"tokens={result['usage']['total_tokens']}, "
                    f"finish={result['finish_reason']}")

                results.append({
                    "chunk": idx,
                    "chunk_start_s": start_s,
                    "chunk_end_s": end_s,
                    "text": text,
                    "chunk_size_mb": round(chunk_size_mb, 3),
                    "elapsed_seconds": round(elapsed, 3),
                    "usage": result["usage"],
                    "finish_reason": result["finish_reason"],
                })
            except Exception as exc:
                elapsed = time.monotonic() - t0
                msg = str(exc).strip().splitlines()[0] if str(exc).strip() else repr(exc)
                log(f"  FAILED ({elapsed:.1f}s): {msg}")
                results.append({
                    "chunk": idx,
                    "chunk_start_s": start_s,
                    "chunk_end_s": end_s,
                    "text": "",
                    "chunk_size_mb": round(chunk_size_mb, 3),
                    "elapsed_seconds": round(elapsed, 3),
                    "error": msg,
                })

    return results


def main():
    run = start_run("transcribe_mimo_video")

    parser = argparse.ArgumentParser(
        description="Transcribe video with MiMo v2 Omni, outputting Gemini-compatible raw JSON."
    )
    parser.add_argument("--video", required=True, help="Source video file.")
    parser.add_argument("--output", required=True, help="Output raw JSON path.")
    parser.add_argument("--vad-json", default="", help="Pre-computed VAD segments JSON.")
    parser.add_argument("--chunk-json", default="", help="Pre-computed VAD chunk boundaries JSON.")
    parser.add_argument("--model", default="mimo-v2-omni", help="MiMo model name.")
    parser.add_argument("--context", default="", help="Context string for the prompt.")
    parser.add_argument("--target-chunk-s", type=float, default=90, help="Target chunk duration (default: 90).")
    parser.add_argument("--max-chunk-s", type=float, default=120, help="Max chunk duration (default: 120).")
    parser.add_argument("--max-chunks", type=int, default=0, help="Limit chunks for testing (0 = all).")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max completion tokens (default: 4096).")
    parser.add_argument("--fps", type=float, default=2.0, help="Frame extraction rate for MiMo API (default: 2.0).")
    parser.add_argument("--media-resolution", default="default", choices=["default", "max"],
                        help="MiMo media resolution tier (default: default).")
    parser.add_argument("--video-fps", type=float, default=1.0,
                        help="FPS for video compression before upload (default: 1.0).")
    parser.add_argument("--video-width", type=int, default=None,
                        help="Resize video width for compression (default: no resize).")
    parser.add_argument("--video-crf", type=int, default=36,
                        help="CRF for video compression (default: 36).")
    parser.add_argument("--audio-bitrate", default="24k",
                        help="Audio bitrate for compressed chunks (default: 24k).")
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument("--no-thinking", action="store_true",
                                help="Disable model reasoning/thinking to save tokens.")
    thinking_group.add_argument("--thinking", action="store_true",
                                help="Explicitly enable model reasoning/thinking.")
    args = parser.parse_args()

    video_path = args.video
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # API client
    # -----------------------------------------------------------------------
    api_key = os.environ.get("XIAOMI_API_KEY")
    if not api_key:
        raise SystemExit("XIAOMI_API_KEY not found in environment. Add it to .env.")

    from openai import OpenAI
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.xiaomimimo.com/v1",
    )

    # -----------------------------------------------------------------------
    # Load or compute VAD chunk boundaries
    # -----------------------------------------------------------------------
    if args.chunk_json:
        log(f"Loading chunk boundaries from {args.chunk_json}")
        chunk_bounds = _load_vad_chunk_bounds(args.chunk_json)
    else:
        duration = get_duration(video_path)
        log(f"Duration: {duration:.1f}s")

        if args.vad_json and Path(args.vad_json).exists():
            log(f"Loading VAD segments from {args.vad_json}")
            vad_segments = json.loads(Path(args.vad_json).read_text(encoding="utf-8"))
        else:
            log("Running Silero VAD...")
            with tempfile.TemporaryDirectory() as workdir:
                vad_segments = run_silero_vad(video_path, workdir)
            log(f"  {len(vad_segments)} speech segments detected")

        chunk_bounds = find_chunk_boundaries(
            vad_segments,
            total_duration=duration,
            target_chunk_s=args.target_chunk_s,
            max_chunk_s=args.max_chunk_s,
        )

    log(f"{len(chunk_bounds)} chunks to transcribe")

    # -----------------------------------------------------------------------
    # Build prompt
    # -----------------------------------------------------------------------
    prompt = build_transcription_prompt(context=args.context)

    # -----------------------------------------------------------------------
    # Transcribe
    # -----------------------------------------------------------------------
    t0 = time.monotonic()
    enable_thinking: bool | None = None
    if args.no_thinking:
        enable_thinking = False
    elif args.thinking:
        enable_thinking = True

    results = transcribe_chunks(
        client,
        video_path,
        chunk_bounds,
        prompt=prompt,
        model=args.model,
        fps=args.fps,
        media_resolution=args.media_resolution,
        max_tokens=args.max_tokens,
        enable_thinking=enable_thinking,
        video_fps=args.video_fps,
        video_width=args.video_width,
        video_crf=args.video_crf,
        audio_bitrate=args.audio_bitrate,
        max_chunks=args.max_chunks,
    )
    total_time = time.monotonic() - t0

    # -----------------------------------------------------------------------
    # Write output
    # -----------------------------------------------------------------------
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    total_chars = sum(len(r.get("text", "")) for r in results)
    failed = sum(1 for r in results if "error" in r)
    audio_duration = sum(r["chunk_end_s"] - r["chunk_start_s"] for r in results)
    total_tokens = sum(r.get("usage", {}).get("total_tokens", 0) for r in results)

    log()
    log(f"Wrote {len(results)} chunks to {output_path}")
    log(f"  Total chars: {total_chars}")
    log(f"  Failed chunks: {failed}")
    log(f"  Audio duration: {audio_duration:.1f}s")
    log(f"  Wall time: {total_time:.1f}s")
    log(f"  Total tokens: {total_tokens}")

    metadata = finish_run(
        run,
        inputs={"video": video_path},
        outputs={"raw_json": str(output_path)},
        settings={
            "model": args.model,
            "fps": args.fps,
            "media_resolution": args.media_resolution,
            "max_tokens": args.max_tokens,
            "video_fps": args.video_fps,
            "video_width": args.video_width,
            "video_crf": args.video_crf,
            "audio_bitrate": args.audio_bitrate,
            "target_chunk_s": args.target_chunk_s,
            "max_chunk_s": args.max_chunk_s,
            "max_chunks": args.max_chunks,
        },
        stats={
            "chunks": len(results),
            "total_chars": total_chars,
            "failed_chunks": failed,
            "audio_duration_s": round(audio_duration, 1),
            "wall_time_s": round(total_time, 1),
            "total_tokens": total_tokens,
        },
    )
    write_metadata(str(output_path), metadata)
    log(f"Metadata written: {metadata_path(str(output_path))}")


if __name__ == "__main__":
    main()
