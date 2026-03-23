#!/usr/bin/env python3
"""Transcribe audio using Xiaomi MiMo v2 Omni audio-only mode.

Sends base64-encoded audio chunks to the MiMo API (no video).
Outputs Gemini-compatible raw JSON for comparison.

Requires: XIAOMI_API_KEY in .env or environment.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chigyusubs.audio import get_duration
from chigyusubs.chunking import find_chunk_boundaries
from chigyusubs.env import load_repo_env
from chigyusubs.metadata import finish_run, metadata_path, start_run, write_metadata
from chigyusubs.vad import run_silero_vad

load_repo_env()


def log(msg: str = "", end="\n"):
    print(msg, end=end, flush=True)


def _load_vad_chunk_bounds(chunk_json_path: str) -> list[tuple[float, float]]:
    with open(chunk_json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return [(float(c["start_sec"]), float(c["end_sec"])) for c in chunks]


def _extract_audio_chunk(video_path: str, start_s: float, duration_s: float, out_path: str,
                         *, audio_format: str = "mp3", bitrate: str = "64k"):
    """Extract an audio-only chunk via ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_s),
        "-i", video_path,
        "-t", str(duration_s),
        "-vn", "-ac", "1",
        "-b:a", bitrate,
        "-f", audio_format,
        out_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def _encode_audio_base64(audio_path: str, mime_type: str = "audio/mpeg") -> str:
    with open(audio_path, "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime_type};base64,{b64}"


SYSTEM_PROMPT = (
    "You are a professional Japanese transcription assistant. "
    "You transcribe spoken dialogue from Japanese variety/comedy shows."
)

TRANSCRIPTION_PROMPT = """Transcribe ALL spoken Japanese dialogue in this audio clip faithfully.

Instructions:
1. Output ONLY plain text. Do NOT use JSON or markdown code blocks.
2. Do NOT add timestamps or speaker names/labels.
3. Indicate speaker turns by starting the line with '-- '.
4. INCLUDE standard Japanese punctuation (、。！？) to reflect natural flow.
5. Do NOT translate — keep everything in Japanese.
6. Do NOT skip or summarize — transcribe every utterance verbatim.
7. Output each utterance on a new line.
8. If the audio is silent or background music only, output nothing.
9. Do NOT hallucinate or loop. If speech repeats, transcribe it naturally."""


def transcribe_chunk(client, audio_path: str, prompt: str, *,
                     model: str, max_tokens: int, enable_thinking: bool | None) -> dict:
    audio_data = _encode_audio_base64(audio_path)

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
                        "type": "input_audio",
                        "input_audio": {"data": audio_data},
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        max_completion_tokens=max_tokens,
        frequency_penalty=0.5,
        **({"extra_body": extra_body} if extra_body else {}),
    )

    choice = response.choices[0]
    usage = response.usage
    return {
        "text": (choice.message.content or "").strip(),
        "usage": {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
        },
        "finish_reason": choice.finish_reason,
    }


def main():
    run = start_run("transcribe_mimo_audio")
    parser = argparse.ArgumentParser(description="Transcribe audio with MiMo v2 Omni (audio-only mode).")
    parser.add_argument("--video", required=True, help="Source video/audio file.")
    parser.add_argument("--output", required=True, help="Output raw JSON path.")
    parser.add_argument("--vad-json", default="")
    parser.add_argument("--chunk-json", default="")
    parser.add_argument("--model", default="mimo-v2-omni")
    parser.add_argument("--context", default="")
    parser.add_argument("--target-chunk-s", type=float, default=30)
    parser.add_argument("--max-chunk-s", type=float, default=45)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--audio-bitrate", default="64k")
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--no-thinking", action="store_true")
    thinking.add_argument("--thinking", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("XIAOMI_API_KEY")
    if not api_key:
        raise SystemExit("XIAOMI_API_KEY not found in environment.")

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.xiaomimimo.com/v1")

    # Chunk boundaries
    if args.chunk_json:
        chunk_bounds = _load_vad_chunk_bounds(args.chunk_json)
    else:
        duration = get_duration(args.video)
        log(f"Duration: {duration:.1f}s")
        if args.vad_json and Path(args.vad_json).exists():
            vad_segments = json.loads(Path(args.vad_json).read_text(encoding="utf-8"))
        else:
            with tempfile.TemporaryDirectory() as wd:
                vad_segments = run_silero_vad(args.video, wd)
        chunk_bounds = find_chunk_boundaries(
            vad_segments, total_duration=duration,
            target_chunk_s=args.target_chunk_s, max_chunk_s=args.max_chunk_s,
        )

    total = len(chunk_bounds)
    if args.max_chunks > 0:
        chunk_bounds = chunk_bounds[:args.max_chunks]
    log(f"{len(chunk_bounds)}/{total} chunks to transcribe")

    enable_thinking = None
    if args.no_thinking:
        enable_thinking = False
    elif args.thinking:
        enable_thinking = True

    prompt = TRANSCRIPTION_PROMPT
    if args.context:
        prompt += f"\n\n### CONTEXT ###\n{args.context[:500]}"

    results: list[dict] = []
    t_start = time.monotonic()

    with tempfile.TemporaryDirectory() as workdir:
        for idx, (start_s, end_s) in enumerate(chunk_bounds):
            dur = end_s - start_s
            log(f"Chunk {idx+1}/{len(chunk_bounds)}: {start_s:.1f}-{end_s:.1f}s ({dur:.1f}s)")

            audio_path = os.path.join(workdir, f"mimo_audio_{idx}.mp3")
            _extract_audio_chunk(args.video, start_s, dur, audio_path, bitrate=args.audio_bitrate)
            size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            b64_mb = size_mb * 4 / 3

            if b64_mb > 10.0:
                log(f"  SKIP: b64 {b64_mb:.1f}MB > 10MB")
                results.append({"chunk": idx, "chunk_start_s": start_s, "chunk_end_s": end_s,
                                "text": "", "error": "base64 too large"})
                continue

            t0 = time.monotonic()
            try:
                r = transcribe_chunk(client, audio_path, prompt,
                                     model=args.model, max_tokens=args.max_tokens,
                                     enable_thinking=enable_thinking)
                elapsed = time.monotonic() - t0
                log(f"  {len(r['text'])} chars, {elapsed:.1f}s, "
                    f"tokens={r['usage']['total_tokens']}, finish={r['finish_reason']}")
                results.append({
                    "chunk": idx, "chunk_start_s": start_s, "chunk_end_s": end_s,
                    "text": r["text"], "chunk_size_mb": round(size_mb, 3),
                    "elapsed_seconds": round(elapsed, 3),
                    "usage": r["usage"], "finish_reason": r["finish_reason"],
                })
            except Exception as exc:
                elapsed = time.monotonic() - t0
                msg = str(exc).strip().splitlines()[0] if str(exc).strip() else repr(exc)
                log(f"  FAILED ({elapsed:.1f}s): {msg}")
                results.append({
                    "chunk": idx, "chunk_start_s": start_s, "chunk_end_s": end_s,
                    "text": "", "chunk_size_mb": round(size_mb, 3),
                    "elapsed_seconds": round(elapsed, 3), "error": msg,
                })

    total_time = time.monotonic() - t_start
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    total_chars = sum(len(r.get("text", "")) for r in results)
    failed = sum(1 for r in results if "error" in r)
    audio_dur = sum(r["chunk_end_s"] - r["chunk_start_s"] for r in results)
    total_tokens = sum(r.get("usage", {}).get("total_tokens", 0) for r in results)
    stop_count = sum(1 for r in results if r.get("finish_reason") == "stop")
    length_count = sum(1 for r in results if r.get("finish_reason") == "length")

    log(f"\nWrote {len(results)} chunks to {output_path}")
    log(f"  Chars: {total_chars}, Failed: {failed}, Stop: {stop_count}, Length: {length_count}")
    log(f"  Audio: {audio_dur:.0f}s, Wall: {total_time:.0f}s, Tokens: {total_tokens}")

    metadata = finish_run(run, inputs={"video": args.video}, outputs={"raw_json": str(output_path)},
        settings={"model": args.model, "audio_bitrate": args.audio_bitrate,
                   "max_tokens": args.max_tokens, "no_thinking": args.no_thinking},
        stats={"chunks": len(results), "total_chars": total_chars, "failed": failed,
               "wall_time_s": round(total_time, 1), "total_tokens": total_tokens})
    write_metadata(str(output_path), metadata)
    log(f"Metadata: {metadata_path(str(output_path))}")


if __name__ == "__main__":
    main()
