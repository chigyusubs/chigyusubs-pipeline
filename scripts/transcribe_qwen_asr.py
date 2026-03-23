#!/usr/bin/env python3
"""Transcribe audio using Qwen3-ASR via the qwen-asr HF package.

Processes the same VAD chunks as the Gemini path and outputs a compatible
_gemini_raw.json format for apples-to-apples comparison.

Requires: system python3.12 with qwen-asr and ROCm-enabled PyTorch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chigyusubs.audio import get_duration
from chigyusubs.chunking import find_chunk_boundaries
from chigyusubs.metadata import finish_run, metadata_path, start_run, write_metadata
from chigyusubs.vad import run_silero_vad


def log(msg: str = "", end="\n"):
    print(msg, end=end, flush=True)


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------

def _extract_wav_chunk(video_path: str, start_s: float, duration_s: float, out_path: str):
    """Extract a 16kHz mono WAV chunk via ffmpeg."""
    import subprocess
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(start_s),
            "-i", video_path,
            "-t", str(duration_s),
            "-vn", "-ac", "1", "-ar", "16000",
            "-f", "wav", out_path,
        ],
        capture_output=True, check=True,
    )


# ---------------------------------------------------------------------------
# VAD chunk loading (matches transcribe_gemini_video.py format)
# ---------------------------------------------------------------------------

def _load_vad_chunk_bounds(chunk_json_path: str) -> list[tuple[float, float]]:
    with open(chunk_json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return [(float(c["start_sec"]), float(c["end_sec"])) for c in chunks]


# ---------------------------------------------------------------------------
# Main transcription
# ---------------------------------------------------------------------------

def transcribe_chunks(
    model,
    video_path: str,
    chunk_bounds: list[tuple[float, float]],
    *,
    language: str = "Japanese",
    context: str = "",
    max_chunks: int = 0,
) -> list[dict]:
    """Transcribe each chunk and return records in Gemini raw JSON format."""
    results: list[dict] = []
    total = len(chunk_bounds)
    if max_chunks > 0:
        chunk_bounds = chunk_bounds[:max_chunks]
        log(f"Limiting to {max_chunks}/{total} chunks")

    with tempfile.TemporaryDirectory() as workdir:
        for idx, (start_s, end_s) in enumerate(chunk_bounds):
            duration_s = end_s - start_s
            log(f"Chunk {idx + 1}/{len(chunk_bounds)}: {start_s:.1f}-{end_s:.1f}s ({duration_s:.1f}s)")

            wav_path = os.path.join(workdir, f"qwen_asr_{idx}.wav")
            _extract_wav_chunk(video_path, start_s, duration_s, wav_path)
            wav_size_mb = os.path.getsize(wav_path) / (1024 * 1024)

            t0 = time.monotonic()
            try:
                asr_results = model.transcribe(
                    audio=wav_path,
                    language=language,
                    context=context,
                    return_time_stamps=False,
                )
                text = asr_results[0].text.strip() if asr_results else ""
                detected_lang = asr_results[0].language if asr_results else ""
                elapsed = time.monotonic() - t0

                log(f"  {len(text)} chars, {elapsed:.1f}s, lang={detected_lang}")

                results.append({
                    "chunk": idx,
                    "chunk_start_s": start_s,
                    "chunk_end_s": end_s,
                    "text": text,
                    "chunk_size_mb": round(wav_size_mb, 3),
                    "elapsed_seconds": round(elapsed, 3),
                    "detected_language": detected_lang,
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
                    "chunk_size_mb": round(wav_size_mb, 3),
                    "elapsed_seconds": round(elapsed, 3),
                    "error": msg,
                })

    return results


def main():
    run = start_run("transcribe_qwen_asr")

    parser = argparse.ArgumentParser(
        description="Transcribe audio with Qwen3-ASR, outputting Gemini-compatible raw JSON."
    )
    parser.add_argument("--video", required=True, help="Source video/audio file.")
    parser.add_argument("--output", required=True, help="Output raw JSON path.")
    parser.add_argument(
        "--vad-json", default="",
        help="Pre-computed VAD segments JSON. Computed if not provided.",
    )
    parser.add_argument(
        "--chunk-json", default="",
        help="Pre-computed VAD chunk boundaries JSON (vad_chunks.json). "
             "If provided, uses these exact chunk boundaries instead of computing new ones.",
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3-ASR-1.7B",
        help="HuggingFace model ID (default: Qwen/Qwen3-ASR-1.7B).",
    )
    parser.add_argument("--dtype", default="bfloat16", help="Torch dtype (default: bfloat16).")
    parser.add_argument("--device", default="cuda:0", help="Device (default: cuda:0).")
    parser.add_argument(
        "--language", default="Japanese",
        help="Language hint (default: Japanese). Use empty string for auto-detect.",
    )
    parser.add_argument(
        "--context", default="",
        help="Context string for the model (e.g. glossary names, show title).",
    )
    parser.add_argument(
        "--target-chunk-s", type=float, default=90,
        help="Target chunk duration in seconds (default: 90).",
    )
    parser.add_argument(
        "--max-chunk-s", type=float, default=120,
        help="Maximum chunk duration in seconds (default: 120).",
    )
    parser.add_argument(
        "--max-chunks", type=int, default=0,
        help="Limit number of chunks for smoke tests (0 = all).",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=1024,
        help="Max new tokens for generation (default: 1024).",
    )
    args = parser.parse_args()

    video_path = args.video
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
    # Load model
    # -----------------------------------------------------------------------
    import torch
    from qwen_asr import Qwen3ASRModel

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(args.dtype)
    if dtype is None:
        parser.error(f"Unsupported dtype: {args.dtype}. Use one of: {', '.join(sorted(dtype_map))}")

    log(f"Loading {args.model} ({args.dtype}) on {args.device}...")
    t0 = time.monotonic()
    model = Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    load_time = time.monotonic() - t0
    log(f"Model loaded in {load_time:.1f}s")

    # -----------------------------------------------------------------------
    # Transcribe
    # -----------------------------------------------------------------------
    t0 = time.monotonic()
    results = transcribe_chunks(
        model,
        video_path,
        chunk_bounds,
        language=args.language or None,
        context=args.context,
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

    log()
    log(f"Wrote {len(results)} chunks to {output_path}")
    log(f"  Total chars: {total_chars}")
    log(f"  Failed chunks: {failed}")
    log(f"  Audio duration: {audio_duration:.1f}s")
    log(f"  Wall time: {total_time:.1f}s (model load: {load_time:.1f}s)")
    log(f"  RTF: {total_time / audio_duration:.2f}x" if audio_duration > 0 else "")

    metadata = finish_run(
        run,
        inputs={"video": video_path},
        outputs={"raw_json": str(output_path)},
        settings={
            "model": args.model,
            "dtype": args.dtype,
            "device": args.device,
            "language": args.language,
            "context": args.context[:200] if args.context else "",
            "target_chunk_s": args.target_chunk_s,
            "max_chunk_s": args.max_chunk_s,
            "max_new_tokens": args.max_new_tokens,
            "max_chunks": args.max_chunks,
        },
        stats={
            "chunks": len(results),
            "total_chars": total_chars,
            "failed_chunks": failed,
            "audio_duration_s": round(audio_duration, 1),
            "wall_time_s": round(total_time, 1),
            "model_load_s": round(load_time, 1),
        },
    )
    write_metadata(str(output_path), metadata)
    log(f"Metadata written: {metadata_path(str(output_path))}")


if __name__ == "__main__":
    main()
