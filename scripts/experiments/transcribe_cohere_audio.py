#!/usr/bin/env python3
"""Transcribe audio using CohereLabs/cohere-transcribe-03-2026 via transformers.

Supports two modes:
1. Built-in long-form chunking (--mode native): feed entire audio, let the
   model's processor handle chunking with overlap.
2. VAD-guided chunks (--mode vad): same chunk boundaries as the Gemma4
   experiment for direct comparison.

Requires: transformers>=5.4.0 torch soundfile sentencepiece protobuf
Run with system python3.12 for ROCm access.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Force only the discrete GPU (9070 XT = device 0) — must be set before
# importing torch so the ROCm runtime never sees the iGPU.
os.environ.setdefault("ROCR_VISIBLE_DEVICES", "0")

# Shim numba to avoid "Numba needs NumPy 2.3" crash — librosa.filters.mel
# (used by the Cohere feature extractor) imports numba at module level.
import types as _types
_fake_numba = _types.ModuleType("numba")
_fake_numba.jit = lambda *a, **kw: (lambda f: f)
sys.modules["numba"] = _fake_numba

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from chigyusubs.audio import get_duration
from chigyusubs.chunking import find_chunk_boundaries
from chigyusubs.vad import run_silero_vad


def log(msg: str = "", end="\n"):
    print(msg, end=end, flush=True)


DEFAULT_MODEL = "CohereLabs/cohere-transcribe-03-2026"


def _extract_wav_chunk(video_path: str, start_s: float, duration_s: float, out_path: str):
    """Extract a 16kHz mono WAV chunk via ffmpeg."""
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


def _extract_full_wav(video_path: str, out_path: str):
    """Extract full audio as 16kHz mono WAV."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn", "-ac", "1", "-ar", "16000",
            "-f", "wav", out_path,
        ],
        capture_output=True, check=True,
    )


def load_model(model_id: str, device: str = "cuda:0"):
    """Load Cohere Transcribe model and processor."""
    from transformers import AutoProcessor, CohereAsrForConditionalGeneration

    log(f"Loading processor: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)

    log(f"Loading model: {model_id} on {device}")
    model = CohereAsrForConditionalGeneration.from_pretrained(
        model_id,
    ).to(device)

    return model, processor


def transcribe_native(model, processor, video_path: str, device: str, max_new_tokens: int) -> dict:
    """Transcribe full audio using the model's built-in long-form chunking."""
    import soundfile as sf

    with tempfile.TemporaryDirectory() as workdir:
        wav_path = os.path.join(workdir, "full.wav")
        log("Extracting full audio...")
        _extract_full_wav(video_path, wav_path)
        wav_size_mb = os.path.getsize(wav_path) / (1024 * 1024)
        log(f"  WAV: {wav_size_mb:.1f}MB")

        audio, sr = sf.read(wav_path, dtype="float32")
        duration_s = len(audio) / sr
        log(f"  Duration: {duration_s:.1f}s @ {sr}Hz")

    inputs = processor(
        audio, sampling_rate=sr, return_tensors="pt", language="ja",
    )
    audio_chunk_index = inputs.get("audio_chunk_index")
    inputs.to(device, dtype=model.dtype)

    log("Generating...")
    t0 = time.monotonic()
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    text = processor.decode(
        outputs, skip_special_tokens=True,
        audio_chunk_index=audio_chunk_index, language="ja",
    )
    # decode returns a list for long-form
    if isinstance(text, list):
        text = text[0] if text else ""
    elapsed = time.monotonic() - t0

    del inputs, outputs
    torch.cuda.empty_cache()

    return {
        "text": text.strip(),
        "duration_s": duration_s,
        "elapsed_s": round(elapsed, 3),
    }


def transcribe_vad_chunks(
    model, processor, video_path: str, chunk_bounds: list[tuple[float, float]],
    device: str, max_new_tokens: int,
) -> list[dict]:
    """Transcribe using pre-computed VAD chunk boundaries."""
    import soundfile as sf

    results = []
    with tempfile.TemporaryDirectory() as workdir:
        for idx, (start_s, end_s) in enumerate(chunk_bounds):
            dur = end_s - start_s
            log(f"Chunk {idx+1}/{len(chunk_bounds)}: {start_s:.1f}-{end_s:.1f}s ({dur:.1f}s)", end=" ")

            wav_path = os.path.join(workdir, f"cohere_{idx}.wav")
            _extract_wav_chunk(video_path, start_s, dur, wav_path)

            audio, sr = sf.read(wav_path, dtype="float32")

            inputs = processor(
                audio, sampling_rate=sr, return_tensors="pt", language="ja",
            )
            audio_chunk_index = inputs.pop("audio_chunk_index", None)
            inputs.to(device, dtype=model.dtype)

            t0 = time.monotonic()
            try:
                with torch.inference_mode():
                    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

                decode_kwargs = {"skip_special_tokens": True}
                if audio_chunk_index is not None:
                    decode_kwargs["audio_chunk_index"] = audio_chunk_index
                    decode_kwargs["language"] = "ja"
                text = processor.decode(outputs, **decode_kwargs)
                if isinstance(text, list):
                    text = text[0] if text else ""
                elapsed = time.monotonic() - t0

                log(f"-> {len(text.strip())} chars, {elapsed:.1f}s")
                results.append({
                    "chunk": idx,
                    "chunk_start_s": start_s,
                    "chunk_end_s": end_s,
                    "text": text.strip(),
                    "elapsed_seconds": round(elapsed, 3),
                })
            except Exception as exc:
                elapsed = time.monotonic() - t0
                msg = str(exc).strip().splitlines()[0] if str(exc).strip() else repr(exc)
                log(f"FAILED ({elapsed:.1f}s): {msg}")
                results.append({
                    "chunk": idx,
                    "chunk_start_s": start_s,
                    "chunk_end_s": end_s,
                    "text": "",
                    "elapsed_seconds": round(elapsed, 3),
                    "error": msg,
                })

            del inputs, outputs
            torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio with Cohere Transcribe (2B conformer)."
    )
    parser.add_argument("--video", required=True, help="Source video/audio file.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--mode", choices=["native", "vad"], default="vad",
                        help="Chunking mode: 'native' (model's built-in) or 'vad' (match Gemma4 chunks).")
    parser.add_argument("--chunks-json", default="",
                        help="Pre-computed chunk boundaries JSON (for vad mode). "
                             "If omitted, runs VAD fresh.")
    parser.add_argument("--target-chunk-s", type=float, default=20)
    parser.add_argument("--max-chunk-s", type=float, default=30)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    model, processor = load_model(args.model, device=args.device)
    log(f"Model loaded on {model.device}")

    t_start = time.monotonic()

    if args.mode == "native":
        log("\n--- Native long-form mode ---")
        result = transcribe_native(model, processor, args.video, args.device, args.max_new_tokens)
        total_time = time.monotonic() - t_start

        log(f"\nText length: {len(result['text'])} chars")
        log(f"Duration: {result['duration_s']:.0f}s, Wall: {result['elapsed_s']:.0f}s")
        log(f"RTF: {result['elapsed_s'] / result['duration_s']:.2f}x")

        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    else:
        log("\n--- VAD chunk mode ---")

        if args.chunks_json:
            log(f"Loading chunk bounds from {args.chunks_json}")
            chunks_data = json.load(open(args.chunks_json))
            chunk_bounds = [(c["chunk_start_s"], c["chunk_end_s"]) for c in chunks_data]
        else:
            duration = get_duration(args.video)
            log(f"Duration: {duration:.1f}s")
            log("Running Silero VAD...")
            with tempfile.TemporaryDirectory() as workdir:
                vad_segments = run_silero_vad(args.video, workdir)
            log(f"  {len(vad_segments)} speech segments")
            chunk_bounds = find_chunk_boundaries(
                vad_segments, total_duration=duration,
                target_chunk_s=args.target_chunk_s, max_chunk_s=args.max_chunk_s,
                min_gap_s=0.5, fallback_min_gap_s=0.2,
            )

        total = len(chunk_bounds)
        if args.max_chunks > 0:
            chunk_bounds = chunk_bounds[:args.max_chunks]
        log(f"{len(chunk_bounds)}/{total} chunks to transcribe\n")

        results = transcribe_vad_chunks(
            model, processor, args.video, chunk_bounds,
            args.device, args.max_new_tokens,
        )
        total_time = time.monotonic() - t_start

        total_chars = sum(len(r.get("text", "")) for r in results)
        failed = sum(1 for r in results if "error" in r)
        audio_dur = sum(r["chunk_end_s"] - r["chunk_start_s"] for r in results)

        log(f"\nWrote {len(results)} chunks to {output_path}")
        log(f"  Chars: {total_chars}, Failed: {failed}")
        log(f"  Audio: {audio_dur:.0f}s, Wall: {total_time:.0f}s")
        if audio_dur > 0:
            log(f"  RTF: {total_time / audio_dur:.2f}x")

        output_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    log(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
