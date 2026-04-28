#!/usr/bin/env python3
"""Benchmark openai-whisper variants (large-v2, large-v3, large-v3-turbo).

Runs on the full audio with the same settings as the pipeline pre-pass
(condition_on_previous_text=False, compression_ratio_threshold=2.4) so the
numbers land in the same regime as the other ASR comparison runs.

Requires: openai-whisper torch soundfile
Run with system python3.12 for ROCm access.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types as _types
from pathlib import Path

# Force only the discrete GPU (9070 XT = device 0).
os.environ.setdefault("ROCR_VISIBLE_DEVICES", "0")

# Shim numba to avoid "Numba needs NumPy 2.3" crash.
_fake_numba = _types.ModuleType("numba")
_fake_numba.jit = lambda *a, **kw: (lambda f: f)
sys.modules["numba"] = _fake_numba

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from chigyusubs.audio import get_duration  # noqa: E402


def log(msg: str = "", end: str = "\n"):
    print(msg, end=end, flush=True)


def run_variant(model_name: str, video: str, initial_prompt: str | None,
                device: str) -> dict:
    import whisper

    log(f"\n=== {model_name} ===")
    log(f"Loading model on {device}...")
    t_load = time.monotonic()
    model = whisper.load_model(model_name, device=device)
    log(f"  loaded in {time.monotonic() - t_load:.1f}s "
        f"({torch.cuda.memory_allocated()/1024**3:.1f} GB allocated)")

    log("Transcribing full audio...")
    t_start = time.monotonic()
    result = model.transcribe(
        video,
        language="ja",
        condition_on_previous_text=False,
        compression_ratio_threshold=2.4,
        word_timestamps=False,
        initial_prompt=initial_prompt,
        verbose=False,
    )
    elapsed = time.monotonic() - t_start

    segments = result.get("segments", []) or []
    full_text = "".join(s.get("text", "") for s in segments).strip()

    # Free GPU memory before loading the next variant.
    del model
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "n_segments": len(segments),
        "n_chars": len(full_text),
        "wall_seconds": round(elapsed, 2),
        "text": full_text,
        "segments": [
            {
                "start": round(float(s["start"]), 3),
                "end": round(float(s["end"]), 3),
                "text": s["text"].strip(),
            }
            for s in segments
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark openai-whisper variants on a full episode."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--models",
        default="large-v2,large-v3-turbo",
        help="Comma-separated whisper model names to run.",
    )
    parser.add_argument("--initial-prompt", default="",
                        help="Optional initial_prompt for decoder biasing.")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    duration = get_duration(args.video)
    log(f"Audio duration: {duration:.1f}s")

    initial_prompt = args.initial_prompt or None
    if initial_prompt:
        log(f"Initial prompt ({len(initial_prompt)} chars): {initial_prompt[:80]}...")

    summary = []
    for model_name in args.models.split(","):
        model_name = model_name.strip()
        if not model_name:
            continue
        result = run_variant(model_name, args.video, initial_prompt, args.device)
        rtf = result["wall_seconds"] / duration if duration else 0.0
        chars_per_sec = result["n_chars"] / duration if duration else 0.0
        log(f"  {model_name}: {result['n_chars']} chars, "
            f"{result['n_segments']} segments, "
            f"{result['wall_seconds']:.0f}s wall, "
            f"{rtf:.3f}x RTF, {chars_per_sec:.1f} c/s")

        out_path = output_dir / f"whisper_{model_name.replace('-', '_')}.json"
        out_path.write_text(
            json.dumps({
                "model": result["model"],
                "n_segments": result["n_segments"],
                "n_chars": result["n_chars"],
                "wall_seconds": result["wall_seconds"],
                "rtf": round(rtf, 4),
                "chars_per_sec": round(chars_per_sec, 2),
                "duration_s": round(duration, 2),
                "segments": result["segments"],
                "text": result["text"],
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        log(f"  -> {out_path}")
        summary.append({
            "model": model_name,
            "chars": result["n_chars"],
            "segments": result["n_segments"],
            "wall_s": result["wall_seconds"],
            "rtf": round(rtf, 4),
            "chars_per_sec": round(chars_per_sec, 2),
        })

    log("\n=== Summary ===")
    for row in summary:
        log(f"  {row['model']:18s}  {row['chars']:>6d} chars  "
            f"{row['segments']:>4d} segs  {row['wall_s']:>5.0f}s  "
            f"{row['rtf']:.3f}x  {row['chars_per_sec']:.1f} c/s")

    summary_path = output_dir / "whisper_variants_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    log(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
