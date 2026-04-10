#!/usr/bin/env python3
"""Transcribe audio using ibm-granite/granite-4.0-1b-speech via transformers.

Uses VAD-guided chunks for direct comparison with other ASR experiments.

Requires: transformers>=4.52.1 torch soundfile sentencepiece
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

# Force only the discrete GPU (9070 XT = device 0).
os.environ.setdefault("ROCR_VISIBLE_DEVICES", "0")

# Shim numba to avoid "Numba needs NumPy 2.3" crash.
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


DEFAULT_MODEL = "ibm-granite/granite-4.0-1b-speech"


def _extract_wav_chunk(video_path: str, start_s: float, duration_s: float, out_path: str):
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


def load_model(model_id: str, device: str = "cuda:0"):
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    log(f"Loading processor: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)

    log(f"Loading model: {model_id} (bf16) on {device}")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, device_map=device, dtype=torch.bfloat16,
    )

    return model, processor


def transcribe_chunk(model, processor, wav_path: str, device: str, max_new_tokens: int) -> dict:
    import soundfile as sf

    audio, sr = sf.read(wav_path, dtype="float32")
    wav = torch.tensor(audio).unsqueeze(0)  # [1, samples]

    tokenizer = processor.tokenizer
    prompt_text = "<|audio|>can you transcribe the speech into a written format?"
    chat = [{"role": "user", "content": prompt_text}]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    inputs = processor(prompt, wav, device=device, return_tensors="pt").to(device)
    num_input = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)

    new_tokens = outputs[0, num_input:].unsqueeze(0)
    text = tokenizer.batch_decode(new_tokens, add_special_tokens=False, skip_special_tokens=True)[0]
    tokens_generated = outputs.shape[-1] - num_input

    del inputs, outputs
    torch.cuda.empty_cache()

    return {
        "text": text.strip(),
        "tokens_generated": int(tokens_generated),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio with Granite 4.0-1B Speech."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--chunks-json", default="",
                        help="Pre-computed chunk boundaries JSON. If omitted, runs VAD.")
    parser.add_argument("--target-chunk-s", type=float, default=20)
    parser.add_argument("--max-chunk-s", type=float, default=30)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Chunk boundaries
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
    log(f"{len(chunk_bounds)}/{total} chunks to transcribe")

    # Load model
    model, processor = load_model(args.model, device=args.device)
    log(f"Model loaded on {model.device}")

    # Transcribe
    results: list[dict] = []
    t_start = time.monotonic()

    with tempfile.TemporaryDirectory() as workdir:
        for idx, (start_s, end_s) in enumerate(chunk_bounds):
            dur = end_s - start_s
            log(f"Chunk {idx+1}/{len(chunk_bounds)}: {start_s:.1f}-{end_s:.1f}s ({dur:.1f}s)", end=" ")

            wav_path = os.path.join(workdir, f"granite_{idx}.wav")
            _extract_wav_chunk(args.video, start_s, dur, wav_path)

            t0 = time.monotonic()
            try:
                r = transcribe_chunk(model, processor, wav_path, args.device, args.max_new_tokens)
                elapsed = time.monotonic() - t0
                log(f"-> {len(r['text'])} chars, {elapsed:.1f}s")

                results.append({
                    "chunk": idx,
                    "chunk_start_s": start_s,
                    "chunk_end_s": end_s,
                    "text": r["text"],
                    "elapsed_seconds": round(elapsed, 3),
                    "tokens_generated": r["tokens_generated"],
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

            # Incremental save
            output_path.write_text(
                json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
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

    log(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
