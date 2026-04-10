#!/usr/bin/env python3
"""Transcribe audio using google/gemma-4-E4B-it via transformers (bf16, local).

Splits audio into VAD-guided chunks (max 30s, splitting at silence boundaries)
and transcribes each chunk with Gemma 4's native audio understanding.

Requires: pip install -U transformers torch librosa accelerate
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

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from chigyusubs.audio import get_duration
from chigyusubs.chunking import find_chunk_boundaries
from chigyusubs.metadata import finish_run, metadata_path, start_run, write_metadata
from chigyusubs.vad import run_silero_vad


def log(msg: str = "", end="\n"):
    print(msg, end=end, flush=True)


DEFAULT_MODEL = "google/gemma-4-E4B-it"

TRANSCRIPTION_PROMPT = (
    "Transcribe ALL spoken Japanese dialogue in this audio clip faithfully.\n"
    "Output ONLY plain text — no JSON, no markdown, no code blocks.\n"
    "Do NOT add timestamps or speaker names/labels.\n"
    "Indicate speaker turns by starting the line with '-- '.\n"
    "Include standard Japanese punctuation (、。！？) to reflect natural flow.\n"
    "Do NOT translate — keep everything in Japanese.\n"
    "Do NOT skip or summarize — transcribe every utterance verbatim.\n"
    "Output each utterance on a new line.\n"
    "If the audio is silent or background music only, output nothing."
)


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


def _load_vad_chunk_bounds(chunk_json_path: str) -> list[tuple[float, float]]:
    with open(chunk_json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return [(float(c["start_sec"]), float(c["end_sec"])) for c in chunks]


def load_model(model_id: str, device: str = "cuda:0"):
    """Load Gemma 4 model and processor in bf16 on a specific device."""
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    log(f"Loading processor: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)

    log(f"Loading model: {model_id} (bf16) on {device}")
    model = AutoModelForMultimodalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    ).to(device)

    return model, processor


def transcribe_chunk(
    model,
    processor,
    wav_path: str,
    prompt: str,
    *,
    max_new_tokens: int = 1024,
) -> dict:
    """Transcribe a single audio chunk."""
    import soundfile as sf

    # Load audio as numpy array at native sample rate
    audio, sr = sf.read(wav_path, dtype="float32")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio, "sample_rate": sr},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    tokens_generated = outputs.shape[-1] - input_len
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)

    del inputs, outputs
    torch.cuda.empty_cache()

    return {
        "text": response.strip(),
        "input_tokens": input_len,
        "tokens_generated": int(tokens_generated),
    }


def main():
    run = start_run("transcribe_gemma4_audio")

    parser = argparse.ArgumentParser(
        description="Transcribe audio with Gemma 4 E4B-it via transformers (bf16)."
    )
    parser.add_argument("--video", required=True, help="Source video/audio file.")
    parser.add_argument("--output", required=True, help="Output raw JSON path.")
    parser.add_argument("--vad-json", default="")
    parser.add_argument("--chunk-json", default="")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"HF model ID (default: {DEFAULT_MODEL}).")
    parser.add_argument("--target-chunk-s", type=float, default=20,
                        help="Target chunk duration (default: 20). Split at VAD silences.")
    parser.add_argument("--max-chunk-s", type=float, default=30,
                        help="Hard max chunk duration (default: 30). Gemma 4 limit.")
    parser.add_argument("--max-chunks", type=int, default=0, help="Limit chunks for testing (0 = all).")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0", help="Torch device (default: cuda:0 = discrete GPU).")
    parser.add_argument("--prompt", default=TRANSCRIPTION_PROMPT)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Chunk boundaries (VAD-guided, max 30s)
    # -----------------------------------------------------------------------
    if args.chunk_json:
        log(f"Loading chunk boundaries from {args.chunk_json}")
        chunk_bounds = _load_vad_chunk_bounds(args.chunk_json)
    else:
        duration = get_duration(args.video)
        log(f"Duration: {duration:.1f}s")

        if args.vad_json and Path(args.vad_json).exists():
            log(f"Loading VAD segments from {args.vad_json}")
            vad_segments = json.loads(Path(args.vad_json).read_text(encoding="utf-8"))
        else:
            log("Running Silero VAD...")
            with tempfile.TemporaryDirectory() as workdir:
                vad_segments = run_silero_vad(args.video, workdir)
            log(f"  {len(vad_segments)} speech segments detected")

        chunk_bounds = find_chunk_boundaries(
            vad_segments,
            total_duration=duration,
            target_chunk_s=args.target_chunk_s,
            max_chunk_s=args.max_chunk_s,
            min_gap_s=0.5,
            fallback_min_gap_s=0.2,
        )

    total = len(chunk_bounds)
    if args.max_chunks > 0:
        chunk_bounds = chunk_bounds[:args.max_chunks]
    log(f"{len(chunk_bounds)}/{total} chunks to transcribe")

    durations = [end - start for start, end in chunk_bounds]
    if durations:
        log(f"Chunk durations: min={min(durations):.1f}s avg={sum(durations)/len(durations):.1f}s max={max(durations):.1f}s")
    over_30 = sum(1 for d in durations if d > 30.0)
    if over_30:
        log(f"WARNING: {over_30} chunks exceed 30s — Gemma 4 may truncate audio")

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    model, processor = load_model(args.model, device=args.device)
    log(f"Model loaded on {model.device}")

    # -----------------------------------------------------------------------
    # Transcribe
    # -----------------------------------------------------------------------
    results: list[dict] = []
    t_start = time.monotonic()

    with tempfile.TemporaryDirectory() as workdir:
        for idx, (start_s, end_s) in enumerate(chunk_bounds):
            dur = end_s - start_s
            log(f"Chunk {idx+1}/{len(chunk_bounds)}: {start_s:.1f}-{end_s:.1f}s ({dur:.1f}s)")

            wav_path = os.path.join(workdir, f"gemma4_{idx}.wav")
            _extract_wav_chunk(args.video, start_s, dur, wav_path)
            wav_size_mb = os.path.getsize(wav_path) / (1024 * 1024)

            t0 = time.monotonic()
            try:
                r = transcribe_chunk(
                    model, processor, wav_path, args.prompt,
                    max_new_tokens=args.max_new_tokens,
                )
                elapsed = time.monotonic() - t0

                log(f"  {len(r['text'])} chars, {elapsed:.1f}s, "
                    f"{r['tokens_generated']} tok generated")

                results.append({
                    "chunk": idx,
                    "chunk_start_s": start_s,
                    "chunk_end_s": end_s,
                    "text": r["text"],
                    "chunk_size_mb": round(wav_size_mb, 3),
                    "elapsed_seconds": round(elapsed, 3),
                    "input_tokens": r["input_tokens"],
                    "tokens_generated": r["tokens_generated"],
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

    metadata = finish_run(
        run,
        inputs={"video": args.video},
        outputs={"raw_json": str(output_path)},
        settings={
            "model": args.model,
            "torch_dtype": "bfloat16",
            "target_chunk_s": args.target_chunk_s,
            "max_chunk_s": args.max_chunk_s,
            "max_new_tokens": args.max_new_tokens,
        },
        stats={
            "chunks": len(results),
            "total_chars": total_chars,
            "failed_chunks": failed,
            "audio_duration_s": round(audio_dur, 1),
            "wall_time_s": round(total_time, 1),
        },
    )
    write_metadata(str(output_path), metadata)
    log(f"Metadata: {metadata_path(str(output_path))}")


if __name__ == "__main__":
    main()
