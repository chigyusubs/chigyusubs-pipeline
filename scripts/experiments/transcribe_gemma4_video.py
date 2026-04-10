#!/usr/bin/env python3
"""Transcribe video+audio using google/gemma-4-E2B-it via transformers (bf16).

Extracts frames (0.5fps, capped at 10 per chunk) + audio per VAD chunk,
feeds both modalities to Gemma4 for transcription with on-screen text capture.

Requires: transformers torch soundfile Pillow
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

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from chigyusubs.audio import get_duration
from chigyusubs.chunking import find_chunk_boundaries
from chigyusubs.vad import run_silero_vad


def log(msg: str = "", end="\n"):
    print(msg, end=end, flush=True)


DEFAULT_MODEL = "google/gemma-4-E2B-it"
MAX_FRAMES = 10
TARGET_FPS = 0.5
FRAME_HEIGHT = 720

TRANSCRIPTION_PROMPT = (
    "Transcribe ALL spoken Japanese dialogue in this video clip faithfully.\n"
    "Also transcribe any on-screen Japanese text you see, prefixed with [画面: ...].\n"
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
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(start_s), "-i", video_path, "-t", str(duration_s),
         "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", out_path],
        capture_output=True, check=True,
    )


def _extract_frames(video_path: str, start_s: float, duration_s: float, out_dir: str,
                    max_frames: int = MAX_FRAMES, target_fps: float = TARGET_FPS,
                    height: int = FRAME_HEIGHT) -> list[Path]:
    fps = min(target_fps, max_frames / duration_s)
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(start_s), "-i", video_path, "-t", str(duration_s),
         "-vf", f"fps={fps},scale=-2:{height}", "-q:v", "3", f"{out_dir}/frame_%03d.jpg"],
        capture_output=True, check=True,
    )
    return sorted(Path(out_dir).glob("*.jpg"))


def load_model(model_id: str, device: str = "cuda:0"):
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    log(f"Loading processor: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)

    log(f"Loading model: {model_id} (bf16) on {device}")
    model = AutoModelForMultimodalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
    ).to(device)

    return model, processor


def transcribe_chunk(model, processor, wav_path: str, frame_paths: list[Path],
                     prompt: str, *, max_new_tokens: int = 1024) -> dict:
    import soundfile as sf
    from PIL import Image

    audio, sr = sf.read(wav_path, dtype="float32")
    frames = [Image.open(p) for p in frame_paths]

    content = []
    for frame in frames:
        content.append({"type": "image", "image": frame})
    content.append({"type": "audio", "audio": audio, "sample_rate": sr})
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]

    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True,
        return_tensors="pt", add_generation_prompt=True,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        )

    tokens_generated = outputs.shape[-1] - input_len
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)

    del inputs, outputs, frames
    torch.cuda.empty_cache()

    return {
        "text": response.strip(),
        "input_tokens": input_len,
        "tokens_generated": int(tokens_generated),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe video+audio with Gemma4 E2B-it (frames + audio)."
    )
    parser.add_argument("--video", required=True, help="Source video file.")
    parser.add_argument("--output", required=True, help="Output raw JSON path.")
    parser.add_argument("--chunks-json", default="",
                        help="Pre-computed chunk boundaries JSON. If omitted, runs VAD.")
    parser.add_argument("--vad-json", default="")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--target-chunk-s", type=float, default=20)
    parser.add_argument("--max-chunk-s", type=float, default=30)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--target-fps", type=float, default=TARGET_FPS)
    parser.add_argument("--frame-height", type=int, default=FRAME_HEIGHT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt", default=TRANSCRIPTION_PROMPT)
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

        if args.vad_json and Path(args.vad_json).exists():
            log(f"Loading VAD segments from {args.vad_json}")
            vad_segments = json.loads(Path(args.vad_json).read_text(encoding="utf-8"))
        else:
            log("Running Silero VAD...")
            with tempfile.TemporaryDirectory() as workdir:
                vad_segments = run_silero_vad(args.video, workdir)
            log(f"  {len(vad_segments)} speech segments detected")

        chunk_bounds = find_chunk_boundaries(
            vad_segments, total_duration=duration,
            target_chunk_s=args.target_chunk_s, max_chunk_s=args.max_chunk_s,
            min_gap_s=0.5, fallback_min_gap_s=0.2,
        )

    total = len(chunk_bounds)
    if args.max_chunks > 0:
        chunk_bounds = chunk_bounds[:args.max_chunks]
    log(f"{len(chunk_bounds)}/{total} chunks to transcribe")
    log(f"Frames: max {args.max_frames}, target {args.target_fps}fps, {args.frame_height}p")

    # Load model
    model, processor = load_model(args.model, device=args.device)
    log(f"Model loaded on {model.device} ({torch.cuda.memory_allocated()/1024**3:.1f}GB)")

    # Transcribe
    results: list[dict] = []
    t_start = time.monotonic()

    with tempfile.TemporaryDirectory() as workdir:
        for idx, (start_s, end_s) in enumerate(chunk_bounds):
            dur = end_s - start_s
            log(f"Chunk {idx+1}/{len(chunk_bounds)}: {start_s:.1f}-{end_s:.1f}s ({dur:.1f}s)", end=" ")

            wav_path = os.path.join(workdir, f"chunk_{idx}.wav")
            _extract_wav_chunk(args.video, start_s, dur, wav_path)

            frames_dir = os.path.join(workdir, f"frames_{idx}")
            os.makedirs(frames_dir, exist_ok=True)
            frame_paths = _extract_frames(
                args.video, start_s, dur, frames_dir,
                max_frames=args.max_frames, target_fps=args.target_fps,
                height=args.frame_height,
            )

            t0 = time.monotonic()
            try:
                r = transcribe_chunk(
                    model, processor, wav_path, frame_paths, args.prompt,
                    max_new_tokens=args.max_new_tokens,
                )
                elapsed = time.monotonic() - t0

                log(f"{len(frame_paths)}f -> {len(r['text'])} chars, {elapsed:.1f}s")

                results.append({
                    "chunk": idx,
                    "chunk_start_s": start_s,
                    "chunk_end_s": end_s,
                    "text": r["text"],
                    "n_frames": len(frame_paths),
                    "elapsed_seconds": round(elapsed, 3),
                    "input_tokens": r["input_tokens"],
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
                    "n_frames": len(frame_paths),
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


if __name__ == "__main__":
    main()
