#!/usr/bin/env python3
"""Transcribe audio using openbmb/MiniCPM-o-4_5 via transformers.

Runs audio-only chunked transcription so the output can be compared directly
against the existing local-model experiments on Killah Kuts S01E01.

Requires: system python3.12 with ROCm PyTorch, transformers, soundfile, and
MiniCPM-o's remote-code dependencies.
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

# Force only the discrete GPU before torch import so ROCm never sees the iGPU.
os.environ.setdefault("ROCR_VISIBLE_DEVICES", "0")

# Some audio stacks still import librosa -> numba at module import time.
import types as _types

_fake_numba = _types.ModuleType("numba")
_fake_numba.jit = lambda *a, **kw: (lambda f: f)
sys.modules["numba"] = _fake_numba

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from chigyusubs.audio import get_duration
from chigyusubs.chunking import find_chunk_boundaries
from chigyusubs.metadata import finish_run, metadata_path, start_run, write_metadata
from chigyusubs.vad import run_silero_vad


def log(msg: str = "", end: str = "\n"):
    print(msg, end=end, flush=True)


DEFAULT_MODEL = "openbmb/MiniCPM-o-4_5"

TRANSCRIPTION_PROMPT = (
    "Please listen carefully and transcribe ALL spoken Japanese dialogue in this audio clip faithfully.\n"
    "Output ONLY plain Japanese text.\n"
    "Do NOT translate.\n"
    "Do NOT add timestamps, explanations, JSON, markdown, or speaker names.\n"
    "Indicate speaker turns by starting each new utterance with '-- '.\n"
    "Use natural Japanese punctuation (、。！？).\n"
    "Do NOT skip reactions, filler, or overlapping speech when audible.\n"
    "If the clip contains only silence, music, or unusable noise, output nothing."
)


def _extract_wav_chunk(video_path: str, start_s: float, duration_s: float, out_path: str):
    """Extract a 16 kHz mono WAV chunk via ffmpeg."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(start_s),
            "-i",
            video_path,
            "-t",
            str(duration_s),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            out_path,
        ],
        capture_output=True,
        check=True,
    )


def _load_vad_chunk_bounds(chunk_json_path: str) -> list[tuple[float, float]]:
    with open(chunk_json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return [(float(c["start_sec"]), float(c["end_sec"])) for c in chunks]


def load_model(model_id: str, device: str = "cuda:0"):
    """Load MiniCPM-o in audio-only text-output mode."""
    from transformers import AutoModel

    log(f"Loading model: {model_id} on {device}")
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        init_vision=False,
        init_audio=True,
        init_tts=False,
    )
    model = model.eval().to(device)
    return model


def _normalize_response(response) -> str:
    if isinstance(response, str):
        return response.strip()
    if isinstance(response, dict):
        for key in ("text", "answer", "content", "response"):
            value = response.get(key)
            if isinstance(value, str):
                return value.strip()
    if isinstance(response, (list, tuple)):
        for item in response:
            if isinstance(item, str) and item.strip():
                return item.strip()
            if isinstance(item, dict):
                for key in ("text", "answer", "content", "response"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
    return str(response).strip()


def transcribe_chunk(
    model,
    wav_path: str,
    prompt: str,
    *,
    max_new_tokens: int = 512,
) -> dict:
    """Transcribe a single audio chunk."""
    import soundfile as sf

    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    msgs = [{"role": "user", "content": [prompt, audio]}]

    with torch.inference_mode():
        response = model.chat(
            msgs=msgs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_tts_template=False,
            generate_audio=False,
            enable_thinking=False,
            sampling_rate=sr,
        )

    text = _normalize_response(response)
    torch.cuda.empty_cache()
    return {"text": text}


def main():
    run = start_run("transcribe_minicpm_o_audio")

    parser = argparse.ArgumentParser(
        description="Transcribe audio with MiniCPM-o 4.5 via transformers."
    )
    parser.add_argument("--video", required=True, help="Source video/audio file.")
    parser.add_argument("--output", required=True, help="Output raw JSON path.")
    parser.add_argument("--vad-json", default="")
    parser.add_argument("--chunk-json", default="")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"HF model ID (default: {DEFAULT_MODEL}).")
    parser.add_argument("--target-chunk-s", type=float, default=20)
    parser.add_argument("--max-chunk-s", type=float, default=30)
    parser.add_argument("--max-chunks", type=int, default=0, help="Limit chunks for testing (0 = all).")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", default="cuda:0", help="Torch device (default: cuda:0).")
    parser.add_argument("--prompt", default=TRANSCRIPTION_PROMPT)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
        chunk_bounds = chunk_bounds[: args.max_chunks]
    log(f"{len(chunk_bounds)}/{total} chunks to transcribe")

    durations = [end - start for start, end in chunk_bounds]
    if durations:
        log(
            "Chunk durations: "
            f"min={min(durations):.1f}s avg={sum(durations) / len(durations):.1f}s max={max(durations):.1f}s"
        )

    model = load_model(args.model, device=args.device)
    log(f"Model loaded on {args.device}")

    results: list[dict] = []
    t_start = time.monotonic()

    with tempfile.TemporaryDirectory() as workdir:
        for idx, (start_s, end_s) in enumerate(chunk_bounds):
            dur = end_s - start_s
            log(f"Chunk {idx + 1}/{len(chunk_bounds)}: {start_s:.1f}-{end_s:.1f}s ({dur:.1f}s)")

            wav_path = os.path.join(workdir, f"minicpm_o_{idx}.wav")
            _extract_wav_chunk(args.video, start_s, dur, wav_path)
            wav_size_mb = os.path.getsize(wav_path) / (1024 * 1024)

            t0 = time.monotonic()
            try:
                r = transcribe_chunk(
                    model,
                    wav_path,
                    args.prompt,
                    max_new_tokens=args.max_new_tokens,
                )
                elapsed = time.monotonic() - t0

                log(f"  {len(r['text'])} chars, {elapsed:.1f}s")
                results.append(
                    {
                        "chunk": idx,
                        "chunk_start_s": start_s,
                        "chunk_end_s": end_s,
                        "text": r["text"],
                        "chunk_size_mb": round(wav_size_mb, 3),
                        "elapsed_seconds": round(elapsed, 3),
                    }
                )
            except Exception as exc:
                elapsed = time.monotonic() - t0
                msg = str(exc).strip().splitlines()[0] if str(exc).strip() else repr(exc)
                log(f"  FAILED ({elapsed:.1f}s): {msg}")
                results.append(
                    {
                        "chunk": idx,
                        "chunk_start_s": start_s,
                        "chunk_end_s": end_s,
                        "text": "",
                        "chunk_size_mb": round(wav_size_mb, 3),
                        "elapsed_seconds": round(elapsed, 3),
                        "error": msg,
                    }
                )

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
