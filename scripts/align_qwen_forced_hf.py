#!/usr/bin/env python3
"""Chunk-wise Qwen forced alignment via the official qwen-asr HF package.

This is a benchmark-only ROCm/GPU path for comparing Qwen forced alignment
against the maintained chunked stable-ts baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chigyusubs.metadata import finish_run, metadata_path, start_run, write_metadata


DEFAULT_MODEL = os.environ.get("QWEN_ALIGN_HF_MODEL", "Qwen/Qwen3-ForcedAligner-0.6B")
DEFAULT_DTYPE = os.environ.get("QWEN_ALIGN_HF_DTYPE", "bfloat16")
DEFAULT_DEVICE = os.environ.get("QWEN_ALIGN_HF_DEVICE", "cuda:0")
DEFAULT_LANGUAGE = os.environ.get("QWEN_ALIGN_HF_LANGUAGE", "Japanese")


def _extract_audio_slice(video_path: str, start_s: float, duration_s: float, out_path: str):
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


def _load_chunks(path: str) -> list[dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    chunks: list[dict] = []
    for idx, item in enumerate(raw):
        if "start" in item and "end" in item:
            start = float(item["start"])
            end = float(item["end"])
        elif "chunk_start_s" in item and "chunk_end_s" in item:
            start = float(item["chunk_start_s"])
            end = float(item["chunk_end_s"])
        else:
            raise ValueError(f"Chunk {idx} missing start/end fields")
        text = str(item.get("text", "") or "")
        chunks.append({
            "chunk": int(item.get("chunk", idx)),
            "start": start,
            "end": end,
            "text": text,
        })
    return chunks


def _clean_chunk_text(raw_text: str) -> str:
    lines: list[str] = []
    for line in raw_text.splitlines():
        text = line.strip()
        if not text:
            continue
        if text.startswith("[画面:") or text.startswith("[画面："):
            continue
        if text.startswith("-- "):
            text = text[3:].strip()
        elif text.startswith("- "):
            text = text[2:].strip()
        if text:
            lines.append(text)
    return "\n".join(lines).strip()


def _parse_dtype(dtype_name: str):
    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return mapping[dtype_name]
    except KeyError as exc:
        raise SystemExit(f"Unsupported dtype: {dtype_name}. Use one of: {', '.join(sorted(mapping))}") from exc


def main():
    run = start_run("align_qwen_forced_hf")
    parser = argparse.ArgumentParser(description="Chunk-wise Qwen forced alignment via qwen-asr (HF/ROCm).")
    parser.add_argument("--video", required=True)
    parser.add_argument("--chunks", required=True, help="Chunk transcript JSON (Phase 2 raw chunks or _chunks.json).")
    parser.add_argument("--output-words", required=True, help="Output JSON with aligned words.")
    parser.add_argument("--output-diag", default="", help="Output JSON with per-chunk diagnostics.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"HF model id or local path (default: {DEFAULT_MODEL}).")
    parser.add_argument("--dtype", default=DEFAULT_DTYPE, help=f"torch dtype for model load (default: {DEFAULT_DTYPE}).")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help=f"Device map target for HF load (default: {DEFAULT_DEVICE}).")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE, help=f"Forced aligner language name (default: {DEFAULT_LANGUAGE}).")
    parser.add_argument("--max-chunks", type=int, default=0, help="Optional limit for benchmark smoke tests.")
    args = parser.parse_args()

    if not args.output_diag:
        stem = Path(args.output_words).stem.replace("_words", "")
        args.output_diag = str(Path(args.output_words).with_name(f"{stem}_alignment_chunks_qwen_hf.json"))

    import torch
    from qwen_asr import Qwen3ForcedAligner

    dtype = _parse_dtype(args.dtype)
    print("Loading Qwen3ForcedAligner:")
    print(f"  model:    {args.model}")
    print(f"  device:   {args.device}")
    print(f"  dtype:    {args.dtype}")
    print(f"  language: {args.language}")
    aligner = Qwen3ForcedAligner.from_pretrained(
        args.model,
        device_map=args.device,
        torch_dtype=dtype,
    )
    print(f"Model device: {aligner.device}")
    print(f"Supported languages: {aligner.get_supported_languages()}")

    chunks = _load_chunks(args.chunks)
    if args.max_chunks > 0:
        chunks = chunks[: args.max_chunks]

    segments: list[dict] = []
    diagnostics: list[dict] = []

    with tempfile.TemporaryDirectory() as workdir:
        for idx, chunk in enumerate(chunks):
            c_start = float(chunk["start"])
            c_end = float(chunk["end"])
            c_dur = c_end - c_start
            cleaned_text = _clean_chunk_text(chunk["text"])
            line_count = len([ln for ln in cleaned_text.splitlines() if ln.strip()])
            text_len = len(cleaned_text)

            if not cleaned_text:
                diagnostics.append({
                    "chunk": idx,
                    "chunk_start_s": c_start,
                    "chunk_end_s": c_end,
                    "line_count": 0,
                    "text_length": 0,
                    "segments": 0,
                    "words": 0,
                    "zero_duration_segments": 0,
                    "zero_duration_words": 0,
                    "needs_review": False,
                    "status": "empty",
                })
                continue

            print(
                f"HF Qwen align chunk {idx + 1}/{len(chunks)}: "
                f"{c_start:.1f}-{c_end:.1f}s, {line_count} lines, {text_len} chars"
            )

            slice_wav = os.path.join(workdir, f"qwen_hf_align_{idx}.wav")
            _extract_audio_slice(args.video, c_start, c_dur, slice_wav)

            try:
                result = aligner.align(audio=slice_wav, text=cleaned_text, language=args.language)[0]
                items = list(getattr(result, "items", []) or [])
                words = []
                for item in items:
                    token = str(getattr(item, "text", "") or "")
                    start = float(getattr(item, "start_time", 0.0))
                    end = float(getattr(item, "end_time", 0.0))
                    if not token:
                        continue
                    words.append({
                        "start": start + c_start,
                        "end": end + c_start,
                        "word": token,
                        "probability": 1.0,
                    })

                if words:
                    seg_start = words[0]["start"]
                    seg_end = words[-1]["end"]
                else:
                    seg_start = c_start
                    seg_end = c_start

                segments.append({
                    "start": seg_start,
                    "end": seg_end,
                    "text": cleaned_text,
                    "words": words,
                })

                zero_duration_segments = int(seg_end <= seg_start and bool(cleaned_text))
                zero_duration_words = sum(1 for w in words if w["end"] <= w["start"])
                diagnostics.append({
                    "chunk": idx,
                    "chunk_start_s": c_start,
                    "chunk_end_s": c_end,
                    "line_count": line_count,
                    "text_length": text_len,
                    "segments": 1,
                    "words": len(words),
                    "zero_duration_segments": zero_duration_segments,
                    "zero_duration_words": zero_duration_words,
                    "needs_review": zero_duration_segments > 0 or zero_duration_words > 0 or len(words) == 0,
                    "status": "ok" if words else "no_words",
                })
            except Exception as exc:
                msg = str(exc).strip().splitlines()[0] if str(exc).strip() else repr(exc)
                diagnostics.append({
                    "chunk": idx,
                    "chunk_start_s": c_start,
                    "chunk_end_s": c_end,
                    "line_count": line_count,
                    "text_length": text_len,
                    "error": msg,
                    "status": "failed",
                })
                print(f"  HF Qwen chunk {idx + 1} failed: {msg}")

    segments.sort(key=lambda seg: (float(seg["start"]), float(seg["end"])))
    Path(args.output_words).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_words).write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.output_diag).write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")

    metadata = finish_run(
        run,
        inputs={
            "video": args.video,
            "chunks_json": args.chunks,
        },
        outputs={
            "words_json": args.output_words,
            "alignment_chunks_json": args.output_diag,
        },
        settings={
            "model": args.model,
            "dtype": args.dtype,
            "device": args.device,
            "language": args.language,
            "max_chunks": args.max_chunks,
        },
        stats={
            "chunks_loaded": len(chunks),
            "segments_written": len(segments),
            "words_written": sum(len(seg.get("words", [])) for seg in segments),
            "failed_chunks": sum(1 for item in diagnostics if item.get("status") == "failed"),
            "review_chunks": sum(1 for item in diagnostics if item.get("needs_review")),
        },
    )
    write_metadata(args.output_words, metadata)
    print(f"Wrote {len(segments)} segments to {args.output_words}")
    print(f"Wrote diagnostics to {args.output_diag}")
    print(f"Metadata written: {metadata_path(args.output_words)}")


if __name__ == "__main__":
    main()
