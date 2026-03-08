#!/usr/bin/env python3
"""Chunk-wise Qwen forced-alignment benchmark.

This is intentionally separate from the maintained stable-ts Phase 3 path.
It benchmarks Qwen alignment on the same chunked transcript/audio inputs and
emits a compatible words JSON plus per-chunk diagnostics.
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


DEFAULT_QWEN_PYTHON = ".venv-qwen-align/bin/python"
DEFAULT_QWEN_ASR_MODEL = os.environ.get("QWEN_ALIGN_ASR_MODEL", "qwen3-asr-0.6b-f16")
DEFAULT_QWEN_ALIGN_MODEL = os.environ.get(
    "QWEN_ALIGN_FORCED_MODEL",
    "qwen3-forced-aligner-0.6b-f16",
)
DEFAULT_QWEN_MODELS_DIR = os.environ.get("QWEN_ALIGN_MODELS_DIR", ".cache/qwen-align")


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


def _driver_main(args: argparse.Namespace) -> int:
    try:
        from py_qwen3_asr_cpp.model import Qwen3ASRModel
    except ImportError as exc:
        raise SystemExit(
            "py-qwen3-asr-cpp is not installed in the active interpreter. "
            "Use scripts/setup_qwen_aligner_env.sh and rerun with "
            f"--python-bin {DEFAULT_QWEN_PYTHON}"
        ) from exc

    print(f"Loading Qwen aligner:")
    print(f"  ASR model:   {args.asr_model}")
    print(f"  align model: {args.align_model}")
    model = Qwen3ASRModel(
        asr_model=args.asr_model,
        align_model=args.align_model,
        models_dir=args.models_dir,
        n_threads=args.threads,
        language=args.language,
        print_timing=False,
    )
    if args.download_only:
        print(f"Qwen models available under {args.models_dir}")
        return 0

    chunks = _load_chunks(args.chunks)

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
                f"Qwen align chunk {idx + 1}/{len(chunks)}: "
                f"{c_start:.1f}-{c_end:.1f}s, {line_count} lines, {text_len} chars"
            )

            slice_wav = os.path.join(workdir, f"qwen_align_{idx}.wav")
            _extract_audio_slice(args.video, c_start, c_dur, slice_wav)

            try:
                alignment = model.align(slice_wav, text=cleaned_text)
                raw_words = list(getattr(alignment, "words", []) or [])
                words = []
                for word in raw_words:
                    token = getattr(word, "word", None)
                    if token is None and isinstance(word, dict):
                        token = word.get("word")
                    start = getattr(word, "start", None)
                    if start is None and isinstance(word, dict):
                        start = word.get("start")
                    end = getattr(word, "end", None)
                    if end is None and isinstance(word, dict):
                        end = word.get("end")
                    if token is None or start is None or end is None:
                        continue
                    start = float(start)
                    end = float(end)
                    # Some bindings/examples describe ms even though CLI JSON uses seconds.
                    if start > c_dur * 5 or end > c_dur * 5:
                        start /= 1000.0
                        end /= 1000.0
                    words.append({
                        "start": start + c_start,
                        "end": end + c_start,
                        "word": str(token),
                        "probability": float(getattr(word, "probability", 1.0))
                        if hasattr(word, "probability") else 1.0,
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
                print(f"  Qwen chunk {idx + 1} failed: {msg}")

    segments.sort(key=lambda seg: (float(seg["start"]), float(seg["end"])))
    Path(args.output_words).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_words).write_text(
        json.dumps(segments, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    Path(args.output_diag).write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(segments)} segments to {args.output_words}")
    print(f"Wrote diagnostics to {args.output_diag}")
    return 0


def main():
    run = start_run("align_qwen_forced")
    parser = argparse.ArgumentParser(description="Chunk-wise Qwen forced-alignment benchmark.")
    parser.add_argument("--video")
    parser.add_argument("--chunks", help="Chunk transcript JSON (Phase 2 raw chunks or _chunks.json).")
    parser.add_argument("--output-words", help="Output JSON with aligned words.")
    parser.add_argument("--output-diag", default="", help="Output JSON with per-chunk diagnostics.")
    parser.add_argument("--python-bin", default=DEFAULT_QWEN_PYTHON, help=f"Python interpreter with py-qwen3-asr-cpp installed (default: {DEFAULT_QWEN_PYTHON}).")
    parser.add_argument(
        "--asr-model",
        default=DEFAULT_QWEN_ASR_MODEL,
        help="ASR GGUF model path or short py-qwen3-asr-cpp model name (for example: qwen3-asr-0.6b-f16).",
    )
    parser.add_argument(
        "--align-model",
        default=DEFAULT_QWEN_ALIGN_MODEL,
        help="Forced aligner GGUF model path or short py-qwen3-asr-cpp model name (for example: qwen3-forced-aligner-0.6b-f16).",
    )
    parser.add_argument(
        "--models-dir",
        default=DEFAULT_QWEN_MODELS_DIR,
        help=f"Directory for downloaded Qwen GGUF models (default: {DEFAULT_QWEN_MODELS_DIR}).",
    )
    parser.add_argument("--language", default="ja", help="Language hint for the Qwen aligner (default: ja).")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for the Qwen backend (default: 4).")
    parser.add_argument("--download-only", action="store_true", help="Download/cache the Qwen models and exit without aligning.")
    parser.add_argument("--driver", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not args.download_only and (not args.video or not args.chunks or not args.output_words):
        parser.error("--video, --chunks, and --output-words are required unless --download-only is set.")

    if not args.download_only and not args.output_diag:
        stem = Path(args.output_words).stem.replace("_words", "")
        args.output_diag = str(Path(args.output_words).with_name(f"{stem}_alignment_chunks_qwen.json"))

    if args.driver:
        sys.exit(_driver_main(args))

    py_bin = Path(args.python_bin)
    if not py_bin.exists():
        raise SystemExit(
            f"Python interpreter not found: {py_bin}\n"
            f"Run scripts/setup_qwen_aligner_env.sh first or pass --python-bin."
        )

    cmd = [
        str(py_bin),
        __file__,
        "--driver",
        "--asr-model", args.asr_model,
        "--align-model", args.align_model,
        "--models-dir", args.models_dir,
        "--language", args.language,
        "--threads", str(args.threads),
    ]
    if not args.download_only:
        cmd.extend([
            "--video", args.video,
            "--chunks", args.chunks,
            "--output-words", args.output_words,
            "--output-diag", args.output_diag,
        ])
    if args.download_only:
        cmd.append("--download-only")
    print(f"Running Qwen aligner via {py_bin}")
    subprocess.run(cmd, check=True)

    if args.download_only:
        return

    segments_data = json.loads(Path(args.output_words).read_text(encoding="utf-8"))
    diagnostics = json.loads(Path(args.output_diag).read_text(encoding="utf-8"))
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
            "python_bin": str(py_bin),
            "asr_model": args.asr_model,
            "align_model": args.align_model,
            "models_dir": args.models_dir,
            "language": args.language,
            "threads": args.threads,
        },
        stats={
            "chunks_loaded": len(_load_chunks(args.chunks)),
            "segments_written": len(segments_data),
            "words_written": sum(len(seg.get("words", [])) for seg in segments_data),
            "failed_chunks": sum(1 for item in diagnostics if item.get("status") == "failed"),
            "review_chunks": sum(1 for item in diagnostics if item.get("needs_review")),
        },
    )
    write_metadata(args.output_words, metadata)
    print(f"Metadata written: {metadata_path(args.output_words)}")


if __name__ == "__main__":
    main()
