#!/usr/bin/env python3
"""Transcribe audio with NVIDIA Parakeet Japanese NeMo ASR.

Example:
  LD_LIBRARY_PATH=/opt/rocm/lib MPLCONFIGDIR=/tmp/matplotlib-parakeet \
    .venv-nemo/bin/python scripts/experiments/transcribe_parakeet_nemo.py \
      --audio samples/episodes/killah_kuts_s01e01/transcription/killah_kuts_s01e01_e4b_chunks_cache/chunk_0000_6s.wav \
      --output /tmp/parakeet.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
import wave
from pathlib import Path
from typing import Any


MODEL_NAME = "nvidia/parakeet-tdt_ctc-0.6b-ja"
CHUNK_RE = re.compile(r"chunk_(\d+)_(\d+(?:\.\d+)?)s\.wav$")


def hypothesis_text(item: Any) -> str:
    if hasattr(item, "text"):
        return str(item.text)
    return str(item)


def audio_duration_s(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception:
        return None


def chunk_metadata(path: Path) -> dict[str, Any]:
    meta: dict[str, Any] = {"audio": str(path)}
    m = CHUNK_RE.search(path.name)
    if m:
        meta["chunk"] = int(m.group(1))
        meta["chunk_start_s"] = float(m.group(2))
    duration = audio_duration_s(path)
    if duration is not None:
        meta["duration_s"] = round(duration, 3)
        if "chunk_start_s" in meta:
            meta["chunk_end_s"] = round(float(meta["chunk_start_s"]) + duration, 3)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe audio with NVIDIA Parakeet TDT-CTC 0.6B JA via NeMo.")
    parser.add_argument("--audio", nargs="+", default=[], help="One or more 16 kHz mono WAV files.")
    parser.add_argument("--audio-glob", default="", help="Glob for WAV files. Results are sorted by path.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--vtt-output", default="", help="Optional VTT output path for chunk-timed inputs.")
    parser.add_argument("--model", default=MODEL_NAME, help=f"NeMo/HF model name. Default: {MODEL_NAME}")
    parser.add_argument("--decoder", choices=["tdt", "ctc"], default="tdt", help="Decoder to use. Default: tdt.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--separate-calls",
        action="store_true",
        help="Call model.transcribe once per file. Slower, but avoids NeMo dataloader/decoder state issues seen on long file lists.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if ROCm/CUDA is visible.")
    args = parser.parse_args()

    import torch
    import nemo.collections.asr as nemo_asr

    audio_inputs = [Path(p) for p in args.audio]
    if args.audio_glob:
        audio_inputs.extend(sorted(Path().glob(args.audio_glob)))
    if not audio_inputs:
        raise SystemExit("pass --audio and/or --audio-glob")
    audio_paths = [p.resolve() for p in audio_inputs]
    for p in audio_paths:
        if not p.exists():
            raise FileNotFoundError(p)

    started = time.perf_counter()
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=args.model)
    model = model.eval()

    if torch.cuda.is_available() and not args.cpu:
        model = model.cuda()

    if args.decoder == "ctc":
        model.change_decoding_strategy(decoder_type="ctc", verbose=False)
    elif hasattr(model, "change_decoding_strategy"):
        model.change_decoding_strategy(decoder_type="rnnt", verbose=False)

    load_s = time.perf_counter() - started
    transcribe_started = time.perf_counter()
    if audio_paths:
        # On this ROCm/NeMo stack, the first call after decoder setup can
        # return blank/garbage while subsequent calls are stable.
        model.transcribe([str(audio_paths[0])], batch_size=1, return_hypotheses=True, verbose=False)
    if args.separate_calls:
        hypotheses = []
        for i, path in enumerate(audio_paths, start=1):
            result = model.transcribe([str(path)], batch_size=1, return_hypotheses=True, verbose=False)
            batch_hypotheses = result[0] if isinstance(result, tuple) else result
            hypotheses.append(batch_hypotheses[0])
            print(f"[{i}/{len(audio_paths)}] {path.name}", flush=True)
    else:
        result = model.transcribe([str(p) for p in audio_paths], batch_size=args.batch_size, return_hypotheses=True)
        hypotheses = result[0] if isinstance(result, tuple) else result
    transcribe_s = time.perf_counter() - transcribe_started

    rows = []
    for path, hyp in zip(audio_paths, hypotheses, strict=True):
        row = chunk_metadata(path)
        row["text"] = hypothesis_text(hyp)
        rows.append(row)

    output = {
        "model": args.model,
        "decoder": args.decoder,
        "device": str(next(model.parameters()).device),
        "batch_size": args.batch_size,
        "separate_calls": args.separate_calls,
        "load_s": load_s,
        "transcribe_s": transcribe_s,
        "n_chunks": len(rows),
        "total_audio_s": round(sum(r.get("duration_s", 0.0) for r in rows), 3),
        "items": rows,
        "chunks": rows,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.vtt_output:
        vtt_path = Path(args.vtt_output)
        vtt_path.parent.mkdir(parents=True, exist_ok=True)
        vtt_path.write_text(render_vtt(rows), encoding="utf-8")

    for row in rows:
        print(row["text"])
    print(f"wrote {out_path}")
    if args.vtt_output:
        print(f"wrote {args.vtt_output}")


def format_vtt_time(seconds: float) -> str:
    ms_total = int(round(seconds * 1000))
    h, rem = divmod(ms_total, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def render_vtt(rows: list[dict[str, Any]]) -> str:
    lines = ["WEBVTT", ""]
    for row in rows:
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        if "chunk_start_s" not in row or "chunk_end_s" not in row:
            continue
        lines.append(f"{format_vtt_time(float(row['chunk_start_s']))} --> {format_vtt_time(float(row['chunk_end_s']))}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
