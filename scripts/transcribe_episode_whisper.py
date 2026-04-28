#!/usr/bin/env python3
"""Full-episode Japanese ASR via transformers — Phase A baseline.

Mirrors transcribe_episode_e4b.py for apples-to-apples comparison: same
VAD packing, same per-chunk WAV cache (auto-shared with the E4B runner),
same JSON + VTT output schema. Difference: runs whisper-class models
via transformers AutoModelForSpeechSeq2Seq instead of vLLM.

Models supported:
- whisper-large-v3 — multilingual baseline (openai/whisper-large-v3).
- anime-whisper — kotoba-whisper-v2.0 fine-tuned on 5300 h of galgame
  voice acting (litagin/anime-whisper). Per its model card, MUST NOT
  use an initial prompt — causes hallucinations.

GPU/ROCm: set `ROCR_VISIBLE_DEVICES=1` for the 7900 XTX (24 GB, gfx1100).

Usage:
    LD_LIBRARY_PATH=/opt/rocm/lib ROCR_VISIBLE_DEVICES=1 \\
    python3.12 scripts/transcribe_episode_whisper.py \\
        --episode killah_kuts_s01e01 \\
        --video samples/episodes/killah_kuts_s01e01/source/foo.mkv \\
        --model whisper-large-v3 \\
        --out  samples/episodes/killah_kuts_s01e01/transcription/\\
                killah_kuts_s01e01_whisper_large_v3_raw.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
# Reuse chunk packing + WAV slicing + VTT writer + validator from the
# E4B runner so both pipelines see identical chunks. Cache filenames
# match (chunk_NNNN_Ss.wav), so passing --cache-dir at the e4b cache
# avoids re-extracting all 131 slices.
from transcribe_episode_e4b import (  # noqa: E402
    pack_vad_segments, extract_wav_slice, write_vtt, validate,
)

MODEL_CONFIGS: dict[str, dict] = {
    "whisper-large-v3": {
        "hf_id": "openai/whisper-large-v3",
        "generate_kwargs": {"language": "ja", "task": "transcribe"},
    },
    "anime-whisper": {
        # Fine-tune of kotoba-whisper-v2.0 on galgame voice acting.
        # Per model card: NO initial prompt, no_repeat_ngram_size=0,
        # repetition_penalty=1.0 are the defaults. The mirror at
        # 9r4n4y/anime-whisper-stt-Japanese-Backup is identical weights.
        "hf_id": "litagin/anime-whisper",
        "generate_kwargs": {
            "language": "ja", "task": "transcribe",
            "no_repeat_ngram_size": 0,
            "repetition_penalty": 1.0,
        },
    },
}

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def load_model(hf_id: str, device: str, dtype: torch.dtype):
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    print(f"Loading {hf_id} → {device} ({dtype})")
    t0 = time.monotonic()
    processor = AutoProcessor.from_pretrained(hf_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        hf_id, torch_dtype=dtype, low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    print(f"  loaded in {time.monotonic() - t0:.1f}s")
    return model, processor


def transcribe_chunk(
    *, model, processor, wav_path: Path, device: str,
    dtype: torch.dtype, generate_kwargs: dict,
) -> tuple[str, dict]:
    audio, sr = sf.read(str(wav_path))
    if sr != 16000:
        raise SystemExit(f"expected 16 kHz mono WAV, got {sr} Hz from {wav_path}")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device=device, dtype=dtype)
    with torch.inference_mode():
        gen = model.generate(input_features, **generate_kwargs)
    text = processor.batch_decode(gen, skip_special_tokens=True)[0]
    n_tokens = int(gen.shape[-1])
    return text.strip(), {"completion_tokens": n_tokens}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vtt", default="")
    ap.add_argument("--model", choices=list(MODEL_CONFIGS), required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    ap.add_argument("--target-dur", type=float, default=18.0)
    ap.add_argument("--max-dur", type=float, default=25.0)
    ap.add_argument("--max-gap", type=float, default=1.5)
    ap.add_argument("--min-dur", type=float, default=10.0)
    ap.add_argument("--cache-dir", default="",
                    help="WAV chunk cache (default: shared e4b cache for "
                         "apples-to-apples)")
    ap.add_argument("--start-s", type=float, default=0.0)
    ap.add_argument("--end-s", type=float, default=0.0,
                    help="0 = no upper bound (smoke-test knob)")
    args = ap.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    dtype = DTYPES[args.dtype]

    episode_dir = REPO / "samples" / "episodes" / args.episode
    vad_path = episode_dir / "transcription" / "silero_vad_segments.json"
    if not vad_path.exists():
        raise SystemExit(f"silero VAD not found at {vad_path}")
    vad = json.loads(vad_path.read_text(encoding="utf-8"))

    chunks = pack_vad_segments(
        vad, target_dur=args.target_dur, max_dur=args.max_dur,
        max_gap=args.max_gap, min_dur=args.min_dur,
    )
    print(f"VAD segments:    {len(vad)}")
    print(f"Packed chunks:   {len(chunks)}")
    durs = [c["duration_s"] for c in chunks]
    print(f"  duration min/mean/max: "
          f"{min(durs):.1f}s / {sum(durs)/len(durs):.1f}s / {max(durs):.1f}s")

    out_path = Path(args.out).resolve()
    cache_dir = (Path(args.cache_dir).resolve() if args.cache_dir
                 else episode_dir / "transcription"
                      / f"{args.episode}_e4b_chunks_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cache dir:       {cache_dir}")

    video = Path(args.video).resolve()
    if not video.exists():
        raise SystemExit(f"video not found: {video}")

    selected = [c for c in chunks
                if c["chunk_start_s"] >= args.start_s
                and (not args.end_s or c["chunk_start_s"] < args.end_s)]
    print(f"Selected:        {len(selected)} chunks")
    print(f"Model:           {args.model}  ({cfg['hf_id']})")
    print(f"Device / dtype:  {args.device}  {args.dtype}")

    model, processor = load_model(cfg["hf_id"], args.device, dtype)

    records: list[dict] = []
    n_flagged = 0
    t_total = time.monotonic()
    for c in selected:
        idx = c["chunk"]
        start, end = c["chunk_start_s"], c["chunk_end_s"]
        wav = cache_dir / f"chunk_{idx:04d}_{start:.0f}s.wav"
        extract_wav_slice(video, wav, start, end - start)
        t0 = time.monotonic()
        try:
            text, meta = transcribe_chunk(
                model=model, processor=processor, wav_path=wav,
                device=args.device, dtype=dtype,
                generate_kwargs=cfg["generate_kwargs"],
            )
        except Exception as e:
            wall = time.monotonic() - t0
            print(f"  chunk {idx:>4} [{start:7.1f}-{end:7.1f}]  "
                  f"ERROR {type(e).__name__}: {e}  wall={wall:.1f}s")
            records.append({**c, "text": "", "wall": round(wall, 2),
                            "error": f"{type(e).__name__}: {e}",
                            "flags": ["error"]})
            n_flagged += 1
            continue
        wall = time.monotonic() - t0
        v = validate(text)
        rec = {**c, "text": text, "wall": round(wall, 2), **meta, **v}
        records.append(rec)
        if v["flags"]:
            n_flagged += 1
        flag_str = f"  ⚠ {','.join(v['flags'])}" if v["flags"] else ""
        print(f"  chunk {idx:>4} [{start:7.1f}-{end:7.1f}]  "
              f"wall={wall:.1f}s  cjk={v['cjk_ratio']}  "
              f"chars={len(text.strip())}{flag_str}")
        if text.strip():
            print(f"    | {text.strip()[:120]}")

    total_wall = time.monotonic() - t_total
    total_audio = sum(c["duration_s"] for c in selected)
    print(f"\n==== done in {total_wall:.1f}s ({total_wall/60:.1f}m) ====")
    print(f"  audio:           {total_audio:.1f}s")
    print(f"  realtime ratio:  {total_wall/max(total_audio,1):.2f}x")
    print(f"  chunks:          {len(records)}")
    print(f"  flagged:         {n_flagged}")
    print(f"  total chars:     "
          f"{sum(len((r.get('text') or '').strip()) for r in records)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({
            "episode": args.episode,
            "video": str(video),
            "model": args.model,
            "hf_id": cfg["hf_id"],
            "config": args.model,
            "dtype": args.dtype,
            "vad_path": str(vad_path),
            "n_chunks": len(records),
            "n_flagged": n_flagged,
            "total_audio_s": round(total_audio, 1),
            "total_wall_s": round(total_wall, 1),
            "chunks": records,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {out_path}")
    vtt_path = Path(args.vtt) if args.vtt else out_path.with_suffix(".vtt")
    write_vtt(records, vtt_path)
    print(f"Wrote {vtt_path}")


if __name__ == "__main__":
    main()
