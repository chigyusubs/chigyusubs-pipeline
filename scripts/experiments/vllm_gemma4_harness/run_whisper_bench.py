#!/usr/bin/env python3
"""Run faster-whisper on a vLLM harness eval spec and score the outputs.

This intentionally consumes the same 8-second WAV clips that run_bench.py
uses, so Whisper and E4B are compared on identical media windows and target
sets instead of mismatched full-episode chunks.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from scoring import aggregate, score_segment  # noqa: E402


def transcribe_wav(model, wav: Path) -> tuple[str, float]:
    t0 = time.monotonic()
    segments, _info = model.transcribe(
        str(wav),
        language="ja",
        condition_on_previous_text=False,
        vad_filter=False,
        beam_size=5,
        compression_ratio_threshold=2.4,
    )
    text = "".join(seg.text.strip() for seg in segments if seg.text.strip())
    return text, time.monotonic() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="eval spec JSON")
    ap.add_argument("--out", required=True, help="output results JSON")
    ap.add_argument("--model", default="large-v3",
                    help="faster-whisper model name/path")
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    spec_path = Path(args.spec).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec_dir = spec_path.parent
    config_name = f"whisper_{args.model.replace('-', '_')}"

    print(f"Spec: {spec_path.name}  ({spec['n_segments']} segments, "
          f"episode={spec['episode']})", flush=True)
    print(f"Model: {args.model}  device={args.device}  "
          f"compute={args.compute_type}", flush=True)

    from faster_whisper import WhisperModel

    t_load = time.monotonic()
    model = WhisperModel(
        args.model, device=args.device, compute_type=args.compute_type,
    )
    print(f"Loaded in {time.monotonic() - t_load:.1f}s", flush=True)

    results: list[dict] = []
    for seg in spec["segments"]:
        wav = (spec_dir / seg["wav_rel"]).resolve()
        print(f"\n### {seg['name']}  "
              f"[{seg['seg_start']:.1f}-{seg['seg_end']:.1f}]",
              flush=True)
        print(f"    target: {seg['text'][:80]}", flush=True)
        try:
            text, wall = transcribe_wav(model, wav)
            s = score_segment(text, seg["text"], seg["kata"], seg["names"])
            entry = {
                "text": text,
                "wall": round(wall, 3),
                **s.as_dict(),
            }
            print(f"  {config_name}: kata={entry['kata_recall']} "
                  f"name={entry['name_recall']} "
                  f"lcs={entry['lcs_ratio']}  wall={wall:.2f}s",
                  flush=True)
            print(f"    out: {text[:150]}", flush=True)
        except Exception as exc:
            entry = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  {config_name}: ERROR {entry['error']}", flush=True)
        results.append({
            "seg": seg["name"],
            "target": seg["text"],
            "kata": seg["kata"],
            "names": seg["names"],
            config_name: entry,
        })

    summary = [aggregate(results, config_name)]
    s = summary[0]
    print("\n\n=========== SUMMARY ===========", flush=True)
    print(f"  {s['config']}:  kata {s['kata_num']}/{s['kata_den']} "
          f"({s['kata_pct']}%)  name {s['name_num']}/{s['name_den']} "
          f"({s['name_pct']}%)  mean lcs {s['mean_lcs']}  "
          f"(n={s['n_segments']})", flush=True)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({
            "spec": str(spec_path),
            "model": args.model,
            "device": args.device,
            "compute_type": args.compute_type,
            "configs": [config_name],
            "summary": summary,
            "results": results,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
