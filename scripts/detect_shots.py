#!/usr/bin/env python3
"""Detect shot boundaries with TransNetV2 and cache them per episode.

Output:
    samples/episodes/<slug>/shots/<media-stem>.shots.json   # raw shot list
    samples/episodes/<slug>/shots/<media-stem>.shots.json.meta.json

The shot list is the model output: each entry has
    {"start_frame", "end_frame", "start_time", "end_time", "probability"}

Usage:
    python3.12 scripts/detect_shots.py --video samples/episodes/<slug>/source/<file>.mp4
    python3.12 scripts/detect_shots.py --video <path> --out <path>.shots.json --force
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chigyusubs.metadata import (
    finish_run,
    inherit_run_id,
    start_run,
    write_metadata,
)
from chigyusubs.paths import find_episode_dir_from_path


def default_output_path(video: Path) -> Path:
    episode_dir = find_episode_dir_from_path(video)
    if episode_dir is None:
        return video.with_suffix(".shots.json")
    out_dir = episode_dir / "shots"
    return out_dir / f"{video.stem}.shots.json"


def detect(video: Path, threshold: float) -> tuple[list[dict], float]:
    from transnetv2_pytorch import TransNetV2

    print(f"loading TransNetV2...")
    t0 = time.time()
    model = TransNetV2()
    model.eval()
    print(f"  init {time.time() - t0:.1f}s, device={next(model.parameters()).device}")

    print(f"detecting shots in {video.name}...")
    t0 = time.time()
    shots = model.detect_scenes(str(video), threshold=threshold)
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s — {len(shots)} shots")

    # Coerce non-JSON-serializable values (numpy scalars) to plain Python.
    cleaned: list[dict] = []
    for s in shots:
        cleaned.append(
            {
                "start_frame": int(s["start_frame"]),
                "end_frame": int(s["end_frame"]),
                "start_time": float(s["start_time"]),
                "end_time": float(s["end_time"]),
                "probability": float(s.get("probability", 0.0)),
            }
        )
    return cleaned, elapsed


def main() -> None:
    ap = argparse.ArgumentParser(description="Detect shot boundaries with TransNetV2.")
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None, help="Output .shots.json (default: <episode>/shots/<stem>.shots.json)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--force", action="store_true", help="Re-detect even if cache exists")
    args = ap.parse_args()

    out_path: Path = args.out or default_output_path(args.video)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not args.force:
        print(f"cached shots already at {out_path} — pass --force to re-detect")
        return

    run = start_run("detect_shots")
    run = inherit_run_id(run, args.video)

    shots, elapsed = detect(args.video, args.threshold)

    payload = {"shots": shots, "threshold": args.threshold}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    durations = [s["end_time"] - s["start_time"] for s in shots]
    durations.sort()

    def pct(p: float) -> float:
        if not durations:
            return 0.0
        i = max(0, min(len(durations) - 1, int(round(p * (len(durations) - 1)))))
        return durations[i]

    stats = {
        "shot_count": len(shots),
        "total_seconds": shots[-1]["end_time"] if shots else 0.0,
        "shots_per_minute": (len(shots) / shots[-1]["end_time"] * 60.0) if shots else 0.0,
        "duration_p10": round(pct(0.10), 3),
        "duration_p50": round(pct(0.50), 3),
        "duration_p90": round(pct(0.90), 3),
        "detect_seconds": round(elapsed, 2),
    }

    run = finish_run(
        run,
        inputs={"video": str(args.video)},
        outputs={"shots": str(out_path)},
        settings={"threshold": args.threshold},
        stats=stats,
    )
    write_metadata(out_path, run)

    print(f"wrote {out_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
