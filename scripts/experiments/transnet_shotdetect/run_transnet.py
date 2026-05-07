"""Detect shot boundaries with TransNetV2 and analyse against reflow cues.

Usage:
  python3.12 scripts/experiments/transnet_shotdetect/run_transnet.py \
      --video samples/episodes/<slug>/source/<file>.mp4 \
      --reflow samples/episodes/<slug>/transcription/<run>_reflow.vtt \
      --out scripts/experiments/transnet_shotdetect/results/<slug>.json
"""

import argparse
import json
import re
import time
from pathlib import Path

from transnetv2_pytorch import TransNetV2

VTT_TS = re.compile(r"(\d+:)?(\d{2}):(\d{2})[.,](\d{1,3})")


def parse_ts(s: str) -> float:
    m = VTT_TS.match(s.strip())
    if not m:
        raise ValueError(s)
    h = int((m.group(1) or "0:")[:-1])
    return h * 3600 + int(m.group(2)) * 60 + int(m.group(3)) + int(m.group(4)) / 10 ** len(m.group(4))


def parse_vtt(path: Path) -> list[dict]:
    cues = []
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        if "-->" in lines[i]:
            a, b = [x.strip() for x in lines[i].split("-->")]
            j = i + 1
            text_lines = []
            while j < len(lines) and lines[j].strip() != "":
                text_lines.append(lines[j])
                j += 1
            cues.append({"start": parse_ts(a), "end": parse_ts(b), "text": "\n".join(text_lines)})
            i = j
        else:
            i += 1
    return cues


def shots_to_cuts(shots: list[dict]) -> list[float]:
    """Convert shot list to cut times (transitions between adjacent shots)."""
    cuts = []
    for i in range(len(shots) - 1):
        cuts.append(float(shots[i]["end_time"]))
    return cuts


def nearest_cut_dist(t: float, cuts: list[float]) -> float:
    if not cuts:
        return float("inf")
    import bisect
    i = bisect.bisect_left(cuts, t)
    cands = []
    if i < len(cuts):
        cands.append(abs(cuts[i] - t))
    if i > 0:
        cands.append(abs(cuts[i - 1] - t))
    return min(cands)


def analyse(cues: list[dict], cuts: list[float], snap_window_s: float = 0.5) -> dict:
    if not cues:
        return {"cues": 0, "cuts": len(cuts)}

    starts_in_window = 0
    ends_in_window = 0
    starts_aligned = 0  # within 1 frame (~42 ms)
    ends_aligned = 0
    near_miss_starts = 0  # in window but not aligned
    near_miss_ends = 0
    cue_spans_cut = 0

    aligned_thresh = 1.0 / 24  # ~42 ms

    for c in cues:
        sd = nearest_cut_dist(c["start"], cuts)
        ed = nearest_cut_dist(c["end"], cuts)
        if sd <= snap_window_s:
            starts_in_window += 1
            if sd <= aligned_thresh:
                starts_aligned += 1
            else:
                near_miss_starts += 1
        if ed <= snap_window_s:
            ends_in_window += 1
            if ed <= aligned_thresh:
                ends_aligned += 1
            else:
                near_miss_ends += 1
        # spans a cut if any cut falls strictly inside (start, end)
        import bisect
        i = bisect.bisect_right(cuts, c["start"])
        if i < len(cuts) and cuts[i] < c["end"]:
            cue_spans_cut += 1

    cut_gaps = [cuts[i + 1] - cuts[i] for i in range(len(cuts) - 1)] if len(cuts) > 1 else []

    return {
        "cues": len(cues),
        "cuts": len(cuts),
        "shots_per_min": (len(cuts) / max(1.0, cues[-1]["end"]) * 60) if cues else 0.0,
        "cut_gap_p50": sorted(cut_gaps)[len(cut_gaps) // 2] if cut_gaps else None,
        "cut_gap_p10": sorted(cut_gaps)[max(0, len(cut_gaps) // 10)] if cut_gaps else None,
        "cut_gap_p90": sorted(cut_gaps)[min(len(cut_gaps) - 1, len(cut_gaps) * 9 // 10)] if cut_gaps else None,
        "starts_in_window": starts_in_window,
        "ends_in_window": ends_in_window,
        "starts_aligned": starts_aligned,
        "ends_aligned": ends_aligned,
        "near_miss_starts": near_miss_starts,
        "near_miss_ends": near_miss_ends,
        "cue_spans_cut": cue_spans_cut,
        "snap_window_s": snap_window_s,
        "aligned_thresh_s": aligned_thresh,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--reflow", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cache = args.out.with_suffix(".shots.json")

    if cache.exists() and not args.force:
        print(f"loading cached shots from {cache}")
        shots = json.loads(cache.read_text())
        elapsed = 0.0
    else:
        print(f"loading TransNetV2...")
        t0 = time.time()
        model = TransNetV2()
        model.eval()
        print(f"  init {time.time() - t0:.1f}s, device={next(model.parameters()).device}")

        print(f"detecting shots in {args.video.name}...")
        t0 = time.time()
        shots = model.detect_scenes(str(args.video), threshold=args.threshold)
        elapsed = time.time() - t0
        print(f"  done in {elapsed:.1f}s — {len(shots)} shots")
        cache.write_text(json.dumps(shots, indent=2, default=str))

    cuts = shots_to_cuts(shots)
    cues = parse_vtt(args.reflow)
    stats = analyse(cues, cuts)
    stats["video"] = str(args.video)
    stats["reflow"] = str(args.reflow)
    stats["threshold"] = args.threshold
    stats["detect_seconds"] = elapsed

    payload = {
        "stats": stats,
        "cuts": cuts,
        "shots": shots,
    }
    args.out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote {args.out}")
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
