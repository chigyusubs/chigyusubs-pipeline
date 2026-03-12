#!/usr/bin/env python3
"""On-demand keyframe extraction for Codex.

Quick frame extraction that Codex can invoke during reflow or translation
when it needs visual context for a specific moment or cue range.

Usage:
  # Single frame at a timestamp
  python scripts/extract_keyframes.py \
    --video episode/source/video.mp4 \
    --at 42.5 \
    --output-dir episode/keyframes

  # Start/mid/end frames for specific cues (default: all three)
  python scripts/extract_keyframes.py \
    --video episode/source/video.mp4 \
    --cues episode/transcription/reflow.vtt \
    --cue-ids 15,16,17 \
    --output-dir episode/keyframes

  # Only midpoint frames (telop check)
  python scripts/extract_keyframes.py \
    --video episode/source/video.mp4 \
    --cues episode/transcription/reflow.vtt \
    --cue-ids 15,16,17 --cue-points mid \
    --output-dir episode/keyframes

  # Frames at multiple timestamps
  python scripts/extract_keyframes.py \
    --video episode/source/video.mp4 \
    --at 42.5,85.0,120.3 \
    --output-dir episode/keyframes

  # Frame every N seconds in a time range
  python scripts/extract_keyframes.py \
    --video episode/source/video.mp4 \
    --range 40.0-50.0 --interval 2.0 \
    --output-dir episode/keyframes
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chigyusubs.translation import parse_vtt, parse_srt


def extract_frame(video_path: str, timestamp: float, output_path: str, width: int = 640) -> bool:
    """Extract a single frame. Returns True on success."""
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", f"{timestamp:.3f}",
                "-i", video_path,
                "-frames:v", "1",
                "-vf", f"scale={width}:-2",
                "-qscale:v", "3",
                output_path,
            ],
            capture_output=True, check=True, timeout=15,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def main():
    parser = argparse.ArgumentParser(
        description="On-demand keyframe extraction for Codex visual context."
    )
    parser.add_argument("--video", required=True, help="Input video file.")
    parser.add_argument("--output-dir", required=True, help="Output directory for frames.")
    parser.add_argument("--width", type=int, default=640, help="Frame width (default: 640).")

    # Extraction modes (pick one)
    parser.add_argument(
        "--at", default="",
        help="Comma-separated timestamps in seconds (e.g. 42.5,85.0).",
    )
    parser.add_argument(
        "--cues", default="",
        help="VTT/SRT file for cue-based extraction.",
    )
    parser.add_argument(
        "--cue-ids", default="",
        help="Comma-separated 1-based cue IDs to extract frames for.",
    )
    parser.add_argument(
        "--cue-points", default="start,mid,end",
        help="Which points per cue: any combo of start,mid,end (default: start,mid,end).",
    )
    parser.add_argument(
        "--range", default="",
        help="Time range as START-END in seconds (e.g. 40.0-50.0).",
    )
    parser.add_argument(
        "--interval", type=float, default=2.0,
        help="Seconds between frames when using --range (default: 2.0).",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamps: list[tuple[float, str]] = []  # (time, label)

    if args.at:
        for raw in args.at.split(","):
            ts = float(raw.strip())
            timestamps.append((ts, f"t{ts:.3f}"))

    elif args.cues and args.cue_ids:
        raw = Path(args.cues).read_text(encoding="utf-8")
        is_srt = args.cues.lower().endswith(".srt")
        cues = parse_srt(raw) if is_srt else parse_vtt(raw)
        points = {p.strip() for p in args.cue_points.split(",")}
        for raw_id in args.cue_ids.split(","):
            cue_id = int(raw_id.strip())
            if cue_id < 1 or cue_id > len(cues):
                print(f"Cue {cue_id} out of range (1-{len(cues)})", file=sys.stderr)
                continue
            cue = cues[cue_id - 1]
            if "start" in points:
                timestamps.append((cue.start, f"cue{cue_id}_start_{cue.start:.3f}"))
            if "mid" in points:
                mid = (cue.start + cue.end) / 2
                timestamps.append((mid, f"cue{cue_id}_mid_{mid:.3f}"))
            if "end" in points:
                timestamps.append((cue.end, f"cue{cue_id}_end_{cue.end:.3f}"))

    elif args.range:
        parts = args.range.split("-")
        if len(parts) != 2:
            print("--range must be START-END (e.g. 40.0-50.0)", file=sys.stderr)
            return 1
        start = float(parts[0])
        end = float(parts[1])
        t = start
        while t <= end:
            timestamps.append((t, f"t{t:.3f}"))
            t += args.interval

    else:
        print("Specify --at, --cue-ids (with --cues), or --range.", file=sys.stderr)
        return 1

    if not timestamps:
        print("No timestamps to extract.", file=sys.stderr)
        return 1

    extracted = []
    for ts, label in timestamps:
        frame_path = out_dir / f"frame_{label}.jpg"
        if extract_frame(args.video, ts, str(frame_path), args.width):
            extracted.append(str(frame_path))
            print(f"  {ts:.3f}s -> {frame_path}")
        else:
            print(f"  {ts:.3f}s -> FAILED")

    print(f"\nExtracted {len(extracted)}/{len(timestamps)} frames -> {out_dir}")
    for path in extracted:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
