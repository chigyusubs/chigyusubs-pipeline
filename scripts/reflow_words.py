#!/usr/bin/env python3
"""Semantic reflow: re-segment word-level timestamps into subtitle cues
based on natural speech pauses rather than arbitrary duration splits.

Standalone CLI + importable function.

Usage:
  python scripts/reflow_words.py \
    --input episode/transcription/406_faster_v2_words.json \
    --output episode/transcription/406_reflow.vtt \
    --pause-ms 300 --max-cue-s 10 --min-cue-s 0.3
"""

import argparse
import json
import sys
from pathlib import Path


from chigyusubs.reflow import reflow_words  # noqa: F401
from chigyusubs.vtt import format_ts as _format_ts, write_vtt as write_vtt  # noqa: F401


def main():
    parser = argparse.ArgumentParser(
        description="Re-segment word timestamps into semantically-aligned subtitle cues."
    )
    parser.add_argument(
        "--input", required=True,
        help="Word timestamps JSON file (faster-whisper format).",
    )
    parser.add_argument(
        "--output", default="",
        help="Output VTT path. Defaults to <input_stem>_reflow.vtt.",
    )
    parser.add_argument(
        "--pause-ms", type=int, default=300,
        help="Pause threshold in ms to trigger cue break (default: 300).",
    )
    parser.add_argument(
        "--max-cue-s", type=float, default=10.0,
        help="Maximum cue duration in seconds (default: 10).",
    )
    parser.add_argument(
        "--min-cue-s", type=float, default=0.3,
        help="Minimum cue duration in seconds; shorter cues merge (default: 0.3).",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print cue duration statistics.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    if not args.output:
        args.output = str(input_path.parent / f"{input_path.stem}_reflow.vtt")

    cues = reflow_words(
        segments,
        pause_threshold=args.pause_ms / 1000.0,
        max_cue_s=args.max_cue_s,
        min_cue_s=args.min_cue_s,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_vtt(cues, args.output)

    total_words = sum(len(c["words"]) for c in cues)
    durations = [c["end"] - c["start"] for c in cues]
    print(f"Reflowed {total_words} words into {len(cues)} cues -> {args.output}")

    if args.stats and durations:
        avg = sum(durations) / len(durations)
        print(f"  Duration: min={min(durations):.2f}s  avg={avg:.2f}s  max={max(durations):.2f}s")
        over_max = sum(1 for d in durations if d > args.max_cue_s)
        if over_max:
            print(f"  Warning: {over_max} cues still exceed --max-cue-s {args.max_cue_s}")


if __name__ == "__main__":
    main()
