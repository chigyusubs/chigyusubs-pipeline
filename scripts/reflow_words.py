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


def _format_ts(seconds: float) -> str:
    """Format seconds as VTT timestamp HH:MM:SS.mmm or MM:SS.mmm."""
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f"{int(hours):02d}:{int(mins):02d}:{secs:06.3f}"
    return f"{int(mins):02d}:{secs:06.3f}"


def reflow_words(
    segments: list[dict],
    pause_threshold: float = 0.3,
    max_cue_s: float = 10.0,
    min_cue_s: float = 0.3,
) -> list[dict]:
    """Re-segment word timestamps into cues aligned to natural pauses.

    Args:
        segments: List of segment dicts with "words" arrays (faster-whisper format).
        pause_threshold: Gap in seconds between words that triggers a cue break.
        max_cue_s: Maximum cue duration; longer groups get split.
        min_cue_s: Minimum cue duration; shorter groups merge into the next.

    Returns:
        List of cue dicts with keys: start, end, text, words.
    """
    # 1. Flatten all words into one chronological stream.
    all_words = []
    for seg in segments:
        for w in seg.get("words", []):
            all_words.append(w)

    if not all_words:
        return []

    # 2. Find break points: gaps >= pause_threshold.
    groups: list[list[dict]] = []
    current_group: list[dict] = [all_words[0]]

    for i in range(1, len(all_words)):
        gap = all_words[i]["start"] - all_words[i - 1]["end"]
        if gap >= pause_threshold:
            groups.append(current_group)
            current_group = [all_words[i]]
        else:
            current_group.append(all_words[i])

    if current_group:
        groups.append(current_group)

    # 3. Merge very short groups into the next group.
    merged: list[list[dict]] = []
    i = 0
    while i < len(groups):
        g = groups[i]
        dur = g[-1]["end"] - g[0]["start"]
        if dur < min_cue_s and i + 1 < len(groups):
            # Check gap to next group.
            gap_to_next = groups[i + 1][0]["start"] - g[-1]["end"]
            if gap_to_next < pause_threshold * 2:
                # Merge into next group.
                groups[i + 1] = g + groups[i + 1]
                i += 1
                continue
        merged.append(g)
        i += 1

    # 4. Split groups that exceed max_cue_s.
    final_groups: list[list[dict]] = []
    for g in merged:
        _split_group(g, max_cue_s, final_groups)

    # 5. Build cue dicts.
    cues = []
    for g in final_groups:
        text = "".join(w["word"] for w in g).strip()
        if not text:
            continue
        cues.append({
            "start": g[0]["start"],
            "end": g[-1]["end"],
            "text": text,
            "words": g,
        })

    return cues


def _split_group(
    words: list[dict], max_cue_s: float, out: list[list[dict]]
):
    """Recursively split a word group that exceeds max_cue_s."""
    dur = words[-1]["end"] - words[0]["start"]
    if dur <= max_cue_s or len(words) < 2:
        out.append(words)
        return

    # Find the largest internal gap to split at.
    best_gap = -1.0
    best_idx = -1
    for i in range(1, len(words)):
        gap = words[i]["start"] - words[i - 1]["end"]
        if gap > best_gap:
            best_gap = gap
            best_idx = i

    # If no meaningful gap exists, fall back to greedy duration split.
    if best_gap < 0.01:
        _greedy_split(words, max_cue_s, out)
        return

    left = words[:best_idx]
    right = words[best_idx:]
    _split_group(left, max_cue_s, out)
    _split_group(right, max_cue_s, out)


def _greedy_split(
    words: list[dict], max_cue_s: float, out: list[list[dict]]
):
    """Greedy word-boundary split at max_cue_s (fallback)."""
    current: list[dict] = []
    for w in words:
        if current and w["end"] - current[0]["start"] > max_cue_s:
            out.append(current)
            current = []
        current.append(w)
    if current:
        out.append(current)


def write_vtt(cues: list[dict], output_path: str):
    """Write cues as a standard VTT file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for cue in cues:
            start = _format_ts(cue["start"])
            end = _format_ts(cue["end"])
            f.write(f"{start} --> {end}\n")
            f.write(f"{cue['text']}\n\n")


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
