#!/usr/bin/env python3
"""Semantic reflow: re-segment word-level timestamps into subtitle cues
based on natural speech pauses rather than arbitrary duration splits.

Standalone CLI + importable function.

Usage (line-level, recommended for CTC alignment):
  python scripts/reflow_words.py \
    --input episode/transcription/ctc_words.json \
    --output episode/transcription/reflow.vtt \
    --line-level --stats

Usage (word-level, legacy for stable-ts):
  python scripts/reflow_words.py \
    --input episode/transcription/words.json \
    --output episode/transcription/reflow.vtt \
    --pause-ms 300 --max-cue-s 10 --min-cue-s 0.3
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chigyusubs.reflow import reflow_lines, reflow_words  # noqa: F401
from chigyusubs.metadata import (
    build_vtt_note_lines,
    finish_run,
    inherit_run_id,
    lineage_output_path,
    metadata_path,
    start_run,
    update_preferred_manifest,
    write_metadata,
)
from chigyusubs.vtt import format_ts as _format_ts, write_vtt as write_vtt  # noqa: F401


def main():
    run = start_run("reflow")
    parser = argparse.ArgumentParser(
        description="Re-segment word timestamps into semantically-aligned subtitle cues."
    )
    parser.add_argument(
        "--input", required=True,
        help="Word timestamps JSON file (CTC or faster-whisper format).",
    )
    parser.add_argument(
        "--output", default="",
        help="Output VTT path. Defaults to <input_stem>_reflow.vtt.",
    )
    parser.add_argument(
        "--line-level", action="store_true",
        help="Use line-level reflow (recommended for CTC alignment output). "
             "Treats each transcript line as atomic — never splits mid-word.",
    )
    parser.add_argument(
        "--pause-ms", type=int, default=300,
        help="Pause threshold in ms for word-level reflow (default: 300). "
             "Ignored with --line-level.",
    )
    parser.add_argument(
        "--max-cue-s", type=float, default=7.0,
        help="Maximum cue duration in seconds (default: 7).",
    )
    parser.add_argument(
        "--max-cue-chars", type=int, default=45,
        help="Maximum characters per cue (default: 45). Only used with --line-level.",
    )
    parser.add_argument(
        "--max-lines", type=int, default=2,
        help="Maximum lines per cue (default: 2). Only used with --line-level.",
    )
    parser.add_argument(
        "--min-cue-s", type=float, default=1.0,
        help="Minimum cue duration in seconds; shorter cues merge (default: 1.0).",
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

    run = inherit_run_id(run, input_path)
    default_output = lineage_output_path(
        input_path.parent,
        artifact_type="reflow",
        run=run,
        suffix=".vtt",
    )
    if not args.output:
        args.output = str(default_output)
    else:
        requested = Path(args.output)
        if requested.parent.name == "transcription" and not requested.name.startswith(f"{run['run_id']}_"):
            args.output = str(lineage_output_path(requested.parent, artifact_type="reflow", run=run, suffix=requested.suffix or ".vtt"))

    if args.line_level:
        cues = reflow_lines(
            segments,
            max_cue_s=args.max_cue_s,
            max_cue_chars=args.max_cue_chars,
            max_lines=args.max_lines,
            min_cue_s=args.min_cue_s,
        )
    else:
        cues = reflow_words(
            segments,
            pause_threshold=args.pause_ms / 1000.0,
            max_cue_s=args.max_cue_s,
            min_cue_s=args.min_cue_s,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    note_lines = build_vtt_note_lines(
        {
            "run_id": run.get("run_id"),
            "step": "reflow",
            "episode": input_path.parent.parent.name if input_path.parent.name == "transcription" else "",
            "run_started_at": run.get("run_started_at"),
        },
        source_name=input_path.name,
    )
    write_vtt(cues, str(output_path), note_lines=note_lines)

    durations = [c["end"] - c["start"] for c in cues]
    chars = [len(c["text"].replace(" ", "").replace("\n", "")) for c in cues]
    cpss = [ch / max(d, 0.001) for ch, d in zip(chars, durations)]

    metadata = finish_run(
        run,
        inputs={"words_json": str(input_path)},
        outputs={"reflow_vtt": str(output_path)},
        settings={
            "line_level": args.line_level,
            "pause_ms": args.pause_ms,
            "max_cue_s": args.max_cue_s,
            "max_cue_chars": args.max_cue_chars,
            "max_lines": args.max_lines,
            "min_cue_s": args.min_cue_s,
        },
        stats={
            "cues": len(cues),
            "min_duration_seconds": round(min(durations), 3) if durations else None,
            "avg_duration_seconds": round(sum(durations) / len(durations), 3) if durations else None,
            "max_duration_seconds": round(max(durations), 3) if durations else None,
            "avg_chars": round(sum(chars) / len(chars), 3) if chars else None,
            "max_chars": max(chars) if chars else None,
            "avg_cps": round(sum(cpss) / len(cpss), 3) if cpss else None,
            "max_cps": round(max(cpss), 3) if cpss else None,
        },
    )
    write_metadata(output_path, metadata)
    if output_path.parent.name == "transcription":
        update_preferred_manifest(output_path.parent, reflow=output_path.name)

    print(f"Reflowed into {len(cues)} cues -> {output_path}")

    if args.stats and durations:
        avg_dur = sum(durations) / len(durations)
        short = sum(1 for d in durations if d < 1.0)
        print(f"  Duration: min={min(durations):.2f}s  avg={avg_dur:.2f}s  max={max(durations):.2f}s")
        print(f"  Chars/cue: avg={sum(chars)/len(chars):.0f}  max={max(chars)}")
        print(f"  CPS: avg={sum(cpss)/len(cpss):.1f}  max={max(cpss):.1f}  >20: {sum(1 for c in cpss if c > 20)}")
        print(f"  <1s cues: {short} ({short*100//len(cues)}%)")
        over_max = sum(1 for d in durations if d > args.max_cue_s)
        if over_max:
            print(f"  Warning: {over_max} cues still exceed --max-cue-s {args.max_cue_s}")
    print(f"Metadata written: {metadata_path(output_path)}")


if __name__ == "__main__":
    main()
