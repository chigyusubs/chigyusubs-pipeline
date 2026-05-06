#!/usr/bin/env python3
"""Correct CTC drift in `ctc_words.json` against VAD speech regions.

NOTE: As of 2026-05, drift correction is built into `scripts/align_ctc.py`
and runs by default at the end of CTC alignment. This standalone script is
retained for debugging and for re-running the correction on cached CTC output
with different parameters (no full re-alignment needed).

CTC forced alignment must place every transcript token somewhere; in regions
where the model's posterior over blank collapses (pre-roll music, BGM-masked
speech, domain-mismatch pockets) tokens leak across silence and downstream
reflow inherits massively wrong timings (san-nomi cue 0: 12s pre-speech drift).

This pass clusters words by intra-word gap, validates each cluster against
Silero VAD, and either trims the segment to the real-cluster span or
relocates an entirely-ghost segment near the nearest unclaimed VAD onset.

See `docs/timing-architecture-2026-05.md` for thresholds + research citations.

Usage:
  python scripts/correct_ctc_drift.py \\
      --input  episode/transcription/<run>_ctc_words.json \\
      --vad    episode/transcription/silero_vad_segments.json \\
      --output episode/transcription/<run>_ctc_words_drift_corrected.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chigyusubs.ctc_drift import (
    DEFAULT_CLUSTER_GAP_S,
    DEFAULT_GHOST_VAD_OVERLAP_S,
    DEFAULT_MAX_PRE_SPEECH_LEAD_S,
    DEFAULT_MIN_CUE_S,
    DEFAULT_MIN_INTRA_GAP_S,
    DEFAULT_NEAR_VAD_LOOKAHEAD_S,
    DEFAULT_TARGET_JP_CPS,
    DEFAULT_VAD_BRIDGE_S,
    VadRegion,
    repair_segments,
    summarize,
)


def load_vad(path: Path) -> list[VadRegion]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "segments" in data:
        data = data["segments"]
    regs = [VadRegion(float(s["start"]), float(s["end"])) for s in data]
    regs.sort(key=lambda r: r.start)
    return regs


def main():
    p = argparse.ArgumentParser(
        description="Correct CTC drift by clustering words and validating against VAD."
    )
    p.add_argument("--input", required=True, help="Input ctc_words.json")
    p.add_argument("--vad", required=True, help="Silero VAD segments JSON (list of {start, end})")
    p.add_argument("--output", default="",
                   help="Output ctc_words JSON (default: <input_stem>_drift_corrected.json)")
    p.add_argument("--diagnostics", default="",
                   help="Diagnostics JSON path (default: <output>.diagnostics.json)")
    p.add_argument("--cluster-gap-s", type=float, default=DEFAULT_CLUSTER_GAP_S,
                   help=f"Inter-word gap that splits a cluster (default {DEFAULT_CLUSTER_GAP_S}s; "
                        "WhisperX/stable-ts consensus)")
    p.add_argument("--max-pre-speech-lead-s", type=float, default=DEFAULT_MAX_PRE_SPEECH_LEAD_S,
                   help=f"Max lead before VAD onset on relocation (default {DEFAULT_MAX_PRE_SPEECH_LEAD_S}s; "
                        "Netflix 1-2 frames @24fps)")
    p.add_argument("--min-cue-s", type=float, default=DEFAULT_MIN_CUE_S,
                   help=f"Minimum cue duration after relocation (default {DEFAULT_MIN_CUE_S}s; Netflix min)")
    p.add_argument("--target-jp-cps", type=float, default=DEFAULT_TARGET_JP_CPS,
                   help=f"Target Japanese CPS for compression (default {DEFAULT_TARGET_JP_CPS}; "
                        "Netflix JP regular)")
    p.add_argument("--ghost-vad-overlap-s", type=float, default=DEFAULT_GHOST_VAD_OVERLAP_S,
                   help=f"Min cluster-VAD overlap to count cluster as real (default {DEFAULT_GHOST_VAD_OVERLAP_S}s)")
    p.add_argument("--vad-bridge-s", type=float, default=DEFAULT_VAD_BRIDGE_S,
                   help=f"Min VAD speech inside an inter-word gap to treat the gap as natural pause "
                        f"rather than drift (default {DEFAULT_VAD_BRIDGE_S}s)")
    p.add_argument("--near-vad-lookahead-s", type=float, default=DEFAULT_NEAR_VAD_LOOKAHEAD_S,
                   help=f"Tolerance for 'cluster start near VAD onset' (default {DEFAULT_NEAR_VAD_LOOKAHEAD_S}s)")
    p.add_argument("--min-intra-gap-s", type=float, default=DEFAULT_MIN_INTRA_GAP_S,
                   help=f"Min intra-word gap to flag a segment as suspect drift "
                        f"(default {DEFAULT_MIN_INTRA_GAP_S}s; gross-drift signature only)")
    args = p.parse_args()

    input_path = Path(args.input)
    vad_path = Path(args.vad)
    if not input_path.exists():
        sys.exit(f"Input not found: {input_path}")
    if not vad_path.exists():
        sys.exit(f"VAD not found: {vad_path}")

    output_path = Path(args.output) if args.output else input_path.with_name(
        input_path.stem + "_drift_corrected" + input_path.suffix
    )
    diag_path = Path(args.diagnostics) if args.diagnostics else output_path.with_suffix(
        output_path.suffix + ".diagnostics.json"
    )

    segments = json.loads(input_path.read_text(encoding="utf-8"))
    vad = load_vad(vad_path)

    corrected, diags = repair_segments(
        segments, vad,
        cluster_gap_s=args.cluster_gap_s,
        max_pre_speech_lead_s=args.max_pre_speech_lead_s,
        min_cue_s=args.min_cue_s,
        target_jp_cps=args.target_jp_cps,
        ghost_vad_overlap_s=args.ghost_vad_overlap_s,
        vad_bridge_s=args.vad_bridge_s,
        near_vad_lookahead_s=args.near_vad_lookahead_s,
        min_intra_gap_s=args.min_intra_gap_s,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(corrected, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = summarize(diags)
    diag_payload = {
        "input": str(input_path),
        "vad": str(vad_path),
        "output": str(output_path),
        "params": {
            "cluster_gap_s": args.cluster_gap_s,
            "max_pre_speech_lead_s": args.max_pre_speech_lead_s,
            "min_cue_s": args.min_cue_s,
            "target_jp_cps": args.target_jp_cps,
            "ghost_vad_overlap_s": args.ghost_vad_overlap_s,
            "vad_bridge_s": args.vad_bridge_s,
            "near_vad_lookahead_s": args.near_vad_lookahead_s,
            "min_intra_gap_s": args.min_intra_gap_s,
        },
        "summary": summary,
        "segments": [asdict(d) for d in diags],
    }
    diag_path.write_text(json.dumps(diag_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {output_path} ({len(corrected)} segments)")
    print(f"Wrote {diag_path}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
