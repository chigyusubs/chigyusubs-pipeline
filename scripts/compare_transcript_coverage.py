#!/usr/bin/env python3
"""Compare transcript coverage between a primary aligned transcript and a secondary ASR pass."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chigyusubs.transcript_comparison import (
    build_flagged_windows,
    load_segments,
    merge_flagged_windows,
    normalize_text,
    overlap,
    window_text,
)


def vad_overlaps_region(vad_segments: list[dict], start_s: float, end_s: float) -> bool:
    for seg in vad_segments:
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", 0.0))
        if overlap(seg_start, seg_end, start_s, end_s):
            return True
    return False


def enrich_regions(
    regions: list[dict],
    primary: list[dict],
    secondary: list[dict],
    vad_segments: list[dict] | None = None,
) -> list[dict]:
    enriched = []
    for idx, region in enumerate(regions):
        primary_text = window_text(primary, region["start_s"], region["end_s"])
        secondary_text = window_text(secondary, region["start_s"], region["end_s"])
        item = {
            "region_index": idx,
            "start_s": round(region["start_s"], 3),
            "end_s": round(region["end_s"], 3),
            "duration_s": round(region["end_s"] - region["start_s"], 3),
            "primary_chars": len(normalize_text(primary_text)),
            "secondary_chars": len(normalize_text(secondary_text)),
            "coverage_gap_chars": len(normalize_text(secondary_text)) - len(normalize_text(primary_text)),
            "primary_text": primary_text,
            "secondary_text": secondary_text,
        }
        if vad_segments is not None:
            has_speech = vad_overlaps_region(vad_segments, region["start_s"], region["end_s"])
            item["vad_confirmed"] = has_speech
            item["possible_hallucination"] = not has_speech
        enriched.append(item)
    return enriched


def run_coverage_report(
    primary: list[dict],
    secondary: list[dict],
    *,
    primary_path: str = "",
    secondary_path: str = "",
    window_s: float = 2.0,
    step_s: float = 1.0,
    merge_gap_s: float = 0.75,
    min_secondary_chars: int = 12,
    max_primary_chars: int = 4,
    min_extra_chars: int = 8,
    top: int = 20,
    vad_segments: list[dict] | None = None,
) -> dict:
    """Run coverage comparison and return the report dict."""
    flagged = build_flagged_windows(
        primary,
        secondary,
        window_s=window_s,
        step_s=step_s,
        min_secondary_chars=min_secondary_chars,
        max_primary_chars=max_primary_chars,
        min_extra_chars=min_extra_chars,
    )
    regions = enrich_regions(
        merge_flagged_windows(flagged, merge_gap_s=merge_gap_s),
        primary,
        secondary,
        vad_segments,
    )
    regions.sort(key=lambda item: (-item["coverage_gap_chars"], item["start_s"]))

    summary_dict: dict = {
        "flagged_windows": len(flagged),
        "flagged_regions": len(regions),
        "top_regions_kept": min(top, len(regions)),
    }
    if vad_segments is not None:
        vad_confirmed = sum(1 for r in regions if r.get("vad_confirmed", False))
        possible_hallucination = sum(1 for r in regions if r.get("possible_hallucination", False))
        summary_dict["vad_confirmed_regions"] = vad_confirmed
        summary_dict["possible_hallucination_regions"] = possible_hallucination

    return {
        "primary": primary_path,
        "secondary": secondary_path,
        "settings": {
            "window_s": window_s,
            "step_s": step_s,
            "merge_gap_s": merge_gap_s,
            "min_secondary_chars": min_secondary_chars,
            "max_primary_chars": max_primary_chars,
            "min_extra_chars": min_extra_chars,
        },
        "summary": summary_dict,
        "top_regions": regions[:top],
        "all_regions": regions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare transcript coverage across two word-timestamp JSON artifacts.")
    parser.add_argument("--primary", required=True, help="Primary words JSON, usually Gemini+CTC output.")
    parser.add_argument("--secondary", required=True, help="Secondary words JSON, usually faster-whisper output.")
    parser.add_argument("--output", required=True, help="Output JSON report path.")
    parser.add_argument("--window-s", type=float, default=2.0, help="Sliding window size in seconds.")
    parser.add_argument("--step-s", type=float, default=1.0, help="Sliding window step in seconds.")
    parser.add_argument("--merge-gap-s", type=float, default=0.75, help="Merge adjacent flagged windows across gaps up to this size.")
    parser.add_argument("--min-secondary-chars", type=int, default=12, help="Minimum normalized secondary chars before a window can be flagged.")
    parser.add_argument("--max-primary-chars", type=int, default=4, help="Flag immediately when primary coverage is at or below this threshold.")
    parser.add_argument("--min-extra-chars", type=int, default=8, help="Otherwise require at least this many extra chars in the secondary transcript.")
    parser.add_argument("--top", type=int, default=20, help="How many top flagged regions to keep in the summary sample.")
    parser.add_argument("--vad-json", default="", help="Optional VAD segments JSON for speech cross-reference. Regions without VAD speech are marked as possible hallucination.")
    args = parser.parse_args()

    primary_path = Path(args.primary)
    secondary_path = Path(args.secondary)
    output_path = Path(args.output)

    primary = load_segments(primary_path)
    secondary = load_segments(secondary_path)

    vad_segments = None
    if args.vad_json:
        vad_path = Path(args.vad_json)
        if vad_path.exists():
            vad_segments = json.loads(vad_path.read_text(encoding="utf-8"))
            print(f"Loaded {len(vad_segments)} VAD segments from {vad_path}")

    report = run_coverage_report(
        primary,
        secondary,
        primary_path=str(primary_path),
        secondary_path=str(secondary_path),
        window_s=args.window_s,
        step_s=args.step_s,
        merge_gap_s=args.merge_gap_s,
        min_secondary_chars=args.min_secondary_chars,
        max_primary_chars=args.max_primary_chars,
        min_extra_chars=args.min_extra_chars,
        top=args.top,
        vad_segments=vad_segments,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    regions = report["all_regions"]
    flagged_count = report["summary"]["flagged_windows"]
    print(
        f"Flagged {flagged_count} windows and {len(regions)} merged regions; "
        f"wrote report to {output_path}"
    )
    for item in regions[: min(args.top, 5)]:
        print(
            f"- {item['start_s']:.3f}-{item['end_s']:.3f}s "
            f"gap={item['coverage_gap_chars']} "
            f"secondary='{item['secondary_text'][:80]}'"
        )


if __name__ == "__main__":
    main()
