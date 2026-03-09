#!/usr/bin/env python3
"""Compare transcript coverage between a primary aligned transcript and a secondary ASR pass."""

import argparse
import json
import math
import re
from pathlib import Path


_PUNCT_RE = re.compile(r"[\s\u3000、。！？?!…,.，「」『』（）()【】\[\]・:：\-ー]+")


def load_segments(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} did not contain a top-level list")
    return data


def normalize_text(text: str) -> str:
    return _PUNCT_RE.sub("", text or "")


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return a_start < b_end and a_end > b_start


def window_text(segments: list[dict], start_s: float, end_s: float) -> str:
    texts = []
    for seg in segments:
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", 0.0))
        if overlap(seg_start, seg_end, start_s, end_s):
            text = str(seg.get("text", "")).strip()
            if text:
                texts.append(text)
    return " / ".join(texts)


def build_flagged_windows(
    primary: list[dict],
    secondary: list[dict],
    *,
    window_s: float,
    step_s: float,
    min_secondary_chars: int,
    max_primary_chars: int,
    min_extra_chars: int,
) -> list[dict]:
    max_end = max(
        [0.0]
        + [float(seg.get("end", 0.0)) for seg in primary]
        + [float(seg.get("end", 0.0)) for seg in secondary]
    )
    flagged = []
    cursor = 0.0
    while cursor < max_end:
        end_s = min(max_end, cursor + window_s)
        primary_text = window_text(primary, cursor, end_s)
        secondary_text = window_text(secondary, cursor, end_s)
        primary_norm = normalize_text(primary_text)
        secondary_norm = normalize_text(secondary_text)
        primary_chars = len(primary_norm)
        secondary_chars = len(secondary_norm)
        if secondary_chars >= min_secondary_chars and (
            primary_chars <= max_primary_chars
            or secondary_chars >= max(primary_chars * 2, primary_chars + min_extra_chars)
        ):
            flagged.append(
                {
                    "start_s": round(cursor, 3),
                    "end_s": round(end_s, 3),
                    "primary_chars": primary_chars,
                    "secondary_chars": secondary_chars,
                    "primary_text": primary_text,
                    "secondary_text": secondary_text,
                }
            )
        cursor += step_s
    return flagged


def merge_flagged_windows(flagged: list[dict], *, merge_gap_s: float) -> list[dict]:
    if not flagged:
        return []
    merged = [dict(flagged[0])]
    for item in flagged[1:]:
        current = merged[-1]
        if item["start_s"] <= current["end_s"] + merge_gap_s:
            current["end_s"] = max(current["end_s"], item["end_s"])
            current["primary_chars"] = max(current["primary_chars"], item["primary_chars"])
            current["secondary_chars"] = max(current["secondary_chars"], item["secondary_chars"])
        else:
            merged.append(dict(item))
    return merged


def enrich_regions(regions: list[dict], primary: list[dict], secondary: list[dict]) -> list[dict]:
    enriched = []
    for idx, region in enumerate(regions):
        primary_text = window_text(primary, region["start_s"], region["end_s"])
        secondary_text = window_text(secondary, region["start_s"], region["end_s"])
        enriched.append(
            {
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
        )
    return enriched


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
    args = parser.parse_args()

    primary_path = Path(args.primary)
    secondary_path = Path(args.secondary)
    output_path = Path(args.output)

    primary = load_segments(primary_path)
    secondary = load_segments(secondary_path)
    flagged = build_flagged_windows(
        primary,
        secondary,
        window_s=args.window_s,
        step_s=args.step_s,
        min_secondary_chars=args.min_secondary_chars,
        max_primary_chars=args.max_primary_chars,
        min_extra_chars=args.min_extra_chars,
    )
    regions = enrich_regions(merge_flagged_windows(flagged, merge_gap_s=args.merge_gap_s), primary, secondary)
    regions.sort(key=lambda item: (-item["coverage_gap_chars"], item["start_s"]))

    report = {
        "primary": str(primary_path),
        "secondary": str(secondary_path),
        "settings": {
            "window_s": args.window_s,
            "step_s": args.step_s,
            "merge_gap_s": args.merge_gap_s,
            "min_secondary_chars": args.min_secondary_chars,
            "max_primary_chars": args.max_primary_chars,
            "min_extra_chars": args.min_extra_chars,
        },
        "summary": {
            "flagged_windows": len(flagged),
            "flagged_regions": len(regions),
            "top_regions_kept": min(args.top, len(regions)),
        },
        "top_regions": regions[: args.top],
        "all_regions": regions,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        f"Flagged {len(flagged)} windows and {len(regions)} merged regions; "
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
