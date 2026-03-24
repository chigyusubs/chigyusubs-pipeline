#!/usr/bin/env python3
"""Classify likely raw-transcript omissions using Gemini raw chunks plus a Whisper second opinion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chigyusubs.transcript_comparison import (
    assign_item_timings,
    build_flagged_windows,
    char_bigram_jaccard,
    load_json,
    merge_flagged_windows,
    normalize_text,
    overlap,
    select_chunk_for_region,
    window_text,
)


_NARRATION_HINTS = (
    "一方",
    "ここで",
    "時刻",
    "スタートから",
    "ご覧の通り",
    "現在地",
    "現在位置",
    "到着",
    "状況",
    "優勢",
    "劣勢",
    "右狙い",
    "左狙い",
    "ゴール目前",
    "ゴール",
    "引いた",
    "だったが",
    "抜けると",
    "到達",
    "経過",
    "PM",
)


def aggregate_context(items: list[dict], start_s: float, end_s: float, item_type: str) -> list[dict]:
    selected = []
    for item in items:
        if item["type"] != item_type:
            continue
        item_start = float(item.get("start_s", 0.0))
        item_end = float(item.get("end_s", 0.0))
        if overlap(item_start, item_end, start_s, end_s) or abs(item_start - start_s) <= 1.0 or abs(item_end - end_s) <= 1.0:
            selected.append(item)
    return selected[:6]


def contains_narration_hint(text: str) -> bool:
    return any(hint in text for hint in _NARRATION_HINTS)


def classify_region(
    region: dict,
    chunk_index: int,
    chunk: dict,
    items: list[dict],
    primary: list[dict],
    secondary: list[dict],
) -> dict:
    primary_text = window_text(primary, region["start_s"], region["end_s"])
    secondary_text = window_text(secondary, region["start_s"], region["end_s"])
    primary_chars = len(normalize_text(primary_text))
    secondary_chars = len(normalize_text(secondary_text))
    spoken_items = aggregate_context(items, region["start_s"], region["end_s"], "spoken")
    visual_items = aggregate_context(items, region["start_s"], region["end_s"], "visual")
    spoken_text = " / ".join(item["text"] for item in spoken_items)
    visual_text = " / ".join(item["text"] for item in visual_items)
    narration_like = contains_narration_hint(secondary_text) or contains_narration_hint(visual_text)
    spoken_score = char_bigram_jaccard(secondary_text, spoken_text)
    visual_score = char_bigram_jaccard(secondary_text, visual_text)

    if visual_score >= 0.3 and visual_score > spoken_score + 0.08 and narration_like:
        classification = "visual_substituted_narration"
        confidence = "high"
    elif primary_chars <= 4 and narration_like:
        classification = "missing_narration_high_confidence"
        confidence = "high"
    elif spoken_score >= 0.18:
        classification = "compressed_vs_missing_unclear"
        confidence = "medium"
    elif narration_like:
        classification = "missing_narration_high_confidence"
        confidence = "medium"
    else:
        classification = "spoken_coverage_gap"
        confidence = "medium" if primary_chars <= 6 else "low"

    return {
        **region,
        "primary_text": primary_text,
        "secondary_text": secondary_text,
        "primary_chars": primary_chars,
        "secondary_chars": secondary_chars,
        "coverage_gap_chars": secondary_chars - primary_chars,
        "chunk": chunk_index,
        "chunk_start_s": round(float(chunk["chunk_start_s"]), 3),
        "chunk_end_s": round(float(chunk["chunk_end_s"]), 3),
        "classification": classification,
        "confidence": confidence,
        "narration_like": narration_like,
        "spoken_similarity": round(spoken_score, 3),
        "visual_similarity": round(visual_score, 3),
        "gemini_spoken_context": spoken_items,
        "gemini_visual_context": visual_items,
        "gemini_spoken_text": spoken_text,
        "gemini_visual_text": visual_text,
    }


def run_omission_report(
    raw_chunks: list[dict],
    primary: list[dict],
    secondary: list[dict],
    *,
    gemini_raw_path: str = "",
    primary_path: str = "",
    secondary_path: str = "",
    window_s: float = 2.0,
    step_s: float = 1.0,
    merge_gap_s: float = 0.75,
    min_secondary_chars: int = 12,
    max_primary_chars: int = 4,
    min_extra_chars: int = 8,
    top: int = 30,
) -> dict:
    """Run omission classification and return the report dict."""
    flagged = build_flagged_windows(
        primary,
        secondary,
        window_s=window_s,
        step_s=step_s,
        min_secondary_chars=min_secondary_chars,
        max_primary_chars=max_primary_chars,
        min_extra_chars=min_extra_chars,
    )
    merged_regions = merge_flagged_windows(flagged, merge_gap_s=merge_gap_s)

    chunk_items = []
    for chunk in raw_chunks:
        c_start = float(chunk["chunk_start_s"])
        c_end = float(chunk["chunk_end_s"])
        chunk_primary = [seg for seg in primary if float(seg["start"]) >= c_start and float(seg["start"]) < c_end]
        chunk_items.append(assign_item_timings(chunk, chunk_primary))

    classified = []
    for region in merged_regions:
        chunk_index, chunk = select_chunk_for_region(raw_chunks, region["start_s"], region["end_s"])
        if chunk is None:
            continue
        classified.append(classify_region(region, chunk_index, chunk, chunk_items[chunk_index], primary, secondary))

    class_priority = {
        "visual_substituted_narration": 0,
        "missing_narration_high_confidence": 1,
        "compressed_vs_missing_unclear": 2,
        "spoken_coverage_gap": 3,
    }
    confidence_priority = {"high": 0, "medium": 1, "low": 2}
    classified.sort(
        key=lambda item: (
            class_priority.get(item["classification"], 9),
            confidence_priority.get(item["confidence"], 9),
            -item["coverage_gap_chars"],
            item["start_s"],
        )
    )

    summary = {
        "flagged_windows": len(flagged),
        "flagged_regions": len(merged_regions),
        "classified_candidates": len(classified),
        "by_class": {
            key: sum(1 for item in classified if item["classification"] == key)
            for key in class_priority
        },
        "by_confidence": {
            key: sum(1 for item in classified if item["confidence"] == key)
            for key in confidence_priority
        },
        "top_candidates_kept": min(top, len(classified)),
    }
    return {
        "gemini_raw": gemini_raw_path,
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
        "summary": summary,
        "top_candidates": classified[:top],
        "all_candidates": classified,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a raw-chunk omission report from Gemini raw chunks plus a Whisper second opinion.")
    parser.add_argument("--gemini-raw", required=True, help="Gemini raw chunk JSON.")
    parser.add_argument("--primary", required=True, help="Primary aligned words JSON, usually Gemini+CTC output.")
    parser.add_argument("--secondary", required=True, help="Secondary words JSON, usually faster-whisper output.")
    parser.add_argument("--output", required=True, help="Output omission report JSON path.")
    parser.add_argument("--window-s", type=float, default=2.0, help="Sliding window size in seconds.")
    parser.add_argument("--step-s", type=float, default=1.0, help="Sliding window step in seconds.")
    parser.add_argument("--merge-gap-s", type=float, default=0.75, help="Merge adjacent flagged windows across gaps up to this size.")
    parser.add_argument("--min-secondary-chars", type=int, default=12, help="Minimum normalized secondary chars before a window can be flagged.")
    parser.add_argument("--max-primary-chars", type=int, default=4, help="Flag immediately when primary coverage is at or below this threshold.")
    parser.add_argument("--min-extra-chars", type=int, default=8, help="Otherwise require at least this many extra chars in the secondary transcript.")
    parser.add_argument("--top", type=int, default=30, help="Top classified candidates to keep in the summary sample.")
    args = parser.parse_args()

    raw_chunks = load_json(Path(args.gemini_raw))
    primary = load_json(Path(args.primary))
    secondary = load_json(Path(args.secondary))

    report = run_omission_report(
        raw_chunks,
        primary,
        secondary,
        gemini_raw_path=str(Path(args.gemini_raw)),
        primary_path=str(Path(args.primary)),
        secondary_path=str(Path(args.secondary)),
        window_s=args.window_s,
        step_s=args.step_s,
        merge_gap_s=args.merge_gap_s,
        min_secondary_chars=args.min_secondary_chars,
        max_primary_chars=args.max_primary_chars,
        min_extra_chars=args.min_extra_chars,
        top=args.top,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    classified = report["all_candidates"]
    print(f"Classified {len(classified)} omission candidates; wrote report to {output_path}")
    for item in classified[: min(args.top, 8)]:
        print(
            f"- {item['start_s']:.3f}-{item['end_s']:.3f}s "
            f"{item['classification']}[{item['confidence']}] "
            f"gap={item['coverage_gap_chars']}"
        )


if __name__ == "__main__":
    main()
