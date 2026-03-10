#!/usr/bin/env python3
"""Classify likely raw-transcript omissions using Gemini raw chunks plus a Whisper second opinion."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


_VISUAL_RE = re.compile(r"^\[画面:\s*(.*?)\]$")
_SPEAKER_DASH_RE = re.compile(r"^--\s*")
_PUNCT_RE = re.compile(r"[\s\u3000、。！？?!…,.，「」『』（）()【】\[\]・:：\-ー]+")
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


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_text(text: str) -> str:
    return _PUNCT_RE.sub("", text or "")


def char_bigram_jaccard(a: str, b: str) -> float:
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    if len(a_norm) < 2 or len(b_norm) < 2:
        return 0.0
    a_set = {a_norm[i : i + 2] for i in range(len(a_norm) - 1)}
    b_set = {b_norm[i : i + 2] for i in range(len(b_norm) - 1)}
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


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
        primary_chars = len(normalize_text(primary_text))
        secondary_chars = len(normalize_text(secondary_text))
        if secondary_chars >= min_secondary_chars and (
            primary_chars <= max_primary_chars
            or secondary_chars >= max(primary_chars * 2, primary_chars + min_extra_chars)
        ):
            flagged.append(
                {
                    "start_s": round(cursor, 3),
                    "end_s": round(end_s, 3),
                    "primary_text": primary_text,
                    "secondary_text": secondary_text,
                    "primary_chars": primary_chars,
                    "secondary_chars": secondary_chars,
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
        else:
            merged.append(dict(item))
    return merged


def parse_raw_items(raw_text: str) -> list[dict]:
    items = []
    for line_index, raw_line in enumerate(raw_text.split("\n")):
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        visual_match = _VISUAL_RE.match(raw_line)
        if visual_match:
            items.append({"type": "visual", "text": visual_match.group(1).strip(), "line_index": line_index})
            continue
        starts_new_turn = bool(_SPEAKER_DASH_RE.match(raw_line))
        spoken_text = _SPEAKER_DASH_RE.sub("", raw_line).strip()
        if spoken_text:
            items.append(
                {
                    "type": "spoken",
                    "text": spoken_text,
                    "line_index": line_index,
                    "starts_new_turn": starts_new_turn,
                }
            )
    return items


def assign_item_timings(chunk: dict, primary_segments: list[dict]) -> list[dict]:
    items = parse_raw_items(chunk.get("text", ""))
    spoken_items = [item for item in items if item["type"] == "spoken"]
    seg_idx = 0
    for item in spoken_items:
        if seg_idx >= len(primary_segments):
            break
        seg = primary_segments[seg_idx]
        item["start_s"] = round(float(seg["start"]), 3)
        item["end_s"] = round(float(seg["end"]), 3)
        seg_idx += 1

    chunk_start = float(chunk["chunk_start_s"])
    chunk_end = float(chunk["chunk_end_s"])
    i = 0
    while i < len(items):
        if items[i]["type"] != "visual":
            i += 1
            continue
        j = i
        while j < len(items) and items[j]["type"] == "visual":
            j += 1

        prev_anchor = None
        next_anchor = None
        for k in range(i - 1, -1, -1):
            if "start_s" in items[k]:
                prev_anchor = items[k]
                break
        for k in range(j, len(items)):
            if "start_s" in items[k]:
                next_anchor = items[k]
                break

        if prev_anchor and next_anchor:
            gap_start = float(prev_anchor["end_s"])
            gap_end = float(next_anchor["start_s"])
        elif prev_anchor:
            gap_start = float(prev_anchor["end_s"])
            gap_end = min(chunk_end, gap_start + max(0.6, 0.3 * (j - i)))
        elif next_anchor:
            gap_end = float(next_anchor["start_s"])
            gap_start = max(chunk_start, gap_end - max(0.6, 0.3 * (j - i)))
        else:
            gap_start = chunk_start
            gap_end = chunk_end

        available = max(0.05, gap_end - gap_start)
        slot = available / max(1, (j - i))
        cursor = gap_start
        for k in range(i, j):
            items[k]["start_s"] = round(cursor, 3)
            cursor = min(gap_end, cursor + slot)
            items[k]["end_s"] = round(max(items[k]["start_s"] + 0.05, cursor), 3)

        i = j

    return items


def select_chunk_for_region(chunks: list[dict], start_s: float, end_s: float) -> tuple[int, dict] | tuple[None, None]:
    best = None
    best_overlap = 0.0
    for idx, chunk in enumerate(chunks):
        c_start = float(chunk["chunk_start_s"])
        c_end = float(chunk["chunk_end_s"])
        overlap_s = max(0.0, min(end_s, c_end) - max(start_s, c_start))
        if overlap_s > best_overlap:
            best = (idx, chunk)
            best_overlap = overlap_s
    return best if best is not None else (None, None)


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

    flagged = build_flagged_windows(
        primary,
        secondary,
        window_s=args.window_s,
        step_s=args.step_s,
        min_secondary_chars=args.min_secondary_chars,
        max_primary_chars=args.max_primary_chars,
        min_extra_chars=args.min_extra_chars,
    )
    merged_regions = merge_flagged_windows(flagged, merge_gap_s=args.merge_gap_s)

    chunk_items = []
    for chunk in raw_chunks:
        c_start = float(chunk["chunk_start_s"])
        c_end = float(chunk["chunk_end_s"])
        chunk_primary = [seg for seg in primary if float(seg["start"]) < c_end and float(seg["end"]) > c_start]
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
        "top_candidates_kept": min(args.top, len(classified)),
    }
    report = {
        "gemini_raw": str(Path(args.gemini_raw)),
        "primary": str(Path(args.primary)),
        "secondary": str(Path(args.secondary)),
        "settings": {
            "window_s": args.window_s,
            "step_s": args.step_s,
            "merge_gap_s": args.merge_gap_s,
            "min_secondary_chars": args.min_secondary_chars,
            "max_primary_chars": args.max_primary_chars,
            "min_extra_chars": args.min_extra_chars,
        },
        "summary": summary,
        "top_candidates": classified[: args.top],
        "all_candidates": classified,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Classified {len(classified)} omission candidates; wrote report to {output_path}")
    for item in classified[: min(args.top, 8)]:
        print(
            f"- {item['start_s']:.3f}-{item['end_s']:.3f}s "
            f"{item['classification']}[{item['confidence']}] "
            f"gap={item['coverage_gap_chars']}"
        )


if __name__ == "__main__":
    main()
