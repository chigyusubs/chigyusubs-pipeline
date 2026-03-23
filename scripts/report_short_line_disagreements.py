#!/usr/bin/env python3
"""Flag short high-value Gemini raw lines that disagree with a Whisper second opinion."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chigyusubs.transcript_comparison import (
    assign_item_timings,
    char_bigram_jaccard,
    load_json,
    loose_contains,
    loose_normalize_text,
    normalize_text,
    overlap,
    sequence_ratio,
    text_similarity,
    window_text,
)


# ---------------------------------------------------------------------------
# Short-line classification heuristics (specific to this report)
# ---------------------------------------------------------------------------

_NAME_LIKE_RE = re.compile(r"[ぁ-んァ-ヶ一-龯A-Za-z0-9]{1,12}(さん|くん|ちゃん|氏)")
_NAME_FALSE_POSITIVE_SUBSTRINGS = ("くんない", "ちゃんと", "さんざん")
_TRIVIAL_SHORTS = {
    "あ",
    "え",
    "お",
    "うん",
    "はい",
    "へえ",
    "うわ",
    "おお",
    "ええ",
    "マジ",
    "本当",
    "ほんま",
}
_BARE_NAME_STOPWORDS = {
    "オッケー",
    "オーケー",
    "やばい",
    "やだね",
    "ほんでも",
    "ほんでもよ",
}


def has_name_like_marker(text: str) -> bool:
    if any(token in text for token in _NAME_FALSE_POSITIVE_SUBSTRINGS):
        return False
    return bool(_NAME_LIKE_RE.search(text))


def looks_like_bare_name_reference(text: str) -> bool:
    norm = loose_normalize_text(text)
    if not norm or norm in {loose_normalize_text(item) for item in _BARE_NAME_STOPWORDS}:
        return False
    if not re.search(r"[ァ-ヶ一-龯A-Za-z]", text):
        return False
    if not norm.endswith(("か", "さん")):
        return False
    stem = norm.removesuffix("か").removesuffix("さん")
    return 2 <= len(stem) <= 6


def is_high_value_short(text: str) -> bool:
    norm = normalize_text(text)
    if has_name_like_marker(text):
        return True
    if looks_like_bare_name_reference(text):
        return True
    if "？" in text or "?" in text:
        return True
    return any(ch.isdigit() for ch in norm)


def is_trivial_short(text: str) -> bool:
    norm = normalize_text(text)
    return norm in _TRIVIAL_SHORTS


def classify_candidate(primary_text: str, secondary_text: str, similarity: float) -> tuple[str, str]:
    has_name_marker = has_name_like_marker(primary_text) or looks_like_bare_name_reference(primary_text)
    looks_question = "？" in primary_text or "?" in primary_text
    if has_name_marker and similarity < 0.34:
        return "name_like_disagreement", "high"
    if looks_question and similarity < 0.18:
        return "short_question_disagreement", "high"
    if similarity < 0.18:
        return "short_high_value_disagreement", "high"
    return "short_phrase_disagreement", "medium"


# ---------------------------------------------------------------------------
# Secondary segment selection
# ---------------------------------------------------------------------------

def select_best_secondary_segment(
    segments: list[dict],
    primary_text: str,
    start_s: float,
    end_s: float,
    pad_s: float,
    min_secondary_chars: int,
    max_secondary_chars: int,
    max_window_segments: int,
) -> tuple[dict | None, float]:
    best = None
    best_key = None
    best_similarity = 0.0
    target_mid = (start_s + end_s) / 2.0
    search_start = max(0.0, start_s - pad_s)
    search_end = end_s + pad_s
    primary_has_name_signal = has_name_like_marker(primary_text) or looks_like_bare_name_reference(primary_text)
    primary_is_question = "？" in primary_text or "?" in primary_text
    nearby = []
    for seg in segments:
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", 0.0))
        if seg_end < search_start or seg_start > search_end:
            continue
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        nearby.append({"start": seg_start, "end": seg_end, "text": text})

    for start_idx in range(len(nearby)):
        combined_texts: list[str] = []
        window_start = nearby[start_idx]["start"]
        for end_idx in range(start_idx, min(len(nearby), start_idx + max_window_segments)):
            if end_idx > start_idx and nearby[end_idx]["start"] - nearby[end_idx - 1]["end"] > pad_s:
                break
            combined_texts.append(nearby[end_idx]["text"])
            combined_text = " / ".join(combined_texts)
            secondary_norm = normalize_text(combined_text)
            if len(secondary_norm) < min_secondary_chars:
                continue
            if len(secondary_norm) > max_secondary_chars:
                break
            window_end = nearby[end_idx]["end"]
            ov = max(0.0, min(end_s, window_end) - max(start_s, window_start))
            seg_mid = (window_start + window_end) / 2.0
            distance = abs(seg_mid - target_mid)
            similarity = text_similarity(primary_text, combined_text)
            contains = loose_contains(primary_text, combined_text)
            candidate_has_name_signal = has_name_like_marker(combined_text) or looks_like_bare_name_reference(combined_text)
            candidate_is_question = "？" in combined_text or "?" in combined_text
            key = (
                -int(contains),
                -int(primary_has_name_signal and candidate_has_name_signal),
                -int(primary_is_question and candidate_is_question),
                -similarity,
                -ov,
                distance,
                abs((window_end - window_start) - (end_s - start_s)),
            )
            if best is None or key < best_key:
                best = {"start": window_start, "end": window_end, "text": combined_text}
                best_key = key
                best_similarity = similarity
    return best, best_similarity


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def run_short_disagreement_report(
    raw_chunks: list[dict],
    primary: list[dict],
    secondary: list[dict],
    *,
    gemini_raw_path: str = "",
    primary_path: str = "",
    secondary_path: str = "",
    context_pad_s: float = 1.2,
    min_primary_chars: int = 3,
    max_primary_chars: int = 14,
    min_secondary_chars: int = 3,
    max_secondary_chars: int = 18,
    max_window_segments: int = 2,
    max_duration_s: float = 2.2,
    max_similarity: float = 0.34,
    top: int = 30,
) -> dict:
    """Run short-line disagreement analysis and return the report dict."""
    candidates: list[dict] = []
    for chunk_index, chunk in enumerate(raw_chunks):
        c_start = float(chunk["chunk_start_s"])
        c_end = float(chunk["chunk_end_s"])
        chunk_primary = [seg for seg in primary if float(seg["start"]) < c_end and float(seg["end"]) > c_start]
        items = assign_item_timings(chunk, chunk_primary)
        for item in items:
            if item["type"] != "spoken" or "start_s" not in item or "end_s" not in item:
                continue
            primary_text = str(item["text"]).strip()
            primary_norm = normalize_text(primary_text)
            if len(primary_norm) < min_primary_chars or len(primary_norm) > max_primary_chars:
                continue
            if is_trivial_short(primary_text):
                continue
            if not is_high_value_short(primary_text):
                continue
            duration_s = float(item["end_s"]) - float(item["start_s"])
            if duration_s <= 0.0 or duration_s > max_duration_s:
                continue
            secondary_seg, similarity = select_best_secondary_segment(
                secondary,
                primary_text,
                float(item["start_s"]),
                float(item["end_s"]),
                context_pad_s,
                min_secondary_chars,
                max_secondary_chars,
                max_window_segments,
            )
            if secondary_seg is None:
                continue
            secondary_text = str(secondary_seg.get("text", "")).strip()
            secondary_norm = normalize_text(secondary_text)
            if similarity > max_similarity:
                continue
            if primary_norm == secondary_norm or loose_contains(primary_text, secondary_text):
                continue
            classification, confidence = classify_candidate(primary_text, secondary_text, similarity)
            candidates.append(
                {
                    "chunk": chunk_index,
                    "chunk_start_s": round(c_start, 3),
                    "chunk_end_s": round(c_end, 3),
                    "start_s": round(float(item["start_s"]), 3),
                    "end_s": round(float(item["end_s"]), 3),
                    "duration_s": round(duration_s, 3),
                    "classification": classification,
                    "confidence": confidence,
                    "primary_text": primary_text,
                    "secondary_text": secondary_text,
                    "primary_chars": len(primary_norm),
                    "secondary_chars": len(secondary_norm),
                    "similarity": round(similarity, 3),
                }
            )

    class_priority = {
        "name_like_disagreement": 0,
        "short_question_disagreement": 1,
        "short_high_value_disagreement": 2,
        "short_phrase_disagreement": 3,
    }
    confidence_priority = {"high": 0, "medium": 1}
    candidates.sort(
        key=lambda item: (
            class_priority.get(item["classification"], 9),
            confidence_priority.get(item["confidence"], 9),
            item["similarity"],
            item["start_s"],
        )
    )

    summary = {
        "classified_candidates": len(candidates),
        "by_class": {
            key: sum(1 for item in candidates if item["classification"] == key)
            for key in class_priority
        },
        "by_confidence": {
            key: sum(1 for item in candidates if item["confidence"] == key)
            for key in confidence_priority
        },
        "top_candidates_kept": min(top, len(candidates)),
    }
    return {
        "gemini_raw": gemini_raw_path,
        "primary": primary_path,
        "secondary": secondary_path,
        "settings": {
            "context_pad_s": context_pad_s,
            "min_primary_chars": min_primary_chars,
            "max_primary_chars": max_primary_chars,
            "min_secondary_chars": min_secondary_chars,
            "max_secondary_chars": max_secondary_chars,
            "max_window_segments": max_window_segments,
            "max_duration_s": max_duration_s,
            "max_similarity": max_similarity,
        },
        "summary": summary,
        "top_candidates": candidates[:top],
        "all_candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Flag short high-value raw lines that disagree with a Whisper second opinion.")
    parser.add_argument("--gemini-raw", required=True, help="Gemini raw chunk JSON.")
    parser.add_argument("--primary", required=True, help="Primary aligned words JSON, usually Gemini+CTC output.")
    parser.add_argument("--secondary", required=True, help="Secondary words JSON, usually faster-whisper output.")
    parser.add_argument("--output", required=True, help="Output disagreement report JSON path.")
    parser.add_argument("--context-pad-s", type=float, default=1.2, help="Seconds of padding on each side when reading the secondary transcript.")
    parser.add_argument("--min-primary-chars", type=int, default=3, help="Minimum normalized chars for a primary line to be considered.")
    parser.add_argument("--max-primary-chars", type=int, default=14, help="Maximum normalized chars for a primary line to be considered.")
    parser.add_argument("--min-secondary-chars", type=int, default=3, help="Minimum normalized secondary chars for a candidate.")
    parser.add_argument("--max-secondary-chars", type=int, default=18, help="Maximum normalized secondary chars for a candidate.")
    parser.add_argument("--max-window-segments", type=int, default=2, help="Maximum number of nearby secondary segments to merge when looking for a match.")
    parser.add_argument("--max-duration-s", type=float, default=2.2, help="Maximum primary line duration for this short-line check.")
    parser.add_argument("--max-similarity", type=float, default=0.34, help="Maximum bigram Jaccard similarity before a candidate is ignored.")
    parser.add_argument("--top", type=int, default=30, help="Top candidates to keep in the summary sample.")
    args = parser.parse_args()

    raw_chunks = load_json(Path(args.gemini_raw))
    primary = load_json(Path(args.primary))
    secondary = load_json(Path(args.secondary))

    report = run_short_disagreement_report(
        raw_chunks,
        primary,
        secondary,
        gemini_raw_path=str(Path(args.gemini_raw)),
        primary_path=str(Path(args.primary)),
        secondary_path=str(Path(args.secondary)),
        context_pad_s=args.context_pad_s,
        min_primary_chars=args.min_primary_chars,
        max_primary_chars=args.max_primary_chars,
        min_secondary_chars=args.min_secondary_chars,
        max_secondary_chars=args.max_secondary_chars,
        max_window_segments=args.max_window_segments,
        max_duration_s=args.max_duration_s,
        max_similarity=args.max_similarity,
        top=args.top,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    candidates = report["all_candidates"]
    print(f"Classified {len(candidates)} short-line disagreement candidates; wrote report to {output_path}")
    for item in candidates[: min(args.top, 8)]:
        print(
            f"- {item['start_s']:.3f}-{item['end_s']:.3f}s "
            f"{item['classification']}[{item['confidence']}] "
            f"sim={item['similarity']:.3f} "
            f"primary='{item['primary_text']}' secondary='{item['secondary_text']}'"
        )


if __name__ == "__main__":
    main()
