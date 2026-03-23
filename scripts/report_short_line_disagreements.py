#!/usr/bin/env python3
"""Flag short high-value Gemini raw lines that disagree with a Whisper second opinion."""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
import re
import unicodedata
from pathlib import Path


_VISUAL_RE = re.compile(r"^\[画面:\s*(.*?)\]$")
_SPEAKER_DASH_RE = re.compile(r"^--\s*")
_PUNCT_RE = re.compile(r"[\s\u3000、。！？?!…,.，「」『』（）()【】\[\]・:：\-ー]+")
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


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_text(text: str) -> str:
    return _PUNCT_RE.sub("", text or "")


def katakana_to_hiragana(text: str) -> str:
    chars: list[str] = []
    for ch in unicodedata.normalize("NFKC", text or ""):
        codepoint = ord(ch)
        if 0x30A1 <= codepoint <= 0x30F6:
            chars.append(chr(codepoint - 0x60))
        else:
            chars.append(ch)
    return "".join(chars)


def loose_normalize_text(text: str) -> str:
    norm = katakana_to_hiragana(normalize_text(text))
    # Common short-form alternations that otherwise create noisy false positives.
    norm = norm.replace("本当に", "ほんとに")
    norm = norm.replace("本当", "ほんと")
    return norm


def char_bigram_jaccard(a: str, b: str) -> float:
    a_norm = loose_normalize_text(a)
    b_norm = loose_normalize_text(b)
    if len(a_norm) < 2 or len(b_norm) < 2:
        return 0.0
    a_set = {a_norm[i : i + 2] for i in range(len(a_norm) - 1)}
    b_set = {b_norm[i : i + 2] for i in range(len(b_norm) - 1)}
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def sequence_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, loose_normalize_text(a), loose_normalize_text(b)).ratio()


def text_similarity(a: str, b: str) -> float:
    return max(char_bigram_jaccard(a, b), sequence_ratio(a, b))


def loose_contains(a: str, b: str) -> bool:
    a_norm = loose_normalize_text(a)
    b_norm = loose_normalize_text(b)
    return bool(a_norm and b_norm and (a_norm in b_norm or b_norm in a_norm))


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
    return items


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
            if len(primary_norm) < args.min_primary_chars or len(primary_norm) > args.max_primary_chars:
                continue
            if is_trivial_short(primary_text):
                continue
            if not is_high_value_short(primary_text):
                continue
            duration_s = float(item["end_s"]) - float(item["start_s"])
            if duration_s <= 0.0 or duration_s > args.max_duration_s:
                continue
            secondary_seg, similarity = select_best_secondary_segment(
                secondary,
                primary_text,
                float(item["start_s"]),
                float(item["end_s"]),
                args.context_pad_s,
                args.min_secondary_chars,
                args.max_secondary_chars,
                args.max_window_segments,
            )
            if secondary_seg is None:
                continue
            secondary_text = str(secondary_seg.get("text", "")).strip()
            secondary_norm = normalize_text(secondary_text)
            if similarity > args.max_similarity:
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
        "top_candidates_kept": min(args.top, len(candidates)),
    }
    report = {
        "gemini_raw": str(Path(args.gemini_raw)),
        "primary": str(Path(args.primary)),
        "secondary": str(Path(args.secondary)),
        "settings": {
            "context_pad_s": args.context_pad_s,
            "min_primary_chars": args.min_primary_chars,
            "max_primary_chars": args.max_primary_chars,
            "min_secondary_chars": args.min_secondary_chars,
            "max_secondary_chars": args.max_secondary_chars,
            "max_window_segments": args.max_window_segments,
            "max_duration_s": args.max_duration_s,
            "max_similarity": args.max_similarity,
        },
        "summary": summary,
        "top_candidates": candidates[: args.top],
        "all_candidates": candidates,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

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
