"""Shared utilities for transcript comparison, coverage analysis, and disagreement detection.

Consolidates text normalization, time-overlap helpers, sliding-window coverage gap
detection, and Gemini raw chunk parsing that were previously duplicated across
compare_transcript_coverage.py, report_raw_chunk_omissions.py, and
report_short_line_disagreements.py.
"""

from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path


# ---------------------------------------------------------------------------
# Regex constants
# ---------------------------------------------------------------------------

VISUAL_RE = re.compile(r"^\[画面:\s*(.*?)\]$")
SPEAKER_DASH_RE = re.compile(r"^--\s*")
PUNCT_RE = re.compile(r"[\s\u3000、。！？?!…,.，「」『』（）()【】\[\]・:：\-ー]+")


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def load_json(path: Path):
    """Load and return parsed JSON from *path*."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_segments(path: Path) -> list[dict]:
    """Load a JSON array of timed segments from *path*."""
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"{path} did not contain a top-level list")
    return data


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Strip punctuation, whitespace, and brackets for comparison."""
    return PUNCT_RE.sub("", text or "")


def katakana_to_hiragana(text: str) -> str:
    """Convert katakana to hiragana after NFKC normalization."""
    chars: list[str] = []
    for ch in unicodedata.normalize("NFKC", text or ""):
        codepoint = ord(ch)
        if 0x30A1 <= codepoint <= 0x30F6:
            chars.append(chr(codepoint - 0x60))
        else:
            chars.append(ch)
    return "".join(chars)


def loose_normalize_text(text: str) -> str:
    """Normalize text loosely: strip punct, katakana→hiragana, common alternations."""
    norm = katakana_to_hiragana(normalize_text(text))
    norm = norm.replace("本当に", "ほんとに")
    norm = norm.replace("本当", "ほんと")
    return norm


# ---------------------------------------------------------------------------
# Text similarity
# ---------------------------------------------------------------------------

def char_bigram_jaccard(a: str, b: str) -> float:
    """Character bigram Jaccard similarity after loose normalization."""
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
    """SequenceMatcher ratio after loose normalization."""
    return SequenceMatcher(None, loose_normalize_text(a), loose_normalize_text(b)).ratio()


def text_similarity(a: str, b: str) -> float:
    """Best of bigram Jaccard and sequence ratio."""
    return max(char_bigram_jaccard(a, b), sequence_ratio(a, b))


def loose_contains(a: str, b: str) -> bool:
    """True if either loosely-normalized string contains the other."""
    a_norm = loose_normalize_text(a)
    b_norm = loose_normalize_text(b)
    return bool(a_norm and b_norm and (a_norm in b_norm or b_norm in a_norm))


# ---------------------------------------------------------------------------
# Time-overlap utilities
# ---------------------------------------------------------------------------

def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    """True if two time ranges overlap."""
    return a_start < b_end and a_end > b_start


def overlap_duration(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Return overlap duration in seconds (≥ 0)."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def window_text(segments: list[dict], start_s: float, end_s: float) -> str:
    """Concatenate text from *segments* overlapping the [start_s, end_s) window."""
    texts = []
    for seg in segments:
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", 0.0))
        if overlap(seg_start, seg_end, start_s, end_s):
            text = str(seg.get("text", "")).strip()
            if text:
                texts.append(text)
    return " / ".join(texts)


# ---------------------------------------------------------------------------
# Sliding-window coverage gap detection
# ---------------------------------------------------------------------------

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
    """Slide a window across both transcripts and flag regions where secondary has
    significantly more content than primary (potential omissions)."""
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
    """Merge adjacent flagged windows within *merge_gap_s* of each other."""
    if not flagged:
        return []
    merged = [dict(flagged[0])]
    for item in flagged[1:]:
        current = merged[-1]
        if item["start_s"] <= current["end_s"] + merge_gap_s:
            current["end_s"] = max(current["end_s"], item["end_s"])
            if "primary_chars" in current and "primary_chars" in item:
                current["primary_chars"] = max(current["primary_chars"], item["primary_chars"])
            if "secondary_chars" in current and "secondary_chars" in item:
                current["secondary_chars"] = max(current["secondary_chars"], item["secondary_chars"])
        else:
            merged.append(dict(item))
    return merged


# ---------------------------------------------------------------------------
# Gemini raw chunk parsing
# ---------------------------------------------------------------------------

def parse_raw_items(raw_text: str) -> list[dict]:
    """Parse Gemini raw text into a list of spoken/visual items with line indices."""
    items = []
    for line_index, raw_line in enumerate(raw_text.split("\n")):
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        visual_match = VISUAL_RE.match(raw_line)
        if visual_match:
            items.append({"type": "visual", "text": visual_match.group(1).strip(), "line_index": line_index})
            continue
        starts_new_turn = bool(SPEAKER_DASH_RE.match(raw_line))
        spoken_text = SPEAKER_DASH_RE.sub("", raw_line).strip()
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
    """Assign start_s/end_s to parsed raw items using aligned primary segments.

    Spoken items get timings from the primary segments (1:1 mapping).
    Visual items get interpolated timings from surrounding spoken anchors.
    """
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


def select_chunk_for_region(
    chunks: list[dict], start_s: float, end_s: float,
) -> tuple[int, dict] | tuple[None, None]:
    """Find the chunk with the most time overlap for a given region."""
    best = None
    best_overlap = 0.0
    for idx, chunk in enumerate(chunks):
        c_start = float(chunk["chunk_start_s"])
        c_end = float(chunk["chunk_end_s"])
        ov = overlap_duration(start_s, end_s, c_start, c_end)
        if ov > best_overlap:
            best = (idx, chunk)
            best_overlap = ov
    return best if best is not None else (None, None)
