#!/usr/bin/env python3
"""Deterministic pre-filter for noisy OCR candidate terms.

Removes obvious noise (broadcast chrome, dates, currency, structural junk)
without any editorial judgment.  The goal is to reduce input tokens for the
LLM condensation step without losing any term the LLM would have kept.

Usable as a CLI tool or as an importable module:

    python scripts/clean_candidates.py --input raw.txt --output cleaned.txt

    from clean_candidates import clean_candidates
    cleaned = clean_candidates(raw_text)
"""

import argparse
import re
import sys
import unicodedata
from pathlib import Path


# ---------------------------------------------------------------------------
# Universal broadcast chrome — terms that are never glossary-worthy on any
# Japanese variety/owarai show.  Network names, sponsor cards, UI labels.
# ---------------------------------------------------------------------------
BROADCAST_NOISE: set[str] = {
    # Sponsor / UI
    "提供",
    "字幕",
    "次回予告",
    "番組",
    "再生",
    "停止",
    "ご覧のスポンサー",
    "番組の途中ですが",
    "放送",
    "生放送",
    "収録",
    "スタジオ",
    # Networks
    "TBS",
    "テレビ朝日",
    "フジテレビ",
    "日本テレビ",
    "テレビ東京",
    "MBS",
    "ABC",
    "ytv",
    "NHK",
    "BSテレ東",
    "BS-TBS",
    "BS朝日",
    "BSフジ",
    "読売テレビ",
    "関西テレビ",
    "東海テレビ",
    "中京テレビ",
    "CBCテレビ",
    "テレビ大阪",
    # Legal
    "ALL RIGHTS RESERVED",
}

# Pre-compute lowercase for case-insensitive matching.
_BROADCAST_NOISE_LOWER: set[str] = {t.lower() for t in BROADCAST_NOISE}

# ---------------------------------------------------------------------------
# Regex patterns for structural noise
# ---------------------------------------------------------------------------

# Pure punctuation / symbols (no actual word characters or kana/kanji).
_RE_PURE_PUNCT = re.compile(r"^[^\w\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff]+$")

# Pure digits (timestamps, counters, years standing alone).
_RE_PURE_DIGITS = re.compile(r"^\d+$")

# Repeated single character (ーーー, ///, ===, ・・・, etc.).
_RE_REPEATED_CHAR = re.compile(r"^(.)\1{2,}$")

# Date patterns: 2024年, 1月24日, 2024年1月24日, 2024/01/24
_RE_DATE = re.compile(
    r"^\d{2,4}年(\d{1,2}月(\d{1,2}日)?)?$"
    r"|^\d{1,2}月\d{1,2}日$"
    r"|^\d{2,4}[/\-]\d{1,2}([/\-]\d{1,2})?$"
)

# Time patterns: 20:00, 8時30分
_RE_TIME = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$|^\d{1,2}時(\d{1,2}分)?$")

# Currency: ¥1,000,000 or 1000円 or ￥500
_RE_CURRENCY = re.compile(r"^[¥￥]\s?[\d,]+$|^[\d,]+円$")

# 第N回, 第N話 (episode/round counters)
_RE_COUNTER = re.compile(r"^第\d+[回話]$")

# URL fragments
_RE_URL = re.compile(r"https?://|www\.|\.com|\.jp|\.co\.jp", re.IGNORECASE)

# Copyright lines
_RE_COPYRIGHT = re.compile(r"^[©◎]\s*\d{4}", re.IGNORECASE)

# Short pure-ASCII fragments (2 chars or less).
_RE_SHORT_ASCII = re.compile(r"^[A-Za-z0-9]{1,2}$")

# Leading/trailing punctuation to strip.
_RE_EDGE_PUNCT = re.compile(
    r"^[\s\u3000\-\−\—\―\─\–\"\'「」『』【】（）\(\)\[\]《》〈〉・、。,.!！?？:：;；~〜…♪★☆●○◆◇▶▷◀◁△▽→←↑↓※♥❤️\#＃]+|"
    r"[\s\u3000\-\−\—\―\─\–\"\'「」『』【】（）\(\)\[\]《》〈〉・、。,.!！?？:：;；~〜…♪★☆●○◆◇▶▷◀◁△▽→←↑↓※♥❤️\#＃]+$"
)


def _normalize(text: str) -> str:
    """NFKC normalize and collapse whitespace."""
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("\u3000", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _strip_edge_punct(text: str) -> str:
    return _RE_EDGE_PUNCT.sub("", text).strip()


def _is_structural_noise(term: str) -> bool:
    """Return True if the term is obviously not a glossary candidate."""
    if not term or len(term) <= 1:
        return True
    if _RE_PURE_PUNCT.match(term):
        return True
    if _RE_PURE_DIGITS.match(term):
        return True
    if _RE_REPEATED_CHAR.match(term):
        return True
    if _RE_DATE.match(term):
        return True
    if _RE_TIME.match(term):
        return True
    if _RE_CURRENCY.match(term):
        return True
    if _RE_COUNTER.match(term):
        return True
    if _RE_URL.search(term):
        return True
    if _RE_COPYRIGHT.match(term):
        return True
    if _RE_SHORT_ASCII.match(term):
        return True
    return False


def _is_broadcast_noise(term: str) -> bool:
    return term.lower() in _BROADCAST_NOISE_LOWER


def _substring_collapse(terms: list[str]) -> list[str]:
    """Remove terms that are a substring of another kept term."""
    # Process longer terms first so shorter substrings get absorbed.
    ordered = sorted(terms, key=len, reverse=True)
    kept: list[str] = []
    for t in ordered:
        if any(t != k and t in k for k in kept):
            continue
        kept.append(t)
    return kept


def clean_candidates(raw_text: str) -> list[str]:
    """Clean raw OCR candidate text and return deduplicated terms.

    Accepts newline-delimited and/or comma-delimited input (including 、).
    Returns a list of cleaned terms in input order, deduplicated.
    """
    normalized = raw_text.replace("、", ",")
    terms: list[str] = []
    seen: set[str] = set()

    for line in normalized.splitlines():
        line = line.strip()
        if not line:
            continue
        if "," in line:
            items = [x.strip() for x in line.split(",") if x.strip()]
        else:
            items = [line]

        for item in items:
            term = _normalize(item)
            term = _strip_edge_punct(term)

            if _is_structural_noise(term):
                continue
            if _is_broadcast_noise(term):
                continue
            # Catch long sentence-like noise (keep generous — 48 chars).
            if len(term) > 48:
                continue

            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            terms.append(term)

    return _substring_collapse(terms)


def main():
    parser = argparse.ArgumentParser(
        description="Deterministic pre-filter for noisy OCR candidates."
    )
    parser.add_argument(
        "--input", required=True, help="Input candidate file (newline/comma delimited)."
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output file (one term per line). Defaults to stdout.",
    )
    args = parser.parse_args()

    try:
        raw = Path(args.input).read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    cleaned = clean_candidates(raw)

    if args.output:
        Path(args.output).write_text("\n".join(cleaned) + "\n", encoding="utf-8")
        print(f"Wrote {len(cleaned)} terms to {args.output}")
    else:
        for term in cleaned:
            print(term)


if __name__ == "__main__":
    main()
