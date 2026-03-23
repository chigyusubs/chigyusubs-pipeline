#!/usr/bin/env python3
"""Side-by-side comparison of two raw transcript JSONs (Gemini vs Qwen, etc).

Takes two raw JSON files in the Gemini chunk format and produces:
1. A chunk-by-chunk comparison report (JSON)
2. A human-readable side-by-side text diff

Both inputs should be lists of objects with chunk_start_s, chunk_end_s, text.
Chunks are matched by time overlap, not by index.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_PUNCT_RE = re.compile(r"[\s\u3000、。！？?!…,.，「」『』（）()【】\[\]・:：\-ー～〜]+")
_VISUAL_RE = re.compile(r"\[画面[:：].*?\]")
_SPEAKER_RE = re.compile(r"^--\s*", re.MULTILINE)


def normalize(text: str) -> str:
    """Strip punctuation, whitespace, speaker markers, visual markers for comparison."""
    text = _VISUAL_RE.sub("", text)
    text = _SPEAKER_RE.sub("", text)
    return _PUNCT_RE.sub("", text)


def speech_lines(text: str) -> list[str]:
    """Extract only spoken lines (skip visual markers, empty lines)."""
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("[画面:") or line.startswith("[画面："):
            continue
        if line.startswith("-- "):
            line = line[3:].strip()
        elif line.startswith("- "):
            line = line[2:].strip()
        if line:
            lines.append(line)
    return lines


def load_raw_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON array")
    return data


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Return overlap duration in seconds."""
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    return max(0.0, end - start)


def match_chunks(a_chunks: list[dict], b_chunks: list[dict]) -> list[tuple[dict | None, dict | None]]:
    """Match chunks from A and B by time overlap. Returns pairs (a_chunk, b_chunk)."""
    pairs: list[tuple[dict | None, dict | None]] = []
    b_used = set()

    for a in a_chunks:
        a_start = float(a["chunk_start_s"])
        a_end = float(a["chunk_end_s"])
        best_b = None
        best_overlap = 0.0

        for j, b in enumerate(b_chunks):
            if j in b_used:
                continue
            b_start = float(b["chunk_start_s"])
            b_end = float(b["chunk_end_s"])
            ov = overlap(a_start, a_end, b_start, b_end)
            if ov > best_overlap:
                best_overlap = ov
                best_b = j

        if best_b is not None and best_overlap > 0:
            b_used.add(best_b)
            pairs.append((a, b_chunks[best_b]))
        else:
            pairs.append((a, None))

    # Add unmatched B chunks
    for j, b in enumerate(b_chunks):
        if j not in b_used:
            pairs.append((None, b))

    return pairs


def char_overlap_ratio(text_a: str, text_b: str) -> float:
    """Fraction of characters in text_a that also appear in text_b (order-insensitive bag)."""
    if not text_a:
        return 0.0
    bag_a = list(text_a)
    bag_b = list(text_b)
    matched = 0
    for ch in bag_a:
        if ch in bag_b:
            matched += 1
            bag_b.remove(ch)
    return matched / len(bag_a)


def compare_pair(a: dict | None, b: dict | None) -> dict:
    """Compare one matched pair of chunks."""
    a_text = (a.get("text", "") or "") if a else ""
    b_text = (b.get("text", "") or "") if b else ""

    a_norm = normalize(a_text)
    b_norm = normalize(b_text)

    a_lines = speech_lines(a_text)
    b_lines = speech_lines(b_text)

    result: dict = {
        "time_range": {
            "start_s": float(a["chunk_start_s"]) if a else float(b["chunk_start_s"]),
            "end_s": float(a["chunk_end_s"]) if a else float(b["chunk_end_s"]),
        },
    }

    result["a_chars"] = len(a_norm)
    result["b_chars"] = len(b_norm)
    result["char_delta"] = len(b_norm) - len(a_norm)
    result["a_lines"] = len(a_lines)
    result["b_lines"] = len(b_lines)

    # Content overlap (how much of A is found in B and vice versa)
    result["a_in_b_ratio"] = round(char_overlap_ratio(a_norm, b_norm), 3)
    result["b_in_a_ratio"] = round(char_overlap_ratio(b_norm, a_norm), 3)

    # Sample text for human review (first N chars of each)
    result["a_sample"] = a_text[:300] if a_text else ""
    result["b_sample"] = b_text[:300] if b_text else ""

    if a and "error" in a:
        result["a_error"] = a["error"]
    if b and "error" in b:
        result["b_error"] = b["error"]

    return result


def format_side_by_side(pairs: list[dict], a_label: str, b_label: str) -> str:
    """Format a human-readable side-by-side comparison."""
    lines: list[str] = []
    lines.append(f"{'=' * 80}")
    lines.append(f"Side-by-side: {a_label} vs {b_label}")
    lines.append(f"{'=' * 80}")
    lines.append("")

    for i, pair in enumerate(pairs):
        tr = pair["time_range"]
        lines.append(f"--- Chunk {i + 1}: {tr['start_s']:.1f}s - {tr['end_s']:.1f}s ---")
        lines.append(f"  {a_label}: {pair['a_chars']} chars, {pair['a_lines']} lines")
        lines.append(f"  {b_label}: {pair['b_chars']} chars, {pair['b_lines']} lines")
        lines.append(f"  Delta: {pair['char_delta']:+d} chars")
        lines.append(f"  Overlap: {a_label}→{b_label} {pair['a_in_b_ratio']:.0%}, "
                      f"{b_label}→{a_label} {pair['b_in_a_ratio']:.0%}")

        if pair.get("a_error"):
            lines.append(f"  {a_label} ERROR: {pair['a_error']}")
        if pair.get("b_error"):
            lines.append(f"  {b_label} ERROR: {pair['b_error']}")

        # Show first few lines of each
        a_sample = pair["a_sample"]
        b_sample = pair["b_sample"]
        if a_sample or b_sample:
            lines.append(f"  {a_label} text:")
            for sl in a_sample.splitlines()[:5]:
                lines.append(f"    {sl}")
            if a_sample.count("\n") > 5:
                lines.append(f"    ... ({a_sample.count(chr(10)) - 5} more lines)")
            lines.append(f"  {b_label} text:")
            for sl in b_sample.splitlines()[:5]:
                lines.append(f"    {sl}")
            if b_sample.count("\n") > 5:
                lines.append(f"    ... ({b_sample.count(chr(10)) - 5} more lines)")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare two raw transcript JSONs side by side.")
    parser.add_argument("--a", required=True, help="First transcript raw JSON (e.g. Gemini).")
    parser.add_argument("--b", required=True, help="Second transcript raw JSON (e.g. Qwen).")
    parser.add_argument("--a-label", default="gemini", help="Label for first transcript.")
    parser.add_argument("--b-label", default="qwen", help="Label for second transcript.")
    parser.add_argument("--output", required=True, help="Output comparison JSON.")
    parser.add_argument("--output-txt", default="", help="Output human-readable text (default: derived from --output).")
    args = parser.parse_args()

    a_path = Path(args.a)
    b_path = Path(args.b)
    output_path = Path(args.output)
    txt_path = Path(args.output_txt) if args.output_txt else output_path.with_suffix(".txt")

    a_chunks = load_raw_json(a_path)
    b_chunks = load_raw_json(b_path)

    log(f"Loaded {len(a_chunks)} {args.a_label} chunks, {len(b_chunks)} {args.b_label} chunks")

    matched = match_chunks(a_chunks, b_chunks)
    comparisons = [compare_pair(a, b) for a, b in matched]

    # Summary stats
    total_a_chars = sum(c["a_chars"] for c in comparisons)
    total_b_chars = sum(c["b_chars"] for c in comparisons)
    avg_a_in_b = sum(c["a_in_b_ratio"] for c in comparisons) / len(comparisons) if comparisons else 0
    avg_b_in_a = sum(c["b_in_a_ratio"] for c in comparisons) / len(comparisons) if comparisons else 0
    a_errors = sum(1 for c in comparisons if c.get("a_error"))
    b_errors = sum(1 for c in comparisons if c.get("b_error"))

    report = {
        "a_path": str(a_path),
        "b_path": str(b_path),
        "a_label": args.a_label,
        "b_label": args.b_label,
        "summary": {
            "matched_pairs": len(comparisons),
            f"{args.a_label}_total_chars": total_a_chars,
            f"{args.b_label}_total_chars": total_b_chars,
            "char_ratio": round(total_b_chars / total_a_chars, 3) if total_a_chars > 0 else None,
            f"avg_{args.a_label}_in_{args.b_label}": round(avg_a_in_b, 3),
            f"avg_{args.b_label}_in_{args.a_label}": round(avg_b_in_a, 3),
            f"{args.a_label}_errors": a_errors,
            f"{args.b_label}_errors": b_errors,
        },
        "chunks": comparisons,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    side_by_side = format_side_by_side(comparisons, args.a_label, args.b_label)
    txt_path.write_text(side_by_side, encoding="utf-8")

    log(f"\nSummary:")
    log(f"  {args.a_label}: {total_a_chars} chars across {len(a_chunks)} chunks ({a_errors} errors)")
    log(f"  {args.b_label}: {total_b_chars} chars across {len(b_chunks)} chunks ({b_errors} errors)")
    if total_a_chars > 0:
        log(f"  Char ratio ({args.b_label}/{args.a_label}): {total_b_chars / total_a_chars:.2f}x")
    log(f"  Avg overlap: {args.a_label}→{args.b_label} {avg_a_in_b:.0%}, "
        f"{args.b_label}→{args.a_label} {avg_b_in_a:.0%}")
    log(f"\nWrote: {output_path}")
    log(f"Wrote: {txt_path}")


def log(msg: str = ""):
    print(msg, flush=True)


if __name__ == "__main__":
    main()
