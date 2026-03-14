"""Deterministic chunk-level sanity checks for raw transcript artifacts."""

from __future__ import annotations

import re
from collections import Counter

_VISUAL_RE = re.compile(r"^\[画面:.*\]$")
_SPEAKER_RE = re.compile(r"^--\s*")
_MARKDOWN_HEADING_RE = re.compile(r"^\*\*.*\*\*$")
_THOUGHT_LEAK_RE = re.compile(
    r"(transcription|segmenting|processing the audio|plain text|speaker labels|"
    r"following (all )?instructions|omitting silence|utterance per line|"
    r"i(?:'m| am) now|i(?:'ve| have) (?:begun|completed|started)|"
    r"implementing transcription|commencing transcription|finalizing transcription|"
    r"initiating transcription|analyzing transcription|maintaining transcription|"
    r"focusing transcription progress|methodically transcribing)",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[^\s、。！？,.!?\-]+")


def normalize_line(text: str) -> str:
    text = _SPEAKER_RE.sub("", text.strip())
    text = re.sub(r"\s+", " ", text)
    return text


def _repeated_token_issue(lines: list[str]) -> dict | None:
    worst: dict | None = None
    for line in lines:
        tokens = _TOKEN_RE.findall(normalize_line(line))
        if len(tokens) < 12:
            continue
        counts = Counter(tokens)
        token, count = counts.most_common(1)[0]
        share = count / len(tokens)
        if count >= 12 and share >= 0.75:
            issue = {
                "token": token,
                "count": count,
                "share": round(share, 3),
                "line": line[:240],
            }
            if worst is None or count > int(worst["count"]):
                worst = issue
    return worst


def inspect_chunk(chunk: dict) -> dict:
    raw_text = str(chunk.get("text", "") or "")
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    spoken_lines = [line for line in lines if _SPEAKER_RE.match(line)]
    visual_lines = [line for line in lines if _VISUAL_RE.match(line)]
    other_lines = [line for line in lines if line not in spoken_lines and line not in visual_lines]
    non_visual_lines = spoken_lines + other_lines
    normalized_non_visual = [normalize_line(line) for line in non_visual_lines if normalize_line(line)]

    issue_codes: list[str] = []
    red_reasons: list[str] = []
    yellow_reasons: list[str] = []

    thought_lines = [
        line for line in lines
        if _MARKDOWN_HEADING_RE.match(line)
        or line in {"code", "Text"}
        or bool(_THOUGHT_LEAK_RE.search(line))
    ]
    if thought_lines:
        issue_codes.append("thought_leak")
        red_reasons.append("meta transcription/thought text leaked into chunk output")

    if not lines:
        issue_codes.append("empty_chunk")
        red_reasons.append("empty transcript chunk")

    if visual_lines and not non_visual_lines:
        issue_codes.append("visual_only_chunk")
        red_reasons.append("chunk contains only visual lines")

    if other_lines and not spoken_lines and not visual_lines:
        issue_codes.append("missing_turn_markers")
        red_reasons.append("spoken content lost all `-- ` turn markers")
    elif other_lines and spoken_lines:
        coverage = len(spoken_lines) / max(len(non_visual_lines), 1)
        if coverage < 0.85:
            issue_codes.append("partial_turn_marker_loss")
            yellow_reasons.append("mixed marked/unmarked spoken lines")

    repeated_token_issue = _repeated_token_issue(non_visual_lines)
    if repeated_token_issue:
        issue_codes.append("repeated_token_loop")
        red_reasons.append("line contains pathological repeated-token loop")

    longest_line_chars = max((len(line) for line in lines), default=0)
    if longest_line_chars >= 500:
        issue_codes.append("overlong_line")
        red_reasons.append("extremely long line suggests runaway generation")

    short_line_counts = Counter(
        normalized
        for normalized in normalized_non_visual
        if 0 < len(normalized) <= 12
    )
    dominant_short_line = ""
    dominant_short_line_count = 0
    dominant_short_line_share = 0.0
    if short_line_counts and normalized_non_visual:
        dominant_short_line, dominant_short_line_count = short_line_counts.most_common(1)[0]
        dominant_short_line_share = dominant_short_line_count / len(normalized_non_visual)
        if dominant_short_line_count >= 12 and dominant_short_line_share >= 0.35:
            issue_codes.append("dominant_short_line_loop")
            red_reasons.append("one short line dominates the chunk suspiciously")
        elif dominant_short_line_count >= 8 and dominant_short_line_share >= 0.25:
            issue_codes.append("dominant_short_line_heavy")
            yellow_reasons.append("one short line repeats unusually often")

    if visual_lines and non_visual_lines and len(visual_lines) >= len(non_visual_lines) and len(visual_lines) >= 3:
        issue_codes.append("visual_heavy_chunk")
        yellow_reasons.append("visual lines dominate this chunk")

    status = "green"
    reasons: list[str] = []
    if red_reasons:
        status = "red"
        reasons = red_reasons
    elif yellow_reasons:
        status = "yellow"
        reasons = yellow_reasons

    return {
        "chunk": int(chunk.get("chunk", -1)),
        "chunk_start_s": float(chunk.get("chunk_start_s", 0.0)),
        "chunk_end_s": float(chunk.get("chunk_end_s", 0.0)),
        "duration_s": round(float(chunk.get("chunk_end_s", 0.0)) - float(chunk.get("chunk_start_s", 0.0)), 3),
        "status": status,
        "issue_codes": issue_codes,
        "reasons": reasons,
        "line_counts": {
            "total": len(lines),
            "spoken": len(spoken_lines),
            "visual": len(visual_lines),
            "other": len(other_lines),
        },
        "marker_coverage": round(len(spoken_lines) / max(len(non_visual_lines), 1), 3) if non_visual_lines else None,
        "dominant_short_line": dominant_short_line or None,
        "dominant_short_line_count": dominant_short_line_count,
        "dominant_short_line_share": round(dominant_short_line_share, 3) if dominant_short_line else None,
        "longest_line_chars": longest_line_chars,
        "thought_leak_sample": thought_lines[:4],
        "repeated_token_issue": repeated_token_issue,
        "preview": lines[:6],
        "model_version": chunk.get("model_version", ""),
    }


def summarize_chunk_inspections(inspected: list[dict]) -> dict:
    counts = Counter(item["status"] for item in inspected)
    red_chunks = [item["chunk"] for item in inspected if item["status"] == "red"]
    yellow_chunks = [item["chunk"] for item in inspected if item["status"] == "yellow"]
    return {
        "chunk_count": len(inspected),
        "green_chunks": counts.get("green", 0),
        "yellow_chunks": counts.get("yellow", 0),
        "red_chunks": counts.get("red", 0),
        "red_chunk_ids": red_chunks,
        "yellow_chunk_ids": yellow_chunks,
        "recommended_action": (
            "stop_and_repair_raw_chunks"
            if red_chunks
            else "review_yellow_chunks_before_alignment" if yellow_chunks else "safe_to_align"
        ),
    }
