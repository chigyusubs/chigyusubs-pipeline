#!/usr/bin/env python3
"""Deterministic chunk-level sanity gate for raw transcript JSON artifacts.

This catches obvious upstream transcript failures before alignment/reflow:

- missing `-- ` turn markers
- prompt/thought leakage
- visual-only substitution
- pathological repetition loops
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chigyusubs.metadata import finish_run, inherit_run_id, lineage_output_path, metadata_path, start_run, write_metadata

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


def _normalize_line(text: str) -> str:
    text = _SPEAKER_RE.sub("", text.strip())
    text = re.sub(r"\s+", " ", text)
    return text


def _repeated_token_issue(lines: list[str]) -> dict | None:
    worst: dict | None = None
    for line in lines:
        tokens = _TOKEN_RE.findall(_normalize_line(line))
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
    normalized_non_visual = [_normalize_line(line) for line in non_visual_lines if _normalize_line(line)]

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


def main() -> None:
    run = start_run("raw_chunk_sanity")
    parser = argparse.ArgumentParser(description="Check raw transcript chunks for obvious upstream failures before alignment/reflow.")
    parser.add_argument("--input", required=True, help="Raw chunk transcript JSON, usually *_gemini_raw.json.")
    parser.add_argument("--output", default="", help="Output JSON path. Defaults to transcription/diagnostics/<run>_raw_chunk_sanity.json.")
    parser.add_argument("--fail-on-red", action="store_true", help="Exit non-zero when any chunk is red.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    run = inherit_run_id(run, input_path)
    chunks = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(chunks, list):
        raise SystemExit("Expected a list of chunk objects.")

    if not args.output:
        if input_path.parent.name == "transcription":
            output_dir = input_path.parent / "diagnostics"
        else:
            output_dir = input_path.parent
        output_path = lineage_output_path(output_dir, artifact_type="raw_chunk_sanity", run=run, suffix=".json")
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    inspected = [inspect_chunk(chunk) for chunk in chunks]
    counts = Counter(item["status"] for item in inspected)
    red_chunks = [item["chunk"] for item in inspected if item["status"] == "red"]
    yellow_chunks = [item["chunk"] for item in inspected if item["status"] == "yellow"]

    payload = {
        "input_raw_json": str(input_path),
        "summary": {
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
        },
        "policy": {
            "red": [
                "thought / meta transcription leakage",
                "spoken chunks that lost all `-- ` turn markers",
                "visual-only chunk substitution",
                "pathological repetition loops",
            ],
            "yellow": [
                "mixed marked/unmarked spoken lines",
                "visual-heavy mixed chunk",
                "dominant repeated short line",
            ],
        },
        "chunks": inspected,
    }

    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    metadata = finish_run(
        run,
        inputs={"raw_json": str(input_path)},
        outputs={"sanity_report_json": str(output_path)},
        settings={"fail_on_red": args.fail_on_red},
        stats={
            "chunks": len(inspected),
            "green_chunks": counts.get("green", 0),
            "yellow_chunks": counts.get("yellow", 0),
            "red_chunks": counts.get("red", 0),
        },
    )
    write_metadata(output_path, metadata)

    print(f"Wrote raw chunk sanity report: {output_path}")
    print(
        "  "
        f"green={counts.get('green', 0)} "
        f"yellow={counts.get('yellow', 0)} "
        f"red={counts.get('red', 0)}"
    )
    if red_chunks:
        print(f"  Red chunks: {red_chunks}")
    if yellow_chunks:
        print(f"  Yellow chunks: {yellow_chunks}")
    print(f"Metadata written: {metadata_path(output_path)}")

    if args.fail_on_red and red_chunks:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
