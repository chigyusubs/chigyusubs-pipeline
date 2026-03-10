"""Helpers for consuming Gemini speaker-turn boundaries from aligned words JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def discover_words_json_path(
    *,
    explicit_path: str | Path = "",
    words_path: str | Path = "",
    input_path: str | Path = "",
) -> Path | None:
    if explicit_path:
        return Path(explicit_path)

    if words_path:
        candidate = Path(words_path)
        if candidate.exists():
            return candidate

    if not input_path:
        return None

    input_file = Path(input_path)
    candidate_dirs = [input_file.parent]
    if input_file.parent.name == "translation":
        candidate_dirs.append(input_file.parent.parent / "transcription")

    for stem in _candidate_words_stems(input_file.stem):
        for directory in candidate_dirs:
            candidate = directory / f"{stem}.json"
            if candidate.exists():
                return candidate
    return None


def load_turn_segments(words_path: str | Path) -> dict | None:
    if not words_path:
        return None

    path = Path(words_path)
    if not path.exists():
        return None

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return None

    segments: list[dict] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        if "turn_index" not in item and "starts_new_turn" not in item:
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        segments.append(
            {
                "segment_index": idx,
                "text": text,
                "start_s": round(float(item.get("start", 0.0)), 3),
                "end_s": round(float(item.get("end", 0.0)), 3),
                "turn_index": int(item.get("turn_index", idx)),
                "starts_new_turn": bool(item.get("starts_new_turn", False)),
            }
        )

    if not segments:
        return None

    segments.sort(key=lambda item: (item["start_s"], item["segment_index"]))
    return {
        "words_path": str(path),
        "segments": segments,
        "segment_count": len(segments),
        "turn_count": len({int(item["turn_index"]) for item in segments}),
    }


def build_turn_review(
    cues: Iterable[Any],
    words_path: str | Path,
    *,
    cue_id_start: int = 1,
    sample_limit: int = 8,
) -> dict | None:
    turn_data = load_turn_segments(words_path)
    if not turn_data:
        return None

    cue_list = list(cues)
    cue_turn_segments: dict[str, list[dict]] = {}
    unmapped_segments: list[dict] = []
    nearest_mapped_segments = 0

    for segment in turn_data["segments"]:
        overlapping_cue_ids = []
        for offset, cue in enumerate(cue_list):
            cue_id = cue_id_start + offset
            if _overlaps(cue.start, cue.end, segment["start_s"], segment["end_s"]):
                cue_turn_segments.setdefault(str(cue_id), []).append({**segment, "warning_mapping": "overlap"})
                overlapping_cue_ids.append(cue_id)
        if overlapping_cue_ids:
            continue
        if cue_list:
            nearest_index = min(
                range(len(cue_list)),
                key=lambda idx: _cue_distance(
                    cue_list[idx].start,
                    cue_list[idx].end,
                    segment["start_s"],
                    segment["end_s"],
                ),
            )
            nearest_cue_id = cue_id_start + nearest_index
            cue_turn_segments.setdefault(str(nearest_cue_id), []).append(
                {**segment, "warning_mapping": "nearest_cue"}
            )
            nearest_mapped_segments += 1
        else:
            unmapped_segments.append(segment)

    cue_turns: dict[str, dict] = {}
    multi_turn_cue_ids: list[int] = []
    for cue_id_str, segments in cue_turn_segments.items():
        ordered = sorted(segments, key=lambda item: (item["start_s"], item["segment_index"]))
        unique_turn_indices = _ordered_unique(int(item["turn_index"]) for item in ordered)
        contains_internal_turn_start = any(bool(item["starts_new_turn"]) for item in ordered[1:])
        crosses_turn_boundary = len(unique_turn_indices) > 1
        cue_turns[cue_id_str] = {
            "starts_new_turn": bool(ordered[0]["starts_new_turn"]),
            "contains_internal_turn_start": contains_internal_turn_start,
            "crosses_turn_boundary": crosses_turn_boundary,
            "turn_span_count": len(unique_turn_indices),
            "source_line_count": len(ordered),
            "turn_indices": unique_turn_indices[:sample_limit],
            "source_lines_sample": [
                {
                    "text": item["text"],
                    "start_s": item["start_s"],
                    "end_s": item["end_s"],
                    "turn_index": int(item["turn_index"]),
                    "starts_new_turn": bool(item["starts_new_turn"]),
                    "warning_mapping": item["warning_mapping"],
                }
                for item in ordered[: min(sample_limit, 4)]
            ],
        }
        if crosses_turn_boundary:
            multi_turn_cue_ids.append(int(cue_id_str))

    multi_turn_cue_ids.sort()
    sample_multi_turn_cues = [
        {
            "cue_id": cue_id,
            **cue_turns[str(cue_id)],
        }
        for cue_id in multi_turn_cue_ids[:sample_limit]
    ]

    return {
        "words_path": turn_data["words_path"],
        "advisory_only": True,
        "source_turn_segments": int(turn_data["segment_count"]),
        "source_turn_count": int(turn_data["turn_count"]),
        "cues_with_turn_metadata_count": len(cue_turns),
        "multi_turn_cue_ids": multi_turn_cue_ids,
        "multi_turn_cues_count": len(multi_turn_cue_ids),
        "cue_turns": cue_turns,
        "nearest_cue_mapped_segments_count": nearest_mapped_segments,
        "unmapped_turn_segments_count": len(unmapped_segments),
        "sample_multi_turn_cues": sample_multi_turn_cues,
        "sample_unmapped_turn_segments": unmapped_segments[:sample_limit],
    }


def turn_context_for_cue_ids(
    turn_review: dict | None,
    cue_ids: Iterable[int],
    *,
    sample_limit: int = 8,
) -> dict | None:
    if not turn_review:
        return None

    selected: dict[str, dict] = {}
    for cue_id in cue_ids:
        key = str(int(cue_id))
        entry = turn_review.get("cue_turns", {}).get(key)
        if entry:
            selected[key] = entry

    if not selected:
        return None

    selected_ids = sorted(int(key) for key in selected)
    multi_turn_ids = [cue_id for cue_id in selected_ids if selected[str(cue_id)]["crosses_turn_boundary"]]
    return {
        "advisory_only": True,
        "words_path": turn_review["words_path"],
        "affected_cue_ids": selected_ids,
        "affected_cues_count": len(selected_ids),
        "multi_turn_cue_ids": multi_turn_ids,
        "multi_turn_cues_count": len(multi_turn_ids),
        "cue_turns": {key: selected[key] for key in sorted(selected.keys(), key=int)},
        "sample_multi_turn_cues": [
            {"cue_id": cue_id, **selected[str(cue_id)]}
            for cue_id in multi_turn_ids[:sample_limit]
        ],
    }


def turn_summary_payload(turn_review: dict | None) -> dict | None:
    """Format a turn review dict for Codex session status/diagnostics output."""
    if not turn_review:
        return None
    return {
        "advisory_only": True,
        "words_path": turn_review["words_path"],
        "source_turn_segments": turn_review["source_turn_segments"],
        "source_turn_count": turn_review["source_turn_count"],
        "cues_with_turn_metadata_count": turn_review["cues_with_turn_metadata_count"],
        "multi_turn_cues_count": turn_review["multi_turn_cues_count"],
        "multi_turn_cue_ids_sample": turn_review["multi_turn_cue_ids"][:8],
        "sample_multi_turn_cues": turn_review.get("sample_multi_turn_cues", []),
        "nearest_cue_mapped_segments_count": turn_review.get("nearest_cue_mapped_segments_count", 0),
        "unmapped_turn_segments_count": turn_review.get("unmapped_turn_segments_count", 0),
    }


def _candidate_words_stems(stem: str) -> list[str]:
    candidates = [stem]
    for suffix in ("_reflow_repaired", "_repaired", "_reflow"):
        if stem.endswith(suffix):
            candidates.append(stem[: -len(suffix)])
    seen = set()
    unique: list[str] = []
    for item in candidates:
        if item and item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _ordered_unique(values: Iterable[int]) -> list[int]:
    seen = set()
    ordered: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _overlaps(start_a: float, end_a: float, start_b: float, end_b: float) -> bool:
    return float(end_a) > float(start_b) and float(start_a) < float(end_b)


def _cue_distance(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    if _overlaps(start_a, end_a, start_b, end_b):
        return 0.0
    if float(end_a) <= float(start_b):
        return float(start_b) - float(end_a)
    return float(start_a) - float(end_b)
