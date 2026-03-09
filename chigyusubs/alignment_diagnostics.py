"""Helpers for consuming alignment-stage diagnostics sidecars."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def diagnostics_path_for_words(words_path: str | Path) -> Path:
    return Path(f"{words_path}.diagnostics.json")


def discover_alignment_diagnostics_path(
    *,
    explicit_path: str | Path = "",
    words_path: str | Path = "",
    input_path: str | Path = "",
) -> Path | None:
    if explicit_path:
        return Path(explicit_path)

    if words_path:
        candidate = diagnostics_path_for_words(words_path)
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
            candidate = directory / f"{stem}.json.diagnostics.json"
            if candidate.exists():
                return candidate
    return None


def load_alignment_diagnostics(path_value: str | Path) -> dict | None:
    if not path_value:
        return None

    path = Path(path_value)
    if not path.exists():
        return None

    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        chunks = raw
    elif isinstance(raw, dict):
        chunks = raw.get("chunks", [])
    else:
        return None

    repaired_lines = _flatten_repaired_lines(chunks)
    visual_risk_chunks = [
        {
            "chunk": int(chunk.get("chunk", -1)),
            "chunk_start_s": round(float(chunk.get("chunk_start_s", 0.0)), 3),
            "chunk_end_s": round(float(chunk.get("chunk_end_s", 0.0)), 3),
            "stripped_visual_lines": int(chunk.get("stripped_visual_lines", 0)),
            "narration_like_visual_line_count": int(chunk.get("narration_like_visual_line_count", 0)),
            "suspicious_visual_runs": list(chunk.get("suspicious_visual_runs", [])),
            "review_reasons": list(chunk.get("review_reasons", [])),
        }
        for chunk in chunks
        if bool(chunk.get("possible_visual_narration_substitution"))
    ]
    return {
        "diagnostics_path": str(path),
        "chunks": chunks,
        "chunk_count": len(chunks),
        "interpolated_unaligned_segments": sum(
            int(chunk.get("repaired_unaligned_segments", 0)) for chunk in chunks
        ),
        "chunks_with_interpolated_unaligned_segments": sum(
            1 for chunk in chunks if int(chunk.get("repaired_unaligned_segments", 0)) > 0
        ),
        "chunks_with_visual_narration_substitution_risk": len(visual_risk_chunks),
        "visual_narration_risk_chunks": visual_risk_chunks,
        "repaired_lines": repaired_lines,
    }


def build_alignment_review(
    cues: Iterable[Any],
    diagnostics_path: str | Path,
    *,
    cue_id_start: int = 1,
    sample_limit: int = 8,
) -> dict | None:
    diagnostics = load_alignment_diagnostics(diagnostics_path)
    if not diagnostics:
        return None

    cue_list = list(cues)
    cue_warnings: dict[str, list[dict]] = {}
    unmapped_lines: list[dict] = []
    nearest_mapped_lines = 0

    for line in diagnostics["repaired_lines"]:
        overlapping_cue_ids = []
        for offset, cue in enumerate(cue_list):
            cue_id = cue_id_start + offset
            if _overlaps(cue.start, cue.end, line["repaired_start_s"], line["repaired_end_s"]):
                overlapping_cue_ids.append(cue_id)
                cue_warnings.setdefault(str(cue_id), []).append({**line, "warning_mapping": "overlap"})
        if overlapping_cue_ids:
            continue
        if cue_list:
            nearest_index = min(
                range(len(cue_list)),
                key=lambda idx: _cue_distance(
                    cue_list[idx].start,
                    cue_list[idx].end,
                    line["repaired_start_s"],
                    line["repaired_end_s"],
                ),
            )
            nearest_cue_id = cue_id_start + nearest_index
            cue_warnings.setdefault(str(nearest_cue_id), []).append({**line, "warning_mapping": "nearest_cue"})
            nearest_mapped_lines += 1
        else:
            unmapped_lines.append(line)

    affected_cue_ids = sorted(int(cue_id) for cue_id in cue_warnings)
    return {
        "diagnostics_path": diagnostics["diagnostics_path"],
        "advisory_only": True,
        "interpolated_unaligned_segments": diagnostics["interpolated_unaligned_segments"],
        "chunks_with_interpolated_unaligned_segments": diagnostics["chunks_with_interpolated_unaligned_segments"],
        "repaired_line_count": len(diagnostics["repaired_lines"]),
        "affected_cue_ids": affected_cue_ids,
        "affected_cues_count": len(affected_cue_ids),
        "cue_warnings": cue_warnings,
        "nearest_cue_mapped_lines_count": nearest_mapped_lines,
        "sample_repaired_lines": diagnostics["repaired_lines"][:sample_limit],
        "unmapped_repaired_lines_count": len(unmapped_lines),
        "sample_unmapped_repaired_lines": unmapped_lines[:sample_limit],
    }


def alignment_warnings_for_cue_ids(
    alignment_review: dict | None,
    cue_ids: Iterable[int],
    *,
    sample_limit: int = 8,
) -> dict | None:
    if not alignment_review:
        return None

    selected_ids = []
    lines_by_key: dict[str, dict] = {}
    cue_warnings = alignment_review.get("cue_warnings", {})
    for cue_id in cue_ids:
        key = str(int(cue_id))
        lines = cue_warnings.get(key, [])
        if not lines:
            continue
        selected_ids.append(int(cue_id))
        for line in lines:
            lines_by_key[str(line["line_key"])] = line

    if not selected_ids:
        return None

    repaired_lines = sorted(
        lines_by_key.values(),
        key=lambda item: (float(item["repaired_start_s"]), int(item["chunk"]), int(item["line_index_in_chunk"])),
    )
    return {
        "diagnostics_path": alignment_review["diagnostics_path"],
        "advisory_only": True,
        "episode_interpolated_unaligned_segments": int(alignment_review["interpolated_unaligned_segments"]),
        "episode_chunks_with_interpolated_unaligned_segments": int(
            alignment_review["chunks_with_interpolated_unaligned_segments"]
        ),
        "affected_cue_ids": sorted(selected_ids),
        "affected_cues_count": len(selected_ids),
        "repaired_line_count": len(repaired_lines),
        "repaired_lines": repaired_lines[:sample_limit],
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


def _flatten_repaired_lines(chunks: list[dict]) -> list[dict]:
    repaired_lines: list[dict] = []
    for chunk in chunks:
        chunk_index = int(chunk.get("chunk", -1))
        chunk_start = float(chunk.get("chunk_start_s", 0.0))
        chunk_end = float(chunk.get("chunk_end_s", 0.0))
        for item in chunk.get("repaired_unaligned_details", []):
            local_start = float(item.get("repaired_local_start_s", item.get("original_local_start_s", 0.0)))
            local_end = float(item.get("repaired_local_end_s", item.get("original_local_end_s", 0.0)))
            start_s = float(item.get("repaired_start_s", round(chunk_start + local_start, 3)))
            end_s = float(item.get("repaired_end_s", round(chunk_start + local_end, 3)))
            repaired_lines.append(
                {
                    "line_key": f"chunk:{chunk_index}:line:{int(item.get('line_index_in_chunk', -1))}",
                    "chunk": chunk_index,
                    "chunk_start_s": round(chunk_start, 3),
                    "chunk_end_s": round(chunk_end, 3),
                    "line_index_in_chunk": int(item.get("line_index_in_chunk", -1)),
                    "text": str(item.get("text", "")),
                    "repair_mode": str(item.get("repair_mode", "")),
                    "original_local_start_s": round(float(item.get("original_local_start_s", 0.0)), 3),
                    "original_local_end_s": round(float(item.get("original_local_end_s", 0.0)), 3),
                    "repaired_local_start_s": round(local_start, 3),
                    "repaired_local_end_s": round(local_end, 3),
                    "repaired_start_s": round(start_s, 3),
                    "repaired_end_s": round(end_s, 3),
                }
            )
    repaired_lines.sort(key=lambda item: (item["repaired_start_s"], item["chunk"], item["line_index_in_chunk"]))
    return repaired_lines


def _overlaps(start_a: float, end_a: float, start_b: float, end_b: float) -> bool:
    return float(end_a) > float(start_b) and float(start_a) < float(end_b)


def _cue_distance(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    if _overlaps(start_a, end_a, start_b, end_b):
        return 0.0
    if float(end_a) <= float(start_b):
        return float(start_b) - float(end_a)
    return float(start_a) - float(end_b)
