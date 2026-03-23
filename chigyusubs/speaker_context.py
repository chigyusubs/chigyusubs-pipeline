"""Helpers for consuming named speaker maps for translation context."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def discover_named_speaker_map_path(
    *,
    explicit_path: str | Path = "",
    input_path: str | Path = "",
) -> Path | None:
    """Find a named_speaker_map.json relative to the input file.

    Searches sibling transcription/ directory for *_named_speaker_map.json.
    """
    if explicit_path:
        p = Path(explicit_path)
        return p if p.exists() else None

    if not input_path:
        return None

    input_file = Path(input_path)
    candidate_dirs = [input_file.parent]
    if input_file.parent.name == "translation":
        candidate_dirs.append(input_file.parent.parent / "transcription")
    elif input_file.parent.name == "transcription":
        candidate_dirs.append(input_file.parent)

    for directory in candidate_dirs:
        if not directory.exists():
            continue
        candidates = sorted(
            directory.glob("*_named_speaker_map.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]
    return None


def load_named_speaker_map(path: str | Path) -> dict | None:
    """Load and validate a named_speaker_map.json."""
    if not path:
        return None

    p = Path(path)
    if not p.exists():
        return None

    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return None
    if "identifications" not in raw or "effective_speakers" not in raw:
        return None
    return raw


def _load_source_speaker_map(named_map: dict, named_map_path: Path) -> dict | None:
    """Load the anonymous speaker_map.json referenced by the named map."""
    source_name = named_map.get("source_speaker_map", "")
    if not source_name:
        return None

    candidate = named_map_path.parent / source_name
    if not candidate.exists():
        return None

    raw = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "turns" not in raw:
        return None
    return raw


def _resolve_merges(named_map: dict) -> dict[str, str]:
    """Build spk_id -> effective_spk_id mapping from merges.

    Returns a dict where merged IDs map to their target, and non-merged IDs
    map to themselves.
    """
    merges = {}
    for merge in named_map.get("merges", []):
        source = merge.get("source", "")
        target = merge.get("target", "")
        if source and target:
            merges[source] = target
    return merges


def _spk_id_to_speaker_info(named_map: dict) -> dict[str, dict]:
    """Build spk_id -> speaker info dict from effective_speakers."""
    result: dict[str, dict] = {}
    for name, info in named_map.get("effective_speakers", {}).items():
        entry = {
            "speaker": name,
            "role": info.get("role", "unknown"),
        }
        if info.get("group"):
            entry["group"] = info["group"]
        # Look up confidence from identifications
        for spk_id in info.get("spk_ids", []):
            ident = named_map.get("identifications", {}).get(spk_id, {})
            result[spk_id] = {
                **entry,
                "confidence": ident.get("confidence", "medium"),
            }
    return result


def _overlaps(start_a: float, end_a: float, start_b: float, end_b: float) -> bool:
    return float(end_a) > float(start_b) and float(start_a) < float(end_b)


def build_speaker_review(
    cues: Iterable[Any],
    named_map_path: str | Path,
    *,
    cue_id_start: int = 1,
) -> dict | None:
    """Build per-cue speaker assignments from named speaker map + source speaker map.

    For each subtitle cue, finds overlapping speaker turns by timestamp and
    assigns the primary speaker by majority duration overlap.
    """
    named_map_path = Path(named_map_path)
    named_map = load_named_speaker_map(named_map_path)
    if not named_map:
        return None

    source_map = _load_source_speaker_map(named_map, named_map_path)
    if not source_map:
        return None

    merge_map = _resolve_merges(named_map)
    spk_info = _spk_id_to_speaker_info(named_map)

    turns = source_map.get("turns", [])
    cue_list = list(cues)

    cue_speakers: dict[str, dict] = {}

    for offset, cue in enumerate(cue_list):
        cue_id = cue_id_start + offset
        # Find overlapping turns and accumulate speaker durations
        speaker_durations: dict[str, float] = {}
        for turn in turns:
            if not _overlaps(cue.start, cue.end, turn["start"], turn["end"]):
                continue
            spk_id = turn.get("speaker")
            if spk_id is None:
                continue
            # Resolve merges
            resolved_id = merge_map.get(spk_id, spk_id)
            # Compute overlap duration
            overlap_start = max(cue.start, turn["start"])
            overlap_end = min(cue.end, turn["end"])
            overlap = max(0.0, overlap_end - overlap_start)
            speaker_durations[resolved_id] = speaker_durations.get(resolved_id, 0.0) + overlap

        if not speaker_durations:
            continue

        # Primary speaker = most overlap
        primary_id = max(speaker_durations, key=speaker_durations.get)
        info = spk_info.get(primary_id)
        if info:
            cue_speakers[str(cue_id)] = info

    if not cue_speakers:
        return None

    return {
        "named_map_path": str(named_map_path),
        "source_speaker_map": named_map.get("source_speaker_map", ""),
        "effective_speaker_count": len(named_map.get("effective_speakers", {})),
        "cue_speaker_count": len(cue_speakers),
        "cue_speakers": cue_speakers,
    }


def speaker_context_for_cue_ids(
    speaker_review: dict | None,
    cue_ids: Iterable[int],
) -> dict | None:
    """Extract per-cue speaker labels for a set of cue IDs."""
    if not speaker_review:
        return None

    cue_speakers = speaker_review.get("cue_speakers", {})
    selected: dict[str, dict] = {}
    for cue_id in cue_ids:
        key = str(int(cue_id))
        entry = cue_speakers.get(key)
        if entry:
            selected[key] = entry

    if not selected:
        return None

    return {
        "advisory_only": True,
        "cue_count": len(selected),
        "cues": {key: selected[key] for key in sorted(selected.keys(), key=int)},
    }


def speaker_summary_payload(speaker_review: dict | None) -> dict | None:
    """Format a speaker review dict for session status/diagnostics output."""
    if not speaker_review:
        return None
    return {
        "advisory_only": True,
        "named_map_path": speaker_review["named_map_path"],
        "source_speaker_map": speaker_review["source_speaker_map"],
        "effective_speaker_count": speaker_review["effective_speaker_count"],
        "cue_speaker_count": speaker_review["cue_speaker_count"],
    }
