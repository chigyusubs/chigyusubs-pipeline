"""VAD-aware full-coverage chunking for long-form transcription."""

from __future__ import annotations

import json
import re
from pathlib import Path

from chigyusubs.metadata import read_artifact_metadata


def chunk_coverage_issues(
    chunk_bounds: list[tuple[float, float]],
    total_duration: float,
    *,
    tolerance_s: float = 0.05,
) -> list[str]:
    """Return human-readable coverage problems for a chunk plan.

    A valid plan must cover the full ``0 -> total_duration`` span contiguously,
    allowing only tiny floating-point noise within ``tolerance_s``.
    """
    issues: list[str] = []
    if not chunk_bounds:
        return ["no chunk bounds"]

    first_start = float(chunk_bounds[0][0])
    last_end = float(chunk_bounds[-1][1])
    if abs(first_start) > tolerance_s:
        issues.append(f"starts at {first_start:.3f}s instead of 0.000s")
    if abs(last_end - total_duration) > tolerance_s:
        issues.append(f"ends at {last_end:.3f}s instead of {total_duration:.3f}s")

    prev_end = float(chunk_bounds[0][1])
    for idx, (start, end) in enumerate(chunk_bounds[1:], start=1):
        start = float(start)
        end = float(end)
        delta = start - prev_end
        if delta > tolerance_s:
            issues.append(f"gap before chunk {idx}: {prev_end:.3f}-{start:.3f}s ({delta:.3f}s)")
        elif delta < -tolerance_s:
            issues.append(f"overlap before chunk {idx}: {start:.3f}-{prev_end:.3f}s ({-delta:.3f}s)")
        prev_end = end

    return issues


def find_chunk_boundaries(
    rttm_segments: list[dict],
    total_duration: float,
    target_chunk_s: float = 600,
    max_chunk_s: float | None = None,
    min_gap_s: float = 2.0,
) -> list[tuple[float, float]]:
    """Find full-coverage chunk boundaries at natural silence gaps from VAD segments.

    Returns list of contiguous ``(start_s, end_s)`` tuples that cover the full
    ``0 -> total_duration`` range.

    Silence gaps are used only to choose pleasant boundary locations. When a
    usable silence is found, the split point is placed at the midpoint of the
    gap so no part of the episode is dropped from coverage.
    """
    if not rttm_segments:
        return [(0, total_duration)]

    if max_chunk_s is None:
        max_chunk_s = target_chunk_s + 30.0
    max_chunk_s = max(float(max_chunk_s), float(target_chunk_s))

    sorted_segs = sorted(rttm_segments, key=lambda s: s["start"])
    gaps = []
    for i in range(1, len(sorted_segs)):
        gap_start = sorted_segs[i - 1]["end"]
        gap_end = sorted_segs[i]["start"]
        gap_dur = gap_end - gap_start
        if gap_dur >= min_gap_s:
            gaps.append({
                "time": (gap_start + gap_end) / 2,
                "gap_start": gap_start,
                "gap_end": gap_end,
                "duration": gap_dur,
            })

    boundaries: list[tuple[float, float]] = []
    chunk_start = 0.0

    while chunk_start < total_duration:
        target_end = chunk_start + target_chunk_s
        max_end = chunk_start + max_chunk_s
        if target_end >= total_duration - target_chunk_s * 0.3:
            if total_duration - chunk_start <= max_chunk_s:
                boundaries.append((chunk_start, total_duration))
                break
            forced_end = min(max_end, total_duration)
            boundaries.append((chunk_start, forced_end))
            chunk_start = forced_end
            continue

        best_gap = None
        best_dist = float("inf")
        for g in gaps:
            if g["time"] <= chunk_start:
                continue
            if g["time"] > max_end:
                continue
            dist = abs(g["time"] - target_end)
            if dist < target_chunk_s * 0.4 and dist < best_dist:
                best_dist = dist
                best_gap = g

        if best_gap:
            split_time = float(best_gap["time"])
            boundaries.append((chunk_start, split_time))
            chunk_start = split_time
        else:
            forced_end = min(max_end, total_duration)
            boundaries.append((chunk_start, forced_end))
            chunk_start = forced_end

    return boundaries


def chunk_duration_stats(chunk_bounds: list[tuple[float, float]]) -> dict[str, float | int]:
    if not chunk_bounds:
        return {
            "chunks": 0,
            "min_chunk_s": 0.0,
            "avg_chunk_s": 0.0,
            "max_chunk_s": 0.0,
        }
    durations = [float(end) - float(start) for start, end in chunk_bounds]
    return {
        "chunks": len(chunk_bounds),
        "min_chunk_s": round(min(durations), 3),
        "avg_chunk_s": round(sum(durations) / len(durations), 3),
        "max_chunk_s": round(max(durations), 3),
    }


def describe_chunk_plan(chunk_json_path: str | Path, chunk_bounds: list[tuple[float, float]]) -> dict[str, object]:
    path = Path(chunk_json_path)
    stem = path.stem.lower()
    metadata = read_artifact_metadata(path)
    session_payload = None
    session_path = path.with_name(path.stem + ".session.json")
    if session_path.exists():
        try:
            session_payload = json.loads(session_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            session_payload = None

    parts: list[str] = []
    source = "filename"
    if stem == "vad_chunks":
        parts.append("default VAD full-coverage plan")
    elif "semantic" in stem:
        parts.append("semantic reviewed plan")
    elif "exact" in stem:
        parts.append("exact-duration debug plan")
    elif "probe" in stem or "probes" in str(path):
        parts.append("probe chunk plan")
    else:
        parts.append("saved chunk plan")

    if "repair" in stem:
        parts.append("repair split")
    if "latefix" in stem:
        parts.append("late-fix follow-up")

    target_chunk_s = None
    if isinstance(metadata, dict):
        target_chunk_s = (
            metadata.get("chunk_settings", {}) or {}
        ).get("target_chunk_s")
        if target_chunk_s is not None:
            source = "metadata"
    if target_chunk_s is None and isinstance(session_payload, dict):
        target_chunk_s = session_payload.get("target_chunk_s")
        if target_chunk_s is not None:
            source = "session"
    if target_chunk_s is None:
        match = re.search(r"(\d+)s(?:$|[_-])", stem)
        if match:
            target_chunk_s = int(match.group(1))
        else:
            match = re.search(r"semantic_(\d+)", stem)
            if match:
                target_chunk_s = int(match.group(1))
    if target_chunk_s is not None:
        parts.append(f"target {float(target_chunk_s):g}s")

    stats = chunk_duration_stats(chunk_bounds)
    label = "; ".join(parts)
    return {
        "label": label,
        "path": str(path),
        "name": path.name,
        "source": source,
        **stats,
    }
