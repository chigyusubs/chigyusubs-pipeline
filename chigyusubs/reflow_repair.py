"""Shared helpers for Codex-interactive reflow review and repair."""

from __future__ import annotations

import re
from dataclasses import dataclass

from chigyusubs.translation import Cue


_TERMINAL_PUNCT = tuple("。！？?!…」』）)]")
MIN_CUE_DURATION_S = 0.5


@dataclass
class RepairRegion:
    region_id: int
    start_cue_id: int
    end_cue_id: int
    reasons: list[str]


def _sample_region_indices(count: int) -> list[int]:
    if count <= 0:
        return []
    if count == 1:
        return [0]
    if count == 2:
        return [0, 1]
    return [0, count // 2, count - 1]


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text)


def cue_duration(cue: Cue) -> float:
    return max(0.0, cue.end - cue.start)


def cue_chars(cue: Cue) -> int:
    return len(compact_text(cue.text))


def structural_preflight(cues: list[Cue]) -> dict:
    negative_duration = []
    overlaps = []
    micro_cues = []
    for idx, cue in enumerate(cues, 1):
        if cue_duration(cue) < MIN_CUE_DURATION_S:
            micro_cues.append(idx)
        if cue.end < cue.start:
            negative_duration.append(idx)
        if idx < len(cues) and cues[idx].start < cue.end:
            overlaps.append(idx)
    return {
        "negative_duration_cues": negative_duration,
        "overlap_after_cues": overlaps,
        "micro_duration_cues_under_0_5s": micro_cues,
    }


def _cue_flags(cue: Cue) -> dict[str, bool]:
    text = compact_text(cue.text)
    duration = cue_duration(cue)
    chars = len(text)
    terminal = bool(text) and text.endswith(_TERMINAL_PUNCT)
    return {
        "short": duration < 0.8,
        "very_short": duration < MIN_CUE_DURATION_S,
        "tiny": chars <= 4,
        "small": chars <= 8,
        "non_terminal": bool(text) and not terminal,
        "empty": not text,
    }


def _is_artifact_like_short_cluster(cues: list[Cue], idx: int) -> bool:
    current = _cue_flags(cues[idx])
    if not current["short"]:
        return False

    for neighbor_idx in (idx - 1, idx + 1):
        if neighbor_idx < 0 or neighbor_idx >= len(cues):
            continue
        neighbor = _cue_flags(cues[neighbor_idx])
        if not neighbor["short"]:
            continue
        # Treat short clusters as suspicious only when at least one cue reads
        # like an incomplete fragment. Legitimate bursts of terminal reactions
        # should stay advisory-only.
        if current["non_terminal"] or neighbor["non_terminal"]:
            return True
    return False


def detect_regions(cues: list[Cue], context_cues: int = 1) -> list[RepairRegion]:
    flagged: list[tuple[int, set[str]]] = []
    for idx, cue in enumerate(cues):
        flags = _cue_flags(cue)
        reasons: set[str] = set()
        if flags["empty"]:
            reasons.add("empty cue")
        if flags["short"] and flags["non_terminal"]:
            reasons.add("split-like boundary")
        if flags["short"] and flags["small"] and flags["non_terminal"]:
            reasons.add("short fragment cue")
        if _is_artifact_like_short_cluster(cues, idx):
            reasons.add("short cluster")
        if reasons:
            flagged.append((idx, reasons))

    if not flagged:
        return []

    raw_ranges: list[tuple[int, int, set[str]]] = []
    for idx, reasons in flagged:
        start = max(0, idx - context_cues)
        end = min(len(cues) - 1, idx + context_cues)
        raw_ranges.append((start, end, set(reasons)))

    merged: list[tuple[int, int, set[str]]] = []
    for start, end, reasons in raw_ranges:
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end, set(reasons)))
            continue
        prev_start, prev_end, prev_reasons = merged[-1]
        merged[-1] = (prev_start, max(prev_end, end), prev_reasons | reasons)

    regions: list[RepairRegion] = []
    for region_id, (start, end, reasons) in enumerate(merged):
        regions.append(
            RepairRegion(
                region_id=region_id,
                start_cue_id=start + 1,
                end_cue_id=end + 1,
                reasons=sorted(reasons),
            )
        )
    return regions


def region_reason_counts(regions: list[RepairRegion]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for region in regions:
        for reason in region.reasons:
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def sample_regions(cues: list[Cue], regions: list[RepairRegion]) -> list[dict]:
    samples: list[dict] = []
    for idx in _sample_region_indices(len(regions)):
        region = regions[idx]
        region_cues = cues[region.start_cue_id - 1:region.end_cue_id]
        samples.append(
            {
                "region_id": region.region_id,
                "start_cue_id": region.start_cue_id,
                "end_cue_id": region.end_cue_id,
                "cue_count": len(region_cues),
                "reasons": region.reasons,
                "preview": compact_region_text(region_cues)[:80],
            }
        )
    return samples


def build_review(cues: list[Cue], preflight: dict, regions: list[RepairRegion]) -> dict:
    short_count = 0
    tiny_count = 0
    under_one_second = 0
    for cue in cues:
        duration = cue_duration(cue)
        chars = cue_chars(cue)
        short_count += int(duration < 0.8)
        under_one_second += int(duration < 1.0)
        tiny_count += int(chars <= 4)

    reasons: list[str] = []
    review = "green"
    if (
        preflight["negative_duration_cues"]
        or preflight["overlap_after_cues"]
        or preflight["micro_duration_cues_under_0_5s"]
    ):
        review = "red"
        reasons.append("structural timing blocker")
        if preflight["micro_duration_cues_under_0_5s"]:
            reasons.append("cue under 0.5s")
    elif regions:
        review = "yellow"
        reasons.append("artifact-risk cue boundaries need review")

    return {
        "review": review,
        "reasons": reasons,
        "metrics": {
            "total_cues": len(cues),
            "negative_duration_count": len(preflight["negative_duration_cues"]),
            "overlap_count": len(preflight["overlap_after_cues"]),
            "micro_cues_under_0_5s": len(preflight["micro_duration_cues_under_0_5s"]),
            "short_cues_under_0_8s": short_count,
            "short_cues_under_1_0s": under_one_second,
            "tiny_cues_le_4_chars": tiny_count,
            "detected_regions": len(regions),
            "region_reason_counts": region_reason_counts(regions),
        },
        "sampled_regions": sample_regions(cues, regions),
    }


def compact_region_text(cues: list[Cue]) -> str:
    return "".join(compact_text(cue.text) for cue in cues)


def interpolate_region_time(cues: list[Cue], offset_chars: int) -> float:
    if not cues:
        raise ValueError("cannot interpolate an empty cue region")
    total_chars = sum(max(0, cue_chars(cue)) for cue in cues)
    if total_chars <= 0:
        start = cues[0].start
        end = cues[-1].end
        return start + (end - start) * 0.0
    if offset_chars <= 0:
        return cues[0].start
    if offset_chars >= total_chars:
        return cues[-1].end

    cursor = 0
    for cue in cues:
        chars = cue_chars(cue)
        if chars <= 0:
            continue
        next_cursor = cursor + chars
        if offset_chars <= next_cursor:
            span = max(0.0, cue.end - cue.start)
            ratio = (offset_chars - cursor) / max(chars, 1)
            return cue.start + span * ratio
        cursor = next_cursor
    return cues[-1].end


def synthesize_region_cues(source_cues: list[Cue], replacement_texts: list[str]) -> list[Cue]:
    if not replacement_texts:
        raise ValueError("replacement_texts cannot be empty")
    cleaned = [text.rstrip() for text in replacement_texts]
    if any(not compact_text(text) for text in cleaned):
        raise ValueError("replacement cues must be non-empty")

    source_compact = compact_region_text(source_cues)
    replacement_compact = "".join(compact_text(text) for text in cleaned)
    if replacement_compact != source_compact:
        raise ValueError("replacement cues do not preserve source text coverage")

    total_chars = len(source_compact)
    boundaries = [0]
    cursor = 0
    for text in cleaned[:-1]:
        cursor += len(compact_text(text))
        boundaries.append(cursor)
    boundaries.append(total_chars)

    rebuilt: list[Cue] = []
    region_start = source_cues[0].start
    region_end = source_cues[-1].end
    for idx, text in enumerate(cleaned):
        start_offset = boundaries[idx]
        end_offset = boundaries[idx + 1]
        start = region_start if idx == 0 else interpolate_region_time(source_cues, start_offset)
        end = region_end if idx == len(cleaned) - 1 else interpolate_region_time(source_cues, end_offset)
        if end < start:
            raise ValueError("replacement cues created negative duration")
        rebuilt.append(Cue(start, end, text))
    return rebuilt


def render_repaired_cues(
    base_cues: list[Cue],
    regions: list[RepairRegion],
    applied_regions: dict[int, list[Cue]],
) -> list[Cue]:
    if not applied_regions:
        return list(base_cues)

    region_map = {region.region_id: region for region in regions}
    output: list[Cue] = []
    idx = 1
    while idx <= len(base_cues):
        replaced = False
        for region_id, repaired in applied_regions.items():
            region = region_map[region_id]
            if idx == region.start_cue_id:
                output.extend(repaired)
                idx = region.end_cue_id + 1
                replaced = True
                break
        if not replaced:
            output.append(base_cues[idx - 1])
            idx += 1
    return output
