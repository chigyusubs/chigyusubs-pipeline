#!/usr/bin/env python3
"""Project a reference English VTT onto a target subtitle timeline.

This is a pragmatic helper for adapting a finished translation from one cut
of an episode onto a nearby alternate cut. It assumes both cuts preserve scene
order and differ only by small trims, so a simple proportional time warp is a
useful first draft.

The script does not try to be semantically perfect. It can produce:

- a target-timed English VTT draft
- a JSON report showing how reference cues were assigned
- or a retimed copy of the reference English VTT on the target cut

The draft can then be used as `--seed-from` for `translate_vtt_codex.py`.
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import sys
from pathlib import Path
from difflib import SequenceMatcher

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chigyusubs.metadata import (
    build_vtt_note_lines,
    finish_run,
    lineage_output_path,
    metadata_path,
    start_run,
    update_preferred_manifest,
    write_metadata,
)
from chigyusubs.translation import Cue, parse_vtt, serialize_vtt, seconds_to_time


def _load_vtt(path: Path) -> list[Cue]:
    return parse_vtt(path.read_text(encoding="utf-8"))


def _normalize_join(texts: list[str]) -> str:
    parts = []
    for text in texts:
        normalized = " ".join(line.strip() for line in text.splitlines() if line.strip())
        normalized = " ".join(normalized.split())
        if normalized:
            parts.append(normalized)
    return " ".join(parts).strip()


def _cue_midpoint(cue: Cue) -> float:
    return (cue.start + cue.end) / 2.0


_NORM_RE = re.compile(r"[\s\u3000、。！？?!…,.，「」『』（）()【】\[\]・:：\-ー/]+")


def _norm_text(text: str) -> str:
    return _NORM_RE.sub("", text or "")


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _target_index_for_time(target_cues: list[Cue], target_starts: list[float], t: float) -> int:
    idx = bisect.bisect_right(target_starts, t) - 1
    if idx < 0:
        return 0
    if idx >= len(target_cues):
        return len(target_cues) - 1
    cue = target_cues[idx]
    if cue.start <= t < cue.end:
        return idx
    if idx + 1 < len(target_cues):
        next_cue = target_cues[idx + 1]
        if abs(_cue_midpoint(next_cue) - t) < abs(_cue_midpoint(cue) - t):
            return idx + 1
    return idx


def _piecewise_map(value: float, anchors: list[tuple[float, float]]) -> float:
    if value <= anchors[0][0]:
        return anchors[0][1]
    if value >= anchors[-1][0]:
        return anchors[-1][1]
    for idx in range(len(anchors) - 1):
        left_x, left_y = anchors[idx]
        right_x, right_y = anchors[idx + 1]
        if left_x <= value <= right_x:
            if right_x <= left_x:
                return right_y
            frac = (value - left_x) / (right_x - left_x)
            return left_y + frac * (right_y - left_y)
    return anchors[-1][1]


def _build_text_anchors(
    target_bridge: list[Cue],
    reference_bridge: list[Cue],
    *,
    search_window: int = 18,
    min_chars: int = 6,
    min_similarity: float = 0.74,
) -> list[tuple[float, float]]:
    target_last_end = target_bridge[-1].end if target_bridge else 0.0
    reference_last_end = reference_bridge[-1].end if reference_bridge else 0.0
    anchors: list[tuple[float, float]] = [(0.0, 0.0)]
    last_ref_idx = -1
    ref_norm = [_norm_text(cue.text) for cue in reference_bridge]

    for target_idx, target_cue in enumerate(target_bridge):
        target_norm = _norm_text(target_cue.text)
        if len(target_norm) < min_chars:
            continue
        predicted = round((target_idx / max(1, len(target_bridge) - 1)) * max(0, len(reference_bridge) - 1))
        start = max(last_ref_idx + 1, predicted - search_window)
        end = min(len(reference_bridge) - 1, predicted + search_window)
        if start > end:
            continue
        best_ref_idx = None
        best_score = 0.0
        for ref_idx in range(start, end + 1):
            other_norm = ref_norm[ref_idx]
            if len(other_norm) < min_chars:
                continue
            text_score = _similarity(target_norm, other_norm)
            if text_score < min_similarity:
                continue
            target_rel = _cue_midpoint(target_cue) / max(target_last_end, 0.001)
            ref_rel = _cue_midpoint(reference_bridge[ref_idx]) / max(reference_last_end, 0.001)
            position_score = 1.0 - min(abs(target_rel - ref_rel) * 8.0, 1.0)
            score = (text_score * 0.85) + (position_score * 0.15)
            if score > best_score:
                best_score = score
                best_ref_idx = ref_idx
        if best_ref_idx is None:
            continue
        target_mid = _cue_midpoint(target_cue)
        ref_mid = _cue_midpoint(reference_bridge[best_ref_idx])
        if target_mid <= anchors[-1][0] or ref_mid <= anchors[-1][1]:
            continue
        anchors.append((target_mid, ref_mid))
        last_ref_idx = best_ref_idx

    anchors.append((target_last_end, reference_last_end))
    return anchors


def _build_assignments(
    target_cues: list[Cue],
    reference_cues: list[Cue],
    *,
    bridge_target_cues: list[Cue] | None = None,
    bridge_reference_cues: list[Cue] | None = None,
) -> tuple[list[list[int]], list[tuple[float, float]]]:
    target_last_end = target_cues[-1].end if target_cues else 0.0
    reference_last_end = reference_cues[-1].end if reference_cues else 0.0
    if target_last_end <= 0 or reference_last_end <= 0:
        raise ValueError("Input cues must have positive duration span.")

    if bridge_target_cues and bridge_reference_cues:
        anchors = _build_text_anchors(bridge_target_cues, bridge_reference_cues)
    else:
        scale = reference_last_end / target_last_end
        anchors = [(0.0, 0.0), (target_last_end, reference_last_end)]

    inverse_anchors = [(ref_t, target_t) for target_t, ref_t in anchors]
    assignments: list[list[int]] = [[] for _ in target_cues]
    target_starts = [cue.start for cue in target_cues]
    for ref_idx, ref_cue in enumerate(reference_cues):
        mapped_mid = _piecewise_map(_cue_midpoint(ref_cue), inverse_anchors)
        target_idx = _target_index_for_time(target_cues, target_starts, mapped_mid)
        assignments[target_idx].append(ref_idx)
    return assignments, anchors


def _fallback_reference_index(reference_cues: list[Cue], inverse_anchors: list[tuple[float, float]], target_mid: float) -> int:
    best_idx = 0
    best_dist = None
    for idx, cue in enumerate(reference_cues):
        mapped_mid = _piecewise_map(_cue_midpoint(cue), inverse_anchors)
        dist = abs(mapped_mid - target_mid)
        if best_dist is None or dist < best_dist:
            best_idx = idx
            best_dist = dist
    return best_idx


def adapt_draft(
    target_cues: list[Cue],
    reference_cues: list[Cue],
    *,
    bridge_target_cues: list[Cue] | None = None,
    bridge_reference_cues: list[Cue] | None = None,
) -> tuple[list[Cue], dict]:
    assignments, anchors = _build_assignments(
        target_cues,
        reference_cues,
        bridge_target_cues=bridge_target_cues,
        bridge_reference_cues=bridge_reference_cues,
    )
    inverse_anchors = [(ref_t, target_t) for target_t, ref_t in anchors]

    adapted: list[Cue] = []
    cue_report: list[dict] = []
    multi_assign = 0
    fallback_count = 0

    for target_idx, target_cue in enumerate(target_cues):
        ref_indices = assignments[target_idx]
        used_fallback = False
        if not ref_indices:
            fallback_idx = _fallback_reference_index(reference_cues, inverse_anchors, _cue_midpoint(target_cue))
            ref_indices = [fallback_idx]
            used_fallback = True
            fallback_count += 1
        if len(ref_indices) > 1:
            multi_assign += 1

        ref_texts = [reference_cues[idx].text for idx in ref_indices]
        text = _normalize_join(ref_texts)
        if not text:
            fallback_idx = _fallback_reference_index(reference_cues, inverse_anchors, _cue_midpoint(target_cue))
            ref_indices = [fallback_idx]
            ref_texts = [reference_cues[fallback_idx].text]
            text = _normalize_join(ref_texts)
            used_fallback = True
            fallback_count += 1
        if not text:
            text = "..."

        adapted.append(Cue(target_cue.start, target_cue.end, text))
        cue_report.append(
            {
                "target_cue_id": target_idx + 1,
                "target_start_tc": seconds_to_time(target_cue.start),
                "target_end_tc": seconds_to_time(target_cue.end),
                "reference_cue_ids": [idx + 1 for idx in ref_indices],
                "reference_texts": ref_texts,
                "draft_text": text,
                "used_fallback": used_fallback,
            }
        )

    report = {
        "anchor_count": len(anchors),
        "timewarp_anchors": [[round(a, 3), round(b, 3)] for a, b in anchors],
        "target_cues": len(target_cues),
        "reference_cues": len(reference_cues),
        "multi_assigned_target_cues": multi_assign,
        "fallback_target_cues": fallback_count,
        "cue_mapping": cue_report,
    }
    return adapted, report


def retime_reference(
    reference_en_cues: list[Cue],
    target_bridge_cues: list[Cue],
    reference_bridge_cues: list[Cue],
) -> tuple[list[Cue], dict]:
    anchors = _build_text_anchors(target_bridge_cues, reference_bridge_cues)
    inverse_anchors = [(ref_t, target_t) for target_t, ref_t in anchors]
    retimed: list[Cue] = []
    prev_end = 0.0
    target_end = target_bridge_cues[-1].end

    for cue in reference_en_cues:
        start = _piecewise_map(cue.start, inverse_anchors)
        end = _piecewise_map(cue.end, inverse_anchors)
        start = max(start, prev_end)
        end = max(end, start + 0.01)
        end = min(end, target_end)
        retimed.append(Cue(start, end, cue.text))
        prev_end = end

    report = {
        "anchor_count": len(anchors),
        "timewarp_anchors": [[round(a, 3), round(b, 3)] for a, b in anchors],
        "reference_cues": len(reference_en_cues),
        "target_end": round(target_end, 3),
        "retimed_end": round(retimed[-1].end if retimed else 0.0, 3),
    }
    return retimed, report


def main() -> int:
    parser = argparse.ArgumentParser(description="Adapt a reference English VTT onto a nearby target VTT timeline.")
    parser.add_argument("--target-vtt", required=True, help="Target-timed VTT, usually the new TV-cut Japanese Whisper VTT.")
    parser.add_argument("--reference-en", required=True, help="Reference English VTT from the nearby cut.")
    parser.add_argument("--target-bridge-vtt", default="", help="Optional target-language bridge VTT for local text anchors, usually TV Whisper.")
    parser.add_argument("--reference-bridge-vtt", default="", help="Optional reference-language bridge VTT from the same cut as --reference-en, usually YT Whisper.")
    parser.add_argument("--output", required=True, help="Output VTT path.")
    parser.add_argument("--report", default="", help="Optional JSON report path. Defaults to <output>.report.json")
    parser.add_argument(
        "--retime-reference",
        action="store_true",
        help="Retain the reference English cue count/text and only retime it onto the target cut. Requires both bridge VTTs.",
    )
    args = parser.parse_args()

    run = start_run("adapt_translation_by_timewarp")
    target_path = Path(args.target_vtt)
    reference_path = Path(args.reference_en)
    requested_output_path = Path(args.output)
    output_path = requested_output_path
    if output_path.parent.name == "translation" and not output_path.name.startswith(f"{run['run_id']}_"):
        output_path = lineage_output_path(output_path.parent, artifact_type="en", run=run, suffix=output_path.suffix or ".vtt")
    report_path = Path(args.report) if args.report else output_path.with_name(f"{output_path.stem}.report.json")

    target_cues = _load_vtt(target_path)
    reference_cues = _load_vtt(reference_path)
    target_bridge_path = Path(args.target_bridge_vtt) if args.target_bridge_vtt else None
    reference_bridge_path = Path(args.reference_bridge_vtt) if args.reference_bridge_vtt else None
    target_bridge_cues = _load_vtt(target_bridge_path) if target_bridge_path else None
    reference_bridge_cues = _load_vtt(reference_bridge_path) if reference_bridge_path else None
    if not target_cues:
        raise SystemExit(f"No cues found in target VTT: {target_path}")
    if not reference_cues:
        raise SystemExit(f"No cues found in reference VTT: {reference_path}")

    if args.retime_reference:
        if not (target_bridge_cues and reference_bridge_cues):
            raise SystemExit("--retime-reference requires --target-bridge-vtt and --reference-bridge-vtt")
        adapted, report = retime_reference(reference_cues, target_bridge_cues, reference_bridge_cues)
        note_extra = [
            f"reference_en: {reference_path.name}",
            f"reference_bridge: {reference_bridge_path.name}" if reference_bridge_path else "reference_bridge: ",
            "mode: retime_reference",
        ]
        step_name = "translation_adaptation_retime"
        stats = {
            "reference_cues": len(reference_cues),
            "anchor_count": report["anchor_count"],
            "retimed_end": report["retimed_end"],
        }
        settings = {"timewarp_anchor_mode": True, "retime_reference": True}
    else:
        adapted, report = adapt_draft(
            target_cues,
            reference_cues,
            bridge_target_cues=target_bridge_cues,
            bridge_reference_cues=reference_bridge_cues,
        )
        note_extra = [
            f"reference_en: {reference_path.name}",
            f"reference_bridge: {reference_bridge_path.name}" if reference_bridge_path else "reference_bridge: ",
            "mode: adapt_to_target_cues",
        ]
        step_name = "translation_adaptation_seed"
        stats = {
            "target_cues": len(target_cues),
            "reference_cues": len(reference_cues),
            "fallback_target_cues": report["fallback_target_cues"],
            "multi_assigned_target_cues": report["multi_assigned_target_cues"],
            "anchor_count": report["anchor_count"],
        }
        settings = {
            "timewarp_anchor_mode": bool(target_bridge_cues and reference_bridge_cues),
            "retime_reference": False,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    note_lines = build_vtt_note_lines(
        {
            "run_id": run["run_id"],
            "step": step_name,
            "run_started_at": run["run_started_at"],
        },
        source_name=target_path.name,
        extra_lines=note_extra,
    )
    output_path.write_text(serialize_vtt(adapted, note_lines=note_lines), encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    metadata = finish_run(
        run,
        inputs={"target_vtt": str(target_path), "reference_en": str(reference_path)},
        outputs={"draft_vtt": str(output_path), "report_json": str(report_path)},
        stats=stats,
        settings=settings,
    )
    write_metadata(output_path, metadata)
    if args.retime_reference and output_path.parent.name == "translation":
        update_preferred_manifest(output_path.parent, en_draft=output_path.name)

    print(f"Wrote draft: {output_path}")
    print(f"Report: {report_path}")
    print(f"Metadata written: {metadata_path(output_path)}")
    if args.retime_reference:
        print(
            "Summary: "
            f"anchors={report['anchor_count']} "
            f"retimed_end={report['retimed_end']}"
        )
    else:
        print(
            "Summary: "
            f"anchors={report['anchor_count']} "
            f"multi_assign={report['multi_assigned_target_cues']} "
            f"fallback={report['fallback_target_cues']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
