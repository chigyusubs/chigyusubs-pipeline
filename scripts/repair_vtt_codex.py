#!/usr/bin/env python3
"""Codex-interactive helper for reviewing and repairing reflowed VTTs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chigyusubs.alignment_diagnostics import (
    alignment_summary_payload,
    alignment_warnings_for_cue_ids,
    build_alignment_review,
    discover_alignment_diagnostics_path,
)
from chigyusubs.turn_context import (
    build_turn_review,
    discover_words_json_path,
    turn_context_for_cue_ids,
    turn_summary_payload,
)
from chigyusubs.reflow_repair import (
    RepairRegion,
    build_review,
    compact_region_text,
    cue_chars,
    cue_duration,
    detect_regions,
    render_repaired_cues,
    region_reason_counts,
    structural_preflight,
    synthesize_region_cues,
)
from chigyusubs.translation import Cue, checkpoint_path, parse_srt, parse_vtt, seconds_to_time, serialize_srt, serialize_vtt, write_json_atomic


def _load_cues(path: Path) -> list[Cue]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".srt":
        return parse_srt(raw)
    return parse_vtt(raw)


def _episode_repair_output(input_path: Path) -> Path:
    stem = input_path.stem
    if not stem.endswith("_repaired"):
        stem = f"{stem}_repaired"
    return input_path.with_name(f"{stem}{input_path.suffix or '.vtt'}")


def _partial_output_path(output_path: Path) -> Path:
    if output_path.suffix:
        return output_path.with_name(f"{output_path.stem}.partial{output_path.suffix}")
    return output_path.with_name(f"{output_path.name}.partial.vtt")


def _diagnostics_path(output_path: Path) -> Path:
    return Path(f"{output_path}.diagnostics.json")


def _load_session(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_session(path: Path, session: dict) -> None:
    write_json_atomic(path, session)


def _regions_payload(regions: list[RepairRegion]) -> list[dict]:
    return [
        {
            "region_id": region.region_id,
            "start_cue_id": region.start_cue_id,
            "end_cue_id": region.end_cue_id,
            "reasons": region.reasons,
        }
        for region in regions
    ]


def _session_applied(session: dict) -> dict[int, list[Cue]]:
    applied: dict[int, list[Cue]] = {}
    for region_id_str, cues in session.get("applied_regions", {}).items():
        applied[int(region_id_str)] = [
            Cue(float(item["start"]), float(item["end"]), str(item["text"]))
            for item in cues
        ]
    return applied


def _serialize_applied(applied: dict[int, list[Cue]]) -> dict[str, list[dict]]:
    payload: dict[str, list[dict]] = {}
    for region_id, cues in sorted(applied.items()):
        payload[str(region_id)] = [
            {"start": cue.start, "end": cue.end, "text": cue.text}
            for cue in cues
        ]
    return payload


def _pending_regions(session: dict) -> list[dict]:
    completed = {int(value) for value in session.get("completed_regions", [])}
    return [region for region in session["regions"] if int(region["region_id"]) not in completed]


def _render_current(session: dict, base_cues: list[Cue]) -> list[Cue]:
    regions = [
        RepairRegion(
            region_id=int(region["region_id"]),
            start_cue_id=int(region["start_cue_id"]),
            end_cue_id=int(region["end_cue_id"]),
            reasons=list(region.get("reasons", [])),
        )
        for region in session.get("regions", [])
    ]
    return render_repaired_cues(base_cues, regions, _session_applied(session))


def _write_partial(session: dict, base_cues: list[Cue]) -> None:
    partial_path = Path(session["partial_output"])
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    cues = _render_current(session, base_cues)
    if partial_path.suffix.lower() == ".srt":
        partial_path.write_text(serialize_srt(cues), encoding="utf-8")
    else:
        partial_path.write_text(serialize_vtt(cues), encoding="utf-8")


def _write_diagnostics(session: dict, base_cues: list[Cue]) -> None:
    current_cues = _render_current(session, base_cues)
    preflight = structural_preflight(current_cues)
    regions = detect_regions(current_cues, context_cues=int(session["region_context_cues"]))
    review = build_review(current_cues, preflight, regions)
    prepared_review = session["prepared_review"]
    prepared_metrics = prepared_review["metrics"]
    current_metrics = review["metrics"]
    region_reports = list(session.get("region_reports", {}).values())
    region_review_counts: dict[str, int] = {}
    for report in region_reports:
        key = str(report.get("review", "green"))
        region_review_counts[key] = region_review_counts.get(key, 0) + 1
    pending_regions = _pending_regions(session)
    translation_ready = (
        session.get("status") == "completed"
        and review["review"] == "green"
        and not pending_regions
    )
    recommended_translation_input = (
        str(Path(session["output"]))
        if session.get("status") == "completed" and review["review"] != "red" and not pending_regions
        else ""
    )
    diagnostics = {
        "mode": session["mode"],
        "status": session["status"],
        "stop_reason": session.get("stop_reason", ""),
        "input": session["input"],
        "output": session["output"],
        "partial_output": session["partial_output"],
        "words_path": session.get("words_path", ""),
        "alignment_review": alignment_summary_payload(session.get("alignment_review")),
        "turn_review": turn_summary_payload(session.get("turn_review")),
        "prepared_review": prepared_review,
        "current_review": review,
        "regions_total": len(session["regions"]),
        "completed_regions": len(session.get("completed_regions", [])),
        "pending_regions": len(pending_regions),
        "repair_summary": {
            "before": prepared_metrics,
            "after": current_metrics,
            "delta": {
                "short_cues_under_0_8s": current_metrics["short_cues_under_0_8s"] - prepared_metrics["short_cues_under_0_8s"],
                "short_cues_under_1_0s": current_metrics["short_cues_under_1_0s"] - prepared_metrics["short_cues_under_1_0s"],
                "tiny_cues_le_4_chars": current_metrics["tiny_cues_le_4_chars"] - prepared_metrics["tiny_cues_le_4_chars"],
                "detected_regions": current_metrics["detected_regions"] - prepared_metrics["detected_regions"],
            },
        },
        "region_reports_summary": {
            "review_counts": region_review_counts,
            "reason_counts": region_reason_counts(
                [
                    RepairRegion(
                        region_id=int(region["region_id"]),
                        start_cue_id=int(region["start_cue_id"]),
                        end_cue_id=int(region["end_cue_id"]),
                        reasons=list(region.get("reasons", [])),
                    )
                    for region in session.get("regions", [])
                ]
            ),
        },
        "recommended_translation_input": recommended_translation_input,
        "translation_ready": translation_ready,
        "remaining_translation_risks": review["reasons"] + _alignment_review_reasons(session.get("alignment_review")),
        "region_reports": region_reports,
    }
    path = Path(session["diagnostics_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _region_payload(region: dict, cues: list[Cue]) -> dict:
    start_cue_id = int(region["start_cue_id"])
    end_cue_id = int(region["end_cue_id"])
    region_cues = cues[start_cue_id - 1:end_cue_id]
    return {
        "region_id": int(region["region_id"]),
        "start_cue_id": start_cue_id,
        "end_cue_id": end_cue_id,
        "reasons": region.get("reasons", []),
        "compact_source_text": compact_region_text(region_cues),
        "original_cues": [
            {
                "cue_id": cue_id,
                "start": cue.start,
                "end": cue.end,
                "start_tc": seconds_to_time(cue.start),
                "end_tc": seconds_to_time(cue.end),
                "duration": round(cue_duration(cue), 3),
                "chars": cue_chars(cue),
                "text": cue.text,
            }
            for cue_id, cue in zip(range(start_cue_id, end_cue_id + 1), region_cues)
        ],
        "repair_constraints": {
            "preserve_all_source_text": True,
            "do_not_invent_text": True,
            "do_not_drop_text": True,
            "cue_order_must_remain_monotonic": True,
            "timings_within_region_are_rebuilt_deterministically": True,
        },
    }


def cmd_prepare(args) -> int:
    input_path = Path(args.input)
    cues = _load_cues(input_path)
    if not cues:
        print("No cues found in input file.", file=sys.stderr)
        return 1

    output_path = Path(args.output) if args.output else _episode_repair_output(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    session_path = Path(args.session) if args.session else Path(checkpoint_path(str(output_path)))
    if session_path.exists() and not args.force:
        print(f"Session already exists: {session_path}", file=sys.stderr)
        return 1

    preflight = structural_preflight(cues)
    regions = detect_regions(cues, context_cues=args.region_context_cues)
    review = build_review(cues, preflight, regions)
    words_path = discover_words_json_path(words_path=args.words, input_path=input_path)
    alignment_diagnostics = discover_alignment_diagnostics_path(words_path=words_path, input_path=input_path)
    alignment_review = build_alignment_review(cues, alignment_diagnostics) if alignment_diagnostics else None
    turn_review = build_turn_review(cues, words_path) if words_path else None
    status = "ready"
    stop_reason = ""
    if review["review"] == "red":
        status = "stopped"
        stop_reason = "preflight structural blocker"

    session = {
        "version": 1,
        "mode": "codex-reflow-repair",
        "status": status,
        "stop_reason": stop_reason,
        "input": str(input_path),
        "output": str(output_path),
        "partial_output": str(_partial_output_path(output_path)),
        "diagnostics_path": str(_diagnostics_path(output_path)),
        "words_path": str(words_path) if words_path else "",
        "alignment_diagnostics_path": str(alignment_diagnostics) if alignment_diagnostics else "",
        "alignment_review": alignment_review,
        "turn_review": turn_review,
        "region_context_cues": args.region_context_cues,
        "prepared_review": review,
        "regions": _regions_payload(regions),
        "completed_regions": [],
        "applied_regions": {},
        "region_reports": {},
    }
    _save_session(session_path, session)
    _write_partial(session, cues)
    _write_diagnostics(session, cues)
    print(f"Prepared session: {session_path}")
    print(f"Partial output: {session['partial_output']}")
    print(f"Prepared review: {review['review']}")
    if alignment_review:
        print(
            "Alignment advisory: "
            f"{alignment_review['repaired_line_count']} interpolated source lines across "
            f"{alignment_review['affected_cues_count']} cues"
        )
    if turn_review:
        print(
            "Turn advisory: "
            f"{turn_review['multi_turn_cues_count']} cues span multiple source turns"
        )
    if session["status"] == "stopped":
        print(f"Stopped: {session['stop_reason']}", file=sys.stderr)
        return 2
    return 0


def cmd_status(args) -> int:
    session = _load_session(Path(args.session))
    cues = _load_cues(Path(session["input"]))
    pending = _pending_regions(session)
    current = _render_current(session, cues)
    current_review = build_review(
        current,
        structural_preflight(current),
        detect_regions(current, context_cues=int(session["region_context_cues"])),
    )
    payload = {
        "status": session["status"],
        "stop_reason": session.get("stop_reason", ""),
        "input": session["input"],
        "output": session["output"],
        "partial_output": session["partial_output"],
        "prepared_review": session["prepared_review"]["review"],
        "prepared_metrics": session["prepared_review"]["metrics"],
        "current_review": current_review["review"],
        "current_metrics": current_review["metrics"],
        "alignment_review": alignment_summary_payload(session.get("alignment_review")),
        "turn_review": turn_summary_payload(session.get("turn_review")),
        "regions_total": len(session["regions"]),
        "completed_regions": len(session.get("completed_regions", [])),
        "pending_regions": len(pending),
        "next_region_id": None if not pending else pending[0]["region_id"],
        "next_region_range": None if not pending else [pending[0]["start_cue_id"], pending[0]["end_cue_id"]],
        "recommended_translation_input": (
            session["output"]
            if session.get("status") == "completed" and current_review["review"] != "red" and not pending
            else ""
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_next_region(args) -> int:
    session = _load_session(Path(args.session))
    pending = _pending_regions(session)
    if not pending:
        print(json.dumps({"status": session["status"], "message": "no pending region"}, ensure_ascii=False, indent=2))
        return 0
    region = pending[0]
    cues = _load_cues(Path(session["input"]))
    payload = {
        "status": session["status"],
        "prepared_review": session["prepared_review"],
        "region": _region_payload(region, cues),
        "alignment_warnings": alignment_warnings_for_cue_ids(
            session.get("alignment_review"),
            range(int(region["start_cue_id"]), int(region["end_cue_id"]) + 1),
        ),
        "turn_context": turn_context_for_cue_ids(
            session.get("turn_review"),
            range(int(region["start_cue_id"]), int(region["end_cue_id"]) + 1),
        ),
        "review_policy": {
            "allowed_reviews": ["green", "yellow", "red"],
            "green": "apply region and continue",
            "yellow": "apply region and continue with caution",
            "red": "apply region and stop session",
        },
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output_json:
        Path(args.output_json).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


def _load_replacement_texts(payload: dict) -> list[str]:
    if "replacement_texts" in payload:
        items = payload["replacement_texts"]
        if not isinstance(items, list):
            raise ValueError("replacement_texts must be a list")
        return [str(item) for item in items]
    if "replacement_cues" in payload:
        items = payload["replacement_cues"]
        if not isinstance(items, list):
            raise ValueError("replacement_cues must be a list")
        return [str(item["text"]) for item in items]
    raise ValueError("payload must contain replacement_texts or replacement_cues")


def cmd_apply_region(args) -> int:
    session_path = Path(args.session)
    session = _load_session(session_path)
    if session.get("status") in {"stopped", "completed"}:
        print(f"Session is {session['status']}: {session.get('stop_reason', '')}", file=sys.stderr)
        return 1

    pending = _pending_regions(session)
    if not pending:
        print("No pending region.", file=sys.stderr)
        return 1
    region = pending[0]
    payload = json.loads(Path(args.repair_json).read_text(encoding="utf-8"))
    region_id = int(payload.get("region_id", -1))
    if region_id != int(region["region_id"]):
        raise ValueError(f"Expected region_id {region['region_id']}, got {region_id}")

    review = str(payload.get("review", "green"))
    notes = str(payload.get("notes", ""))
    if review not in {"green", "yellow", "red"}:
        raise ValueError("review must be one of: green, yellow, red")

    replacement_texts = _load_replacement_texts(payload)
    cues = _load_cues(Path(session["input"]))
    source_region_cues = cues[int(region["start_cue_id"]) - 1:int(region["end_cue_id"])]
    rebuilt = synthesize_region_cues(source_region_cues, replacement_texts)

    applied = _session_applied(session)
    applied[region_id] = rebuilt
    completed = {int(value) for value in session.get("completed_regions", [])}
    completed.add(region_id)
    session["completed_regions"] = sorted(completed)
    session["applied_regions"] = _serialize_applied(applied)
    session.setdefault("region_reports", {})[str(region_id)] = {
        "region_id": region_id,
        "start_cue_id": int(region["start_cue_id"]),
        "end_cue_id": int(region["end_cue_id"]),
        "review": review,
        "notes": notes,
        "replacement_cue_count": len(rebuilt),
        "replacement_texts": replacement_texts,
    }

    if review == "red":
        session["status"] = "stopped"
        session["stop_reason"] = "region review marked red"
    elif not _pending_regions(session):
        session["status"] = "completed"
        session["stop_reason"] = ""

    _save_session(session_path, session)
    _write_partial(session, cues)
    _write_diagnostics(session, cues)
    print(
        f"Applied region {region_id}: cues {region['start_cue_id']}-{region['end_cue_id']}, "
        f"review={review}, status={session['status']}"
    )
    return 0


def cmd_finalize(args) -> int:
    session_path = Path(args.session)
    session = _load_session(session_path)
    cues = _load_cues(Path(session["input"]))
    pending = _pending_regions(session)
    if pending:
        raise ValueError("Cannot finalize with unresolved regions")
    final_cues = _render_current(session, cues)
    preflight = structural_preflight(final_cues)
    if preflight["negative_duration_cues"] or preflight["overlap_after_cues"]:
        raise ValueError("Cannot finalize a structurally invalid repaired VTT")

    output_path = Path(session["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".srt":
        output_path.write_text(serialize_srt(final_cues), encoding="utf-8")
    else:
        output_path.write_text(serialize_vtt(final_cues), encoding="utf-8")
    session["status"] = "completed"
    session["stop_reason"] = ""
    _save_session(session_path, session)
    _write_diagnostics(session, cues)
    print(f"Wrote final output: {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Codex-interactive reflow review and repair workflow.")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Prepare a Codex-interactive reflow repair session.")
    prepare.add_argument("--input", required=True, help="Input reflowed VTT or SRT file.")
    prepare.add_argument("--output", default="", help="Final repaired output path.")
    prepare.add_argument("--session", default="", help="Session/checkpoint JSON path. Defaults to <output>.checkpoint.json.")
    prepare.add_argument("--words", default="", help="Optional aligned words JSON path for reference.")
    prepare.add_argument("--region-context-cues", type=int, default=1, help="Context cues to include around flagged regions.")
    prepare.add_argument("--force", action="store_true", help="Overwrite an existing session.")
    prepare.set_defaults(func=cmd_prepare)

    status = sub.add_parser("status", help="Show current repair session status.")
    status.add_argument("--session", required=True)
    status.set_defaults(func=cmd_status)

    next_region = sub.add_parser("next-region", help="Emit the next repair region payload.")
    next_region.add_argument("--session", required=True)
    next_region.add_argument("--output-json", default="", help="Optional file to write the region payload.")
    next_region.set_defaults(func=cmd_next_region)

    apply_region = sub.add_parser("apply-region", help="Apply repaired cue text for the current region.")
    apply_region.add_argument("--session", required=True)
    apply_region.add_argument("--repair-json", required=True, help="JSON file with region_id, replacement_texts, review, and notes.")
    apply_region.set_defaults(func=cmd_apply_region)

    finalize = sub.add_parser("finalize", help="Write the final repaired VTT after all regions are handled.")
    finalize.add_argument("--session", required=True)
    finalize.set_defaults(func=cmd_finalize)

    return parser


def _alignment_review_reasons(alignment_review: dict | None) -> list[str]:
    if not alignment_review or not alignment_review.get("affected_cues_count"):
        return []
    return [
        "advisory: "
        f"{alignment_review['repaired_line_count']} source lines were locally interpolated during alignment "
        f"and overlap {alignment_review['affected_cues_count']} cues"
    ]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
