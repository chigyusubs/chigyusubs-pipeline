#!/usr/bin/env python3
"""Codex-interactive subtitle translation helper workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from chigyusubs.translation import (
    Cue,
    batch_cues,
    checkpoint_path,
    cue_diag,
    parse_srt,
    parse_vtt,
    seconds_to_time,
    serialize_srt,
    serialize_vtt,
    text_char_count,
    text_cps,
    wrap_english_text,
    write_json_atomic,
)

DEFAULT_BATCH_TIERS = [84, 60, 48]
DEFAULT_BATCH_SECONDS = 300.0
DEFAULT_CONTEXT_CUES = 2
DEFAULT_TARGET_CPS = 17.0
DEFAULT_HARD_CPS = 20.0
DEFAULT_MAX_LINE_LENGTH = 42


def _load_cues(input_path: Path) -> list[Cue]:
    raw = input_path.read_text(encoding="utf-8")
    if input_path.suffix.lower() == ".srt":
        return parse_srt(raw)
    return parse_vtt(raw)


def _episode_translation_output(input_path: Path, target_lang: str) -> Path:
    parent = input_path.parent
    if parent.name == "transcription":
        parent = parent.parent / "translation"
    stem = input_path.stem
    lang_slug = target_lang.lower().replace(" ", "_")
    return parent / f"{stem}_{lang_slug}_codex.vtt"


def _partial_output_path(output_path: Path) -> Path:
    if output_path.suffix:
        return output_path.with_name(f"{output_path.stem}.partial{output_path.suffix}")
    return output_path.with_name(f"{output_path.name}.partial.vtt")


def _diagnostics_path(output_path: Path) -> Path:
    return Path(f"{output_path}.diagnostics.json")


def _parse_tiers(raw: str) -> list[int]:
    tiers = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not tiers:
        raise ValueError("batch tiers cannot be empty")
    if sorted(tiers, reverse=True) != tiers:
        raise ValueError("batch tiers must be in descending order")
    if any(value <= 0 for value in tiers):
        raise ValueError("batch tiers must be positive")
    return tiers


def _preflight(cues: list[Cue]) -> dict:
    negative_duration = []
    overlaps = []
    for idx, cue in enumerate(cues, 1):
        if cue.end < cue.start:
            negative_duration.append(idx)
        if idx < len(cues) and cues[idx].start < cue.end:
            overlaps.append(idx)
    return {
        "negative_duration_cues": negative_duration,
        "overlap_after_cues": overlaps,
    }


def _load_text_blob(path_value: str) -> str:
    if not path_value:
        return ""
    return Path(path_value).read_text(encoding="utf-8").strip()


def _initial_session(args, input_path: Path, output_path: Path, cues: list[Cue]) -> dict:
    preflight = _preflight(cues)
    status = "ready"
    stop_reason = ""
    if preflight["negative_duration_cues"] or preflight["overlap_after_cues"]:
        status = "stopped"
        stop_reason = "preflight structural blocker"

    return {
        "version": 1,
        "mode": "codex-interactive",
        "status": status,
        "stop_reason": stop_reason,
        "input": str(input_path),
        "output": str(output_path),
        "partial_output": str(_partial_output_path(output_path)),
        "diagnostics_path": str(_diagnostics_path(output_path)),
        "target_language": args.target_lang,
        "glossary_path": args.glossary or "",
        "summary_path": args.summary or "",
        "preferred_model": args.preferred_model,
        "preferred_thinking": args.preferred_thinking,
        "preferred_temperature": args.preferred_temperature,
        "batch_tiers": args.batch_tiers,
        "current_batch_tier": args.batch_tiers[0],
        "batch_seconds": args.batch_seconds,
        "context_cues": args.context_cues,
        "target_cps": args.target_cps,
        "hard_cps": args.hard_cps,
        "max_line_length": args.max_line_length,
        "total_cues": len(cues),
        "translations": {},
        "completed_batches": [],
        "batch_diagnostics": {},
        "preflight": preflight,
    }


def _load_session(path_value: Path) -> dict:
    return json.loads(path_value.read_text(encoding="utf-8"))


def _save_session(path_value: Path, session: dict) -> None:
    write_json_atomic(path_value, session)


def _sorted_completed(session: dict) -> set[int]:
    return {int(idx) for idx in session.get("completed_batches", [])}


def _translated_map(session: dict) -> dict[int, str]:
    return {int(k): v for k, v in session.get("translations", {}).items()}


def _next_batch_from_session(session: dict, cues: list[Cue]) -> dict | None:
    translations = _translated_map(session)
    next_cue_id = 1
    while next_cue_id <= len(cues) and translations.get(next_cue_id, "").strip():
        next_cue_id += 1
    if next_cue_id > len(cues):
        return None

    start_index = next_cue_id - 1
    remaining = cues[start_index:]
    batches = batch_cues(
        remaining,
        max_cues=int(session["current_batch_tier"]),
        max_seconds=float(session["batch_seconds"]),
        context_cues=int(session["context_cues"]),
    )
    batch = batches[0]
    abs_start = start_index + batch.start_index
    abs_end = start_index + batch.end_index
    cue_ids = list(range(abs_start + 1, abs_end + 1))
    prev_start = max(0, abs_start - int(session["context_cues"]))
    next_end = min(len(cues), abs_end + int(session["context_cues"]))
    return {
        "batch_index": len(session.get("completed_batches", [])),
        "start_index": abs_start,
        "end_index": abs_end,
        "cue_ids": cue_ids,
        "cues": cues[abs_start:abs_end],
        "prev_context": cues[prev_start:abs_start],
        "next_context": cues[abs_end:next_end],
    }


def _cue_payload(cue_id: int, cue: Cue) -> dict:
    duration = max(0.001, cue.end - cue.start)
    return {
        "cue_id": cue_id,
        "start": cue.start,
        "end": cue.end,
        "start_tc": seconds_to_time(cue.start),
        "end_tc": seconds_to_time(cue.end),
        "text": cue.text,
        "duration": round(duration, 3),
        "source_cps": round(text_cps(cue.text, duration), 3),
    }


def _render_partial(session: dict, cues: list[Cue]) -> None:
    translations = _translated_map(session)
    partial_cues: list[Cue] = []
    max_line_length = int(session["max_line_length"])
    for cue_id, cue in enumerate(cues, 1):
        text = translations.get(cue_id, "")
        if text:
            final_text = wrap_english_text(text, max_line_length=max_line_length)
        else:
            final_text = cue.text
        partial_cues.append(Cue(cue.start, cue.end, final_text))

    output_path = Path(session["partial_output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".srt":
        output_path.write_text(serialize_srt(partial_cues), encoding="utf-8")
    else:
        output_path.write_text(serialize_vtt(partial_cues), encoding="utf-8")


def _render_final(session: dict, cues: list[Cue]) -> list[Cue]:
    translations = _translated_map(session)
    missing = [cue_id for cue_id in range(1, len(cues) + 1) if not translations.get(cue_id, "").strip()]
    if missing:
        raise ValueError(
            "Cannot finalize with missing translated cues: "
            + ", ".join(map(str, missing[:20]))
            + ("..." if len(missing) > 20 else "")
        )
    final_cues = [
        Cue(cue.start, cue.end, wrap_english_text(translations[idx], int(session["max_line_length"])))
        for idx, cue in enumerate(cues, 1)
    ]
    return final_cues


def _session_diagnostics(session: dict, cues: list[Cue]) -> dict:
    translations = _translated_map(session)
    completed = len(translations)
    return {
        "mode": session["mode"],
        "status": session["status"],
        "stop_reason": session.get("stop_reason", ""),
        "input": session["input"],
        "output": session["output"],
        "partial_output": session["partial_output"],
        "target_language": session["target_language"],
        "preferred_model": session["preferred_model"],
        "preferred_thinking": session["preferred_thinking"],
        "preferred_temperature": session["preferred_temperature"],
        "current_batch_tier": session["current_batch_tier"],
        "batch_tiers": session["batch_tiers"],
        "completed_cues": completed,
        "total_cues": len(cues),
        "completed_batches": len(session.get("completed_batches", [])),
        "batch_diagnostics": list(session.get("batch_diagnostics", {}).values()),
        "preflight": session.get("preflight", {}),
    }


def _write_diagnostics(session: dict, cues: list[Cue]) -> None:
    diagnostics_path = Path(session["diagnostics_path"])
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(
        json.dumps(_session_diagnostics(session, cues), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _review_rank(review: str) -> int:
    return {"green": 0, "yellow": 1, "red": 2}[review]


def _merge_review(user_review: str, *, hard_cps_violations: int, line_violations: int, structural_error: bool, at_min_tier: bool) -> tuple[str, str]:
    if structural_error:
        return "red", "structural validation failed"
    objective = "green"
    if hard_cps_violations or line_violations:
        objective = "yellow"
    final = user_review if _review_rank(user_review) >= _review_rank(objective) else objective
    if final == "yellow" and at_min_tier and (hard_cps_violations >= 3 or line_violations >= 1):
        return "red", "quality threshold failed at minimum tier"
    if final == "yellow" and objective == "yellow":
        return "yellow", "objective subtitle constraints exceeded"
    return final, ""


def cmd_prepare(args) -> int:
    input_path = Path(args.input)
    cues = _load_cues(input_path)
    if not cues:
        print("No cues found in input file.", file=sys.stderr)
        return 1

    output_path = Path(args.output) if args.output else _episode_translation_output(input_path, args.target_lang)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    session_path = Path(args.session) if args.session else Path(checkpoint_path(str(output_path)))
    if session_path.exists() and not args.force:
        print(f"Session already exists: {session_path}", file=sys.stderr)
        return 1

    session = _initial_session(args, input_path, output_path, cues)
    _save_session(session_path, session)
    _render_partial(session, cues)
    _write_diagnostics(session, cues)
    print(f"Prepared session: {session_path}")
    print(f"Partial output: {session['partial_output']}")
    if session["status"] == "stopped":
        print(f"Stopped: {session['stop_reason']}", file=sys.stderr)
        return 2
    return 0


def cmd_next_batch(args) -> int:
    session_path = Path(args.session)
    session = _load_session(session_path)
    cues = _load_cues(Path(session["input"]))
    batch = _next_batch_from_session(session, cues)
    if batch is None:
        print(json.dumps({"status": session["status"], "message": "no pending batch"}, ensure_ascii=False, indent=2))
        return 0
    payload = {
        "batch_index": batch["batch_index"],
        "current_batch_tier": session["current_batch_tier"],
        "target_cues": [
            _cue_payload(cue_id, cue)
            for cue_id, cue in zip(batch["cue_ids"], batch["cues"])
        ],
        "previous_context": [
            _cue_payload(batch["start_index"] - len(batch["prev_context"]) + idx + 1, cue)
            for idx, cue in enumerate(batch["prev_context"])
        ],
        "next_context": [
            _cue_payload(batch["end_index"] + idx + 1, cue)
            for idx, cue in enumerate(batch["next_context"])
        ],
        "glossary": _load_text_blob(session.get("glossary_path", "")),
        "summary": _load_text_blob(session.get("summary_path", "")),
        "review_policy": {
            "allowed_reviews": ["green", "yellow", "red"],
            "green": "keep current tier",
            "yellow": "apply batch and reduce next tier",
            "red": "apply batch and stop episode",
        },
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output_json:
        Path(args.output_json).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


def cmd_apply_batch(args) -> int:
    session_path = Path(args.session)
    session = _load_session(session_path)
    cues = _load_cues(Path(session["input"]))
    if session.get("status") in {"stopped", "completed"}:
        print(f"Session is {session['status']}: {session.get('stop_reason', '')}", file=sys.stderr)
        return 1
    batch = _next_batch_from_session(session, cues)
    if batch is None:
        print("No pending batch.", file=sys.stderr)
        return 1

    payload = json.loads(Path(args.translations_json).read_text(encoding="utf-8"))
    items = payload.get("translations", [])
    user_review = payload.get("review", "green")
    notes = payload.get("notes", "")
    if user_review not in {"green", "yellow", "red"}:
        raise ValueError("review must be one of: green, yellow, red")
    if not isinstance(items, list):
        raise ValueError("translations must be a list")

    expected_ids = batch["cue_ids"]
    seen: dict[int, str] = {}
    for item in items:
        cue_id = int(item["cue_id"])
        text = str(item["text"])
        seen[cue_id] = text
    structural_error = sorted(seen.keys()) != expected_ids
    if structural_error:
        raise ValueError(f"Translated cue IDs do not match current batch: expected {expected_ids}, got {sorted(seen.keys())}")

    translations = _translated_map(session)
    translated_batch: list[Cue] = []
    hard_cps_violations = 0
    line_violations = 0
    cue_diags: list[dict] = []
    for cue_id, source in zip(expected_ids, batch["cues"]):
        text = wrap_english_text(seen[cue_id], max_line_length=int(session["max_line_length"]))
        if not text.strip():
            raise ValueError(f"Empty translation for cue {cue_id}")
        translated = Cue(source.start, source.end, text)
        diag = cue_diag(cue_id, source, translated, False)
        hard_cps_violations += int(diag["translated_cps"] > float(session["hard_cps"]))
        line_violations += int(diag["line_count"] > 2)
        cue_diags.append(diag)
        translated_batch.append(translated)
        translations[cue_id] = text

    at_min_tier = int(session["current_batch_tier"]) == int(session["batch_tiers"][-1])
    final_review, auto_reason = _merge_review(
        user_review,
        hard_cps_violations=hard_cps_violations,
        line_violations=line_violations,
        structural_error=False,
        at_min_tier=at_min_tier,
    )

    batch_index = batch["batch_index"]
    session["translations"] = {str(k): v for k, v in sorted(translations.items())}
    completed = _sorted_completed(session)
    completed.add(batch_index)
    session["completed_batches"] = sorted(completed)
    session.setdefault("batch_diagnostics", {})[str(batch_index)] = {
        "batch_index": batch_index,
        "start_cue_id": expected_ids[0],
        "end_cue_id": expected_ids[-1],
        "cue_count": len(expected_ids),
        "review": final_review,
        "user_review": user_review,
        "auto_reason": auto_reason,
        "notes": notes,
        "hard_cps_violations": hard_cps_violations,
        "line_violations": line_violations,
        "cues": cue_diags,
    }

    if final_review == "yellow":
        tiers = session["batch_tiers"]
        current = int(session["current_batch_tier"])
        for tier in tiers:
            if tier < current:
                session["current_batch_tier"] = tier
                break
    elif final_review == "red":
        session["status"] = "stopped"
        session["stop_reason"] = auto_reason or "batch review marked red"

    if len(translations) == len(cues):
        session["status"] = "completed"
        session["stop_reason"] = ""

    _save_session(session_path, session)
    _render_partial(session, cues)
    _write_diagnostics(session, cues)
    print(
        f"Applied batch {batch_index}: cues {expected_ids[0]}-{expected_ids[-1]}, "
        f"review={final_review}, next_tier={session['current_batch_tier']}, status={session['status']}"
    )
    return 0


def cmd_status(args) -> int:
    session = _load_session(Path(args.session))
    cues = _load_cues(Path(session["input"]))
    translations = _translated_map(session)
    pending_batch = _next_batch_from_session(session, cues)
    payload = {
        "status": session["status"],
        "stop_reason": session.get("stop_reason", ""),
        "input": session["input"],
        "output": session["output"],
        "partial_output": session["partial_output"],
        "completed_cues": len(translations),
        "total_cues": len(cues),
        "completed_batches": len(session.get("completed_batches", [])),
        "current_batch_tier": session["current_batch_tier"],
        "next_batch_index": None if pending_batch is None else pending_batch["batch_index"],
        "next_batch_range": None if pending_batch is None else [pending_batch["cue_ids"][0], pending_batch["cue_ids"][-1]],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_finalize(args) -> int:
    session_path = Path(args.session)
    session = _load_session(session_path)
    cues = _load_cues(Path(session["input"]))
    if session.get("status") not in {"completed", "ready"} and _next_batch_from_session(session, cues) is not None:
        raise ValueError("Cannot finalize an incomplete session")
    final_cues = _render_final(session, cues)
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
    parser = argparse.ArgumentParser(description="Codex-interactive subtitle translation workflow.")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Prepare a Codex-interactive translation session.")
    prepare.add_argument("--input", required=True, help="Input VTT or SRT file.")
    prepare.add_argument("--output", default="", help="Final output path in translation/.")
    prepare.add_argument("--session", default="", help="Session/checkpoint JSON path. Defaults to <output>.checkpoint.json.")
    prepare.add_argument("--target-lang", default="English")
    prepare.add_argument("--glossary", default="", help="Optional glossary text file.")
    prepare.add_argument("--summary", default="", help="Optional summary text file.")
    prepare.add_argument("--preferred-model", default="gpt-5.4")
    prepare.add_argument("--preferred-thinking", default="medium")
    prepare.add_argument("--preferred-temperature", type=float, default=0.2)
    prepare.add_argument("--batch-tiers", default="84,60,48", type=_parse_tiers)
    prepare.add_argument("--batch-seconds", type=float, default=DEFAULT_BATCH_SECONDS)
    prepare.add_argument("--context-cues", type=int, default=DEFAULT_CONTEXT_CUES)
    prepare.add_argument("--target-cps", type=float, default=DEFAULT_TARGET_CPS)
    prepare.add_argument("--hard-cps", type=float, default=DEFAULT_HARD_CPS)
    prepare.add_argument("--max-line-length", type=int, default=DEFAULT_MAX_LINE_LENGTH)
    prepare.add_argument("--force", action="store_true", help="Overwrite an existing session.")
    prepare.set_defaults(func=cmd_prepare)

    next_batch = sub.add_parser("next-batch", help="Emit the next batch payload for Codex translation.")
    next_batch.add_argument("--session", required=True)
    next_batch.add_argument("--output-json", default="", help="Optional file to write the batch payload.")
    next_batch.set_defaults(func=cmd_next_batch)

    apply_batch = sub.add_parser("apply-batch", help="Apply translated text for the current batch.")
    apply_batch.add_argument("--session", required=True)
    apply_batch.add_argument("--translations-json", required=True, help="JSON file with translations + review.")
    apply_batch.set_defaults(func=cmd_apply_batch)

    status = sub.add_parser("status", help="Show session progress.")
    status.add_argument("--session", required=True)
    status.set_defaults(func=cmd_status)

    finalize = sub.add_parser("finalize", help="Write the final VTT/SRT when translation is complete.")
    finalize.add_argument("--session", required=True)
    finalize.set_defaults(func=cmd_finalize)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
