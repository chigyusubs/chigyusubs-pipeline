#!/usr/bin/env python3
"""Codex-interactive CPS compression pass.

Finds cues exceeding the hard CPS limit in a translated VTT and presents them
to Codex for compression via a prepare/next-cue/apply-cue/finalize workflow.

Usage:
  # Prepare a compression session
  python scripts/compress_cps_overruns.py prepare \
    --input subs_en.vtt

  # Get the next overrun cue for Codex to compress
  python scripts/compress_cps_overruns.py next-cue --session <path>

  # Apply Codex's compressed text
  python scripts/compress_cps_overruns.py apply-cue --session <path> \
    --repair-json <path>

  # Write final output
  python scripts/compress_cps_overruns.py finalize --session <path>
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chigyusubs.translation import (
    Cue,
    checkpoint_path,
    cue_duration,
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

DEFAULT_HARD_CPS = 20.0
DEFAULT_MAX_LINE_LENGTH = 42
CONTEXT_CUES = 2


def _load_cues(path: Path) -> list[Cue]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".srt":
        return parse_srt(raw)
    return parse_vtt(raw)


def _find_overruns(cues: list[Cue], hard_cps: float) -> list[int]:
    """Return 1-based cue IDs exceeding hard CPS."""
    overruns = []
    for i, cue in enumerate(cues):
        dur = cue_duration(cue)
        cps = text_cps(cue.text, dur)
        if cps > hard_cps:
            overruns.append(i + 1)
    return overruns


def _cue_payload(cue_id: int, cue: Cue, hard_cps: float) -> dict:
    dur = cue_duration(cue)
    max_chars = int(dur * hard_cps)
    return {
        "cue_id": cue_id,
        "start": cue.start,
        "end": cue.end,
        "start_tc": seconds_to_time(cue.start),
        "end_tc": seconds_to_time(cue.end),
        "duration": round(dur, 3),
        "text": cue.text,
        "chars": text_char_count(cue.text),
        "cps": round(text_cps(cue.text, dur), 1),
        "max_chars": max_chars,
        "hard_cps": hard_cps,
    }


def _output_path(input_path: Path) -> Path:
    stem = input_path.stem
    if not stem.endswith("_compressed"):
        stem = f"{stem}_compressed"
    return input_path.with_name(f"{stem}{input_path.suffix or '.vtt'}")


def _partial_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.partial{output_path.suffix or '.vtt'}")


def _diagnostics_path(output_path: Path) -> Path:
    return Path(f"{output_path}.diagnostics.json")


def _load_session(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_session(path: Path, session: dict) -> None:
    write_json_atomic(path, session)


def _render_partial(session: dict, cues: list[Cue]) -> None:
    applied = {int(k): v for k, v in session.get("applied_cues", {}).items()}
    max_ll = int(session["max_line_length"])
    partial: list[Cue] = []
    for cue_id, cue in enumerate(cues, 1):
        if cue_id in applied:
            text = wrap_english_text(applied[cue_id], max_line_length=max_ll)
            partial.append(Cue(cue.start, cue.end, text))
        else:
            partial.append(cue)
    out_path = Path(session["partial_output"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    is_srt = out_path.suffix.lower() == ".srt"
    out_path.write_text(
        serialize_srt(partial) if is_srt else serialize_vtt(partial),
        encoding="utf-8",
    )


def _write_diagnostics(session: dict, cues: list[Cue]) -> None:
    applied = {int(k): v for k, v in session.get("applied_cues", {}).items()}
    hard_cps = float(session["hard_cps"])
    max_ll = int(session["max_line_length"])

    # Build current cue list with applied compressions
    current: list[Cue] = []
    for cue_id, cue in enumerate(cues, 1):
        if cue_id in applied:
            text = wrap_english_text(applied[cue_id], max_line_length=max_ll)
            current.append(Cue(cue.start, cue.end, text))
        else:
            current.append(cue)

    remaining = _find_overruns(current, hard_cps)
    diag = {
        "mode": "codex-cps-compression",
        "status": session["status"],
        "input": session["input"],
        "output": session["output"],
        "hard_cps": hard_cps,
        "original_overruns": len(session["overrun_cue_ids"]),
        "completed": len(session.get("completed_cue_ids", [])),
        "remaining_overruns": len(remaining),
        "remaining_cue_ids": remaining,
        "cue_reports": list(session.get("cue_reports", {}).values()),
    }
    path = Path(session["diagnostics_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(diag, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def cmd_prepare(args) -> int:
    input_path = Path(args.input)
    cues = _load_cues(input_path)
    if not cues:
        print("No cues found.", file=sys.stderr)
        return 1

    hard_cps = args.hard_cps
    overruns = _find_overruns(cues, hard_cps)
    if not overruns:
        print(f"No cues exceed {hard_cps} CPS. Nothing to compress.")
        return 0

    output_path = Path(args.output) if args.output else _output_path(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    session_path = Path(args.session) if args.session else Path(checkpoint_path(str(output_path)))

    if session_path.exists() and not args.force:
        print(f"Session exists: {session_path}", file=sys.stderr)
        return 1

    session = {
        "version": 1,
        "mode": "codex-cps-compression",
        "status": "ready",
        "input": str(input_path),
        "output": str(output_path),
        "partial_output": str(_partial_output_path(output_path)),
        "diagnostics_path": str(_diagnostics_path(output_path)),
        "hard_cps": hard_cps,
        "max_line_length": args.max_line_length,
        "context_cues": CONTEXT_CUES,
        "total_cues": len(cues),
        "overrun_cue_ids": overruns,
        "completed_cue_ids": [],
        "applied_cues": {},
        "cue_reports": {},
    }
    _save_session(session_path, session)
    _render_partial(session, cues)
    _write_diagnostics(session, cues)

    print(f"Prepared CPS compression session: {session_path}")
    print(f"Found {len(overruns)}/{len(cues)} cues exceeding {hard_cps} CPS")
    for cue_id in overruns[:10]:
        cue = cues[cue_id - 1]
        dur = cue_duration(cue)
        cps = text_cps(cue.text, dur)
        print(f"  [{cue_id}] {cps:.1f} CPS ({dur:.2f}s): {cue.text[:60]}")
    if len(overruns) > 10:
        print(f"  ... and {len(overruns) - 10} more")
    return 0


def cmd_next_cue(args) -> int:
    session = _load_session(Path(args.session))
    cues = _load_cues(Path(session["input"]))
    completed = set(session.get("completed_cue_ids", []))
    pending = [cid for cid in session["overrun_cue_ids"] if cid not in completed]

    if not pending:
        print(json.dumps({"status": session["status"], "message": "no pending cues"}, indent=2))
        return 0

    cue_id = pending[0]
    cue = cues[cue_id - 1]
    hard_cps = float(session["hard_cps"])

    # Build context
    ctx = int(session.get("context_cues", CONTEXT_CUES))
    prev_start = max(0, cue_id - 1 - ctx)
    next_end = min(len(cues), cue_id + ctx)

    payload = {
        "pending_count": len(pending),
        "completed_count": len(completed),
        "total_overruns": len(session["overrun_cue_ids"]),
        "target_cue": _cue_payload(cue_id, cue, hard_cps),
        "previous_context": [
            {"cue_id": i + 1, "text": cues[i].text}
            for i in range(prev_start, cue_id - 1)
        ],
        "next_context": [
            {"cue_id": i + 1, "text": cues[i].text}
            for i in range(cue_id, next_end)
        ],
        "instruction": (
            f"Compress this cue to at most {_cue_payload(cue_id, cue, hard_cps)['max_chars']} characters "
            f"(currently {text_char_count(cue.text)} chars, {text_cps(cue.text, cue_duration(cue)):.1f} CPS). "
            "Keep natural subtitle English. Preserve punchlines and meaning."
        ),
    }

    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output_json:
        Path(args.output_json).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


def cmd_apply_cue(args) -> int:
    session_path = Path(args.session)
    session = _load_session(session_path)
    cues = _load_cues(Path(session["input"]))

    completed = set(session.get("completed_cue_ids", []))
    pending = [cid for cid in session["overrun_cue_ids"] if cid not in completed]
    if not pending:
        print("No pending cue.", file=sys.stderr)
        return 1

    payload = json.loads(Path(args.repair_json).read_text(encoding="utf-8"))
    cue_id = int(payload["cue_id"])
    if cue_id != pending[0]:
        raise ValueError(f"Expected cue_id {pending[0]}, got {cue_id}")

    new_text = str(payload["text"]).strip()
    review = str(payload.get("review", "green"))
    notes = str(payload.get("notes", ""))

    if not new_text:
        raise ValueError("Compressed text cannot be empty")

    cue = cues[cue_id - 1]
    dur = cue_duration(cue)
    hard_cps = float(session["hard_cps"])
    max_ll = int(session["max_line_length"])
    wrapped = wrap_english_text(new_text, max_line_length=max_ll)
    new_cps = text_cps(wrapped, dur)
    old_cps = text_cps(cue.text, dur)

    # Apply
    applied = dict(session.get("applied_cues", {}))
    applied[str(cue_id)] = wrapped
    completed.add(cue_id)

    session["applied_cues"] = applied
    session["completed_cue_ids"] = sorted(completed)
    session.setdefault("cue_reports", {})[str(cue_id)] = {
        "cue_id": cue_id,
        "review": review,
        "notes": notes,
        "old_text": cue.text,
        "new_text": wrapped,
        "old_cps": round(old_cps, 1),
        "new_cps": round(new_cps, 1),
        "improved": new_cps < old_cps,
        "under_limit": new_cps <= hard_cps,
    }

    remaining = [cid for cid in session["overrun_cue_ids"] if cid not in completed]
    if not remaining:
        session["status"] = "completed"

    if review == "red":
        session["status"] = "stopped"
        session["stop_reason"] = "cue review marked red"

    _save_session(session_path, session)
    _render_partial(session, cues)
    _write_diagnostics(session, cues)

    status_msg = "done" if not remaining else f"{len(remaining)} remaining"
    print(f"Applied cue {cue_id}: {old_cps:.1f} -> {new_cps:.1f} CPS, {status_msg}")
    return 0


def cmd_status(args) -> int:
    session = _load_session(Path(args.session))
    completed = set(session.get("completed_cue_ids", []))
    pending = [cid for cid in session["overrun_cue_ids"] if cid not in completed]
    payload = {
        "status": session["status"],
        "input": session["input"],
        "output": session["output"],
        "hard_cps": session["hard_cps"],
        "total_overruns": len(session["overrun_cue_ids"]),
        "completed": len(completed),
        "pending": len(pending),
        "next_cue_id": pending[0] if pending else None,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_finalize(args) -> int:
    session_path = Path(args.session)
    session = _load_session(session_path)
    cues = _load_cues(Path(session["input"]))
    applied = {int(k): v for k, v in session.get("applied_cues", {}).items()}
    max_ll = int(session["max_line_length"])

    final: list[Cue] = []
    for cue_id, cue in enumerate(cues, 1):
        if cue_id in applied:
            text = wrap_english_text(applied[cue_id], max_line_length=max_ll)
            final.append(Cue(cue.start, cue.end, text))
        else:
            final.append(cue)

    output_path = Path(session["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    is_srt = output_path.suffix.lower() == ".srt"
    output_path.write_text(
        serialize_srt(final) if is_srt else serialize_vtt(final),
        encoding="utf-8",
    )

    session["status"] = "completed"
    _save_session(session_path, session)
    _write_diagnostics(session, cues)

    hard_cps = float(session["hard_cps"])
    remaining = _find_overruns(final, hard_cps)
    reports = session.get("cue_reports", {})
    improved = sum(1 for r in reports.values() if r.get("under_limit"))
    print(f"Wrote {output_path}")
    print(f"  {improved}/{len(reports)} cues compressed under {hard_cps} CPS")
    if remaining:
        print(f"  {len(remaining)} cues still exceed limit")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Codex-interactive CPS compression workflow.")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Prepare a CPS compression session.")
    prepare.add_argument("--input", required=True, help="Translated VTT or SRT file.")
    prepare.add_argument("--output", default="", help="Output path (default: <input>_compressed.<ext>).")
    prepare.add_argument("--session", default="", help="Session JSON path.")
    prepare.add_argument("--hard-cps", type=float, default=DEFAULT_HARD_CPS)
    prepare.add_argument("--max-line-length", type=int, default=DEFAULT_MAX_LINE_LENGTH)
    prepare.add_argument("--force", action="store_true")
    prepare.set_defaults(func=cmd_prepare)

    next_cue = sub.add_parser("next-cue", help="Emit the next overrun cue for Codex compression.")
    next_cue.add_argument("--session", required=True)
    next_cue.add_argument("--output-json", default="")
    next_cue.set_defaults(func=cmd_next_cue)

    apply_cue = sub.add_parser("apply-cue", help="Apply compressed text for the current cue.")
    apply_cue.add_argument("--session", required=True)
    apply_cue.add_argument("--repair-json", required=True, help="JSON with cue_id, text, review, notes.")
    apply_cue.set_defaults(func=cmd_apply_cue)

    status = sub.add_parser("status", help="Show compression session progress.")
    status.add_argument("--session", required=True)
    status.set_defaults(func=cmd_status)

    finalize = sub.add_parser("finalize", help="Write the final compressed VTT/SRT.")
    finalize.add_argument("--session", required=True)
    finalize.set_defaults(func=cmd_finalize)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
