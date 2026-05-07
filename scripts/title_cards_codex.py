#!/usr/bin/env python3
"""Codex-interactive helper for adding title-card cues to a translated VTT.

Cross-references TransNetV2 shot brackets with Flash Lite chunk OCR. For every
shot that looks card-shaped (right duration, no dialogue overlap), it offers
Codex the chunk's OCR menu so Codex can decide whether the shot is a card
worth captioning, pick the best matching JA text, and supply terse EN wording.

Subcommands:
    prepare      Build candidate list, write session checkpoint
    next-card    Return next pending candidate as JSON
    apply-card   Record Codex's decision for a candidate
    status       Print progress summary
    finalize     Write *_en_with_cards.vtt and update preferred.json

Usage:
    python3.12 scripts/title_cards_codex.py prepare \\
        --en-vtt samples/episodes/<slug>/translation/<run>_en.vtt \\
        --reflow samples/episodes/<slug>/transcription/<run>_reflow.vtt \\
        --shots samples/episodes/<slug>/shots/<stem>.shots.json \\
        --chunk-ocr samples/episodes/<slug>/ocr/<stem>_flash_lite_chunk_ocr.json

    python3.12 scripts/title_cards_codex.py next-card --session <session>
    python3.12 scripts/title_cards_codex.py apply-card --session <session> --decision-json /tmp/d.json
    python3.12 scripts/title_cards_codex.py finalize --session <session>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chigyusubs.metadata import (
    finish_run,
    inherit_run_id,
    start_run,
    update_preferred_manifest,
    write_metadata,
)
from chigyusubs.title_cards import (
    find_candidates,
    inject_cards,
    load_chunk_ocr,
    load_shots,
)
from chigyusubs.translation import (
    Cue,
    parse_vtt,
    serialize_vtt,
    write_json_atomic,
)


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------


def _load_session(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_session(path: Path, session: dict) -> None:
    write_json_atomic(path, session)


def _read_vtt(path: Path) -> list[Cue]:
    return parse_vtt(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_prepare(args: argparse.Namespace) -> None:
    en_vtt_path = Path(args.en_vtt)
    reflow_path = Path(args.reflow)
    shots_path = Path(args.shots)
    chunk_ocr_path = Path(args.chunk_ocr)

    en_cues = _read_vtt(en_vtt_path)
    reflow_cues = _read_vtt(reflow_path)
    shots = load_shots(shots_path)
    chunks = load_chunk_ocr(chunk_ocr_path)

    candidates = find_candidates(
        shots=shots,
        chunk_ocr=chunks,
        reflow_cues=reflow_cues,
        en_cues=en_cues,
        min_dur=args.min_dur,
        max_dur=args.max_dur,
        dialog_overlap_max=args.dialog_overlap_max,
    )

    if args.output:
        output_path = Path(args.output)
    else:
        stem = en_vtt_path.stem
        if stem.endswith("_en"):
            output_stem = stem[:-3] + "_en_with_cards"
        else:
            output_stem = stem + "_with_cards"
        output_path = en_vtt_path.with_name(f"{output_stem}.vtt")

    session_path = Path(str(output_path) + ".session.json")

    kind_counts: dict[str, int] = {}
    for c in candidates:
        for entry in c.ocr_menu:
            kind_counts[entry["kind_guess"]] = kind_counts.get(entry["kind_guess"], 0) + 1

    session = {
        "en_vtt": str(en_vtt_path),
        "reflow_vtt": str(reflow_path),
        "shots": str(shots_path),
        "chunk_ocr": str(chunk_ocr_path),
        "output": str(output_path),
        "settings": {
            "min_dur": args.min_dur,
            "max_dur": args.max_dur,
            "dialog_overlap_max": args.dialog_overlap_max,
        },
        "candidates": [
            {**c.to_dict(), "status": "pending", "decision": None}
            for c in candidates
        ],
        "stats": {
            "candidate_count": len(candidates),
            "menu_entries_total": sum(len(c.ocr_menu) for c in candidates),
            "kind_counts": kind_counts,
        },
    }
    _save_session(session_path, session)

    print(f"Session: {session_path}")
    print(f"Candidates: {len(candidates)}")
    print(f"Total menu entries: {session['stats']['menu_entries_total']}")
    print(f"Menu kind counts: {kind_counts}")
    print(f"Output will be: {output_path}")


def cmd_next_card(args: argparse.Namespace) -> None:
    session_path = Path(args.session)
    session = _load_session(session_path)

    pending = [c for c in session["candidates"] if c["status"] == "pending"]
    if not pending:
        print(json.dumps({"done": True, "message": "All candidates reviewed."}))
        return

    cand = pending[0]
    payload = {
        "done": False,
        "remaining": len(pending) - 1,
        **{k: v for k, v in cand.items() if k not in ("status", "decision")},
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_apply_card(args: argparse.Namespace) -> None:
    session_path = Path(args.session)
    session = _load_session(session_path)

    decision = json.loads(Path(args.decision_json).read_text(encoding="utf-8"))
    cand_id = decision.get("candidate_id")
    if cand_id is None:
        print("Error: decision JSON must include 'candidate_id'", file=sys.stderr)
        sys.exit(1)

    target = None
    for c in session["candidates"]:
        if c["id"] == cand_id:
            target = c
            break
    if target is None:
        print(f"Error: candidate {cand_id} not found", file=sys.stderr)
        sys.exit(1)
    if target["status"] != "pending":
        print(f"Error: candidate {cand_id} status is '{target['status']}'", file=sys.stderr)
        sys.exit(1)

    action = decision.get("action")
    if action not in ("include", "skip"):
        print("Error: action must be 'include' or 'skip'", file=sys.stderr)
        sys.exit(1)

    if action == "include":
        text = str(decision.get("text", "")).strip()
        if not text:
            print("Error: include decision must supply non-empty 'text'", file=sys.stderr)
            sys.exit(1)
        target["decision"] = {
            "action": "include",
            "text": text,
            "picked_ja": decision.get("picked_ja", ""),
            "reason": decision.get("reason", ""),
        }
    else:
        target["decision"] = {
            "action": "skip",
            "reason": decision.get("reason", ""),
        }
    target["status"] = "completed"

    _save_session(session_path, session)

    completed = sum(1 for c in session["candidates"] if c["status"] == "completed")
    pending = sum(1 for c in session["candidates"] if c["status"] == "pending")
    print(f"Candidate {cand_id}: {action}")
    print(f"Progress: {completed} completed, {pending} pending")


def cmd_status(args: argparse.Namespace) -> None:
    session_path = Path(args.session)
    session = _load_session(session_path)

    completed = [c for c in session["candidates"] if c["status"] == "completed"]
    pending = [c for c in session["candidates"] if c["status"] == "pending"]
    included = [c for c in completed if c["decision"] and c["decision"]["action"] == "include"]
    skipped = [c for c in completed if c["decision"] and c["decision"]["action"] == "skip"]

    print(f"Session: {args.session}")
    print(f"Candidates: {len(session['candidates'])}")
    print(f"Completed: {len(completed)}  (included {len(included)} / skipped {len(skipped)})")
    print(f"Pending: {len(pending)}")

    if completed:
        print("\nLast 5 decisions:")
        for c in completed[-5:]:
            d = c["decision"]
            shot = f"{c['shot_start']:.2f}-{c['shot_end']:.2f}s"
            if d["action"] == "include":
                print(f"  [{c['id']}] {shot} INCLUDE  {d['text']!r}")
            else:
                reason = d.get("reason") or "(no reason)"
                print(f"  [{c['id']}] {shot} SKIP     {reason}")
    if pending:
        nxt = pending[0]
        print(f"\nNext: candidate {nxt['id']} at {nxt['shot_start']:.2f}-{nxt['shot_end']:.2f}s")


def cmd_finalize(args: argparse.Namespace) -> None:
    session_path = Path(args.session)
    session = _load_session(session_path)

    pending = [c for c in session["candidates"] if c["status"] == "pending"]
    if pending:
        print(
            f"Warning: {len(pending)} candidates still pending — they will be skipped.",
            file=sys.stderr,
        )

    en_vtt_path = Path(session["en_vtt"])
    output_path = Path(session["output"])

    en_cues = _read_vtt(en_vtt_path)

    decisions: list[dict] = []
    for c in session["candidates"]:
        d = c.get("decision")
        if not d:
            continue
        if d["action"] != "include":
            continue
        decisions.append(
            {
                "candidate_id": c["id"],
                "action": "include",
                "shot_start": c["shot_start"],
                "shot_end": c["shot_end"],
                "text": d["text"],
            }
        )

    merged_cues, audit = inject_cards(en_cues, decisions)

    # Sanity assertions before any disk write — bail out if the splice misbehaves.
    _assert_sanity(merged_cues, en_cues, audit)

    note_lines = [
        f"source_en: {en_vtt_path.name}",
        f"cards_added: {sum(1 for a in audit if a['action'] == 'included')}",
        f"cards_dropped: {sum(1 for a in audit if a['action'] != 'included')}",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(serialize_vtt(merged_cues, note_lines=note_lines), encoding="utf-8")
    tmp_path.replace(output_path)

    run = start_run("title_cards")
    run = inherit_run_id(run, en_vtt_path)
    stats = {
        "candidates_total": len(session["candidates"]),
        "candidates_included": sum(1 for a in audit if a["action"] == "included"),
        "candidates_skipped": sum(
            1 for c in session["candidates"]
            if c.get("decision") and c["decision"]["action"] == "skip"
        ),
        "candidates_dropped_no_room": sum(1 for a in audit if a["action"] == "dropped_no_room"),
        "candidates_pending": len(pending),
        "en_cue_count_before": len(en_cues),
        "en_cue_count_after": len(merged_cues),
    }
    run = finish_run(
        run,
        inputs={
            "en_vtt": str(en_vtt_path),
            "reflow": session.get("reflow_vtt"),
            "shots": session.get("shots"),
            "chunk_ocr": session.get("chunk_ocr"),
        },
        outputs={"en_with_cards": str(output_path)},
        settings=session.get("settings"),
        stats=stats,
        audit=audit,
    )
    write_metadata(output_path, run)

    if output_path.parent.name == "translation":
        update_preferred_manifest(output_path.parent, en_with_cards=output_path.name)
        print(f"Updated preferred.json: en_with_cards = {output_path.name}")

    print(f"Wrote {output_path}")
    for k, v in stats.items():
        print(f"  {k}: {v}")


def _assert_sanity(merged: list[Cue], original: list[Cue], audit: list[dict]) -> None:
    sorted_merged = sorted(merged, key=lambda c: c.start)
    for prev, cur in zip(sorted_merged, sorted_merged[1:]):
        if cur.start < prev.end - 1e-3:
            raise AssertionError(
                f"Overlap after splice: cue ending {prev.end:.3f} overlaps cue starting {cur.start:.3f}"
            )
    for entry in audit:
        if entry["action"] != "included":
            continue
        if entry["end"] - entry["start"] < 0.5:
            raise AssertionError(f"Added card cue under 0.5 s after splice: {entry}")
    added = sum(1 for a in audit if a["action"] == "included")
    if added > 80:
        raise AssertionError(f"Sanity ceiling tripped: {added} added cues (> 80)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Codex-interactive title-card augmentation for translated VTT.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser("prepare", help="Build candidate list and write session")
    p_prep.add_argument("--en-vtt", required=True, help="Translated EN VTT (input)")
    p_prep.add_argument("--reflow", required=True, help="JA reflow VTT")
    p_prep.add_argument("--shots", required=True, help="*.shots.json from detect_shots.py")
    p_prep.add_argument("--chunk-ocr", required=True, help="*_flash_lite_chunk_ocr.json")
    p_prep.add_argument("--output", default="", help="Output VTT path (default: <run>_en_with_cards.vtt)")
    p_prep.add_argument("--min-dur", type=float, default=0.6)
    p_prep.add_argument("--max-dur", type=float, default=8.0)
    p_prep.add_argument("--dialog-overlap-max", type=float, default=0.25)

    p_next = sub.add_parser("next-card", help="Get next pending candidate")
    p_next.add_argument("--session", required=True)

    p_apply = sub.add_parser("apply-card", help="Apply a decision for a candidate")
    p_apply.add_argument("--session", required=True)
    p_apply.add_argument("--decision-json", required=True)

    p_status = sub.add_parser("status", help="Print progress summary")
    p_status.add_argument("--session", required=True)

    p_final = sub.add_parser("finalize", help="Write augmented VTT and update preferred.json")
    p_final.add_argument("--session", required=True)

    args = parser.parse_args()

    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "next-card":
        cmd_next_card(args)
    elif args.command == "apply-card":
        cmd_apply_card(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "finalize":
        cmd_finalize(args)


if __name__ == "__main__":
    main()
