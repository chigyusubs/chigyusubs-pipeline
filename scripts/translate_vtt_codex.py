#!/usr/bin/env python3
"""Codex-interactive subtitle translation helper workflow."""

from __future__ import annotations

import argparse
import hashlib
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
from chigyusubs.speaker_context import (
    build_speaker_review,
    discover_named_speaker_map_path,
    speaker_context_for_cue_ids,
    speaker_summary_payload,
)
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
from chigyusubs.metadata import (
    build_vtt_note_lines,
    finish_run,
    inherit_run_id,
    lineage_output_path,
    metadata_path,
    start_run,
    update_preferred_manifest,
    write_metadata,
)

DEFAULT_BATCH_TIERS = [84, 60, 48]
DEFAULT_BATCH_SECONDS = 300.0
DEFAULT_CONTEXT_CUES = 2
DEFAULT_TARGET_CPS = 17.0
DEFAULT_HARD_CPS = 20.0
DEFAULT_MAX_LINE_LENGTH = 42
MIN_CUE_DURATION_S = 0.5


def _load_cues(input_path: Path) -> list[Cue]:
    raw = input_path.read_text(encoding="utf-8")
    if input_path.suffix.lower() == ".srt":
        return parse_srt(raw)
    return parse_vtt(raw)


def _cue_timing_signature(cue: Cue) -> tuple[int, int]:
    return (round(cue.start * 1000), round(cue.end * 1000))


def _source_text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _validate_seed_timeline(source_cues: list[Cue], seed_cues: list[Cue], seed_path: Path) -> None:
    if len(seed_cues) != len(source_cues):
        raise ValueError(
            f"Seed draft cue count mismatch for {seed_path}: "
            f"source has {len(source_cues)} cues, seed has {len(seed_cues)}"
        )
    for cue_id, (source, seed) in enumerate(zip(source_cues, seed_cues), 1):
        if _cue_timing_signature(source) != _cue_timing_signature(seed):
            raise ValueError(
                "Seed draft cue timing mismatch for "
                f"{seed_path} at cue {cue_id}: "
                f"source={seconds_to_time(source.start)} --> {seconds_to_time(source.end)} "
                f"seed={seconds_to_time(seed.start)} --> {seconds_to_time(seed.end)}"
            )


def _seed_translations_from_path(source_cues: list[Cue], seed_path: Path) -> dict[int, str]:
    seed_cues = _load_cues(seed_path)
    _validate_seed_timeline(source_cues, seed_cues, seed_path)
    return {
        cue_id: cue.text
        for cue_id, cue in enumerate(seed_cues, 1)
        if cue.text.strip()
    }


def _episode_translation_output(input_path: Path, target_lang: str) -> Path:
    parent = input_path.parent
    if parent.name == "transcription":
        parent = parent.parent / "translation"
    lang_slug = target_lang.lower().replace(" ", "_")
    probe_run = inherit_run_id(start_run("translate_vtt_codex"), input_path)
    artifact_type = "en" if lang_slug == "english" else lang_slug
    if parent.name == "translation":
        return lineage_output_path(parent, artifact_type=artifact_type, run=probe_run, suffix=".vtt")
    return parent / f"{input_path.stem}_{lang_slug}_codex.vtt"


def _partial_output_path(output_path: Path) -> Path:
    if output_path.suffix:
        return output_path.with_name(f"{output_path.stem}.partial{output_path.suffix}")
    return output_path.with_name(f"{output_path.name}.partial.vtt")


def _diagnostics_path(output_path: Path) -> Path:
    return Path(f"{output_path}.diagnostics.json")


def _discover_visual_cues_path(input_path: Path) -> str:
    parent = input_path.parent
    if parent.name in {"transcription", "translation"}:
        episode_dir = parent.parent
    else:
        episode_dir = parent
    ocr_dir = episode_dir / "ocr"
    if not ocr_dir.exists():
        return ""
    candidates = sorted(
        ocr_dir.glob("*_flash_lite_chunk_ocr.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return ""
    return str(candidates[0])


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
    micro_cues = []
    for idx, cue in enumerate(cues, 1):
        if max(0.0, cue.end - cue.start) < MIN_CUE_DURATION_S:
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


def _load_visual_cues_for_batch(visual_cues_path: str, batch_start: float, batch_end: float) -> list[dict]:
    if not visual_cues_path:
        return []
    data = json.loads(Path(visual_cues_path).read_text(encoding="utf-8"))
    if isinstance(data, list) and data and isinstance(data[0], dict) and "items" in data[0]:
        visual_cues: list[dict] = []
        for chunk in data:
            chunk_start = float(chunk["chunk_start_s"])
            chunk_end = float(chunk["chunk_end_s"])
            if chunk_end <= batch_start or chunk_start >= batch_end:
                continue
            for item in chunk.get("items", []):
                kind = str(item.get("kind_guess", "other"))
                importance = str(item.get("importance", "medium"))
                if kind not in {"title_card", "name_card", "info_card", "rule_text"} and not (
                    kind == "label" and importance in {"high", "medium"}
                ):
                    continue
                text = str(item.get("text", "")).strip()
                if not text:
                    continue
                visual_cues.append(
                    {
                        "chunk_start_s": chunk_start,
                        "chunk_end_s": chunk_end,
                        "text": text,
                        "kind_guess": kind,
                        "importance": importance,
                    }
                )
        return visual_cues
    return [
        {"chunk_start_s": vc["chunk_start_s"], "chunk_end_s": vc["chunk_end_s"], "text": vc["text"]}
        for vc in data
        if vc["chunk_end_s"] > batch_start and vc["chunk_start_s"] < batch_end
    ]


def _load_text_blob(path_value: str) -> str:
    if not path_value:
        return ""
    return Path(path_value).read_text(encoding="utf-8").strip()


def _initial_session(args, input_path: Path, output_path: Path, cues: list[Cue]) -> dict:
    preflight = _preflight(cues)
    visual_cues_path = args.visual_cues or _discover_visual_cues_path(input_path)
    words_path = discover_words_json_path(input_path=input_path)
    alignment_diagnostics = discover_alignment_diagnostics_path(
        explicit_path=args.alignment_diagnostics,
        input_path=input_path,
    )
    alignment_review = build_alignment_review(cues, alignment_diagnostics) if alignment_diagnostics else None
    turn_review = build_turn_review(cues, words_path) if words_path else None
    named_speaker_map_path = discover_named_speaker_map_path(
        explicit_path=args.speaker_map,
        input_path=input_path,
    )
    speaker_review = build_speaker_review(cues, named_speaker_map_path) if named_speaker_map_path else None
    status = "ready"
    stop_reason = ""
    if preflight["negative_duration_cues"] or preflight["overlap_after_cues"]:
        status = "stopped"
        stop_reason = "preflight structural blocker"
    elif preflight["micro_duration_cues_under_0_5s"]:
        status = "stopped"
        stop_reason = "preflight cue under 0.5s"

    return {
        "version": 1,
        "mode": "codex-interactive",
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
        "speaker_review": speaker_review,
        "speaker_map_path": str(named_speaker_map_path) if named_speaker_map_path else "",
        "target_language": args.target_lang,
        "glossary_path": args.glossary or "",
        "summary_path": args.summary or "",
        "visual_cues_path": visual_cues_path,
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
        "seed_from": args.seed_from or "",
        "seeded_cues": 0,
    }


def _load_session(path_value: Path) -> dict:
    return json.loads(path_value.read_text(encoding="utf-8"))


def _save_session(path_value: Path, session: dict) -> None:
    write_json_atomic(path_value, session)


def _remove_if_exists(path_value: Path) -> None:
    if path_value.exists():
        path_value.unlink()


def _cleanup_prepare_outputs(session_path: Path, output_path: Path) -> None:
    _remove_if_exists(session_path)
    _remove_if_exists(output_path)
    _remove_if_exists(_partial_output_path(output_path))
    _remove_if_exists(_diagnostics_path(output_path))


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


def _cue_payload(cue_id: int, cue: Cue, turn_review: dict | None = None) -> dict:
    duration = max(0.001, cue.end - cue.start)
    payload = {
        "cue_id": cue_id,
        "start": cue.start,
        "end": cue.end,
        "start_tc": seconds_to_time(cue.start),
        "end_tc": seconds_to_time(cue.end),
        "text": cue.text,
        "source_text_hash": _source_text_hash(cue.text),
        "duration": round(duration, 3),
        "source_cps": round(text_cps(cue.text, duration), 3),
    }
    if turn_review:
        cue_turn = turn_review.get("cue_turns", {}).get(str(cue_id))
        if cue_turn:
            payload["turn_index"] = cue_turn["turn_indices"][0] if cue_turn["turn_indices"] else None
            payload["starts_new_turn"] = cue_turn["starts_new_turn"]
    return payload


def _validate_translation_source_signature(item: dict, cue_id: int, source: Cue) -> None:
    expected_text = source.text
    expected_hash = _source_text_hash(expected_text)
    provided_text = item.get("source_text")
    provided_hash = item.get("source_text_hash")

    if provided_text is None and provided_hash is None:
        raise ValueError(
            f"Translation for cue {cue_id} is missing source_text/source_text_hash. "
            "Copy source_text_hash from next-batch target_cues into each translation item."
        )

    if provided_text is not None and str(provided_text) != expected_text:
        raise ValueError(
            f"Translation source_text mismatch for cue {cue_id}: "
            f"expected {expected_text!r}, got {str(provided_text)!r}"
        )

    if provided_hash is not None and str(provided_hash) != expected_hash:
        raise ValueError(
            f"Translation source_text_hash mismatch for cue {cue_id}: "
            f"expected {expected_hash}, got {str(provided_hash)}"
        )


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
    batch_diags = sorted(
        session.get("batch_diagnostics", {}).values(),
        key=lambda item: int(item.get("batch_index", 0)),
    )
    batch_review_counts: dict[str, int] = {}
    hard_cps_total = 0
    line_violations_total = 0
    structural_red_batches = 0
    warning_batches = 0
    alignment_warning_batches = 0
    alignment_warning_cues_total = 0
    alignment_warning_lines_total = 0
    for batch in batch_diags:
        review = str(batch.get("review", "green"))
        batch_review_counts[review] = batch_review_counts.get(review, 0) + 1
        hard_cps_total += int(batch.get("hard_cps_violations", 0))
        line_violations_total += int(batch.get("line_violations", 0))
        warning_batches += int(review == "yellow")
        alignment_warning_batches += int(int(batch.get("alignment_warning_cues", 0)) > 0)
        alignment_warning_cues_total += int(batch.get("alignment_warning_cues", 0))
        alignment_warning_lines_total += int(batch.get("alignment_warning_lines", 0))
        structural_red_batches += int(
            review == "red"
            and str(batch.get("auto_reason", "")).startswith("structural")
        )
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
        "seed_from": session.get("seed_from", ""),
        "seeded_cues": int(session.get("seeded_cues", 0)),
        "current_batch_tier": session["current_batch_tier"],
        "batch_tiers": session["batch_tiers"],
        "alignment_warning_summary": alignment_summary_payload(session.get("alignment_review")),
        "turn_context_summary": turn_summary_payload(session.get("turn_review")),
        "speaker_context_summary": speaker_summary_payload(session.get("speaker_review")),
        "completed_cues": completed,
        "total_cues": len(cues),
        "completed_batches": len(session.get("completed_batches", [])),
        "batch_summary": {
            "review_counts": batch_review_counts,
            "warning_batches": warning_batches,
            "cps_only_yellow_batches": sum(
                1 for batch in batch_diags if batch.get("yellow_class") == "cps_only"
            ),
            "structural_yellow_batches": sum(
                1 for batch in batch_diags if batch.get("yellow_class") == "structural"
            ),
            "structural_red_batches": structural_red_batches,
            "alignment_warning_batches": alignment_warning_batches,
            "alignment_warning_cues_total": alignment_warning_cues_total,
            "alignment_warning_lines_total": alignment_warning_lines_total,
            "multi_turn_batches": sum(
                1 for batch in batch_diags if int(batch.get("multi_turn_cues", 0)) > 0
            ),
            "multi_turn_cues_total": sum(int(batch.get("multi_turn_cues", 0)) for batch in batch_diags),
            "hard_cps_violations_total": hard_cps_total,
            "line_violations_total": line_violations_total,
        },
        "batch_diagnostics": batch_diags,
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


def _merge_review(
    user_review: str,
    *,
    hard_cps_violations: int,
    line_violations: int,
    structural_error: bool,
) -> tuple[str, str, str | None, list[str]]:
    """Merge user review with objective constraint checks.

    Returns (review, auto_reason, yellow_class, yellow_reasons).

    yellow_class is one of:
      - "cps_only"    — only hard CPS violations, no line/structural issues (no tier downgrade)
      - "structural"  — line-count violations, possibly with CPS too (triggers tier downgrade)
      - None          — not yellow
    """
    if structural_error:
        return "red", "structural validation failed", None, []

    yellow_reasons: list[str] = []
    if hard_cps_violations:
        yellow_reasons.append("hard_cps")
    if line_violations:
        yellow_reasons.append("line_count")

    objective = "yellow" if yellow_reasons else "green"
    final = user_review if _review_rank(user_review) >= _review_rank(objective) else objective

    if final != "yellow":
        return final, "", None, []

    yellow_class = "structural" if line_violations else "cps_only"
    auto_reason = "objective subtitle constraints exceeded" if objective == "yellow" else ""
    return final, auto_reason, yellow_class, yellow_reasons


def cmd_prepare(args) -> int:
    input_path = Path(args.input)
    cues = _load_cues(input_path)
    if not cues:
        print("No cues found in input file.", file=sys.stderr)
        return 1

    output_path = Path(args.output) if args.output else _episode_translation_output(input_path, args.target_lang)
    probe_run = inherit_run_id(start_run("translate_vtt_codex"), input_path)
    if output_path.parent.name == "translation" and not output_path.name.startswith(f"{probe_run['run_id']}_"):
        lang_slug = args.target_lang.lower().replace(" ", "_")
        artifact_type = "en" if lang_slug == "english" else lang_slug
        output_path = lineage_output_path(output_path.parent, artifact_type=artifact_type, run=probe_run, suffix=output_path.suffix or ".vtt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    session_path = Path(args.session) if args.session else Path(checkpoint_path(str(output_path)))
    seed_path = Path(args.seed_from) if args.seed_from else None
    if seed_path and args.force:
        reserved = {
            output_path.resolve(),
            _partial_output_path(output_path).resolve(),
            _diagnostics_path(output_path).resolve(),
            session_path.resolve(),
        }
        if seed_path.resolve() in reserved:
            print(
                "--seed-from cannot point at the session/output files being reset by prepare --force.",
                file=sys.stderr,
            )
            return 1
    if session_path.exists() and not args.force:
        print(f"Session already exists: {session_path}", file=sys.stderr)
        return 1
    if args.force:
        _cleanup_prepare_outputs(session_path, output_path)

    session = _initial_session(args, input_path, output_path, cues)
    try:
        if seed_path:
            seeded = _seed_translations_from_path(cues, seed_path)
            session["translations"] = {str(k): v for k, v in sorted(seeded.items())}
            session["seeded_cues"] = len(seeded)
            session["seed_from"] = str(seed_path)
            if session["status"] == "ready" and len(seeded) == len(cues):
                session["status"] = "completed"
                session["stop_reason"] = ""
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    _save_session(session_path, session)
    _render_partial(session, cues)
    _write_diagnostics(session, cues)
    print(f"Prepared session: {session_path}")
    print(f"Partial output: {session['partial_output']}")
    if session.get("visual_cues_path"):
        print(f"Visual cues: {session['visual_cues_path']}")
    if session.get("alignment_review"):
        print(
            "Alignment advisory: "
            f"{session['alignment_review']['repaired_line_count']} interpolated source lines across "
            f"{session['alignment_review']['affected_cues_count']} cues"
        )
    if session.get("turn_review"):
        print(
            "Turn advisory: "
            f"{session['turn_review']['multi_turn_cues_count']} cues span multiple source turns"
        )
    if session.get("speaker_review"):
        print(
            "Speaker advisory: "
            f"{session['speaker_review']['effective_speaker_count']} speakers identified, "
            f"{session['speaker_review']['cue_speaker_count']} cues with speaker context"
        )
    if seed_path:
        print(f"Seeded from: {seed_path} ({session['seeded_cues']} cues)")
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
    batch_start = batch["cues"][0].start
    batch_end = batch["cues"][-1].end
    visual_cues = _load_visual_cues_for_batch(session.get("visual_cues_path", ""), batch_start, batch_end)
    target_alignment = alignment_warnings_for_cue_ids(session.get("alignment_review"), batch["cue_ids"])
    target_turn_context = turn_context_for_cue_ids(session.get("turn_review"), batch["cue_ids"])
    target_speaker_context = speaker_context_for_cue_ids(session.get("speaker_review"), batch["cue_ids"])
    context_ids = [
        batch["start_index"] - len(batch["prev_context"]) + idx + 1
        for idx, _ in enumerate(batch["prev_context"])
    ] + [
        batch["end_index"] + idx + 1
        for idx, _ in enumerate(batch["next_context"])
    ]
    context_alignment = alignment_warnings_for_cue_ids(session.get("alignment_review"), context_ids)
    context_turn_context = turn_context_for_cue_ids(session.get("turn_review"), context_ids)
    context_speaker_context = speaker_context_for_cue_ids(session.get("speaker_review"), context_ids)
    full_turn_review = session.get("turn_review")
    payload = {
        "batch_index": batch["batch_index"],
        "current_batch_tier": session["current_batch_tier"],
        "target_cues": [
            _cue_payload(cue_id, cue, full_turn_review)
            for cue_id, cue in zip(batch["cue_ids"], batch["cues"])
        ],
        "previous_context": [
            _cue_payload(batch["start_index"] - len(batch["prev_context"]) + idx + 1, cue, full_turn_review)
            for idx, cue in enumerate(batch["prev_context"])
        ],
        "next_context": [
            _cue_payload(batch["end_index"] + idx + 1, cue, full_turn_review)
            for idx, cue in enumerate(batch["next_context"])
        ],
        "visual_cues": visual_cues,
        "glossary": _load_text_blob(session.get("glossary_path", "")),
        "summary": _load_text_blob(session.get("summary_path", "")),
        "alignment_warnings": _batch_alignment_payload(target_alignment, context_alignment),
        "turn_context": _batch_turn_payload(target_turn_context, context_turn_context),
        "speaker_context": _batch_speaker_payload(target_speaker_context, context_speaker_context),
        "review_policy": {
            "allowed_reviews": ["green", "yellow", "red"],
            "green": "apply batch and continue",
            "yellow": "apply batch, warn, and continue",
            "red": "apply batch and stop episode",
        },
        "turn_policy": _build_turn_policy(target_turn_context),
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
    raw_items: dict[int, dict] = {}
    for item in items:
        cue_id = int(item["cue_id"])
        text = str(item["text"])
        seen[cue_id] = text
        raw_items[cue_id] = item
    structural_error = sorted(seen.keys()) != expected_ids
    if structural_error:
        raise ValueError(f"Translated cue IDs do not match current batch: expected {expected_ids}, got {sorted(seen.keys())}")

    translations = _translated_map(session)
    translated_batch: list[Cue] = []
    hard_cps_violations = 0
    line_violations = 0
    cue_diags: list[dict] = []
    for cue_id, source in zip(expected_ids, batch["cues"]):
        _validate_translation_source_signature(raw_items[cue_id], cue_id, source)
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

    final_review, auto_reason, yellow_class, yellow_reasons = _merge_review(
        user_review,
        hard_cps_violations=hard_cps_violations,
        line_violations=line_violations,
        structural_error=False,
    )

    batch_index = batch["batch_index"]
    batch_alignment = alignment_warnings_for_cue_ids(session.get("alignment_review"), expected_ids)
    batch_turn_context = turn_context_for_cue_ids(session.get("turn_review"), expected_ids)
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
        "yellow_class": yellow_class,
        "yellow_reasons": yellow_reasons,
        "notes": notes,
        "hard_cps_violations": hard_cps_violations,
        "line_violations": line_violations,
        "alignment_warning_cues": 0 if not batch_alignment else batch_alignment["affected_cues_count"],
        "alignment_warning_lines": 0 if not batch_alignment else batch_alignment["repaired_line_count"],
        "alignment_warning_cue_ids": [] if not batch_alignment else batch_alignment["affected_cue_ids"],
        "alignment_warning_lines_sample": [] if not batch_alignment else batch_alignment["repaired_lines"],
        "multi_turn_cues": 0 if not batch_turn_context else batch_turn_context["multi_turn_cues_count"],
        "multi_turn_cue_ids": [] if not batch_turn_context else batch_turn_context["multi_turn_cue_ids"],
        "multi_turn_cues_sample": [] if not batch_turn_context else batch_turn_context["sample_multi_turn_cues"],
        "cues": cue_diags,
    }

    if final_review == "yellow" and yellow_class != "cps_only":
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
    yellow_tag = f" ({yellow_class})" if yellow_class else ""
    print(
        f"Applied batch {batch_index}: cues {expected_ids[0]}-{expected_ids[-1]}, "
        f"review={final_review}{yellow_tag}, next_tier={session['current_batch_tier']}, status={session['status']}"
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
        "alignment_warning_summary": alignment_summary_payload(session.get("alignment_review")),
        "turn_context_summary": turn_summary_payload(session.get("turn_review")),
        "speaker_context_summary": speaker_summary_payload(session.get("speaker_review")),
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
    run = inherit_run_id(start_run("translate_vtt_codex_finalize"), session["input"])
    source_name = Path(session["input"]).name
    note_lines = build_vtt_note_lines(
        {
            "run_id": run.get("run_id"),
            "step": "translation",
            "episode": Path(session["input"]).parent.parent.name if Path(session["input"]).parent.name == "transcription" else "",
            "run_started_at": run.get("run_started_at"),
        },
        source_name=source_name,
        extra_lines=[f"target_language: {session['target_language']}"],
    )
    if output_path.suffix.lower() == ".srt":
        output_path.write_text(serialize_srt(final_cues), encoding="utf-8")
    else:
        output_path.write_text(serialize_vtt(final_cues, note_lines=note_lines), encoding="utf-8")
    session["status"] = "completed"
    session["stop_reason"] = ""
    _save_session(session_path, session)
    _write_diagnostics(session, cues)
    metadata = finish_run(
        run,
        inputs={"japanese_vtt": session["input"], "session_json": str(session_path)},
        outputs={"english_vtt": str(output_path), "diagnostics_json": session["diagnostics_path"]},
        settings={
            "target_language": session["target_language"],
            "preferred_model": session["preferred_model"],
            "preferred_thinking": session["preferred_thinking"],
            "preferred_temperature": session["preferred_temperature"],
            "batch_tiers": session["batch_tiers"],
            "visual_cues_path": session.get("visual_cues_path", ""),
        },
        stats={
            "total_cues": len(cues),
            "completed_batches": len(session.get("completed_batches", [])),
        },
    )
    write_metadata(output_path, metadata)
    if output_path.parent.name == "translation":
        update_preferred_manifest(output_path.parent, en_draft=output_path.name)
    print(f"Wrote final output: {output_path}")
    print(f"Metadata written: {metadata_path(output_path)}")
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
    prepare.add_argument(
        "--visual-cues",
        default="",
        help="Optional visual cues JSON file. Defaults to latest sibling *_flash_lite_chunk_ocr.json if present.",
    )
    prepare.add_argument(
        "--alignment-diagnostics",
        default="",
        help="Optional alignment diagnostics sidecar. If omitted, prepare auto-discovers it from the input VTT.",
    )
    prepare.add_argument(
        "--speaker-map",
        default="",
        help="Optional named speaker map JSON. If omitted, prepare auto-discovers it from the input VTT.",
    )
    prepare.add_argument("--preferred-model", default="gpt-5.4")
    prepare.add_argument("--preferred-thinking", default="medium")
    prepare.add_argument("--preferred-temperature", type=float, default=0.2)
    prepare.add_argument("--batch-tiers", default="84,60,48", type=_parse_tiers)
    prepare.add_argument("--batch-seconds", type=float, default=DEFAULT_BATCH_SECONDS)
    prepare.add_argument("--context-cues", type=int, default=DEFAULT_CONTEXT_CUES)
    prepare.add_argument("--target-cps", type=float, default=DEFAULT_TARGET_CPS)
    prepare.add_argument("--hard-cps", type=float, default=DEFAULT_HARD_CPS)
    prepare.add_argument("--max-line-length", type=int, default=DEFAULT_MAX_LINE_LENGTH)
    prepare.add_argument(
        "--seed-from",
        default="",
        help="Optional existing translated VTT/SRT to seed from. Requires exact cue count and cue timings.",
    )
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


def _batch_alignment_payload(target_alignment: dict | None, context_alignment: dict | None) -> dict | None:
    if not target_alignment and not context_alignment:
        return None
    return {
        "advisory_only": True,
        "target": target_alignment,
        "context": context_alignment,
    }


def _build_turn_policy(target_turn_context: dict | None) -> dict | None:
    if not target_turn_context:
        return None
    multi_turn = target_turn_context.get("multi_turn_cues_count", 0)
    affected = target_turn_context.get("affected_cues_count", 0)
    has_rapid_exchange = affected >= 3 and any(
        entry.get("starts_new_turn", False)
        for entry in target_turn_context.get("cue_turns", {}).values()
    )
    if not has_rapid_exchange and multi_turn == 0:
        return None
    return {
        "instructions": (
            "This batch contains speaker turn boundaries. "
            "Cues with starts_new_turn=true mark where a different person starts speaking. "
            "Use the alternation pattern to track pronouns: in a back-and-forth between two people, "
            "odd turns and even turns are different speakers. If speaker A says something about/to "
            "speaker B, speaker B's reply should use first-person ('I') not third-person ('he'). "
            "Avoid defaulting to third-person pronouns when the dialogue is clearly addressed between "
            "the speakers present. "
            "If you cannot confidently determine who is addressing whom and it affects "
            "pronoun choice, set review to 'yellow' and add a note explaining the ambiguity."
        ),
        "has_rapid_exchange": has_rapid_exchange,
        "multi_turn_cue_count": multi_turn,
    }


def _batch_turn_payload(target_turn_context: dict | None, context_turn_context: dict | None) -> dict | None:
    if not target_turn_context and not context_turn_context:
        return None
    return {
        "advisory_only": True,
        "target": target_turn_context,
        "context": context_turn_context,
    }


def _batch_speaker_payload(target_speaker_context: dict | None, context_speaker_context: dict | None) -> dict | None:
    if not target_speaker_context and not context_speaker_context:
        return None
    return {
        "advisory_only": True,
        "target": target_speaker_context,
        "context": context_speaker_context,
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
