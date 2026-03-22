#!/usr/bin/env python3
"""Codex-interactive helper for semantic chunk boundary review.

Two-pass chunking: run faster-whisper to get a rough transcript, then let Codex
decide at each candidate VAD silence gap whether it falls between sentences
(good split) or mid-sentence (skip).

Subcommands:
  prepare      Run faster-whisper on full audio, collect candidate gaps, write session
  next-candidate  Emit the next candidate gap with surrounding transcript context
  apply-candidate Record Codex's split/skip decision for the current candidate
  status       Show current session progress
  finalize     Produce vad_chunks.json from accepted splits

Usage:
  # 1. Prepare session (runs faster-whisper + Silero VAD)
  python scripts/build_semantic_chunks.py prepare \
    --video samples/episodes/<slug>/source/video.mp4

  # 2. Loop in Codex: next-candidate -> decision -> apply-candidate
  python scripts/build_semantic_chunks.py next-candidate --session <session.json>
  python scripts/build_semantic_chunks.py apply-candidate --session <session.json> \
    --decision-json /tmp/decision.json

  # 3. Finalize
  python scripts/build_semantic_chunks.py finalize --session <session.json>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chigyusubs.audio import get_duration
from chigyusubs.chunking import choose_split_gap, chunk_coverage_issues, collect_vad_gaps
from chigyusubs.paths import find_episode_dir_from_path, find_latest_episode_dir, find_latest_episode_video
from chigyusubs.rocm import apply_rocm_env
from chigyusubs.translation import checkpoint_path, write_json_atomic
from chigyusubs.vad import run_silero_vad


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_episode_dir(args: argparse.Namespace) -> Path:
    if args.episode_dir:
        return Path(args.episode_dir)
    if args.video:
        found = find_episode_dir_from_path(Path(args.video))
        if found:
            return found
    found = find_latest_episode_dir()
    if found is None:
        raise SystemExit("Could not infer episode dir. Pass --episode-dir.")
    return found


def _resolve_video(args: argparse.Namespace, episode_dir: Path) -> Path:
    if args.video:
        return Path(args.video)
    source_dir = episode_dir / "source"
    videos = sorted(
        p for pattern in ("*.mp4", "*.webm", "*.mkv", "*.mov") for p in source_dir.glob(pattern)
    )
    if videos:
        return videos[-1]
    found = find_latest_episode_video()
    if found is None:
        raise SystemExit("Could not infer episode video. Pass --video.")
    return found


def _session_path(output_path: Path) -> Path:
    return Path(checkpoint_path(str(output_path)))


def _collect_candidate_gaps(
    vad_segments: list[dict],
    total_duration: float,
    min_gap_s: float,
) -> list[dict]:
    """Collect silence gaps >= min_gap_s from VAD segments."""
    gaps = []
    for raw_gap in collect_vad_gaps(vad_segments, min_gap_s=min_gap_s):
        gaps.append({
            "candidate_id": len(gaps),
            "gap_start": round(raw_gap["gap_start"], 3),
            "gap_end": round(raw_gap["gap_end"], 3),
            "gap_duration": round(raw_gap["duration"], 3),
            "midpoint": round(raw_gap["time"], 3),
        })
    return gaps


def _transcript_context_around(
    segments: list[dict],
    time_s: float,
    window_s: float = 30.0,
) -> dict:
    """Extract transcript text before and after a timestamp."""
    before_lines = []
    after_lines = []
    for seg in segments:
        seg_mid = (seg["start"] + seg["end"]) / 2
        if seg["end"] <= time_s and seg["start"] >= time_s - window_s:
            before_lines.append({
                "start": round(seg["start"], 3),
                "end": round(seg["end"], 3),
                "text": seg["text"],
            })
        elif seg["start"] >= time_s and seg["end"] <= time_s + window_s:
            after_lines.append({
                "start": round(seg["start"], 3),
                "end": round(seg["end"], 3),
                "text": seg["text"],
            })
    return {
        "before": before_lines,
        "after": after_lines,
    }


def _count_chars_in_interval(
    segments: list[dict],
    start_s: float,
    end_s: float,
) -> int:
    """Approximate transcript density in an interval using segment midpoints."""
    total = 0
    for seg in segments:
        seg_mid = (seg["start"] + seg["end"]) / 2
        if start_s <= seg_mid < end_s:
            total += len(seg["text"])
    return total


def _run_whisper_prepass(video_path: str, model_name: str, compute_type: str) -> list[dict]:
    """Run faster-whisper on the full audio and return segment-level results."""
    apply_rocm_env()
    from faster_whisper import WhisperModel

    print("  Loading faster-whisper model...", flush=True)
    model = WhisperModel(model_name, device="cuda", compute_type=compute_type)

    print("  Transcribing full audio (pre-pass)...", flush=True)
    segments, info = model.transcribe(
        video_path,
        language="ja",
        condition_on_previous_text=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        word_timestamps=False,
    )

    results = []
    for seg in segments:
        results.append({
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
        })
    print(f"  Pre-pass: {len(results)} segments", flush=True)
    return results


def _load_cached_transcript(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    segments = payload.get("segments")
    if not isinstance(segments, list):
        return None
    cleaned = []
    for item in segments:
        if not isinstance(item, dict):
            return None
        try:
            cleaned.append({
                "start": round(float(item["start"]), 3),
                "end": round(float(item["end"]), 3),
                "text": str(item["text"]).strip(),
            })
        except Exception:
            return None
    return cleaned


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_prepare(args: argparse.Namespace) -> int:
    episode_dir = _resolve_episode_dir(args)
    video_path = _resolve_video(args, episode_dir)
    output_path = Path(args.output) if args.output else (episode_dir / "transcription" / "vad_chunks.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    session_file = Path(args.session) if args.session else _session_path(output_path)

    if session_file.exists() and not args.force:
        raise SystemExit(f"Session already exists: {session_file}. Use --force to overwrite.")

    # Clear stale artifacts on --force
    if args.force:
        for stale in (session_file, output_path):
            if stale.exists():
                stale.unlink()

    duration = get_duration(str(video_path))
    print(f"Video: {video_path}", flush=True)
    print(f"Duration: {duration:.0f}s ({duration / 60:.1f} min)", flush=True)

    # Phase 1: Silero VAD
    print("\n=== Phase 1: Silero VAD ===", flush=True)

    # Check for cached VAD segments
    vad_cache = episode_dir / "transcription" / "silero_vad_segments.json"
    if vad_cache.exists():
        print(f"  Using cached VAD: {vad_cache}", flush=True)
        vad_segments = json.loads(vad_cache.read_text(encoding="utf-8"))
    else:
        with tempfile.TemporaryDirectory() as work_dir:
            vad_segments = run_silero_vad(str(video_path), work_dir=work_dir)
        vad_cache.write_text(
            json.dumps(vad_segments, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"  Saved VAD cache: {vad_cache}", flush=True)

    total_speech = sum(s["end"] - s["start"] for s in vad_segments)
    print(f"Speech: {total_speech:.0f}s / {duration:.0f}s ({total_speech / duration * 100:.0f}%)", flush=True)

    # Phase 2: Collect candidate gaps
    print("\n=== Phase 2: Candidate gaps ===", flush=True)
    candidates = _collect_candidate_gaps(vad_segments, duration, min_gap_s=args.min_gap_s)
    print(f"Candidates: {len(candidates)} gaps >= {args.min_gap_s}s", flush=True)

    # Phase 3: faster-whisper pre-pass
    print("\n=== Phase 3: faster-whisper pre-pass ===", flush=True)
    transcript_path = output_path.parent / "whisper_prepass_transcript.json"
    transcript = None if args.rerun_whisper else _load_cached_transcript(transcript_path)
    if transcript is not None:
        print(f"  Reusing cached pre-pass transcript: {transcript_path}", flush=True)
    else:
        print(
            "  Running faster-whisper pre-pass"
            f" (CT2_CUDA_ALLOCATOR={os.environ.get('CT2_CUDA_ALLOCATOR', 'cub_caching via helper')})...",
            flush=True,
        )
        transcript = _run_whisper_prepass(str(video_path), args.model, args.compute_type)
    transcript_char_count = sum(len(seg["text"]) for seg in transcript)
    transcript_chars_per_s = transcript_char_count / duration if duration > 0 else 0.0
    resolved_max_chunk_s = args.max_chunk_s if args.max_chunk_s > 0 else args.target_chunk_s + 30.0
    target_chars = int(round(args.target_chars)) if args.target_chars > 0 else int(round(transcript_chars_per_s * args.target_chunk_s))
    max_chars = int(round(args.max_chars)) if args.max_chars > 0 else int(round(target_chars * 1.2))
    max_chars = max(max_chars, target_chars)

    # Save transcript alongside session for reference
    write_json_atomic(transcript_path, {"segments": transcript})
    print(f"Transcript saved: {transcript_path}", flush=True)

    session = {
        "version": 1,
        "mode": "codex-semantic-chunks",
        "status": "ready",
        "video": str(video_path),
        "episode_dir": str(episode_dir),
        "output": str(output_path),
        "transcript_path": str(transcript_path),
        "duration_seconds": round(duration, 3),
        "target_chunk_s": args.target_chunk_s,
        "max_chunk_s": resolved_max_chunk_s,
        "target_chars": target_chars,
        "max_chars": max_chars,
        "transcript_char_count": transcript_char_count,
        "transcript_chars_per_s": round(transcript_chars_per_s, 3),
        "min_gap_s": args.min_gap_s,
        "fallback_min_gap_s": args.fallback_min_gap_s,
        "context_window_s": args.context_window_s,
        "candidates": candidates,
        "decisions": {},
        "completed_candidates": [],
    }
    write_json_atomic(session_file, session)
    print(f"\nSession written: {session_file}")
    print(f"Candidates to review: {len(candidates)}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    session = json.loads(Path(args.session).read_text(encoding="utf-8"))
    completed = set(session.get("completed_candidates", []))
    pending = [c for c in session["candidates"] if c["candidate_id"] not in completed]
    accepted = [cid for cid, d in session.get("decisions", {}).items() if d["decision"] == "split"]

    payload = {
        "status": session["status"],
        "total_candidates": len(session["candidates"]),
        "completed": len(completed),
        "pending": len(pending),
        "accepted_splits": len(accepted),
        "target_chunk_s": session["target_chunk_s"],
        "max_chunk_s": session["max_chunk_s"],
        "target_chars": session.get("target_chars"),
        "max_chars": session.get("max_chars"),
        "transcript_chars_per_s": session.get("transcript_chars_per_s"),
        "duration_seconds": session["duration_seconds"],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_next_candidate(args: argparse.Namespace) -> int:
    session_file = Path(args.session)
    session = json.loads(session_file.read_text(encoding="utf-8"))
    completed = set(session.get("completed_candidates", []))
    pending = [c for c in session["candidates"] if c["candidate_id"] not in completed]

    if not pending:
        print(json.dumps({
            "status": session["status"],
            "message": "no pending candidates",
        }, ensure_ascii=False, indent=2))
        return 0

    candidate = pending[0]

    # Load transcript for context
    transcript = json.loads(Path(session["transcript_path"]).read_text(encoding="utf-8"))["segments"]
    context = _transcript_context_around(
        transcript,
        candidate["midpoint"],
        window_s=session.get("context_window_s", 30.0),
    )

    # Calculate time since last accepted split (or start)
    accepted_candidates = sorted(
        (
            session["candidates"][int(cid)]
            for cid, d in session.get("decisions", {}).items()
            if d["decision"] == "split"
        ),
        key=lambda candidate_item: candidate_item["candidate_id"],
    )
    last_split_end = accepted_candidates[-1]["gap_end"] if accepted_candidates else 0.0
    time_since_last_split = round(candidate["midpoint"] - last_split_end, 3)
    time_to_end = round(session["duration_seconds"] - candidate["midpoint"], 3)
    target_chars = int(session.get("target_chars", 0) or 0)
    max_chars = int(session.get("max_chars", target_chars) or target_chars)
    chars_since_last_split = _count_chars_in_interval(
        transcript,
        last_split_end,
        candidate["gap_start"],
    )
    chars_to_end = _count_chars_in_interval(
        transcript,
        candidate["gap_end"],
        session["duration_seconds"],
    )
    approaching_max_duration = time_since_last_split >= session["max_chunk_s"] * 0.8
    approaching_max_chars = max_chars > 0 and chars_since_last_split >= max_chars * 0.8

    payload = {
        "status": session["status"],
        "candidate": candidate,
        "progress": {
            "completed": len(completed),
            "total": len(session["candidates"]),
            "pending": len(pending),
        },
        "chunk_state": {
            "time_since_last_split_s": time_since_last_split,
            "time_to_end_s": time_to_end,
            "chars_since_last_split": chars_since_last_split,
            "chars_to_end": chars_to_end,
            "target_chunk_s": session["target_chunk_s"],
            "max_chunk_s": session["max_chunk_s"],
            "target_chars": target_chars,
            "max_chars": max_chars,
            "approaching_max_duration": approaching_max_duration,
            "approaching_max_chars": approaching_max_chars,
            "approaching_target_chars": target_chars > 0 and chars_since_last_split >= target_chars,
            "approaching_max": approaching_max_duration or approaching_max_chars,
        },
        "transcript_context": context,
        "decision_policy": {
            "allowed_decisions": ["split", "skip"],
            "split": "accept this gap as a chunk boundary",
            "skip": "skip this gap, not a good sentence boundary",
            "note": "if approaching_max is true because of duration or transcript density, prefer splitting even at an imperfect boundary",
        },
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output_json:
        Path(args.output_json).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


def cmd_apply_candidate(args: argparse.Namespace) -> int:
    session_file = Path(args.session)
    session = json.loads(session_file.read_text(encoding="utf-8"))

    if session.get("status") in {"completed", "stopped"}:
        print(f"Session is {session['status']}", file=sys.stderr)
        return 1

    completed = set(session.get("completed_candidates", []))
    pending = [c for c in session["candidates"] if c["candidate_id"] not in completed]
    if not pending:
        print("No pending candidates.", file=sys.stderr)
        return 1

    current = pending[0]
    decision_payload = json.loads(Path(args.decision_json).read_text(encoding="utf-8"))

    candidate_id = int(decision_payload.get("candidate_id", -1))
    if candidate_id != current["candidate_id"]:
        raise ValueError(f"Expected candidate_id {current['candidate_id']}, got {candidate_id}")

    decision = str(decision_payload.get("decision", ""))
    if decision not in {"split", "skip"}:
        raise ValueError("decision must be 'split' or 'skip'")

    notes = str(decision_payload.get("notes", ""))

    session.setdefault("decisions", {})[str(candidate_id)] = {
        "decision": decision,
        "notes": notes,
    }
    completed.add(candidate_id)
    session["completed_candidates"] = sorted(completed)

    # Check if all done
    remaining = [c for c in session["candidates"] if c["candidate_id"] not in completed]
    if not remaining:
        session["status"] = "review_complete"

    write_json_atomic(session_file, session)

    accepted_count = sum(1 for d in session["decisions"].values() if d["decision"] == "split")
    print(
        f"Candidate {candidate_id}: {decision}"
        f" | accepted splits: {accepted_count}"
        f" | remaining: {len(remaining)}"
    )
    return 0


def cmd_finalize(args: argparse.Namespace) -> int:
    session_file = Path(args.session)
    session = json.loads(session_file.read_text(encoding="utf-8"))
    duration = session["duration_seconds"]
    max_chunk_s = session["max_chunk_s"]
    fallback_min_gap_s = float(session.get("fallback_min_gap_s", 0.75) or 0.75)

    # Collect accepted split midpoints (full-coverage: split in the middle of
    # the silence gap so no audio is dropped between chunks)
    split_midpoints = []
    for cid_str, d in session.get("decisions", {}).items():
        if d["decision"] == "split":
            candidate = session["candidates"][int(cid_str)]
            split_midpoints.append(candidate["midpoint"])

    split_midpoints.sort()

    # Build chunk boundaries — contiguous, full-coverage
    chunks = []
    chunk_start = 0.0
    for midpoint in split_midpoints:
        chunks.append({
            "chunk_id": len(chunks),
            "start_sec": round(chunk_start, 3),
            "end_sec": round(midpoint, 3),
            "duration_sec": round(midpoint - chunk_start, 3),
        })
        chunk_start = midpoint

    # Final chunk
    if chunk_start < duration:
        chunks.append({
            "chunk_id": len(chunks),
            "start_sec": round(chunk_start, 3),
            "end_sec": round(duration, 3),
            "duration_sec": round(duration - chunk_start, 3),
        })

    # Enforce a hard max chunk duration by inserting deterministic fallback splits
    # when semantic review left an oversized span with no accepted midpoint.
    if chunks:
        episode_dir = Path(session["episode_dir"])
        vad_cache = episode_dir / "transcription" / "silero_vad_segments.json"
        fallback_gaps = []
        if vad_cache.exists():
            try:
                vad_segments = json.loads(vad_cache.read_text(encoding="utf-8"))
                fallback_gaps = collect_vad_gaps(vad_segments, min_gap_s=0.0)
            except json.JSONDecodeError:
                fallback_gaps = []
        enforced_chunks = []
        forced_splits = 0
        fallback_splits = 0
        for chunk in chunks:
            start = float(chunk["start_sec"])
            end = float(chunk["end_sec"])
            while end - start > max_chunk_s:
                latest_gap_that_fits = None
                for gap in fallback_gaps:
                    gap_time = float(gap["time"])
                    gap_duration = float(gap["duration"])
                    if gap_time <= start or gap_time > start + max_chunk_s:
                        continue
                    if gap_duration < fallback_min_gap_s:
                        continue
                    if end - gap_time <= max_chunk_s:
                        latest_gap_that_fits = gap
                target_end = start + session.get("target_chunk_s", max_chunk_s)
                max_end = start + max_chunk_s
                fallback_gap = latest_gap_that_fits
                if fallback_gap is None:
                    fallback_gap = choose_split_gap(
                        fallback_gaps,
                        chunk_start=start,
                        target_end=target_end,
                        max_end=max_end,
                        target_chunk_s=session.get("target_chunk_s", max_chunk_s),
                        min_gap_s=float(session.get("min_gap_s", 1.5) or 1.5),
                        fallback_min_gap_s=fallback_min_gap_s,
                    )
                if fallback_gap is not None:
                    forced_end = round(float(fallback_gap["time"]), 3)
                    fallback_splits += 1
                else:
                    forced_end = round(start + max_chunk_s, 3)
                    forced_splits += 1
                enforced_chunks.append({
                    "chunk_id": len(enforced_chunks),
                    "start_sec": round(start, 3),
                    "end_sec": forced_end,
                    "duration_sec": round(forced_end - start, 3),
                })
                start = forced_end
            enforced_chunks.append({
                "chunk_id": len(enforced_chunks),
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "duration_sec": round(end - start, 3),
            })
        chunks = enforced_chunks
        if fallback_splits:
            print(
                f"Inserted {fallback_splits} fallback split(s) at shorter real silence gaps "
                f"(down to {fallback_min_gap_s}s) before forcing any mid-speech cuts.",
                file=sys.stderr,
            )
        if forced_splits:
            print(
                f"Inserted {forced_splits} forced split(s) at max_chunk_s ({max_chunk_s}s) "
                "to enforce the hard chunk-duration cap.",
                file=sys.stderr,
            )

    # Validate full coverage
    bounds = [(c["start_sec"], c["end_sec"]) for c in chunks]
    issues = chunk_coverage_issues(bounds, duration)
    if issues:
        print("Coverage issues:", file=sys.stderr)
        for issue in issues:
            print(f"  - {issue}", file=sys.stderr)

    # Warn about oversized chunks (should now be impossible unless rounding drift remains)
    oversized = [c for c in chunks if c["duration_sec"] > max_chunk_s]
    if oversized:
        print(f"Warning: {len(oversized)} chunks exceed max_chunk_s ({max_chunk_s}s):", file=sys.stderr)
        for c in oversized:
            print(f"  chunk {c['chunk_id']}: {c['duration_sec']:.1f}s", file=sys.stderr)

    output_path = Path(session["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    session["status"] = "completed"
    write_json_atomic(session_file, session)

    print(f"Wrote {len(chunks)} chunks: {output_path}")
    if chunks:
        durations = [c["duration_sec"] for c in chunks]
        print(f"Durations: min={min(durations):.1f}s avg={sum(durations)/len(durations):.1f}s max={max(durations):.1f}s")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Codex-interactive semantic chunk boundary review.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Run whisper pre-pass, collect candidates, create session.")
    prepare.add_argument("--video", default="", help="Input video path.")
    prepare.add_argument("--episode-dir", default="", help="Episode root directory.")
    prepare.add_argument("--output", default="", help="Output vad_chunks.json path.")
    prepare.add_argument("--session", default="", help="Session JSON path.")
    prepare.add_argument("--model", default="large-v3", help="faster-whisper model (default: large-v3).")
    prepare.add_argument("--compute-type", default="float16", help="CTranslate2 compute type.")
    prepare.add_argument("--target-chunk-s", type=float, default=240.0, help="Target chunk duration in seconds.")
    prepare.add_argument(
        "--max-chunk-s",
        type=float,
        default=0.0,
        help="Maximum chunk duration before forcing a split. Default: target_chunk_s + 30s.",
    )
    prepare.add_argument("--target-chars", type=float, default=0.0, help="Soft target transcript character budget. Default: derive from pre-pass density and target chunk duration.")
    prepare.add_argument("--max-chars", type=float, default=0.0, help="Maximum transcript character budget before preferring a split. Default: 1.2x target chars.")
    prepare.add_argument("--min-gap-s", type=float, default=1.5, help="Minimum silence gap to consider as candidate.")
    prepare.add_argument(
        "--fallback-min-gap-s",
        type=float,
        default=0.75,
        help="Short-gap fallback before a forced split during finalize.",
    )
    prepare.add_argument("--context-window-s", type=float, default=30.0, help="Transcript context window around each candidate.")
    prepare.add_argument("--rerun-whisper", action="store_true", help="Ignore cached whisper_prepass_transcript.json and rerun faster-whisper.")
    prepare.add_argument("--force", action="store_true", help="Overwrite existing session.")
    prepare.set_defaults(func=cmd_prepare)

    status = sub.add_parser("status", help="Show session progress.")
    status.add_argument("--session", required=True)
    status.set_defaults(func=cmd_status)

    next_c = sub.add_parser("next-candidate", help="Emit the next candidate gap with transcript context.")
    next_c.add_argument("--session", required=True)
    next_c.add_argument("--output-json", default="", help="Optional file to write the payload.")
    next_c.set_defaults(func=cmd_next_candidate)

    apply_c = sub.add_parser("apply-candidate", help="Record split/skip decision for the current candidate.")
    apply_c.add_argument("--session", required=True)
    apply_c.add_argument("--decision-json", required=True, help="JSON with candidate_id, decision, notes.")
    apply_c.set_defaults(func=cmd_apply_candidate)

    finalize = sub.add_parser("finalize", help="Produce vad_chunks.json from accepted splits.")
    finalize.add_argument("--session", required=True)
    finalize.set_defaults(func=cmd_finalize)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
