#!/usr/bin/env python3
"""Codex-interactive helper for merging Flash Lite audio additions into Gemini transcripts.

Flash Lite re-transcribes from audio and catches missing reactions/interjections
that Gemini's video pass compresses away, but it rewrites line boundaries and drops
visual annotations. This script diffs the two per chunk and lets Codex decide which
additions to accept.

Subcommands:
    prepare      Diff gemini_raw vs corrected, write session checkpoint
    next-chunk   Return next pending chunk's diff payload as JSON
    apply-chunk  Record Codex's merge decisions for a chunk
    status       Print progress summary
    finalize     Build augmented gemini_raw and update preferred.json

Usage:
    python scripts/augment_transcript_codex.py prepare \
        --gemini-raw <path>_gemini_raw.json \
        --corrected <path>_corrected.json

    python scripts/augment_transcript_codex.py next-chunk --session <session.json>

    python scripts/augment_transcript_codex.py apply-chunk \
        --session <session.json> --merge-json /tmp/merge.json

    python scripts/augment_transcript_codex.py finalize --session <session.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chigyusubs.transcript_comparison import (
    SPEAKER_DASH_RE,
    loose_contains,
    normalize_text,
    parse_raw_items,
    text_similarity,
)
from chigyusubs.translation import write_json_atomic


# ---------------------------------------------------------------------------
# Diff algorithm
# ---------------------------------------------------------------------------

def _spoken_text(item: dict) -> str:
    """Get the spoken text from a parsed item."""
    return item.get("text", "")


def _raw_line(item: dict) -> str:
    """Reconstruct the raw line from a parsed item."""
    if item["type"] == "visual":
        return f"[画面: {item['text']}]"
    prefix = "-- " if item.get("starts_new_turn") else ""
    return f"{prefix}{item['text']}"


def _match_lines(
    orig_items: list[dict],
    corr_items: list[dict],
    *,
    match_threshold: float = 0.85,
    modify_threshold: float = 0.5,
) -> dict:
    """Match original items to corrected items, maintaining relative order.

    Returns dict with keys: matched, modified, added, dropped.
    """
    n_orig = len(orig_items)
    n_corr = len(corr_items)

    # For each corrected item, find the best original match (order-preserving)
    # Use a greedy forward scan: for each corr item, only consider orig items
    # at or after the last matched orig index.
    orig_matched: set[int] = set()
    corr_matched: dict[int, int] = {}  # corr_idx -> orig_idx

    # First pass: match high-confidence pairs greedily in order
    last_orig = 0
    for ci in range(n_corr):
        if corr_items[ci]["type"] == "visual":
            # Visual items: match by exact text
            for oi in range(last_orig, n_orig):
                if oi in orig_matched:
                    continue
                if orig_items[oi]["type"] == "visual" and orig_items[oi]["text"] == corr_items[ci]["text"]:
                    orig_matched.add(oi)
                    corr_matched[ci] = oi
                    last_orig = oi + 1
                    break
            continue

        best_oi = -1
        best_sim = 0.0
        ct = _spoken_text(corr_items[ci])
        for oi in range(last_orig, n_orig):
            if oi in orig_matched:
                continue
            if orig_items[oi]["type"] != "spoken":
                continue
            ot = _spoken_text(orig_items[oi])
            # Exact normalized match is fastest
            if normalize_text(ct) == normalize_text(ot):
                best_oi = oi
                best_sim = 1.0
                break
            # Check containment
            if loose_contains(ct, ot):
                sim = text_similarity(ct, ot)
                if sim > best_sim:
                    best_oi = oi
                    best_sim = max(sim, match_threshold)  # containment counts as match
            else:
                sim = text_similarity(ct, ot)
                if sim > best_sim:
                    best_oi = oi
                    best_sim = sim

        if best_oi >= 0 and best_sim >= modify_threshold:
            orig_matched.add(best_oi)
            corr_matched[ci] = best_oi
            last_orig = best_oi + 1

    # Second pass: try to match remaining orig items that were skipped
    # (Flash Lite might reorder slightly)
    for oi in range(n_orig):
        if oi in orig_matched:
            continue
        if orig_items[oi]["type"] == "visual":
            continue
        ot = _spoken_text(orig_items[oi])
        best_ci = -1
        best_sim = 0.0
        for ci in range(n_corr):
            if ci in corr_matched:
                continue
            if corr_items[ci]["type"] != "spoken":
                continue
            ct = _spoken_text(corr_items[ci])
            sim = text_similarity(ot, ct)
            if loose_contains(ot, ct):
                sim = max(sim, match_threshold)
            if sim > best_sim:
                best_ci = ci
                best_sim = sim
        if best_ci >= 0 and best_sim >= modify_threshold:
            orig_matched.add(oi)
            corr_matched[best_ci] = oi

    # Classify
    matched = []
    modified = []
    for ci, oi in sorted(corr_matched.items()):
        sim = text_similarity(_spoken_text(orig_items[oi]), _spoken_text(corr_items[ci]))
        if orig_items[oi]["type"] == "visual":
            sim = 1.0  # visual items matched by exact text
        if sim >= match_threshold or loose_contains(
            _spoken_text(orig_items[oi]), _spoken_text(corr_items[ci])
        ):
            matched.append({"orig_idx": oi, "corr_idx": ci})
        else:
            modified.append({
                "orig_idx": oi,
                "corr_idx": ci,
                "original": _raw_line(orig_items[oi]),
                "corrected": _raw_line(corr_items[ci]),
                "similarity": round(sim, 3),
            })

    # Additions: corrected items with no match
    added = []
    for ci in range(n_corr):
        if ci in corr_matched:
            continue
        # Determine insert position: find the nearest matched corr item before this one
        insert_after = -1  # before all original lines
        for prev_ci in range(ci - 1, -1, -1):
            if prev_ci in corr_matched:
                insert_after = corr_matched[prev_ci]
                break
        added.append({
            "addition_idx": len(added),
            "insert_after_orig": insert_after,
            "text": _raw_line(corr_items[ci]),
            "type": corr_items[ci]["type"],
        })

    # Dropped: original items with no match
    dropped = []
    for oi in range(n_orig):
        if oi in orig_matched:
            continue
        dropped.append({
            "orig_idx": oi,
            "text": _raw_line(orig_items[oi]),
            "type": orig_items[oi]["type"],
            "warning": "visual annotation not in audio" if orig_items[oi]["type"] == "visual" else "",
        })

    return {
        "matched": matched,
        "modified": modified,
        "added": added,
        "dropped": dropped,
    }


def _diff_chunk(orig_text: str, corr_text: str) -> dict:
    """Diff a single chunk's original vs corrected text."""
    orig_items = parse_raw_items(orig_text)
    corr_items = parse_raw_items(corr_text)
    return _match_lines(orig_items, corr_items)


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def _load_session(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_session(path: Path, session: dict) -> None:
    write_json_atomic(path, session)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_prepare(args: argparse.Namespace) -> None:
    gemini_raw_path = Path(args.gemini_raw)
    corrected_path = Path(args.corrected)

    orig_chunks = json.loads(gemini_raw_path.read_text(encoding="utf-8"))
    corr_chunks = json.loads(corrected_path.read_text(encoding="utf-8"))

    if len(orig_chunks) != len(corr_chunks):
        print(f"Error: chunk count mismatch ({len(orig_chunks)} vs {len(corr_chunks)})", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else gemini_raw_path.with_name(
        gemini_raw_path.stem.replace("_gemini_raw", "") + "_augmented.json"
    )
    session_path = Path(str(output_path) + ".session.json")

    chunk_diffs = []
    n_skip = 0
    n_with_additions = 0
    total_additions = 0

    for i, (orig, corr) in enumerate(zip(orig_chunks, corr_chunks)):
        orig_text = orig.get("text", "")
        corr_text = corr.get("corrected_text", corr.get("text", ""))

        # Handle error chunks or missing corrected text
        if not corr_text or "correction_error" in corr:
            chunk_diffs.append({
                "chunk_index": i,
                "chunk_start_s": orig.get("chunk_start_s", 0),
                "chunk_end_s": orig.get("chunk_end_s", 0),
                "status": "skipped",
                "reason": "no corrected text" if not corr_text else "correction error",
            })
            n_skip += 1
            continue

        diff = _diff_chunk(orig_text, corr_text)

        if not diff["added"]:
            chunk_diffs.append({
                "chunk_index": i,
                "chunk_start_s": orig.get("chunk_start_s", 0),
                "chunk_end_s": orig.get("chunk_end_s", 0),
                "status": "skipped",
                "reason": "no additions",
            })
            n_skip += 1
            continue

        n_with_additions += 1
        total_additions += len(diff["added"])

        # Build original lines list for payload
        orig_items = parse_raw_items(orig_text)
        original_lines = []
        for idx, item in enumerate(orig_items):
            original_lines.append({"idx": idx, "text": _raw_line(item)})

        chunk_diffs.append({
            "chunk_index": i,
            "chunk_start_s": orig.get("chunk_start_s", 0),
            "chunk_end_s": orig.get("chunk_end_s", 0),
            "status": "pending",
            "original_lines": original_lines,
            "additions": diff["added"],
            "modifications": diff["modified"],
            "dropped_originals": [d for d in diff["dropped"] if d["warning"]],
            "merge_result": None,
        })

    session = {
        "gemini_raw_path": str(gemini_raw_path),
        "corrected_path": str(corrected_path),
        "output_path": str(output_path),
        "chunks": chunk_diffs,
        "stats": {
            "total_chunks": len(orig_chunks),
            "chunks_with_additions": n_with_additions,
            "chunks_skipped": n_skip,
            "total_additions": total_additions,
        },
    }

    _save_session(session_path, session)

    print(f"Session: {session_path}")
    print(f"Total chunks: {len(orig_chunks)}")
    print(f"Chunks with additions: {n_with_additions}")
    print(f"Chunks skipped (no additions): {n_skip}")
    print(f"Total candidate additions: {total_additions}")
    print(f"Output will be: {output_path}")


def cmd_next_chunk(args: argparse.Namespace) -> None:
    session_path = Path(args.session)
    session = _load_session(session_path)

    pending = [c for c in session["chunks"] if c["status"] == "pending"]
    if not pending:
        print(json.dumps({"done": True, "message": "All chunks reviewed."}))
        return

    chunk = pending[0]
    remaining = len(pending) - 1

    payload = {
        "done": False,
        "chunk_index": chunk["chunk_index"],
        "chunk_time": f"{chunk['chunk_start_s']:.1f}-{chunk['chunk_end_s']:.1f}s",
        "original_lines": chunk["original_lines"],
        "additions": chunk["additions"],
        "modifications": chunk.get("modifications", []),
        "dropped_originals": chunk.get("dropped_originals", []),
        "remaining": remaining,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_apply_chunk(args: argparse.Namespace) -> None:
    session_path = Path(args.session)
    session = _load_session(session_path)

    merge = json.loads(Path(args.merge_json).read_text(encoding="utf-8"))
    target_idx = merge["chunk_index"]

    # Find the chunk in session
    chunk = None
    for c in session["chunks"]:
        if c["chunk_index"] == target_idx:
            chunk = c
            break

    if chunk is None:
        print(f"Error: chunk {target_idx} not found in session", file=sys.stderr)
        sys.exit(1)

    if chunk["status"] != "pending":
        print(f"Error: chunk {target_idx} status is '{chunk['status']}', expected 'pending'", file=sys.stderr)
        sys.exit(1)

    accepted = set(merge.get("accepted_additions", []))
    rejected = set(merge.get("rejected_additions", []))

    # Build merged text: start with all original lines, insert accepted additions
    orig_lines = [item["text"] for item in chunk["original_lines"]]

    # Group additions by insert_after position
    insertions: dict[int, list[str]] = {}
    for addition in chunk["additions"]:
        aidx = addition["addition_idx"]
        if aidx in accepted:
            pos = addition["insert_after_orig"]
            insertions.setdefault(pos, []).append(addition["text"])

    # Build result: walk through original lines, inserting additions after each
    merged_lines: list[str] = []

    # Insert additions that go before all original lines (insert_after = -1)
    for line in insertions.get(-1, []):
        merged_lines.append(line)

    for i, orig_line in enumerate(orig_lines):
        merged_lines.append(orig_line)
        for line in insertions.get(i, []):
            merged_lines.append(line)

    merged_text = "\n".join(merged_lines)
    chunk["merge_result"] = merged_text
    chunk["status"] = "completed"
    chunk["accepted_count"] = len(accepted)
    chunk["rejected_count"] = len(rejected)
    chunk["notes"] = merge.get("notes", "")

    _save_session(session_path, session)

    total_completed = sum(1 for c in session["chunks"] if c["status"] == "completed")
    total_pending = sum(1 for c in session["chunks"] if c["status"] == "pending")
    print(f"Chunk {target_idx}: accepted {len(accepted)}/{len(accepted) + len(rejected)} additions")
    print(f"Progress: {total_completed} completed, {total_pending} pending")


def cmd_status(args: argparse.Namespace) -> None:
    session_path = Path(args.session)
    session = _load_session(session_path)

    completed = [c for c in session["chunks"] if c["status"] == "completed"]
    pending = [c for c in session["chunks"] if c["status"] == "pending"]
    skipped = [c for c in session["chunks"] if c["status"] == "skipped"]

    total_accepted = sum(c.get("accepted_count", 0) for c in completed)
    total_rejected = sum(c.get("rejected_count", 0) for c in completed)

    print(f"Session: {args.session}")
    print(f"Total chunks: {session['stats']['total_chunks']}")
    print(f"Completed: {len(completed)}")
    print(f"Pending: {len(pending)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Additions accepted: {total_accepted}")
    print(f"Additions rejected: {total_rejected}")

    if pending:
        next_chunk = pending[0]
        print(f"Next: chunk {next_chunk['chunk_index']} ({next_chunk['chunk_start_s']:.1f}-{next_chunk['chunk_end_s']:.1f}s)")


def cmd_finalize(args: argparse.Namespace) -> None:
    session_path = Path(args.session)
    session = _load_session(session_path)

    pending = [c for c in session["chunks"] if c["status"] == "pending"]
    if pending:
        print(f"Warning: {len(pending)} chunks still pending. They will use original text.", file=sys.stderr)

    from chigyusubs.metadata import (
        finish_run,
        inherit_run_id,
        start_run,
        update_preferred_manifest,
        write_metadata,
    )

    gemini_raw_path = Path(session["gemini_raw_path"])
    output_path = Path(session["output_path"])
    orig_chunks = json.loads(gemini_raw_path.read_text(encoding="utf-8"))

    # Build augmented output
    augmented = []
    total_accepted = 0
    chunks_modified = 0

    # Index session chunks by chunk_index
    session_by_idx = {c["chunk_index"]: c for c in session["chunks"]}

    for i, orig in enumerate(orig_chunks):
        sc = session_by_idx.get(i)
        if sc and sc["status"] == "completed" and sc.get("merge_result"):
            augmented.append({
                **{k: v for k, v in orig.items()},
                "text": sc["merge_result"],
                "augmented": True,
                "augmentation_stats": {
                    "accepted": sc.get("accepted_count", 0),
                    "rejected": sc.get("rejected_count", 0),
                },
            })
            total_accepted += sc.get("accepted_count", 0)
            chunks_modified += 1
        else:
            augmented.append(dict(orig))

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(augmented, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Metadata & preferred.json
    run = start_run("transcript_augment")
    run = inherit_run_id(run, gemini_raw_path)
    run = finish_run(
        run,
        inputs={"gemini_raw": str(gemini_raw_path), "corrected": session["corrected_path"]},
        outputs={"augmented": str(output_path)},
        stats={
            "chunks_modified": chunks_modified,
            "total_accepted": total_accepted,
        },
    )
    write_metadata(output_path, run)

    # Update preferred.json
    tx_dir = output_path.parent
    if tx_dir.name == "transcription":
        update_preferred_manifest(tx_dir, gemini_raw=output_path.name)
        print(f"Updated preferred.json: gemini_raw = {output_path.name}")

    print(f"Wrote {output_path}")
    print(f"Chunks modified: {chunks_modified}/{len(orig_chunks)}")
    print(f"Total additions accepted: {total_accepted}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Codex-interactive transcript augmentation: merge Flash Lite audio additions into Gemini transcripts.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="Diff gemini_raw vs corrected, write session")
    p_prepare.add_argument("--gemini-raw", required=True, help="Path to gemini_raw.json")
    p_prepare.add_argument("--corrected", required=True, help="Path to Flash Lite corrected.json")
    p_prepare.add_argument("--output", default="", help="Output augmented JSON path")

    p_next = sub.add_parser("next-chunk", help="Get next pending chunk diff payload")
    p_next.add_argument("--session", required=True, help="Path to session JSON")

    p_apply = sub.add_parser("apply-chunk", help="Apply merge decisions for a chunk")
    p_apply.add_argument("--session", required=True, help="Path to session JSON")
    p_apply.add_argument("--merge-json", required=True, help="Path to merge decision JSON")

    p_status = sub.add_parser("status", help="Print progress summary")
    p_status.add_argument("--session", required=True, help="Path to session JSON")

    p_finalize = sub.add_parser("finalize", help="Build augmented output and update preferred.json")
    p_finalize.add_argument("--session", required=True, help="Path to session JSON")

    args = parser.parse_args()

    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "next-chunk":
        cmd_next_chunk(args)
    elif args.command == "apply-chunk":
        cmd_apply_chunk(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "finalize":
        cmd_finalize(args)


if __name__ == "__main__":
    main()
