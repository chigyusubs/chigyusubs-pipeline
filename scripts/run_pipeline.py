#!/usr/bin/env python3
"""Transcription pipeline orchestrator.

Runs the automated first half of the subtitle pipeline — from raw video through
transcription, alignment, reflow, and second-opinion diagnostics. Each phase is
delegated to an existing script via subprocess; artifact paths are chained through
the preferred.json lineage system (see chigyusubs/metadata.py).

Phases:
  1. Whisper pre-pass + VAD              (GPU, ROCm)
  2. Semantic chunk auto-review           (CPU, instant)
  3. Gemini transcription                 (Network, concurrent)
  4. CTC alignment                        (GPU, ROCm)
  5. Reflow                               (CPU, instant)
  6. Second opinion report                (CPU, compares pre-pass vs gemini)

Usage:
  python scripts/run_pipeline.py --video /path/to/video.mp4
  python scripts/run_pipeline.py --slug great_escape_s03e07_tvcut
  python scripts/run_pipeline.py --video /path/to/video.mp4 --stop-after 3
  python scripts/run_pipeline.py --slug my_episode --force-from 4
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chigyusubs.audio import get_duration
from chigyusubs.env import load_repo_env
from chigyusubs.metadata import preferred_manifest_path
from chigyusubs.paths import ensure_episode_layout, infer_episode_dir_from_video

load_repo_env()

# ROCm environment for GPU subprocesses
_ROCM_ENV = {
    **os.environ,
    "LD_LIBRARY_PATH": "/opt/rocm/lib",
    "CT2_CUDA_ALLOCATOR": "cub_caching",
}

PYTHON = "python3.12"

# preferred.json keys written by each phase (used for skip logic and force cascade)
_PREFERRED_KEYS = {3: "gemini_raw", 4: "ctc_words", 5: "reflow"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str = ""):
    print(msg, flush=True)


def phase_header(n: int, title: str):
    log()
    log("=" * 60)
    log(f"PHASE {n}: {title}")
    log("=" * 60)


def run_cmd(cmd: list[str], env: dict | None = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess with live output."""
    log(f"  $ {' '.join(cmd[:6])}{'...' if len(cmd) > 6 else ''}")
    result = subprocess.run(cmd, env=env or os.environ)
    if check and result.returncode != 0:
        log(f"  Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    return result


def read_preferred(area_dir: Path, key: str) -> Path | None:
    """Read a key from preferred.json, return full path if file exists on disk."""
    manifest = preferred_manifest_path(area_dir)
    if not manifest.exists():
        return None
    name = json.loads(manifest.read_text(encoding="utf-8")).get(key)
    if not name:
        return None
    candidate = area_dir / name
    return candidate if candidate.exists() else None


def clear_preferred_keys(area_dir: Path, *keys: str) -> None:
    """Remove keys from preferred.json (for --force)."""
    manifest = preferred_manifest_path(area_dir)
    if not manifest.exists():
        return
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    changed = any(payload.pop(k, None) is not None for k in keys)
    if changed:
        manifest.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        log(f"  Cleared preferred keys: {', '.join(keys)}")


# ---------------------------------------------------------------------------
# Video / episode resolution
# ---------------------------------------------------------------------------

def find_video_for_slug(slug: str, base: Path) -> Path:
    """Find the video file for an existing episode slug."""
    source_dir = base / "samples" / "episodes" / slug / "source"
    if not source_dir.exists():
        log(f"Error: source dir not found: {source_dir}")
        sys.exit(1)
    videos = [f for f in source_dir.iterdir() if f.suffix in (".mp4", ".mkv", ".webm", ".avi")]
    if not videos:
        log(f"Error: no video files in {source_dir}")
        sys.exit(1)
    if len(videos) > 1:
        log(f"Warning: multiple videos in {source_dir}, using {videos[0].name}")
    return videos[0]


def setup_episode_from_video(video: Path) -> tuple[Path, Path]:
    """Create episode directory structure and symlink video into source/.

    Returns (video_in_source, episode_dir).
    """
    episode_dir = infer_episode_dir_from_video(video)
    ensure_episode_layout(episode_dir)

    source_dir = episode_dir / "source"
    target = source_dir / video.name

    if target.exists() or target.is_symlink():
        return target.resolve(), episode_dir

    target.symlink_to(video.resolve())
    log(f"  Linked {video.name} -> {episode_dir.name}/source/")
    return target, episode_dir


# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------

def phase_1_whisper_prepass(video: Path, episode_dir: Path, target_s: float, max_s: float, force: bool = False):
    """Run whisper pre-pass and VAD via build_semantic_chunks.py prepare."""
    phase_header(1, "Whisper Pre-pass + VAD")

    session_path = episode_dir / "transcription" / "vad_chunks.json.checkpoint.json"
    prepass_path = episode_dir / "transcription" / "whisper_prepass_transcript.json"

    if session_path.exists() and prepass_path.exists() and not force:
        session = json.loads(session_path.read_text())
        if session.get("status") in ("review_complete", "completed"):
            log(f"  Pre-pass already done (session status: {session['status']})")
            log("  Skipping. Use --force to rerun.")
            return str(session_path)

    cmd = [
        PYTHON, "scripts/build_semantic_chunks.py", "prepare",
        "--video", str(video),
        "--target-chunk-s", str(target_s),
        "--max-chunk-s", str(max_s),
    ]
    if force:
        cmd.append("--force")
        cmd.append("--rerun-whisper")
    run_cmd(cmd, env=_ROCM_ENV)

    if not session_path.exists():
        chunks_path = episode_dir / "transcription" / "vad_chunks.json"
        alt_session = Path(str(chunks_path) + ".checkpoint.json")
        if alt_session.exists():
            session_path = alt_session
        else:
            log(f"  Error: session file not found at {session_path}")
            sys.exit(1)

    log(f"  Session: {session_path}")
    return str(session_path)


def phase_2_auto_chunk(session_path: str):
    """Run deterministic auto-review and finalize chunks."""
    phase_header(2, "Semantic Chunk Auto-Review")

    session = json.loads(Path(session_path).read_text())
    if session.get("status") == "completed":
        log("  Chunks already finalized. Skipping.")
        return session["output"]

    if session.get("status") != "review_complete":
        run_cmd([PYTHON, "scripts/build_semantic_chunks.py", "auto-review", "--session", session_path])

    run_cmd([PYTHON, "scripts/build_semantic_chunks.py", "finalize", "--session", session_path])

    session = json.loads(Path(session_path).read_text())
    log(f"  Output: {session['output']}")
    return session["output"]


def phase_3_transcription(video: Path, tx_dir: Path, chunks_path: str, preset: str) -> Path:
    """Run Gemini transcription. Returns resolved gemini_raw path from preferred.json."""
    phase_header(3, "Gemini Transcription")

    # Output hint — script auto-prefixes with run ID
    output_hint = str(tx_dir / "gemini_raw.json")

    result = run_cmd([
        PYTHON, "scripts/transcribe_gemini_video.py",
        "--video", str(video),
        "--output", output_hint,
        "--chunk-json", chunks_path,
        "--preset", preset,
    ], check=False)

    if result.returncode == 75:
        # Incomplete — some chunks failed (likely RPD quota).
        # Progress is saved in per-chunk files; re-run to resume.
        log()
        log("  Transcription incomplete (some chunks hit rate limits).")
        log(f"  Re-run with a different preset to resume, e.g.:")
        log(f"    python3.12 scripts/run_pipeline.py --slug ... --transcript-preset flash25_free_default --force-from 3")
        sys.exit(75)
    elif result.returncode != 0:
        log(f"  Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    gemini_raw = read_preferred(tx_dir, "gemini_raw")
    if gemini_raw is None:
        log("  Error: transcription completed but preferred.json has no gemini_raw entry")
        sys.exit(1)

    log(f'  preferred["gemini_raw"] = {gemini_raw.name}')
    return gemini_raw


def phase_4_ctc_alignment(video: Path, gemini_raw_path: Path, tx_dir: Path) -> Path:
    """Run CTC forced alignment. Returns resolved ctc_words path from preferred.json."""
    phase_header(4, "CTC Alignment")

    # Output hint — script inherits run ID from gemini_raw, auto-prefixes
    output_hint = str(tx_dir / "ctc_words.json")

    run_cmd([
        PYTHON, "scripts/align_ctc.py",
        "--video", str(video),
        "--chunks", str(gemini_raw_path),
        "--output-words", output_hint,
    ], env=_ROCM_ENV)

    ctc_words = read_preferred(tx_dir, "ctc_words")
    if ctc_words is None:
        log("  Error: alignment completed but preferred.json has no ctc_words entry")
        sys.exit(1)

    log(f'  preferred["ctc_words"] = {ctc_words.name}')
    return ctc_words


def phase_5_reflow(ctc_words_path: Path, tx_dir: Path) -> Path:
    """Reflow aligned words into subtitle cues. Returns resolved reflow path."""
    phase_header(5, "Reflow")

    # Output hint — script inherits run ID, auto-prefixes
    output_hint = str(tx_dir / "reflow.vtt")

    run_cmd([
        PYTHON, "scripts/reflow_words.py",
        "--input", str(ctc_words_path),
        "--output", output_hint,
        "--line-level",
        "--stats",
    ])

    reflow = read_preferred(tx_dir, "reflow")
    if reflow is None:
        log("  Error: reflow completed but preferred.json has no reflow entry")
        sys.exit(1)

    log(f'  preferred["reflow"] = {reflow.name}')
    return reflow


def phase_6_second_opinion(ctc_words_path: Path, force: bool = False):
    """Run second opinion comparison (pre-pass vs gemini)."""
    phase_header(6, "Second Opinion Report")

    cmd = [
        PYTHON, "scripts/pre_reflow_second_opinion.py",
        "--words", str(ctc_words_path),
    ]
    if force:
        cmd.append("--force")
    run_cmd(cmd, env=_ROCM_ENV)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Transcription pipeline: raw video to aligned subtitles + diagnostics.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", help="Input video file path.")
    group.add_argument("--slug", help="Episode slug (looks up video in samples/episodes/<slug>/source/).")

    parser.add_argument("--target-chunk-s", type=float, default=90.0, help="Target chunk duration (default: 90).")
    parser.add_argument("--max-chunk-s", type=float, default=120.0, help="Max chunk duration (default: 120).")
    parser.add_argument(
        "--transcript-preset", default="flash_free_default",
        help="Gemini transcription preset (default: flash_free_default).",
    )
    parser.add_argument(
        "--stop-after", type=int, default=6, choices=range(1, 7),
        help="Stop after phase N (default: 6).",
    )
    parser.add_argument("--force", action="store_true", help="Force rerun of all phases.")
    parser.add_argument(
        "--force-from", type=int, default=None, choices=range(1, 7),
        help="Force rerun from phase N onward (clears downstream preferred entries).",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    # Resolve video and episode directory
    if args.slug:
        video = find_video_for_slug(args.slug, repo_root)
        episode_dir = infer_episode_dir_from_video(video)
    else:
        video = Path(args.video).resolve()
        if not video.exists():
            log(f"Error: video not found: {video}")
            sys.exit(1)
        video, episode_dir = setup_episode_from_video(video)

    tx_dir = episode_dir / "transcription"
    force_from = 1 if args.force else args.force_from

    log(f"Video: {video}")
    log(f"Episode: {episode_dir}")
    log(f"Duration: {get_duration(str(video)):.0f}s")

    # Force cascade: clear downstream preferred keys before starting
    if force_from and force_from <= 6:
        keys_to_clear = [v for k, v in _PREFERRED_KEYS.items() if k >= force_from]
        if keys_to_clear:
            clear_preferred_keys(tx_dir, *keys_to_clear)

    t0 = time.time()
    is_forced = lambda n: force_from is not None and n >= force_from

    # Phase 1: Whisper pre-pass + VAD
    session_path = phase_1_whisper_prepass(
        video, episode_dir, args.target_chunk_s, args.max_chunk_s, force=is_forced(1),
    )
    if args.stop_after < 2:
        return

    # Phase 2: Deterministic chunk boundary selection
    chunks_path = phase_2_auto_chunk(session_path)
    if args.stop_after < 3:
        return

    # Phase 3: Gemini transcription
    existing = read_preferred(tx_dir, "gemini_raw")
    if existing and not is_forced(3):
        phase_header(3, "Gemini Transcription")
        log(f"  Already done: {existing.name}")
        log("  Skipping. Use --force-from 3 to rerun.")
        gemini_raw_path = existing
    else:
        gemini_raw_path = phase_3_transcription(video, tx_dir, chunks_path, args.transcript_preset)
    if args.stop_after < 4:
        return

    # Phase 4: CTC alignment
    existing = read_preferred(tx_dir, "ctc_words")
    if existing and not is_forced(4):
        phase_header(4, "CTC Alignment")
        log(f"  Already done: {existing.name}")
        log("  Skipping. Use --force-from 4 to rerun.")
        ctc_words_path = existing
    else:
        ctc_words_path = phase_4_ctc_alignment(video, gemini_raw_path, tx_dir)
    if args.stop_after < 5:
        return

    # Phase 5: Reflow
    existing = read_preferred(tx_dir, "reflow")
    if existing and not is_forced(5):
        phase_header(5, "Reflow")
        log(f"  Already done: {existing.name}")
        log("  Skipping. Use --force-from 5 to rerun.")
    else:
        phase_5_reflow(ctc_words_path, tx_dir)
    if args.stop_after < 6:
        return

    # Phase 6: Second opinion (always runs)
    phase_6_second_opinion(ctc_words_path, force=is_forced(6))

    elapsed = time.time() - t0
    log()
    log("=" * 60)
    log(f"Pipeline complete in {elapsed/60:.1f} minutes")
    log("=" * 60)


if __name__ == "__main__":
    main()
