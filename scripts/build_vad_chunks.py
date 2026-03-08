#!/usr/bin/env python3
"""Build reusable chunk boundaries from saved VAD segments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chigyusubs.audio import get_duration
from chigyusubs.chunking import find_chunk_boundaries
from chigyusubs.metadata import finish_run, metadata_path, start_run, write_metadata
from chigyusubs.paths import find_episode_dir_from_path, find_latest_episode_dir, find_latest_episode_video


def resolve_episode_dir(args: argparse.Namespace) -> Path:
    if args.episode_dir:
        return Path(args.episode_dir)
    for attr in ("in_json", "video", "out_json"):
        raw = getattr(args, attr, "")
        if not raw:
            continue
        found = find_episode_dir_from_path(Path(raw))
        if found:
            return found
    found = find_latest_episode_dir()
    if found is None:
        raise SystemExit("Could not infer episode dir. Pass --episode-dir.")
    return found


def resolve_video_path(args: argparse.Namespace, episode_dir: Path) -> Path:
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


def main():
    parser = argparse.ArgumentParser(description="Build reusable chunk boundaries from saved VAD segments.")
    parser.add_argument("--episode-dir", default="", help="Episode root, e.g. samples/episodes/<episode_slug>")
    parser.add_argument("--video", default="", help="Input media path. Defaults to episode source media.")
    parser.add_argument("--in-json", default="", help="Defaults to <episode>/transcription/silero_vad_segments.json")
    parser.add_argument("--out-json", default="", help="Defaults to <episode>/transcription/vad_chunks.json")
    parser.add_argument("--target-chunk-s", type=float, default=240.0)
    parser.add_argument("--min-gap-s", type=float, default=2.0)
    args = parser.parse_args()

    run = start_run("vad_chunks")
    episode_dir = resolve_episode_dir(args)
    video_path = resolve_video_path(args, episode_dir)
    in_json = Path(args.in_json) if args.in_json else (episode_dir / "transcription" / "silero_vad_segments.json")
    out_json = Path(args.out_json) if args.out_json else (episode_dir / "transcription" / "vad_chunks.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if not in_json.exists():
        raise SystemExit(f"VAD JSON not found: {in_json}")

    vad_segments = json.loads(in_json.read_text(encoding="utf-8"))
    duration = get_duration(str(video_path))
    chunk_bounds = find_chunk_boundaries(
        vad_segments,
        duration,
        target_chunk_s=args.target_chunk_s,
        min_gap_s=args.min_gap_s,
    )
    chunks = [
        {
            "chunk_id": idx,
            "start_sec": start,
            "end_sec": end,
            "duration_sec": round(end - start, 3),
        }
        for idx, (start, end) in enumerate(chunk_bounds)
    ]

    out_json.write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Chunk boundaries written: {out_json}")
    print(f"Chunks: {len(chunks)}")
    if chunks:
        durations = [chunk["duration_sec"] for chunk in chunks]
        print(
            f"Chunk durations: min={min(durations):.1f}s avg={sum(durations) / len(durations):.1f}s max={max(durations):.1f}s"
        )

    metadata = finish_run(
        run,
        episode_dir=str(episode_dir),
        inputs={
            "video": str(video_path),
            "vad_json": str(in_json),
        },
        outputs={
            "chunk_json": str(out_json),
        },
        chunk_settings={
            "target_chunk_s": args.target_chunk_s,
            "min_gap_s": args.min_gap_s,
        },
        stats={
            "vad_segments": len(vad_segments),
            "chunks": len(chunks),
            "duration_seconds": round(duration, 3),
        },
    )
    write_metadata(out_json, metadata)
    print(f"Metadata written: {metadata_path(out_json)}")


if __name__ == "__main__":
    main()
