#!/usr/bin/env python3
"""Run reusable Silero VAD for an episode and write stable JSON artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chigyusubs.audio import get_duration
from chigyusubs.metadata import finish_run, metadata_path, start_run, write_metadata
from chigyusubs.paths import find_episode_dir_from_path, find_latest_episode_dir, find_latest_episode_video
from chigyusubs.vad import run_silero_vad


def resolve_episode_dir(args: argparse.Namespace) -> Path:
    if args.episode_dir:
        return Path(args.episode_dir)
    for attr in ("video", "out_json"):
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
    parser = argparse.ArgumentParser(description="Run standalone Silero VAD and persist reusable JSON.")
    parser.add_argument("--episode-dir", default="", help="Episode root, e.g. samples/episodes/<episode_slug>")
    parser.add_argument("--video", default="", help="Input media path. Defaults to episode source media.")
    parser.add_argument("--out-json", default="", help="Defaults to <episode>/transcription/silero_vad_segments.json")
    parser.add_argument("--work-dir", default="", help="Defaults to <episode>/transcription")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-speech-ms", type=int, default=250)
    parser.add_argument("--min-silence-ms", type=int, default=100)
    args = parser.parse_args()

    run = start_run("vad")
    episode_dir = resolve_episode_dir(args)
    video_path = resolve_video_path(args, episode_dir)
    work_dir = Path(args.work_dir) if args.work_dir else (episode_dir / "transcription")
    out_json = Path(args.out_json) if args.out_json else (episode_dir / "transcription" / "silero_vad_segments.json")
    work_dir.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    duration = get_duration(str(video_path))
    print(f"Video: {video_path}")
    print(f"Duration: {duration:.1f}s")

    segments = run_silero_vad(
        str(video_path),
        work_dir=str(work_dir),
        threshold=args.threshold,
        min_speech_duration_ms=args.min_speech_ms,
        min_silence_duration_ms=args.min_silence_ms,
    )
    total_speech = sum(seg["end"] - seg["start"] for seg in segments)
    speech_ratio = (total_speech / duration) if duration > 0 else 0.0

    out_json.write_text(json.dumps(segments, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"VAD segments written: {out_json}")
    print(f"Segments: {len(segments)}")
    print(f"Speech: {total_speech:.1f}s / {duration:.1f}s ({speech_ratio * 100:.1f}%)")

    metadata = finish_run(
        run,
        episode_dir=str(episode_dir),
        inputs={
            "video": str(video_path),
        },
        outputs={
            "vad_json": str(out_json),
        },
        vad_settings={
            "threshold": args.threshold,
            "min_speech_ms": args.min_speech_ms,
            "min_silence_ms": args.min_silence_ms,
            "work_dir": str(work_dir),
        },
        stats={
            "duration_seconds": round(duration, 3),
            "vad_segments": len(segments),
            "speech_seconds": round(total_speech, 3),
            "speech_ratio": round(speech_ratio, 4),
        },
    )
    write_metadata(out_json, metadata)
    print(f"Metadata written: {metadata_path(out_json)}")


if __name__ == "__main__":
    main()
