#!/usr/bin/env python3
"""Copy a VTT to the episode source folder with a name matching the source video.

mpv auto-discovers subtitle files that share the video's filename stem.

Usage:
    python scripts/publish_vtt.py samples/episodes/<slug>/translation/output.vtt

Finds the video in source/, copies the VTT next to it as <video_stem>.vtt.
"""

import argparse
import shutil
import sys
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".webm", ".ts"}


def find_video(source_dir: Path) -> Path | None:
    for f in source_dir.iterdir():
        if f.suffix.lower() in VIDEO_EXTS and f.is_file():
            return f
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("vtt", type=Path, help="VTT file to publish")
    parser.add_argument("--episode-dir", type=Path, help="Episode directory (auto-detected from VTT path if omitted)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be copied without doing it")
    args = parser.parse_args()

    if not args.vtt.exists():
        print(f"Error: {args.vtt} not found", file=sys.stderr)
        sys.exit(1)

    # Auto-detect episode dir: walk up from the VTT until we find source/
    episode_dir = args.episode_dir
    if episode_dir is None:
        for parent in args.vtt.resolve().parents:
            if (parent / "source").is_dir():
                episode_dir = parent
                break
        if episode_dir is None:
            print("Error: could not find episode directory (no source/ folder in parents)", file=sys.stderr)
            sys.exit(1)

    source_dir = episode_dir / "source"
    video = find_video(source_dir)
    if video is None:
        print(f"Error: no video file found in {source_dir}", file=sys.stderr)
        sys.exit(1)

    dest = video.with_suffix(".vtt")

    if args.dry_run:
        print(f"Would copy:\n  {args.vtt}\n  -> {dest}")
        return

    shutil.copy2(args.vtt, dest)
    print(f"Copied to {dest}")


if __name__ == "__main__":
    main()
