#!/usr/bin/env python3
"""Create an episode workspace from one or more media files."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chigyusubs.metadata import finish_run, metadata_path, start_run, write_metadata
from chigyusubs.paths import (
    DEFAULT_EPISODES_ROOT,
    ensure_episode_layout,
    find_episode_dir_from_path,
    slugify_episode_name,
)


def import_media(src: Path, episode_dir: Path, mode: str) -> Path:
    dst = episode_dir / "source" / src.name
    if src.resolve() == dst.resolve():
        return dst
    if dst.exists():
        raise SystemExit(f"Destination already exists: {dst}")
    if mode == "move":
        shutil.move(str(src), str(dst))
    elif mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        dst.hardlink_to(src)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return dst


def extract_frames(video_path: Path, frames_dir: Path, fps: float, qscale: int) -> int:
    frames_dir.mkdir(parents=True, exist_ok=True)
    pattern = frames_dir / "frame_%05d.jpg"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"fps={fps}",
            "-qscale:v",
            str(qscale),
            str(pattern),
        ],
        capture_output=True,
        check=True,
    )
    return len(list(frames_dir.glob("*.jpg")))


def main():
    parser = argparse.ArgumentParser(description="Create episode workspace(s) from media files.")
    parser.add_argument("media", nargs="+", help="Media file(s) to import.")
    parser.add_argument("--episodes-root", default=str(DEFAULT_EPISODES_ROOT))
    parser.add_argument("--slug", default="", help="Episode slug override (single input only).")
    parser.add_argument(
        "--mode",
        choices=["move", "copy", "hardlink"],
        default="move",
        help="How to place media into source/ (default: move).",
    )
    parser.add_argument("--extract-frames", action="store_true", help="Extract fixed-rate JPG frames after import.")
    parser.add_argument("--fps", type=float, default=0.5, help="Frame extraction rate (default: 0.5 fps).")
    parser.add_argument("--qscale", type=int, default=2, help="JPEG qscale for ffmpeg (default: 2).")
    args = parser.parse_args()

    if args.slug and len(args.media) != 1:
        raise SystemExit("--slug can only be used with a single media input.")

    episodes_root = Path(args.episodes_root)
    episodes_root.mkdir(parents=True, exist_ok=True)

    for raw_media in args.media:
        run = start_run("init_episode")
        src = Path(raw_media)
        if not src.exists():
            raise SystemExit(f"Media not found: {src}")

        existing_episode_dir = find_episode_dir_from_path(src)
        if args.slug:
            episode_dir = ensure_episode_layout(episodes_root / args.slug)
            slug = args.slug
        elif existing_episode_dir:
            episode_dir = ensure_episode_layout(existing_episode_dir)
            slug = episode_dir.name
        else:
            slug = slugify_episode_name(src.stem)
            episode_dir = ensure_episode_layout(episodes_root / slug)
        imported_media = import_media(src, episode_dir, args.mode)

        outputs = {
            "episode_dir": str(episode_dir),
            "source_media": str(imported_media),
        }
        stats = {}

        if args.extract_frames:
            frames_dir = episode_dir / "frames" / "raw_2s"
            frame_count = extract_frames(imported_media, frames_dir, args.fps, args.qscale)
            outputs["frames_dir"] = str(frames_dir)
            stats["frames_written"] = frame_count

        metadata = finish_run(
            run,
            inputs={"original_media": str(src)},
            outputs=outputs,
            init_settings={
                "mode": args.mode,
                "slug": slug,
                "extract_frames": args.extract_frames,
                "fps": args.fps,
                "qscale": args.qscale,
            },
            stats=stats,
        )
        meta_out = episode_dir / "logs" / "init_episode"
        write_metadata(meta_out, metadata)
        print(f"Episode ready: {episode_dir}")
        print(f"Metadata written: {metadata_path(meta_out)}")


if __name__ == "__main__":
    main()
