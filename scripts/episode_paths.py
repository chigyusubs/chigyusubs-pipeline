from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


DEFAULT_EPISODES_ROOT = Path("samples/episodes")
LEGACY_DEFAULT_VIDEO = Path("samples/WEDNESDAY_DOWNTOWN_2024-01-24 _#363.mp4")


def _slugify_episode_name(name: str) -> str:
    text = name.strip().lower()
    text = text.replace("#", "")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "episode"


def find_episode_dir_from_path(path: Path) -> Optional[Path]:
    parts = path.parts
    lower_parts = [p.lower() for p in parts]
    if "episodes" not in lower_parts:
        return None
    idx = lower_parts.index("episodes")
    if idx + 1 >= len(parts):
        return None
    return Path(*parts[: idx + 2])


def infer_episode_dir_from_video(video_path: Path) -> Path:
    found = find_episode_dir_from_path(video_path)
    if found:
        return found
    return DEFAULT_EPISODES_ROOT / _slugify_episode_name(video_path.stem)


def find_latest_episode_video(root: Path = DEFAULT_EPISODES_ROOT) -> Optional[Path]:
    videos = sorted(root.glob("*/source/*.mp4"))
    if videos:
        return videos[-1]
    if LEGACY_DEFAULT_VIDEO.exists():
        return LEGACY_DEFAULT_VIDEO
    return None


def find_latest_episode_dir(root: Path = DEFAULT_EPISODES_ROOT) -> Optional[Path]:
    dirs = sorted(p for p in root.glob("*") if p.is_dir())
    return dirs[-1] if dirs else None

