"""Minimal repo-local .env loading for CLI scripts.

This avoids a python-dotenv dependency while making Codex/terminal runs pick up
the repo's `.env` file consistently.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_repo_env(*, override: bool = False) -> Path | None:
    """Load KEY=VALUE lines from the repo's `.env` file into ``os.environ``.

    Existing environment variables win unless ``override`` is true.
    Returns the loaded path, or ``None`` if no `.env` file was found.
    """
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value
    return env_path
