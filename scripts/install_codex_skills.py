#!/usr/bin/env python3
"""Install repo-tracked Codex skills into a live Codex home."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLS_ROOT = REPO_ROOT / "codex" / "skills"
DEFAULT_DEST = Path.home() / ".codex" / "skills"
VALIDATOR = Path.home() / ".codex" / "skills" / ".system" / "skill-creator" / "scripts" / "quick_validate.py"


def tracked_skills() -> list[Path]:
    if not SKILLS_ROOT.exists():
        return []
    return sorted(
        path for path in SKILLS_ROOT.iterdir()
        if path.is_dir() and (path / "SKILL.md").exists()
    )


def validate_skill(path: Path) -> None:
    if not (path / "SKILL.md").exists():
        raise SystemExit(f"Missing SKILL.md: {path}")
    if VALIDATOR.exists():
        subprocess.run([sys.executable, str(VALIDATOR), str(path)], check=True)
    else:
        print(f"warning: validator not found, skipping structured validation for {path.name}", file=sys.stderr)


def install_skill(src: Path, dest_root: Path, *, force: bool) -> Path:
    dest = dest_root / src.name
    if dest.exists():
        if not force:
            raise SystemExit(f"Destination already exists, rerun with --force: {dest}")
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description="Install repo-tracked Codex skills into ~/.codex/skills.")
    parser.add_argument("--skill", action="append", default=[], help="Skill name to install. May be passed multiple times.")
    parser.add_argument("--dest", default=str(DEFAULT_DEST), help="Destination skills directory. Defaults to ~/.codex/skills.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing destination skills.")
    args = parser.parse_args()

    available = {path.name: path for path in tracked_skills()}
    if not available:
        raise SystemExit(f"No tracked skills found under {SKILLS_ROOT}")

    names = args.skill or sorted(available)
    unknown = [name for name in names if name not in available]
    if unknown:
        raise SystemExit(f"Unknown skill(s): {', '.join(unknown)}")

    dest_root = Path(args.dest).expanduser()
    dest_root.mkdir(parents=True, exist_ok=True)

    installed: list[Path] = []
    for name in names:
        src = available[name]
        validate_skill(src)
        installed.append(install_skill(src, dest_root, force=args.force))

    print("Installed skills:")
    for path in installed:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
