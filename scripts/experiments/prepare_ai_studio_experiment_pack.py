#!/usr/bin/env python3
"""Build a manual AI Studio experiment pack with clips, prompts, and notes."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from chigyusubs.audio import extract_audio_chunk, extract_inline_video_chunk
from transcribe_gemini import build_prompt


VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm"}


def _find_source_video(episode_dir: Path) -> Path:
    source_dir = episode_dir / "source"
    candidates = sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS
    )
    if not candidates:
        raise SystemExit(f"No source video found under {source_dir}")
    return candidates[0]


def _system_and_user_prompt(*, include_visual_brackets: bool) -> tuple[str, str]:
    prompt = build_prompt([], prev_context=None, include_visual_brackets=include_visual_brackets)
    lines = prompt.splitlines()
    if lines and lines[-1].strip() == "Transcribe the audio now.":
        lines = lines[:-1]
    system_prompt = "\n".join(lines).strip()
    user_prompt = "Transcribe the attached media now."
    return system_prompt, user_prompt


def _ocr_only_prompts() -> tuple[str, str]:
    system_prompt = "\n".join(
        [
            "You are extracting visible on-screen text from a Japanese variety/comedy show clip.",
            "",
            "Instructions:",
            "1. Extract only text that is visibly present on screen.",
            "2. Do NOT transcribe spoken dialogue unless the same words are also visibly shown on screen.",
            "3. Output ONLY plain text. Do NOT use JSON, markdown, commentary, timestamps, or speaker labels.",
            "4. Put each extracted item on its own line as `[画面: ...]`.",
            "5. Preserve exact visible wording when readable, including kanji, kana, katakana, digits, and Latin text.",
            "6. Prefer exact text over paraphrase or summary.",
            "7. If text is only partially readable, output only the confidently readable portion. Do NOT guess missing characters.",
            "8. Ignore scenery, faces, clothing, and actions unless there is visible text attached to them.",
            "9. Prioritize cast/name cards, rule cards, mission prompts, labels, counters, maps, and signs.",
            "10. Keep duplicate lines minimal. Do not repeat the same visible text unless the screen state clearly changed.",
            "11. If there is no meaningful readable on-screen text, output nothing.",
            "",
            "The goal is reusable visual-text extraction, not speech transcription.",
        ]
    )
    user_prompt = "\n".join(
        [
            "Extract only meaningful readable on-screen text from this clip.",
            "",
            "Return one line per item as `[画面: ...]`.",
            "",
            "Do not transcribe speech unless the same words are visibly shown on screen.",
            "Do not summarize.",
            "Do not guess unreadable characters.",
        ]
    )
    return system_prompt, user_prompt


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _scene_readme(scene: dict) -> str:
    lines = [
        f"# {scene['label']}",
        "",
        f"- Start: `{scene['start_s']}` seconds",
        f"- Duration: `{scene['duration_s']}` seconds",
        "",
        "## Why This Scene",
        "",
        scene["description"].strip(),
        "",
    ]
    focus = scene.get("focus") or []
    if focus:
        lines.extend(["## Focus", ""])
        lines.extend(f"- {item}" for item in focus)
        lines.append("")
    if scene.get("notes"):
        lines.extend(["## Notes", "", scene["notes"].strip(), ""])
    return "\n".join(lines)


def _top_readme(spec: dict, video_path: Path, pack_dir: Path) -> str:
    lines = [
        f"# {spec['pack_name']}",
        "",
        spec.get("description", "").strip(),
        "",
        "## Source",
        "",
        f"- Episode dir: `{spec['episode_dir']}`",
        f"- Source video: `{video_path}`",
        "",
        "## Prompt Files",
        "",
        "- `prompts/spoken_only_system.txt`",
        "- `prompts/spoken_only_user.txt`",
        "- `prompts/spoken_plus_visual_system.txt`",
        "- `prompts/spoken_plus_visual_user.txt`",
        "- `prompts/ocr_only_system.txt`",
        "- `prompts/ocr_only_user.txt`",
        "",
        "## Scene Clips",
        "",
    ]
    for scene in spec["scenes"]:
        lines.append(f"- `scenes/{scene['label']}/video.mp4`")
        lines.append(f"- `scenes/{scene['label']}/audio.mp3`")
    lines.extend(
        [
            "",
            "## Recommended Manual AI Studio Procedure",
            "",
            "1. Pick a scene clip under `scenes/<scene>/`.",
            "2. Choose a model, media mode, and prompt profile from `manifest.json`.",
            "3. In AI Studio, choose the model and set the matching controls.",
            "4. Paste the matching system prompt and user prompt from `prompts/`.",
            "5. Upload the matching `video.mp4` or `audio.mp3` clip from `scenes/<scene>/`.",
            "6. If the output is worth keeping, save it under `results/<scene>/<model>__<mode>__<prompt_profile>.md`.",
            "",
            "Suggested naming examples:",
            "",
            "- `results/<scene>/gemini-3.1-pro-preview__low__video__spoken_only.md`",
            "- `results/<scene>/gemini-3.1-pro-preview__low__video__spoken_plus_visual.md`",
            "- `results/<scene>/gemini-3.1-pro-preview__low__audio__spoken_only.md`",
            "",
            "Notes:",
            "",
            "- `spoken_only` and `spoken_plus_visual` are for video; `spoken_only` also works for audio.",
            "- `ocr_only` is for video clips only.",
            "- The pack no longer pre-creates empty result files; save only the runs you actually want to keep.",
            "",
        ]
    )
    for scene in spec["scenes"]:
        lines.append(f"- `results/{scene['label']}/...`")
    lines.append("")
    return "\n".join(lines)


def build_pack(
    episode_dir: Path,
    spec_path: Path,
    pack_dir: Path,
    *,
    force: bool,
    video_width: int | None = None,
) -> None:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["episode_dir"] = str(episode_dir)
    video_path = _find_source_video(episode_dir)

    if pack_dir.exists():
        if not force:
            raise SystemExit(f"Pack directory already exists: {pack_dir} (use --force to rebuild)")
        shutil.rmtree(pack_dir)

    pack_dir.mkdir(parents=True, exist_ok=True)
    (pack_dir / "prompts").mkdir(parents=True, exist_ok=True)
    (pack_dir / "scenes").mkdir(parents=True, exist_ok=True)
    (pack_dir / "results").mkdir(parents=True, exist_ok=True)

    spoken_system, spoken_user = _system_and_user_prompt(include_visual_brackets=False)
    visual_system, visual_user = _system_and_user_prompt(include_visual_brackets=True)
    ocr_system, ocr_user = _ocr_only_prompts()
    _write_text(pack_dir / "prompts" / "spoken_only_system.txt", spoken_system)
    _write_text(pack_dir / "prompts" / "spoken_only_user.txt", spoken_user)
    _write_text(pack_dir / "prompts" / "spoken_plus_visual_system.txt", visual_system)
    _write_text(pack_dir / "prompts" / "spoken_plus_visual_user.txt", visual_user)
    _write_text(pack_dir / "prompts" / "ocr_only_system.txt", ocr_system)
    _write_text(pack_dir / "prompts" / "ocr_only_user.txt", ocr_user)

    resolved_scenes = []
    for scene in spec["scenes"]:
        scene_dir = pack_dir / "scenes" / scene["label"]
        scene_dir.mkdir(parents=True, exist_ok=True)
        scene_video_width = video_width if video_width is not None else scene.get("video_width")
        extract_inline_video_chunk(
            str(video_path),
            str(scene_dir / "video.mp4"),
            start_s=scene["start_s"],
            duration_s=scene["duration_s"],
            fps=1.0,
            width=scene_video_width,
            audio_bitrate="24k",
            crf=36,
        )
        extract_audio_chunk(
            str(video_path),
            str(scene_dir / "audio.mp3"),
            start_s=scene["start_s"],
            duration_s=scene["duration_s"],
        )
        scene_meta = dict(scene)
        scene_meta["video_path"] = str((scene_dir / "video.mp4").relative_to(pack_dir))
        scene_meta["audio_path"] = str((scene_dir / "audio.mp3").relative_to(pack_dir))
        _write_text(scene_dir / "README.md", _scene_readme(scene))
        _write_text(scene_dir / "scene.json", json.dumps(scene_meta, ensure_ascii=False, indent=2))
        resolved_scenes.append(scene_meta)

        result_scene_dir = pack_dir / "results" / scene["label"]
        result_scene_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "pack_name": spec["pack_name"],
        "description": spec.get("description", ""),
        "episode_dir": str(episode_dir),
        "source_video": str(video_path),
        "spec_path": str(spec_path),
        "models": spec["models"],
        "scenes": resolved_scenes,
        "prompts": {
            "spoken_only_system": "prompts/spoken_only_system.txt",
            "spoken_only_user": "prompts/spoken_only_user.txt",
            "spoken_plus_visual_system": "prompts/spoken_plus_visual_system.txt",
            "spoken_plus_visual_user": "prompts/spoken_plus_visual_user.txt",
            "ocr_only_system": "prompts/ocr_only_system.txt",
            "ocr_only_user": "prompts/ocr_only_user.txt",
        },
    }
    _write_text(pack_dir / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    _write_text(pack_dir / "README.md", _top_readme(manifest, video_path, pack_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a manual AI Studio experiment pack.")
    parser.add_argument("--episode-dir", required=True, help="Episode workspace under samples/episodes/<slug>.")
    parser.add_argument("--scene-spec", required=True, help="JSON spec describing scenes and model scaffolding.")
    parser.add_argument("--pack-dir", required=True, help="Output experiment pack directory.")
    parser.add_argument(
        "--video-width",
        type=int,
        default=None,
        help="Optional output width for all extracted video clips. Default keeps source width.",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild if the pack directory already exists.")
    args = parser.parse_args()

    build_pack(
        Path(args.episode_dir),
        Path(args.scene_spec),
        Path(args.pack_dir),
        force=args.force,
        video_width=args.video_width,
    )
    print(args.pack_dir)


if __name__ == "__main__":
    main()
