#!/usr/bin/env python3
"""Build a manual AI Studio experiment pack with clips, prompts, and result templates."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _settings_block(model: dict, mode: str, prompt_profile: str) -> dict:
    settings = {
        "temperature": 0.0,
        "top_p": 0.95,
        "output_length": 65536,
        "thinking_level": model.get("thinking_level", "Default"),
        "media_resolution": "High" if mode == "video" else "N/A",
        "prompt_profile": prompt_profile,
    }
    if mode == "audio":
        settings["media_resolution"] = "N/A"
    return settings


def _result_template(scene: dict, model: dict, mode: str, prompt_profile: str) -> str:
    settings = _settings_block(model, mode, prompt_profile)
    comment = [
        "<!--",
        f"scene: {scene['label']}",
        f"source_start_s: {scene['start_s']}",
        f"source_duration_s: {scene['duration_s']}",
        f"model: {model['id']}",
        f"mode: {mode}",
        f"prompt_profile: {prompt_profile}",
        f"temperature: {settings['temperature']}",
        f"media_resolution: {settings['media_resolution']}",
        f"thinking_level: {settings['thinking_level']}",
        f"top_p: {settings['top_p']}",
        f"output_length: {settings['output_length']}",
        "-->",
        "",
    ]
    body = [
        f"# {scene['label']} / {model['id']} / {mode}",
        "",
        "## Settings",
        "",
        f"- Model: `{model['display_name']}`",
        f"- Temperature: `{settings['temperature']}`",
        f"- Media resolution: `{settings['media_resolution']}`",
        f"- Thinking level: `{settings['thinking_level']}`",
        f"- Prompt profile: `{prompt_profile}`",
        "",
        "## Paste Output",
        "",
        "Paste the raw model output below this line.",
        "",
    ]
    return "\n".join(comment + body)


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
        "1. Pick the result template under `results/<scene>/<model>__<mode>__<prompt_profile>.md`.",
        "2. In AI Studio, choose the model named in that file.",
        "3. Paste the matching system prompt and user prompt from `prompts/`.",
        "4. Upload the matching `video.mp4` or `audio.mp3` clip from `scenes/<scene>/`.",
        "5. Set the controls to the values listed in the template.",
        "6. Paste the raw output back into the template file under `## Paste Output`.",
        "",
        "## Result Templates",
        "",
        "Each result file starts with a metadata comment block so settings stay attached to the pasted output.",
        "Both `spoken_only` and `spoken_plus_visual` prompt profiles are scaffolded.",
        "",
    ]
    )
    for scene in spec["scenes"]:
        lines.append(f"- `results/{scene['label']}/...`")
    lines.append("")
    return "\n".join(lines)


def build_pack(episode_dir: Path, spec_path: Path, pack_dir: Path, *, force: bool) -> None:
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
    _write_text(pack_dir / "prompts" / "spoken_only_system.txt", spoken_system)
    _write_text(pack_dir / "prompts" / "spoken_only_user.txt", spoken_user)
    _write_text(pack_dir / "prompts" / "spoken_plus_visual_system.txt", visual_system)
    _write_text(pack_dir / "prompts" / "spoken_plus_visual_user.txt", visual_user)

    resolved_scenes = []
    for scene in spec["scenes"]:
        scene_dir = pack_dir / "scenes" / scene["label"]
        scene_dir.mkdir(parents=True, exist_ok=True)
        extract_inline_video_chunk(
            str(video_path),
            str(scene_dir / "video.mp4"),
            start_s=scene["start_s"],
            duration_s=scene["duration_s"],
            fps=1.0,
            width=640,
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
        for model in spec["models"]:
            for mode in ("video", "audio"):
                prompt_profiles = ("spoken_only", "spoken_plus_visual") if mode == "video" else ("spoken_only",)
                for prompt_profile in prompt_profiles:
                    filename = f"{model['id']}__{mode}__{prompt_profile}.md"
                    _write_text(
                        result_scene_dir / filename,
                        _result_template(scene, model, mode, prompt_profile),
                    )

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
        },
    }
    _write_text(pack_dir / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    _write_text(pack_dir / "README.md", _top_readme(manifest, video_path, pack_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a manual AI Studio experiment pack.")
    parser.add_argument("--episode-dir", required=True, help="Episode workspace under samples/episodes/<slug>.")
    parser.add_argument("--scene-spec", required=True, help="JSON spec describing scenes and model scaffolding.")
    parser.add_argument("--pack-dir", required=True, help="Output experiment pack directory.")
    parser.add_argument("--force", action="store_true", help="Rebuild if the pack directory already exists.")
    args = parser.parse_args()

    build_pack(
        Path(args.episode_dir),
        Path(args.scene_spec),
        Path(args.pack_dir),
        force=args.force,
    )
    print(args.pack_dir)


if __name__ == "__main__":
    main()
