#!/usr/bin/env python3
"""Build an eval spec from CTC-aligned gold transcripts.

Given an episode slug and a target count, this picks hard segments from
the episode's CTC words file — segments containing katakana jargon and
proper nouns — and extracts tight audio+frame windows around each. The
resulting spec.json is consumed by run_bench.py.

Usage:
    python3 build_eval_spec.py \\
        --episode killah_kuts_s01e01 \\
        --video /path/to/source.mkv \\
        --out eval_specs/killah_kuts_s01e01.json \\
        --count 10

The CTC file is expected at:
    samples/episodes/<slug>/transcription/<slug>_*_ctc_words.json

Frames are extracted at 1 fps × 8 s = 8 frames per window. This matches
config C_vision_primed_1fps280 and is close to the sweet spot we found
in the 2026-04-11 session.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

KATAKANA_RE = re.compile(r"[\u30A0-\u30FF]{3,}")

# Default named-entity list. Override with --names for other episodes.
DEFAULT_NAMES = [
    "しんいち", "みなみかわ", "設楽", "大崎", "道尾", "道雄",
    "バナナマン", "しんち",
]


@dataclass
class Segment:
    name: str
    seg_start: float
    seg_end: float
    window_start: float
    window_dur: float
    text: str
    kata: list[str]
    names: list[str]
    wav_rel: str = ""
    frames_dir_rel: str = ""


def find_ctc_file(episode_dir: Path) -> Path:
    transcription_dir = episode_dir / "transcription"
    candidates = sorted(transcription_dir.glob("*_ctc_words.json"))
    if not candidates:
        raise FileNotFoundError(f"no *_ctc_words.json in {transcription_dir}")
    # Prefer non-repaired variant for cleaner gold
    for c in candidates:
        if "repair" not in c.name:
            return c
    return candidates[0]


def pick_segments(ctc_segments: list[dict], names: list[str],
                  count: int, min_dur: float = 1.0,
                  max_dur: float = 4.0) -> list[tuple]:
    scored = []
    for s in ctc_segments:
        dur = s["end"] - s["start"]
        if dur < min_dur or dur > max_dur:
            continue
        kata_all = KATAKANA_RE.findall(s["text"])
        kata = sorted(set(kata_all))  # dedupe — don't reward repetition
        name_hits = [n for n in names if n in s["text"]]
        if not kata and not name_hits:
            continue
        score = len(kata) * 2 + len(name_hits) * 3
        scored.append((score, s["start"], s["end"], s["text"],
                       kata, name_hits))
    scored.sort(reverse=True)
    # De-duplicate segments whose start/end overlap to avoid picking
    # the same scene twice. Skip anything within 5s of a prior pick.
    chosen: list[tuple] = []
    for row in scored:
        _, st, en, *_ = row
        if any(abs(st - cst) < 5.0 for _, cst, *_ in chosen):
            continue
        chosen.append(row)
        if len(chosen) >= count:
            break
    return chosen


def extract_window(video: Path, out_wav: Path, out_frames_dir: Path,
                   start_s: float, dur_s: float, fps: float = 1.0) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    out_frames_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{start_s:.2f}", "-i", str(video),
         "-t", f"{dur_s:.2f}", "-vn", "-ac", "1", "-ar", "16000",
         str(out_wav)],
        capture_output=True, check=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{start_s:.2f}", "-i", str(video),
         "-t", f"{dur_s:.2f}", "-vf", f"fps={fps},scale=-2:480",
         "-q:v", "3", str(out_frames_dir / "f_%02d.jpg")],
        capture_output=True, check=True,
    )
    # co-located mp4 clip for video_url configs (J/K). 480p, mono 16k
    # audio muxed in — same quality as the wav/frames above so the
    # only variable vs hand-picked frames is the packing.
    out_mp4 = out_wav.with_suffix(".mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{start_s:.2f}", "-i", str(video),
         "-t", f"{dur_s:.2f}", "-vf", "scale=-2:480",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
         "-ac", "1", "-ar", "16000", "-c:a", "aac", "-b:a", "64k",
         str(out_mp4)],
        capture_output=True, check=True,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episode", required=True,
                   help="episode slug under samples/episodes/")
    p.add_argument("--video", required=True, help="source video path")
    p.add_argument("--out", required=True,
                   help="output eval spec JSON path")
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--window-dur", type=float, default=8.0,
                   help="window duration in seconds (default 8)")
    p.add_argument("--fps", type=float, default=1.0,
                   help="frame sampling rate (default 1 fps)")
    p.add_argument("--min-dur", type=float, default=1.0)
    p.add_argument("--max-dur", type=float, default=4.0)
    p.add_argument("--names", nargs="*", default=DEFAULT_NAMES)
    p.add_argument("--cache-dir", default="",
                   help="dir to write wav+frames (default: alongside spec)")
    args = p.parse_args()

    episode_dir = REPO_ROOT / "samples" / "episodes" / args.episode
    ctc_file = find_ctc_file(episode_dir)
    ctc = json.load(open(ctc_file))

    chosen = pick_segments(
        ctc, args.names, count=args.count,
        min_dur=args.min_dur, max_dur=args.max_dur,
    )
    print(f"Picked {len(chosen)} segments from {ctc_file.name}")

    spec_path = Path(args.out).resolve()
    cache_dir = (Path(args.cache_dir).resolve() if args.cache_dir
                 else spec_path.parent / (spec_path.stem + "_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    segments: list[Segment] = []
    for i, (_, st, en, text, kata, names) in enumerate(chosen, 1):
        mid = (st + en) / 2
        w_start = max(0, mid - args.window_dur / 2)
        name = f"seg{i:02d}_{int(st)}s"
        wav_path = cache_dir / f"{name}.wav"
        frames_path = cache_dir / f"{name}_frames"
        extract_window(Path(args.video), wav_path, frames_path,
                       w_start, args.window_dur, fps=args.fps)
        segments.append(Segment(
            name=name,
            seg_start=st,
            seg_end=en,
            window_start=w_start,
            window_dur=args.window_dur,
            text=text,
            kata=kata,
            names=names,
            wav_rel=str(wav_path.relative_to(spec_path.parent)),
            frames_dir_rel=str(frames_path.relative_to(spec_path.parent)),
        ))
        print(f"  {name}  [{st:.1f}-{en:.1f}]  {text[:50]}")

    spec = {
        "episode": args.episode,
        "video": args.video,
        "ctc_file": str(ctc_file),
        "window_dur": args.window_dur,
        "fps": args.fps,
        "n_segments": len(segments),
        "segments": [asdict(s) for s in segments],
    }
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(
        json.dumps(spec, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {spec_path}")


if __name__ == "__main__":
    main()
