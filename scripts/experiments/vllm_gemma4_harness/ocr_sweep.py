#!/usr/bin/env python3
"""Whole-episode OCR sweep via E4B at mst=140 (OCR-aggregate mode).

Extracts frames at a low fps, batches them into multi-image requests,
and asks the model to transcribe every on-screen text block it sees.
Raw per-batch outputs + a deduped flat telop list are written to
``--out``. The deduped list is stage-2 input for the glossary-build
step (not implemented here — inspect the output first).

Usage:
    python3 ocr_sweep.py \\
        --video samples/episodes/killah_kuts_s01e01/source/foo.mkv \\
        --episode killah_kuts_s01e01 \\
        --out scripts/experiments/vllm_gemma4_harness/results/killah_kuts_s01e01_ocr_sweep.json
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_MODEL = "google/gemma-4-E4B-it"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1/chat/completions"

OCR_SYSTEM = (
    "You are reading on-screen Japanese text from frames of a Japanese "
    "variety show. Telops (chyrons, lower-thirds), name plates, "
    "scoreboards, brackets, title cards, and decorative gag captions "
    "all count. Do NOT transcribe spoken dialogue — only text physically "
    "printed on the frame."
)

OCR_PROMPT = (
    "For the frames in this batch, list the distinct Japanese on-screen "
    "text blocks you can read.\n"
    "\n"
    "Rules:\n"
    "- Japanese verbatim, exactly as written (preserve kanji / hiragana / "
    "katakana / punctuation as shown).\n"
    "- One text block per line.\n"
    "- DEDUPE across all frames in the batch: if the same text appears "
    "in multiple frames, list it ONCE. Never repeat a line.\n"
    "- No translation, no romanization, no commentary.\n"
    "- Cap the output at 40 distinct lines. If more than 40 blocks are "
    "visible, prioritize name plates, scoreboards, and title cards over "
    "small decorative text.\n"
    "- If the batch contains no readable on-screen text at all, output "
    "the single word: (none)"
)


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def extract_frames(video: Path, out_dir: Path, fps: float,
                   height: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Only (re-)extract if the dir looks empty or stale
    existing = sorted(out_dir.glob("f_*.jpg"))
    if existing:
        return existing
    cmd = [
        "ffmpeg", "-y", "-i", str(video),
        "-vf", f"fps={fps},scale=-2:{height}",
        "-q:v", "3", str(out_dir / "f_%05d.jpg"),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return sorted(out_dir.glob("f_*.jpg"))


def post(base_url: str, payload: dict, timeout: int = 600) -> tuple[dict, float]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url, data=data,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
        return body, time.monotonic() - t0
    except urllib.error.HTTPError as e:
        return ({"error": f"HTTP {e.code}",
                 "body": e.read().decode("utf-8", "replace")[:500]},
                time.monotonic() - t0)


def run_batch(*, base_url: str, model: str, frames: list[Path],
              mst: int, max_tokens: int,
              temperature: float, top_p: float, top_k: int,
              frequency_penalty: float,
              backend: str) -> tuple[str, dict, float]:
    content: list[dict] = [
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64,{b64(f)}"}}
        for f in frames
    ]
    content.append({"type": "text", "text": OCR_PROMPT})
    payload: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "frequency_penalty": frequency_penalty,
        "messages": [
            {"role": "system", "content": OCR_SYSTEM},
            {"role": "user", "content": content},
        ],
    }
    if backend == "vllm":
        payload["mm_processor_kwargs"] = {"max_soft_tokens": mst}
    else:
        # Gemma 4's chat template defaults to thinking-on, which burns
        # the whole completion budget on a reasoning trace before any
        # OCR output appears.
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    body, wall = post(base_url, payload)
    if "error" in body:
        return "", {"error": body["error"], "body": body.get("body", "")}, wall
    text = body["choices"][0]["message"]["content"]
    usage = body.get("usage", {})
    return text, usage, wall


def dedupe_lines(lines: list[str]) -> list[str]:
    """Drop whitespace-only lines and exact duplicates, preserving
    first-seen order. Trims surrounding whitespace but keeps internal
    punctuation as the model emitted it (callers can normalize further).
    """
    seen: set[str] = set()
    out: list[str] = []
    for raw in lines:
        s = raw.strip()
        if not s or s == "(none)":
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def write_result(
    *,
    out_path: Path,
    video: Path,
    args: argparse.Namespace,
    frames: list[Path],
    batch_results: list[dict],
    total_wall: float,
) -> None:
    all_lines: list[str] = []
    for batch in batch_results:
        all_lines.extend(batch.get("lines", []))
    deduped = dedupe_lines(all_lines)
    out_path.write_text(
        json.dumps({
            "video": str(video),
            "episode": args.episode,
            "model": args.model,
            "fps": args.fps,
            "height": args.height,
            "batch_size": args.batch_size,
            "mst": args.mst,
            "backend": args.backend,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "frequency_penalty": args.frequency_penalty,
            "n_frames": len(frames),
            "n_batches": len(batch_results),
            "total_wall": round(total_wall, 1),
            "unique_lines": deduped,
            "batches": batch_results,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--episode", required=True,
                    help="episode slug, used for frame cache path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=0.25,
                    help="frame sample rate (default 0.25 = every 4s)")
    ap.add_argument("--height", type=int, default=720,
                    help="frame height in px (default 720)")
    ap.add_argument("--batch-size", type=int, default=30)
    ap.add_argument("--mst", type=int, default=140)
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="Gemma 4 recommended sampling temperature.")
    ap.add_argument("--top-p", type=float, default=0.95,
                    help="Gemma 4 recommended top-p.")
    ap.add_argument("--top-k", type=int, default=64,
                    help="Gemma 4 recommended top-k.")
    ap.add_argument("--frequency-penalty", type=float, default=0.0,
                    help="Optional anti-loop knob. Gemma 4's recommended "
                         "sampling (temp=1.0/top_p=0.95/top_k=64) already "
                         "breaks the static-scene repetition loops we saw "
                         "at temp=0; only raise this if loops still appear.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--backend", choices=["vllm", "llama-cpp"], default="vllm")
    ap.add_argument("--frame-cache-dir", default="",
                    help="where to store extracted frames "
                         "(default: alongside --out under _frames/)")
    ap.add_argument("--limit-batches", type=int, default=0,
                    help="stop after N batches (0 = no limit); for testing")
    args = ap.parse_args()

    video = Path(args.video).resolve()
    if not video.exists():
        raise SystemExit(f"video not found: {video}")

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.frame_cache_dir:
        frame_dir = Path(args.frame_cache_dir).resolve()
    else:
        frame_dir = out_path.parent / f"{args.episode}_ocr_frames_fps{args.fps}_h{args.height}"

    print(f"Video:        {video.name}")
    print(f"Episode:      {args.episode}")
    print(f"Frame cache:  {frame_dir}")
    print(f"Extracting frames at {args.fps} fps, height {args.height}px...")
    t0 = time.monotonic()
    frames = extract_frames(video, frame_dir, args.fps, args.height)
    print(f"  {len(frames)} frames ({time.monotonic() - t0:.1f}s)")

    batches: list[list[Path]] = [
        frames[i:i + args.batch_size]
        for i in range(0, len(frames), args.batch_size)
    ]
    if args.limit_batches > 0:
        batches = batches[: args.limit_batches]
    print(f"Batches:      {len(batches)} × up to {args.batch_size} frames"
          f" (mst={args.mst})")

    batch_results: list[dict] = []
    done_batches: set[int] = set()
    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        if (
            existing.get("episode") == args.episode
            and existing.get("fps") == args.fps
            and existing.get("height") == args.height
            and existing.get("batch_size") == args.batch_size
        ):
            batch_results = existing.get("batches", [])
            done_batches = {
                int(b["batch_idx"])
                for b in batch_results
                if "batch_idx" in b and "error" not in b
            }
            if done_batches:
                print(f"Resuming: {len(done_batches)} completed batches already saved")
        else:
            raise SystemExit(
                f"Refusing to resume {out_path}: settings do not match. "
                "Move the old output or use a new --out path."
            )

    all_lines: list[str] = []
    for b in batch_results:
        all_lines.extend(b.get("lines", []))
    t_sweep = time.monotonic()
    for i, batch in enumerate(batches, 1):
        if i in done_batches:
            continue
        # Describe the time range this batch covers (frames are evenly
        # sampled, so frame index → time = index / fps).
        start_idx = frames.index(batch[0])
        end_idx = frames.index(batch[-1])
        t_start = start_idx / args.fps
        t_end = end_idx / args.fps
        print(f"\n--- batch {i}/{len(batches)}  "
              f"frames {start_idx + 1}..{end_idx + 1}  "
              f"({t_start:.0f}s - {t_end:.0f}s) ---")

        text, usage, wall = run_batch(
            base_url=args.base_url, model=args.model,
            frames=batch, mst=args.mst, max_tokens=args.max_tokens,
            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
            frequency_penalty=args.frequency_penalty,
            backend=args.backend,
        )
        if "error" in usage:
            print(f"  ERROR {usage['error']}  {usage.get('body', '')[:200]}")
            batch_results.append({
                "batch_idx": i,
                "frames": [f.name for f in batch],
                "t_start": t_start,
                "t_end": t_end,
                "error": usage["error"],
                "wall": wall,
            })
            write_result(
                out_path=out_path,
                video=video,
                args=args,
                frames=frames,
                batch_results=batch_results,
                total_wall=time.monotonic() - t_sweep,
            )
            continue

        lines = [ln for ln in text.splitlines() if ln.strip()]
        all_lines.extend(lines)
        batch_results.append({
            "batch_idx": i,
            "frames": [f.name for f in batch],
            "t_start": t_start,
            "t_end": t_end,
            "wall": round(wall, 3),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "text": text,
            "lines": lines,
        })
        print(f"  wall={wall:.2f}s  "
              f"prompt_tok={usage.get('prompt_tokens')}  "
              f"completion_tok={usage.get('completion_tokens')}  "
              f"lines={len(lines)}")
        # Peek at first few lines so the run is legible in the terminal
        for ln in lines[:6]:
            print(f"    | {ln[:120]}")
        if len(lines) > 6:
            print(f"    | ... (+{len(lines) - 6} more)")
        write_result(
            out_path=out_path,
            video=video,
            args=args,
            frames=frames,
            batch_results=batch_results,
            total_wall=time.monotonic() - t_sweep,
        )

    total_wall = time.monotonic() - t_sweep
    deduped = dedupe_lines(all_lines)

    print(f"\n==== OCR sweep done in {total_wall:.1f}s ====")
    print(f"  batches:        {len(batch_results)}")
    print(f"  raw lines:      {len(all_lines)}")
    print(f"  unique lines:   {len(deduped)}")

    write_result(
        out_path=out_path,
        video=video,
        args=args,
        frames=frames,
        batch_results=batch_results,
        total_wall=total_wall,
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
