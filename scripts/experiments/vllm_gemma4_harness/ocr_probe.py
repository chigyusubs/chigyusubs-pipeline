#!/usr/bin/env python3
"""Probe E4B's OCR ability on chyron frames from the eval segments.

Sends each frame to the vLLM endpoint with a simple "read the text"
prompt at several max_soft_tokens levels (70, 280, 560). Reports what
names the model extracts vs the gold names for that segment.

Usage:
    python3 scripts/experiments/vllm_gemma4_harness/ocr_probe.py \
        --spec scripts/experiments/vllm_gemma4_harness/eval_specs/killah_kuts_s01e01.json
"""
from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_MODEL = "google/gemma-4-E4B-it"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1/chat/completions"

OCR_PROMPT = (
    "Read ALL Japanese text visible on screen — chyrons, lower-thirds, "
    "name plates, scoreboards, brackets, captions. "
    "Output each distinct text block on its own line, exactly as written. "
    "Do not translate or interpret. If no text is visible, output: (none)"
)

MST_LEVELS = [70, 280, 560]


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def post(base_url: str, payload: dict, timeout: int = 120) -> tuple[dict, float]:
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


def check_names(text: str, names: list[str]) -> dict[str, bool]:
    return {n: n in text for n in names}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--out", default="")
    ap.add_argument("--mst", default="",
                    help="comma-separated mst levels (default: 70,280,560)")
    ap.add_argument("--stride", type=int, default=2,
                    help="frame stride (1=all, 2=every other)")
    args = ap.parse_args()

    mst_levels = [int(x) for x in args.mst.split(",")] if args.mst else MST_LEVELS

    spec_path = Path(args.spec).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec_dir = spec_path.parent

    print(f"Spec: {spec_path.name}  ({spec['n_segments']} segments)")
    print(f"Model: {args.model}")
    print(f"MST levels: {mst_levels}, frame stride: {args.stride}")

    results: list[dict] = []

    for seg in spec["segments"]:
        name = seg["name"]
        gold_names = seg["names"]
        frames_dir = (spec_dir / seg["frames_dir_rel"]).resolve()
        frames = sorted(frames_dir.glob("*.jpg"))[::args.stride]

        print(f"\n### {name}  gold_names={gold_names}")
        print(f"    frames: {len(frames)}")

        seg_result: dict = {
            "seg": name,
            "gold_names": gold_names,
            "n_frames": len(frames),
            "mst_results": {},
        }

        for mst in mst_levels:
            # Send all picked frames in one request
            content: list[dict] = []
            for f in frames:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64(f)}"},
                })
            content.append({"type": "text", "text": OCR_PROMPT})

            payload = {
                "model": args.model,
                "max_tokens": 512,
                "temperature": 0.0,
                "messages": [{"role": "user", "content": content}],
                "mm_processor_kwargs": {"max_soft_tokens": mst},
            }

            body, wall = post(args.base_url, payload)
            if "error" in body:
                print(f"  mst={mst}: ERROR {body['error']}")
                seg_result["mst_results"][mst] = {"error": body["error"]}
                continue

            text = body["choices"][0]["message"]["content"]
            usage = body.get("usage", {})
            hits = check_names(text, gold_names)
            n_hit = sum(hits.values())
            n_total = len(gold_names)

            seg_result["mst_results"][mst] = {
                "text": text,
                "wall": round(wall, 3),
                "prompt_tokens": usage.get("prompt_tokens"),
                "name_hits": hits,
                "name_recall": f"{n_hit}/{n_total}",
            }

            print(f"  mst={mst:>3}: names={n_hit}/{n_total} {hits}  "
                  f"wall={wall:.2f}s  prompt_tok={usage.get('prompt_tokens')}")
            # Show first 200 chars of output
            for line in text.strip().split("\n")[:6]:
                print(f"    | {line[:120]}")

        results.append(seg_result)

    # Summary
    print("\n\n=========== OCR PROBE SUMMARY ===========")
    for mst in mst_levels:
        total_names = 0
        total_hits = 0
        for r in results:
            mr = r["mst_results"].get(mst, {})
            hits = mr.get("name_hits", {})
            total_names += len(hits)
            total_hits += sum(hits.values())
        pct = round(100 * total_hits / total_names) if total_names else 0
        print(f"  mst={mst:>3}: {total_hits}/{total_names} ({pct}%) name recall")

    if args.out:
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({
                "spec": str(spec_path),
                "model": args.model,
                "mst_levels": mst_levels,
                "frame_stride": args.stride,
                "results": results,
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
