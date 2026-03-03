#!/usr/bin/env python3
"""Test Qwen3-VL vision for OCR on Japanese variety show frames.

Compares against EasyOCR results from ocr_results.jsonl.
Uses llama.cpp server with OpenAI-compatible API.

Usage:
    python pipeline/scripts/test_qwen_ocr.py
    python pipeline/scripts/test_qwen_ocr.py --frames frame_0004.jpg frame_0009.jpg
    python pipeline/scripts/test_qwen_ocr.py --all
    python pipeline/scripts/test_qwen_ocr.py --url http://localhost:8787
"""

import argparse
import base64
import json
import time
import urllib.request
from pathlib import Path

FRAMES_DIR = Path("samples/frames")
OCR_RESULTS = Path("samples/ocr_results.jsonl")

# Frames with interesting OCR content (name cards, titles, on-screen text)
DEFAULT_FRAMES = ["frame_0004.jpg", "frame_0009.jpg", "frame_0010.jpg", "frame_0020.jpg"]

SYSTEM_PROMPT = """\
You are an OCR system for Japanese TV variety shows. Extract ALL visible text from the image.
Return one line per distinct text region. Include:
- On-screen captions and telop
- Name cards (人名テロップ)
- Show titles and logos
- Any other readable text

Output the text exactly as it appears. Do not translate. Do not explain."""


def load_easyocr_results() -> dict:
    results = {}
    if OCR_RESULTS.exists():
        with open(OCR_RESULTS) as f:
            for line in f:
                entry = json.loads(line)
                results[entry["frame"]] = entry["results"]
    return results


def image_to_data_url(path: Path) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    suffix = path.suffix.lower()
    mime = {"jpg": "image/jpeg", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(suffix, "image/jpeg")
    return f"data:{mime};base64,{b64}"


def call_vision(image_path: Path, url: str) -> str:
    data_url = image_to_data_url(image_path)

    payload = json.dumps({
        "model": "qwen3-vl",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": "Extract all Japanese text visible in this image."},
                ],
            },
        ],
        "temperature": 0.1,
        "max_tokens": 512,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", nargs="+", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--url", default="http://localhost:8080")
    args = parser.parse_args()

    if args.all:
        frames = sorted(f.name for f in FRAMES_DIR.glob("*.jpg"))
    elif args.frames:
        frames = args.frames
    else:
        frames = DEFAULT_FRAMES

    easyocr = load_easyocr_results()

    print(f"Using llama.cpp server at {args.url}")
    print()

    for frame_name in frames:
        frame_path = FRAMES_DIR / frame_name
        if not frame_path.exists():
            print(f"SKIP {frame_name}: file not found")
            continue

        print(f"{'='*60}")
        print(f"Frame: {frame_name}")

        # EasyOCR baseline
        easyocr_texts = easyocr.get(frame_name, [])
        if easyocr_texts:
            print(f"EasyOCR:  {', '.join(r['text'] for r in easyocr_texts)}")
        else:
            print(f"EasyOCR:  (no text detected)")

        # Qwen3-VL vision
        t0 = time.time()
        try:
            result = call_vision(frame_path, args.url)
            elapsed = time.time() - t0
            print(f"Qwen3VL: {result}")
            print(f"Time: {elapsed:.1f}s")
        except Exception as e:
            print(f"ERROR: {e}")
        print()


if __name__ == "__main__":
    main()
