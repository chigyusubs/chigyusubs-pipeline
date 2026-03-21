#!/usr/bin/env python3
"""Test stable-ts text-conditioned alignment on known weak-anchor lines.

Extracts narrow audio clips around weak-anchor segments from the CTC alignment
and runs stable_whisper.model.align() to see if Whisper can place the text
that wav2vec2 couldn't tokenize (digits, units, Latin script).

Usage:
  python scripts/experiments/test_stable_ts_weak_anchors.py
"""

import json
import subprocess
import tempfile
from pathlib import Path

import stable_whisper

SAMPLE_RATE = 16000
VIDEO = "samples/episodes/great_escape_s03e02_youtube/source/great_escape_s03e02_youtube.mp4"
CTC_WORDS = "samples/episodes/great_escape_s03e02_youtube/transcription/great_escape_s03e02_youtube_video_only_v2_ctc_words.json"

# Each test case: the weak-anchor text, the audio window to search in,
# and context about what's expected.
# Window is chosen from neighboring aligned segments in the CTC output.
TEST_CASES = [
    {
        "text": "190kg。",
        "clip_start": 1428.0,
        "clip_end": 1438.0,
        "context": "Answer to 大鶴肥満の本日の体重を答えよ — CTC orphan at 1185.826",
    },
    {
        "text": "161cm。",
        "clip_start": 1455.0,
        "clip_end": 1470.0,
        "context": "Answer to 北島三郎像の身長 — CTC orphan at 1436.546",
    },
    {
        "text": "70。",
        "clip_start": 1455.0,
        "clip_end": 1470.0,
        "context": "Weight answer — CTC orphan at 1436.546",
    },
    # Broader window test: give stable-ts more context around the prompt+answer
    {
        "text": "本日の体重を答えよ。\n190kg。",
        "clip_start": 1425.0,
        "clip_end": 1440.0,
        "context": "Prompt+answer together — does aligning both help?",
    },
]


def extract_clip(video: str, start_s: float, end_s: float, out_path: str):
    duration = end_s - start_s
    subprocess.run(
        [
            "ffmpeg", "-y", "-ss", str(start_s), "-i", video,
            "-t", str(duration), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
            "-f", "wav", out_path,
        ],
        capture_output=True, check=True,
    )


def format_result(result, clip_start: float) -> list[dict]:
    """Extract word-level timestamps from a stable-ts result, offset to episode time."""
    words = []
    if result is None:
        return words
    for seg in result.segments:
        for w in seg.words:
            words.append({
                "word": w.word,
                "start_clip": round(w.start, 3),
                "end_clip": round(w.end, 3),
                "start_episode": round(w.start + clip_start, 3),
                "end_episode": round(w.end + clip_start, 3),
                "duration": round(w.end - w.start, 3),
                "probability": round(getattr(w, "probability", 0.0), 4),
            })
    return words


def main():
    print(f"Loading stable-ts model (medium for speed)...")
    model = stable_whisper.load_model("medium", device="cuda")

    with tempfile.TemporaryDirectory() as work_dir:
        for i, case in enumerate(TEST_CASES):
            print(f"\n{'='*70}")
            print(f"Test {i+1}: {case['text']!r}")
            print(f"Context: {case['context']}")
            print(f"Clip: {case['clip_start']:.1f}s - {case['clip_end']:.1f}s")
            print(f"{'='*70}")

            clip_path = f"{work_dir}/clip_{i}.wav"
            extract_clip(VIDEO, case["clip_start"], case["clip_end"], clip_path)

            result = model.align(
                clip_path,
                case["text"],
                language="ja",
            )

            words = format_result(result, case["clip_start"])

            if not words:
                print("  RESULT: No words returned (alignment failed)")
                continue

            total_dur = sum(w["duration"] for w in words)
            zero_dur = sum(1 for w in words if w["duration"] == 0)
            print(f"  Words: {len(words)}, zero-duration: {zero_dur}, total duration: {total_dur:.3f}s")
            print()
            for w in words:
                flag = " *** ZERO" if w["duration"] == 0 else ""
                print(
                    f"  {w['start_episode']:>10.3f} - {w['end_episode']:<10.3f} "
                    f"({w['duration']:.3f}s) p={w['probability']:.4f}  "
                    f"{w['word']}{flag}"
                )

    print(f"\n{'='*70}")
    print("Done.")


if __name__ == "__main__":
    main()
