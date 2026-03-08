#!/usr/bin/env python3
"""Align a speaker-labeled transcript to audio using stable-ts.

Takes a Gemini transcript (Speaker: text format) and produces a word-timestamped
VTT file, preserving speaker labels.

Usage:
  python scripts/align_stable_ts.py \
    --video samples/episodes/.../source/video.mp4 \
    --transcript samples/episodes/.../transcription/406_gemini31_transcript.txt \
    --output samples/episodes/.../transcription/406_gemini31_aligned.vtt \
    --model large-v3
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chigyusubs.metadata import finish_run, metadata_path, start_run, write_metadata

# Speaker label pattern: "Name: text" or "(sound effect)"
_SPEAKER_RE = re.compile(r"^([\w\u3000-\u9fff\uff00-\uffef]+):\s*(.+)$")
_SFX_RE = re.compile(r"^\(.*\)$")


def parse_transcript(text: str) -> list[dict]:
    """Parse speaker-labeled transcript into utterances."""
    utterances = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        m = _SPEAKER_RE.match(line)
        if m:
            utterances.append({
                "speaker": m.group(1),
                "text": m.group(2).strip(),
            })
        elif _SFX_RE.match(line):
            utterances.append({
                "speaker": "",
                "text": line,
            })
        else:
            utterances.append({
                "speaker": "",
                "text": line,
            })
    return utterances


def _format_ts(seconds: float) -> str:
    total_ms = round(seconds * 1000)
    h = total_ms // 3600000
    mi = (total_ms % 3600000) // 60000
    s = (total_ms % 60000) // 1000
    ms = total_ms % 1000
    return f"{h:02d}:{mi:02d}:{s:02d}.{ms:03d}"


def main():
    run = start_run("align_stable_ts")
    parser = argparse.ArgumentParser(
        description="Align a transcript to audio using stable-ts (Whisper cross-attention)."
    )
    parser.add_argument("--video", required=True, help="Input video/audio file.")
    parser.add_argument("--transcript", required=True, help="Speaker-labeled transcript (.txt).")
    parser.add_argument("--output", default="", help="Output VTT. Defaults to <transcript>_aligned.vtt.")
    parser.add_argument("--output-words-json", default="", help="Output word timestamps JSON.")
    parser.add_argument(
        "--model", default="large-v3",
        help="Whisper model for alignment (default: large-v3). Smaller models use less VRAM.",
    )
    args = parser.parse_args()

    if not args.output:
        stem = Path(args.transcript).stem
        args.output = str(Path(args.transcript).parent / f"{stem}_aligned.vtt")
    if not args.output_words_json:
        args.output_words_json = args.output.replace(".vtt", "_words.json")

    # Parse transcript
    transcript_text = Path(args.transcript).read_text(encoding="utf-8")
    utterances = parse_transcript(transcript_text)
    print(f"Parsed {len(utterances)} utterances from transcript")

    # Build plain text for alignment (strip speaker labels and SFX lines)
    speech_utterances = [u for u in utterances if u["text"] and not _SFX_RE.match(u["text"])]
    plain_text = "\n".join(u["text"] for u in speech_utterances)
    print(f"Speech utterances for alignment: {len(speech_utterances)}")

    # Load model and align
    import stable_whisper
    print(f"Loading Whisper model: {args.model}")
    model = stable_whisper.load_model(args.model, device="cuda")

    print("Aligning transcript to audio...")
    result = model.align(
        args.video,
        plain_text,
        language="ja",
    )

    # Write raw aligned VTT
    print(f"Writing aligned VTT to {args.output}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_srt_vtt(args.output, vtt=True)

    # Also write word-level JSON for reflow_words.py
    segments_data = []
    for seg in result.segments:
        words_data = []
        for w in seg.words:
            words_data.append({
                "start": w.start,
                "end": w.end,
                "word": w.word,
                "probability": getattr(w, "probability", 0.0),
            })
        segments_data.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "words": words_data,
        })

    print(f"Writing word timestamps JSON to {args.output_words_json}")
    with open(args.output_words_json, "w", encoding="utf-8") as f:
        json.dump(segments_data, f, ensure_ascii=False, indent=2)

    # Stats
    total_words = sum(len(s["words"]) for s in segments_data)
    total_segs = len(segments_data)
    print(f"\nDone: {total_segs} segments, {total_words} words aligned")
    metadata = finish_run(
        run,
        inputs={
            "video": args.video,
            "transcript": args.transcript,
        },
        outputs={
            "vtt": args.output,
            "words_json": args.output_words_json,
        },
        settings={
            "model": args.model,
        },
        stats={
            "utterances_parsed": len(utterances),
            "speech_utterances": len(speech_utterances),
            "segments_written": total_segs,
            "words_written": total_words,
        },
    )
    write_metadata(args.output, metadata)
    print(f"Metadata written: {metadata_path(args.output)}")


if __name__ == "__main__":
    main()
