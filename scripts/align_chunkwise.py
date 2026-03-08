#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chigyusubs.metadata import finish_run, metadata_path, start_run, write_metadata


def extract_audio_slice(video_path, start_s, duration_s, out_path):
    subprocess.run([
        "ffmpeg", "-y", "-ss", str(start_s), "-i", video_path,
        "-t", str(duration_s), "-vn", "-ac", "1", "-ar", "16000",
        "-f", "wav", out_path
    ], capture_output=True, check=True)

def main():
    run = start_run("align_chunkwise")
    parser = argparse.ArgumentParser(description="Chunk-wise stable-ts alignment.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--chunks", required=True, help="Path to _chunks.json from transcription.")
    parser.add_argument("--output-words", required=True, help="Output JSON with aligned words.")
    parser.add_argument("--model", default="large-v3")
    args = parser.parse_args()

    with open(args.chunks, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    import stable_whisper
    print(f"Loading Whisper model: {args.model}")
    model = stable_whisper.load_model(args.model, device="cuda")

    all_segments = []
    
    with tempfile.TemporaryDirectory() as work_dir:
        for i, chunk in enumerate(chunks_data):
            c_start = chunk["start"]
            c_end = chunk["end"]
            c_dur = c_end - c_start
            print(f"\nAligning chunk {i+1}/{len(chunks_data)} (Start: {c_start:.1f}s, Dur: {c_dur:.1f}s)")
            
            # Clean text: strip leading dashes and spaces
            raw_text = chunk["text"]
            clean_lines = []
            for line in raw_text.split("\n"):
                line = line.strip()
                if line.startswith("-"):
                    line = line.lstrip("- ").strip()
                if line:
                    clean_lines.append(line)
            clean_text = "\n".join(clean_lines)

            if not clean_text:
                print("Skipping empty chunk.")
                continue

            slice_wav = os.path.join(work_dir, f"slice_{i}.wav")
            extract_audio_slice(args.video, c_start, c_dur, slice_wav)

            # Align the chunk
            result = model.align(slice_wav, clean_text, language="ja")

            # Offset the timestamps
            for seg in result.segments:
                words_data = []
                for w in seg.words:
                    words_data.append({
                        "start": w.start + c_start,
                        "end": w.end + c_start,
                        "word": w.word,
                        "probability": getattr(w, "probability", 0.0),
                    })
                all_segments.append({
                    "start": seg.start + c_start,
                    "end": seg.end + c_start,
                    "text": seg.text,
                    "words": words_data,
                })

    print(f"\nWriting word timestamps JSON to {args.output_words}")
    with open(args.output_words, "w", encoding="utf-8") as f:
        json.dump(all_segments, f, ensure_ascii=False, indent=2)
    metadata = finish_run(
        run,
        inputs={
            "video": args.video,
            "chunks_json": args.chunks,
        },
        outputs={
            "words_json": args.output_words,
        },
        settings={
            "model": args.model,
        },
        stats={
            "chunks_loaded": len(chunks_data),
            "segments_written": len(all_segments),
            "words_written": sum(len(seg.get("words", [])) for seg in all_segments),
        },
    )
    write_metadata(args.output_words, metadata)
    print(f"Metadata written: {metadata_path(args.output_words)}")

if __name__ == "__main__":
    main()
