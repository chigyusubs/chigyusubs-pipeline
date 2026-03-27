#!/usr/bin/env python3
"""Run Flash Lite audio correction pass on Gemini raw transcript chunks.

Extracts audio per chunk, sends to Flash Lite with the existing transcript,
and saves the corrected version alongside the original for diffing.

Usage:
    GOOGLE_GENAI_USE_VERTEXAI=false python3.12 scripts/correct_transcript_flash_lite.py \
        --gemini-raw samples/episodes/<slug>/transcription/<run>_gemini_raw.json \
        --video samples/episodes/<slug>/source/video.mp4

    # Resume from chunk N:
    ... --start-chunk 15

    # Process specific chunks only:
    ... --chunks 5,12,20
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chigyusubs.env import load_repo_env

load_repo_env()


def extract_chunk_audio(video_path: str, start_s: float, end_s: float, dest: str) -> bool:
    """Extract audio for a chunk as m4a."""
    duration = end_s - start_s
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ss", str(start_s), "-t", str(duration),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "aac", dest,
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def _build_thinking_config(thinking_level: str):
    """Build a ThinkingConfig from a level string, or None if unspecified."""
    from google.genai import types

    thinking_level = thinking_level.lower()
    if thinking_level == "unspecified":
        return None
    level_map = {
        "minimal": types.ThinkingLevel.MINIMAL,
        "low": types.ThinkingLevel.LOW,
        "medium": types.ThinkingLevel.MEDIUM,
        "high": types.ThinkingLevel.HIGH,
    }
    if thinking_level not in level_map:
        raise ValueError(f"Unsupported thinking_level: {thinking_level}")
    return types.ThinkingConfig(thinking_level=level_map[thinking_level])


def correct_chunk(
    client,
    model: str,
    audio_path: str,
    transcript_text: str,
    temperature: float = 0.1,
    thinking_level: str = "unspecified",
) -> str:
    """Send audio + transcript to Flash Lite and return corrected transcript."""
    from google.genai import types

    with open(audio_path, "rb") as f:
        audio_data = f.read()

    prompt = f"""You are a Japanese audio transcriber. I will give you an audio clip and a reference transcript.

Listen to the audio carefully and transcribe EVERYTHING you hear — every reaction, interjection, back-channel response, and overlapping dialogue. The reference transcript may be incomplete; your job is to produce the most complete transcription possible.

Rules:
1. Transcribe all spoken content from the audio, including short reactions (うん, えー, おお) and filler
2. Use -- to mark each speaker turn, matching the reference transcript's style
3. Use the reference transcript as a guide for names, terminology, and context
4. It is OK to restructure or split lines to match what you actually hear

Output ONLY the transcript. Do not add commentary or explanations.

REFERENCE TRANSCRIPT:
{transcript_text}"""

    config_kwargs: dict = {"temperature": temperature}
    thinking_config = _build_thinking_config(thinking_level)
    if thinking_config is not None:
        config_kwargs["thinking_config"] = thinking_config

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Content(parts=[
                types.Part.from_bytes(data=audio_data, mime_type="audio/mp4"),
                types.Part.from_text(text=prompt),
            ])
        ],
        config=types.GenerateContentConfig(**config_kwargs),
    )
    return response.text or ""


def main():
    parser = argparse.ArgumentParser(description="Flash Lite audio correction pass on Gemini transcript chunks.")
    parser.add_argument("--gemini-raw", required=True, help="Path to gemini_raw.json")
    parser.add_argument("--video", default="", help="Source video path (auto-detected from episode dir if omitted)")
    parser.add_argument("--model", default="gemini-3.1-flash-lite-preview", help="Flash Lite model name")
    parser.add_argument("--output", default="", help="Output corrected JSON path (default: <gemini_raw_stem>_corrected.json)")
    parser.add_argument("--start-chunk", type=int, default=0, help="Start from this chunk index")
    parser.add_argument("--chunks", default="", help="Comma-separated chunk indices to process (default: all)")
    parser.add_argument("--temperature", type=float, default=0.1, help="LLM temperature")
    parser.add_argument("--thinking-level", default="unspecified",
                        choices=["unspecified", "minimal", "low", "medium", "high"],
                        help="Gemini thinking level (default: unspecified)")
    parser.add_argument("--rpm-limit", type=int, default=10, help="Max requests per minute (default: 10)")
    args = parser.parse_args()

    gemini_raw_path = Path(args.gemini_raw)
    chunks = json.loads(gemini_raw_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(chunks)} chunks from {gemini_raw_path}", flush=True)

    # Auto-detect video
    video_path = args.video
    if not video_path:
        for parent in gemini_raw_path.resolve().parents:
            source_dir = parent / "source"
            if source_dir.is_dir():
                for f in source_dir.iterdir():
                    if f.suffix.lower() in {".mp4", ".mkv", ".avi", ".webm", ".ts"}:
                        video_path = str(f)
                        break
                break
    if not video_path:
        print("Error: could not find video. Pass --video explicitly.", file=sys.stderr)
        sys.exit(1)
    print(f"Video: {video_path}", flush=True)

    # Output path
    output_path = Path(args.output) if args.output else gemini_raw_path.with_name(
        gemini_raw_path.stem + "_corrected.json"
    )

    # Determine which chunks to process
    if args.chunks:
        chunk_indices = set(int(x.strip()) for x in args.chunks.split(","))
    else:
        chunk_indices = set(range(args.start_chunk, len(chunks)))

    # Load existing output for resume
    corrected = []
    if output_path.exists():
        try:
            corrected = json.loads(output_path.read_text(encoding="utf-8"))
            completed = {c["chunk"] for c in corrected if "corrected_text" in c}
            print(f"Resuming: {len(completed)} chunks already corrected", flush=True)
            chunk_indices -= completed
        except Exception:
            corrected = []

    if not corrected:
        corrected = [dict(c) for c in chunks]

    # Build index
    corrected_by_idx = {c["chunk"]: c for c in corrected}
    for c in chunks:
        if c["chunk"] not in corrected_by_idx:
            corrected_by_idx[c["chunk"]] = dict(c)
            corrected.append(corrected_by_idx[c["chunk"]])

    if not chunk_indices:
        print("All chunks already corrected.", flush=True)
        sys.exit(0)

    print(f"Chunks to process: {sorted(chunk_indices)} ({len(chunk_indices)} total)", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"RPM limit: {args.rpm_limit}", flush=True)

    # Init Gemini client
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("Error: GEMINI_API_KEY not set in environment", file=sys.stderr)
        sys.exit(1)

    from google import genai
    client = genai.Client(api_key=api_key)

    request_times: list[float] = []
    rpm_window = 60.0

    with tempfile.TemporaryDirectory() as tmpdir:
        for idx in sorted(chunk_indices):
            chunk = chunks[idx]
            start_s = chunk["chunk_start_s"]
            end_s = chunk["chunk_end_s"]
            text = chunk["text"]

            print(f"\n[Chunk {idx}/{len(chunks)-1}] {start_s:.1f}-{end_s:.1f}s ({end_s-start_s:.1f}s, {len(text)} chars)", flush=True)

            # Rate limiting
            now = time.time()
            request_times = [t for t in request_times if now - t < rpm_window]
            if len(request_times) >= args.rpm_limit:
                wait = rpm_window - (now - request_times[0]) + 0.5
                print(f"  Rate limit: waiting {wait:.1f}s", flush=True)
                time.sleep(wait)

            # Extract audio
            audio_path = os.path.join(tmpdir, f"chunk_{idx}.m4a")
            if not extract_chunk_audio(video_path, start_s, end_s, audio_path):
                print(f"  ERROR: ffmpeg failed, skipping", flush=True)
                continue

            audio_size = os.path.getsize(audio_path)
            print(f"  Audio: {audio_size/1024:.0f}KB", flush=True)

            # Call Flash Lite
            try:
                request_times.append(time.time())
                result = correct_chunk(client, args.model, audio_path, text, args.temperature, args.thinking_level)
                print(f"  Corrected: {len(result)} chars (was {len(text)})", flush=True)

                # Store result
                corrected_by_idx[idx]["corrected_text"] = result
                corrected_by_idx[idx]["correction_model"] = args.model

                # Preview changes
                orig_lines = set(text.strip().split("\n"))
                new_lines = set(result.strip().split("\n"))
                added = new_lines - orig_lines
                removed = orig_lines - new_lines
                if added:
                    print(f"  + {len(added)} new/changed lines", flush=True)
                    for line in list(added)[:3]:
                        print(f"    + {line[:70]}", flush=True)
                if removed:
                    print(f"  - {len(removed)} removed/changed lines", flush=True)
                if not added and not removed:
                    print(f"  = No changes", flush=True)

            except Exception as e:
                print(f"  ERROR: {e}", flush=True)
                corrected_by_idx[idx]["correction_error"] = str(e)

            # Save after each chunk (resumable)
            corrected_sorted = sorted(corrected, key=lambda c: c["chunk"])
            output_path.write_text(
                json.dumps(corrected_sorted, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    # Final summary
    n_corrected = sum(1 for c in corrected if "corrected_text" in c)
    n_errors = sum(1 for c in corrected if "correction_error" in c)
    print(f"\nDone: {n_corrected} corrected, {n_errors} errors, saved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
