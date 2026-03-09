#!/usr/bin/env python3
"""Transcribe audio using Gemini's multimodal audio understanding, returning RAW text.

This strips out the JSON schema requirements, asking the model to just return text.
It sends the audio (optionally split into chunks via VAD).
It can dynamically pull context from an OCR JSONL file based on the chunk's time window.

Saves a .txt file for review and a _chunks.json file for chunk-wise alignment.
"""

import argparse
import json
import os
import tempfile
import time
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chigyusubs.audio import extract_audio_chunk, get_duration
from chigyusubs.chunking import find_chunk_boundaries
from chigyusubs.glossary import load_glossary_names
from chigyusubs.metadata import finish_run, metadata_path, start_run, write_metadata
from chigyusubs.ocr import (
    filter_ocr_terms_with_llm,
    get_ocr_context_for_chunk,
    load_ocr_data,
)
from chigyusubs.vad import run_silero_vad


def log(msg: str = "", end="\n"):
    print(msg, end=end, flush=True)


def build_raw_prompt(
    glossary_entries: list[str] = None,
    ocr_lines: list[str] = None,
    prev_context: str | None = None,
) -> str:
    lines = [
        "You are transcribing a Japanese variety/comedy show (水曜日のダウンタウン / Wednesday Downtown).",
        "",
        "Instructions:",
        "1. Transcribe ALL spoken Japanese dialogue faithfully.",
        "2. Output ONLY plain text format. Do NOT use JSON or markdown code blocks.",
        "3. Do NOT add timestamps or speaker names/labels.",
        "4. Indicate speaker turns by starting the line with a hyphen and a space ('- ').",
        "5. INCLUDE standard Japanese punctuation (、 and 。) to reflect the natural flow and pauses.",
        "6. Do NOT translate — keep everything in Japanese.",
        "7. Do NOT skip or summarize — transcribe every utterance verbatim.",
        "8. Output each utterance on a new line.",
        "9. CRITICAL: DO NOT hallucinate infinite loops. If people repeat a word like '待って' rapidly, transcribe it naturally but DO NOT get stuck in an endless loop.",
        "10. If the audio becomes silent or background music only, DO NOT output anything.",
        "",
    ]

    if glossary_entries:
        lines.append("### GLOBAL GLOSSARY (names and terms) ###")
        for entry in glossary_entries:
            lines.append(f"- {entry}")
        lines.append("")

    if ocr_lines:
        lines.append("### ON-SCREEN TEXT CONTEXT ###")
        lines.append("The following text appeared on screen during this specific audio chunk.")
        lines.append("Use this to help correctly spell names, locations, or topics being discussed:")
        # Limit to 50 lines just to prevent insane prompt bloat if a frame had a wall of text
        for entry in ocr_lines[:50]: 
            lines.append(f"- {entry}")
        lines.append("")

    if prev_context:
        lines.append("### PREVIOUS CONTEXT (for continuity) ###")
        lines.append(prev_context[-1000:])
        lines.append("")

    return "\n".join(lines)


def transcribe_raw_chunk(
    audio_bytes: bytes,
    prompt: str,
    model: str,
    max_retries: int = 10,
    api_key: str = "",
    location: str = "europe-west4",
) -> str:
    from google import genai
    from google.genai import types

    if api_key:
        client = genai.Client(api_key=api_key)
    else:
        client = genai.Client(vertexai=True, location=location)

    audio_part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/mpeg")
    text_part = types.Part.from_text(text=prompt)

    config = types.GenerateContentConfig(
        temperature=0.2,
        response_mime_type="text/plain",
        max_output_tokens=65536,
        httpOptions=types.HttpOptions(timeout=180_000),
    )

    for attempt in range(max_retries):
        attempt_label = f"[attempt {attempt + 1}/{max_retries}]"
        try:
            log(f"  {attempt_label} Requesting...", end="")
            t0 = time.time()

            chunks: list[str] = []
            char_count = 0
            
            response = client.models.generate_content_stream(
                model=model,
                contents=[audio_part, text_part],
                config=config,
            )

            for chunk in response:
                text = chunk.text
                if text:
                    chunks.append(text)
                    new_chars = len(text)
                    char_count += new_chars
                    if char_count < 50:
                        log(text, end="")
                    elif char_count == new_chars:
                        log("...", end="")

            full_text = "".join(chunks)
            dur = time.time() - t0
            log(f"\n  {attempt_label} Success: {len(full_text)} chars in {dur:.1f}s")
            return full_text

        except Exception as e:
            log(f"\n  {attempt_label} ERROR: {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                delay = 60.0 * (attempt + 1) / ((attempt + 1) + 2.0)
                log(f"  Waiting {delay:.1f}s...")
                time.sleep(delay)
            else:
                log("  Max retries reached.")
                raise


def main():
    run = start_run("transcribe_gemini_raw")
    parser = argparse.ArgumentParser(description="Raw text Gemini transcription.")
    parser.add_argument("--video", required=True, help="Input video/audio file.")
    parser.add_argument("--output", required=True, help="Output TXT path.")
    parser.add_argument("--glossary", default="", help="Global Glossary TSV for names/terms.")
    parser.add_argument("--ocr-jsonl", default="", help="JSONL from Qwen OCR for dynamic chunk context.")
    parser.add_argument(
        "--model",
        default=os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview"),
        help="Gemini model.",
    )
    parser.add_argument(
        "--chunk-minutes", type=float, default=5,
        help="Target chunk duration in min. 0 = send whole file inline.",
    )
    parser.add_argument(
        "--ocr-filter-url", default="",
        help="Local OpenAI-compatible URL for OCR filtering (e.g. http://127.0.0.1:8080). "
             "If empty, uses Vertex gemini-2.5-flash.",
    )
    parser.add_argument(
        "--ocr-filter-model", default="",
        help="Model name for local OCR filter endpoint (e.g. gemma3-27b, qwen3.5-9b).",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GEMINI_API_KEY", ""),
        help="Gemini API key for Google AI Studio free tier. "
        "Defaults to GEMINI_API_KEY env var. When set, uses the Gemini API instead of Vertex AI.",
    )
    args = parser.parse_args()

    video_path = args.video
    if not os.path.exists(video_path):
        log(f"Error: Video not found: {video_path}")
        sys.exit(1)

    output_path = Path(args.output)
    out_dir = output_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    chunks_json_path = out_dir / f"{output_path.stem}_chunks.json"

    log(f"Video: {video_path}")
    log(f"Output: {output_path}")
    duration = get_duration(video_path)
    log(f"Duration: {duration:.0f}s ({duration / 60:.1f} min)")

    glossary_entries = []
    if args.glossary and os.path.exists(args.glossary):
        glossary_entries = load_glossary_names(args.glossary)
        log(f"Global Glossary: {len(glossary_entries)} entries")

    ocr_frames = []
    if args.ocr_jsonl and os.path.exists(args.ocr_jsonl):
        ocr_frames = load_ocr_data(args.ocr_jsonl)
        log(f"OCR Data: Loaded {len(ocr_frames)} frames for dynamic context.")

    with tempfile.TemporaryDirectory() as work_dir:
        chunk_bounds = []
        if args.chunk_minutes > 0:
            log("Running VAD for chunking...")
            vad_segments = run_silero_vad(video_path, work_dir=work_dir)
            target_chunk_s = args.chunk_minutes * 60
            chunk_bounds = find_chunk_boundaries(
                vad_segments, duration,
                target_chunk_s=target_chunk_s,
                min_gap_s=2.0,
            )
            log(f"Split into {len(chunk_bounds)} chunks.")
        else:
            log("No chunking requested. Sending whole file inline.")
            chunk_bounds = [(0, duration)]

        all_text = []
        chunks_data = []
        prev_context = ""

        with open(output_path, "w", encoding="utf-8") as f:
            for i, (c_start, c_end) in enumerate(chunk_bounds):
                c_dur = c_end - c_start
                log(f"\n--- Chunk {i + 1}/{len(chunk_bounds)} ---")
                log(f"Time: {c_start / 60:.1f} - {c_end / 60:.1f} min ({c_dur:.0f}s)")

                chunk_path = os.path.join(work_dir, f"chunk_{i}.mp3")
                log(f"Extracting audio to {chunk_path}...")
                extract_audio_chunk(
                    video_path, chunk_path,
                    start_s=c_start, duration_s=c_dur,
                )
                chunk_bytes = Path(chunk_path).read_bytes()
                log(f"Audio size: {len(chunk_bytes) / (1024 * 1024):.1f} MB")

                # Get dynamic OCR context for this specific time window
                chunk_ocr_lines = []
                if ocr_frames:
                    raw_chunk_ocr_lines = get_ocr_context_for_chunk(ocr_frames, c_start, c_end)
                    chunk_ocr_lines = filter_ocr_terms_with_llm(
                        raw_chunk_ocr_lines,
                        ocr_filter_url=args.ocr_filter_url or None,
                        ocr_filter_model=args.ocr_filter_model,
                    )
                    log(f"  -> Extracted {len(chunk_ocr_lines)} key terms for this chunk: {chunk_ocr_lines[:5]}...")

                prompt = build_raw_prompt(
                    glossary_entries=glossary_entries, 
                    ocr_lines=chunk_ocr_lines, 
                    prev_context=prev_context
                )
                
                text_result = transcribe_raw_chunk(
                    chunk_bytes, prompt, args.model, max_retries=10,
                    api_key=args.api_key,
                )
                
                all_text.append(text_result)
                prev_context = text_result
                
                chunks_data.append({
                    "chunk_index": i,
                    "start": c_start,
                    "end": c_end,
                    "text": text_result
                })

                f.write(text_result + "\n\n")
                f.flush()
                
                with open(chunks_json_path, "w", encoding="utf-8") as jf:
                    json.dump(chunks_data, jf, ensure_ascii=False, indent=2)

    log(f"\nFinished! Output saved to {output_path}")
    log(f"Chunks JSON saved to {chunks_json_path}")
    metadata = finish_run(
        run,
        inputs={
            "video": args.video,
            "glossary": args.glossary or None,
            "ocr_jsonl": args.ocr_jsonl or None,
        },
        outputs={
            "text": str(output_path),
            "chunks_json": str(chunks_json_path),
        },
        settings={
            "model": args.model,
            "chunk_minutes": args.chunk_minutes,
            "ocr_filter_url": args.ocr_filter_url or None,
            "ocr_filter_model": args.ocr_filter_model or None,
        },
        stats={
            "duration_seconds": round(duration, 3),
            "glossary_entries": len(glossary_entries),
            "ocr_frames_loaded": len(ocr_frames),
            "chunks": len(chunk_bounds),
            "text_chunks_written": len(chunks_data),
            "characters_written": sum(len(t) for t in all_text),
        },
    )
    write_metadata(output_path, metadata)
    log(f"Metadata written: {metadata_path(output_path)}")

if __name__ == "__main__":
    main()
