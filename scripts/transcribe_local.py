#!/usr/bin/env python3
"""Fully local transcription pipeline: Silero VAD + chunk-wise OCR filtering + faster-whisper.

Instead of a global initial_prompt, this script:
1. Runs Silero VAD to find speech boundaries and chunk the audio
2. For each chunk, filters OCR terms via a local LLM (e.g. Gemma on llama.cpp)
3. Transcribes each chunk with faster-whisper using a focused initial_prompt
4. Merges word-level timestamps and reflows into a final VTT

Zero API calls — everything runs locally.

Usage:
  python scripts/transcribe_local.py \
    --video samples/episodes/.../source/video.mp4 \
    --ocr-jsonl samples/episodes/.../ocr/qwen_ocr_results.jsonl \
    --llm-url http://127.0.0.1:8080 \
    --llm-model "bartowski/google_gemma-3-27b-it-qat-GGUF:Q4_K_M"

  # With a global glossary merged into each chunk prompt
  python scripts/transcribe_local.py \
    --video ... --ocr-jsonl ... --llm-url ... \
    --glossary samples/episodes/.../glossary/whisper_prompt_condensed.txt
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from chigyusubs.audio import extract_audio_chunk, get_duration
from chigyusubs.chunking import find_chunk_boundaries
from chigyusubs.ocr import filter_ocr_terms_with_llm, get_ocr_context_for_chunk, load_ocr_data
from chigyusubs.paths import find_latest_episode_video, infer_episode_dir_from_video
from chigyusubs.reflow import reflow_words
from chigyusubs.vad import run_silero_vad
from chigyusubs.vtt import write_standard_vtt, write_word_timestamps_json


def log(msg: str = "", end="\n"):
    print(msg, end=end, flush=True)


def transcribe_chunk_whisper(
    model,
    audio_path: str,
    initial_prompt: str,
    hotwords: str | None = None,
) -> list[dict]:
    """Transcribe a single audio chunk with faster-whisper, return word-level segments."""
    segments, info = model.transcribe(
        audio_path,
        language="ja",
        initial_prompt=initial_prompt or None,
        hotwords=hotwords or None,
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500, max_speech_duration_s=15.0),
        compression_ratio_threshold=2.4,
        word_timestamps=True,
    )

    results = []
    for seg in segments:
        words = []
        if seg.words:
            words = [{"start": w.start, "end": w.end, "word": w.word,
                       "probability": w.probability} for w in seg.words]
        results.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "words": words,
        })
    return results


def build_hotwords(ocr_terms: list[str]) -> str:
    """Build a hotwords string from chunk-specific OCR terms.

    Hotwords are comma-separated and boost decoder probability for these tokens.
    """
    if not ocr_terms:
        return ""
    return ", ".join(ocr_terms)


def main():
    parser = argparse.ArgumentParser(
        description="Fully local transcription: Silero VAD + OCR filter + faster-whisper."
    )
    parser.add_argument("--video", default="", help="Input video path.")
    parser.add_argument("--ocr-jsonl", default="", help="OCR JSONL from Qwen for chunk-wise context.")
    parser.add_argument("--glossary", default="", help="Global glossary text (whisper_prompt_condensed.txt).")
    parser.add_argument("--output", default="", help="Output VTT path.")
    parser.add_argument("--model", default="large-v3", help="faster-whisper model name.")
    parser.add_argument("--compute-type", default="float16", help="ctranslate2 compute type.")
    parser.add_argument(
        "--llm-url", default="http://127.0.0.1:8080",
        help="Local LLM endpoint for OCR filtering.",
    )
    parser.add_argument(
        "--llm-model", default="",
        help="Model name for OCR filter LLM (e.g. bartowski/google_gemma-3-27b-it-qat-GGUF:Q4_K_M).",
    )
    parser.add_argument(
        "--chunk-minutes", type=float, default=3,
        help="Target chunk duration in minutes (default: 3).",
    )
    parser.add_argument("--reflow", action="store_true", default=True,
                        help="Reflow words into cues (default: on).")
    parser.add_argument("--no-reflow", action="store_false", dest="reflow")
    parser.add_argument("--reflow-pause-ms", type=int, default=300)
    parser.add_argument("--max-cue-s", type=float, default=10.0)
    args = parser.parse_args()

    # --- Resolve paths ---
    if not args.video:
        default_video = find_latest_episode_video()
        if not default_video:
            raise SystemExit("No default video found. Pass --video explicitly.")
        args.video = str(default_video)

    episode_dir = infer_episode_dir_from_video(Path(args.video))

    if not args.ocr_jsonl:
        # Try to find OCR JSONL in episode dir
        ocr_dir = episode_dir / "ocr"
        if ocr_dir.exists():
            jsonls = sorted(ocr_dir.glob("*_results.jsonl"))
            if jsonls:
                args.ocr_jsonl = str(jsonls[-1])
                log(f"Auto-detected OCR JSONL: {args.ocr_jsonl}")

    if not args.glossary:
        default_glossary = episode_dir / "glossary" / "whisper_prompt_condensed.txt"
        if default_glossary.exists():
            args.glossary = str(default_glossary)

    if not args.output:
        args.output = str(
            episode_dir / "transcription" / f"{Path(args.video).stem}_local_chunked.vtt"
        )

    os.makedirs(Path(args.output).parent, exist_ok=True)

    log(f"Video:    {args.video}")
    log(f"OCR:      {args.ocr_jsonl or '(none)'}")
    log(f"Glossary: {args.glossary or '(none)'}")
    log(f"LLM:      {args.llm_url} / {args.llm_model or '(default)'}")
    log(f"Output:   {args.output}")

    duration = get_duration(args.video)
    log(f"Duration: {duration:.0f}s ({duration / 60:.1f} min)")

    # --- Load OCR data ---
    ocr_frames = []
    if args.ocr_jsonl and os.path.exists(args.ocr_jsonl):
        ocr_frames = load_ocr_data(args.ocr_jsonl)
        log(f"OCR data: {len(ocr_frames)} frames loaded")

    # --- Load global glossary ---
    global_glossary = ""
    if args.glossary and os.path.exists(args.glossary):
        with open(args.glossary, "r", encoding="utf-8") as f:
            global_glossary = f.read().strip()
        log(f"Global glossary: {len(global_glossary)} chars")

    # --- Phase 1: Silero VAD ---
    log("\n=== Phase 1: Silero VAD ===")
    with tempfile.TemporaryDirectory() as work_dir:
        vad_segments = run_silero_vad(args.video, work_dir=work_dir)

    total_speech = sum(s["end"] - s["start"] for s in vad_segments)
    log(f"Speech: {total_speech:.0f}s / {duration:.0f}s ({total_speech / duration * 100:.0f}%)")

    target_chunk_s = args.chunk_minutes * 60
    chunk_bounds = find_chunk_boundaries(
        vad_segments, duration,
        target_chunk_s=target_chunk_s,
        min_gap_s=2.0,
    )
    log(f"Chunks: {len(chunk_bounds)}")
    for i, (cs, ce) in enumerate(chunk_bounds):
        log(f"  [{i + 1}] {cs / 60:.1f} - {ce / 60:.1f} min ({ce - cs:.0f}s)")

    # --- Phase 2: Load Whisper model ---
    log("\n=== Phase 2: Loading faster-whisper ===")
    from faster_whisper import WhisperModel
    model = WhisperModel(args.model, device="cuda", compute_type=args.compute_type)
    log(f"Model loaded: {args.model} ({args.compute_type})")

    # --- Phase 3: Chunk-wise transcription ---
    log("\n=== Phase 3: Chunk-wise transcription ===")
    all_segments = []

    with tempfile.TemporaryDirectory() as audio_tmp:
        for i, (c_start, c_end) in enumerate(chunk_bounds):
            c_dur = c_end - c_start
            log(f"\n--- Chunk {i + 1}/{len(chunk_bounds)} "
                f"[{c_start / 60:.1f}-{c_end / 60:.1f} min] ---")

            # Extract audio chunk
            chunk_path = os.path.join(audio_tmp, f"chunk_{i}.mp3")
            extract_audio_chunk(args.video, chunk_path, start_s=c_start, duration_s=c_dur)
            chunk_mb = os.path.getsize(chunk_path) / (1024 * 1024)
            log(f"  Audio: {chunk_mb:.1f} MB")

            # Filter OCR terms for this chunk
            ocr_terms = []
            if ocr_frames:
                raw_lines = get_ocr_context_for_chunk(ocr_frames, c_start, c_end)
                if raw_lines and args.llm_url:
                    ocr_terms = filter_ocr_terms_with_llm(
                        raw_lines,
                        ocr_filter_url=args.llm_url,
                        ocr_filter_model=args.llm_model or None,
                    )
                    log(f"  OCR terms: {ocr_terms[:8]}{'...' if len(ocr_terms) > 8 else ''}")
                elif raw_lines:
                    # No LLM — pass raw Japanese lines as-is
                    ocr_terms = raw_lines[:20]

            # Build chunk-specific hotwords from OCR terms
            hotwords = build_hotwords(ocr_terms)
            if hotwords:
                log(f"  Hotwords: {hotwords[:120]}{'...' if len(hotwords) > 120 else ''}")

            # Transcribe: global glossary as initial_prompt, OCR as hotwords
            chunk_segments = transcribe_chunk_whisper(
                model, chunk_path,
                initial_prompt=global_glossary,
                hotwords=hotwords or None,
            )

            # Offset timestamps to absolute time
            for seg in chunk_segments:
                seg["start"] += c_start
                seg["end"] += c_start
                for w in seg.get("words", []):
                    w["start"] += c_start
                    w["end"] += c_start

            text_len = sum(len(s["text"]) for s in chunk_segments)
            log(f"  Result: {len(chunk_segments)} segments, {text_len} chars")
            for seg in chunk_segments[:3]:
                log(f"    {seg['text'][:80]}")
            if len(chunk_segments) > 3:
                log(f"    ... ({len(chunk_segments) - 3} more)")

            all_segments.extend(chunk_segments)

    log(f"\nTotal: {len(all_segments)} segments")

    # --- Phase 4: Write outputs ---
    log("\n=== Phase 4: Output ===")

    # Word timestamps JSON
    json_path = args.output.replace(".vtt", "_words.json")
    log(f"Writing word timestamps: {json_path}")
    write_word_timestamps_json(all_segments, json_path)

    # Reflow or direct VTT
    if args.reflow:
        cues = reflow_words(
            all_segments,
            pause_threshold=args.reflow_pause_ms / 1000.0,
            max_cue_s=args.max_cue_s,
            min_cue_s=0.3,
        )
        log(f"Reflowed: {len(all_segments)} segments -> {len(cues)} cues")
        write_standard_vtt(cues, args.output)
    else:
        write_standard_vtt(all_segments, args.output)

    log(f"VTT written: {args.output}")
    log("\nDone!")


if __name__ == "__main__":
    main()
