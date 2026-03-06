#!/usr/bin/env python3
"""Transcribe audio using Gemini's multimodal audio understanding.

Sends audio chunks + glossary to Gemini and gets back structured JSON with
speaker labels. No timestamps — those are recovered via forced alignment.

Output JSON format per utterance:
  {"speaker": "フジモン", "text": "みなさんよろしくお願いします!", "type": "speech"}
  {"speaker": "", "text": "(拍手)", "type": "sfx"}

Usage:
  python scripts/transcribe_gemini.py \
    --video samples/episodes/.../source/video.mp4 \
    --glossary samples/episodes/.../glossary/translation_glossary_v2.tsv \
    --output samples/episodes/.../transcription/406_gemini_transcript.json \
    --model gemini-3.1-pro-preview \
    --chunk-minutes 10
"""

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

from chigyusubs.audio import extract_audio_chunk, get_duration
from chigyusubs.chunking import find_chunk_boundaries
from chigyusubs.glossary import load_glossary_names


def _backoff_delay(attempt: int, cap: float = 60.0, k: float = 2.0) -> float:
    """Hyperbolic backoff: ramps fast, asymptotes to cap."""
    return cap * attempt / (attempt + k)


def build_prompt(
    glossary_entries: list[str],
    prev_context: list[dict] | None = None,
) -> str:
    """Build the transcription prompt with JSON output format."""
    lines = [
        "You are transcribing a Japanese variety/comedy show (水曜日のダウンタウン / Wednesday Downtown).",
        "",
        "Instructions:",
        "1. Transcribe ALL spoken Japanese dialogue faithfully.",
        "2. Identify speakers using the glossary. If unsure, use generic labels (Speaker A, etc.).",
        "3. Output a JSON array of objects. Each object has:",
        '   - "speaker": speaker name (empty string for non-speech)',
        '   - "text": the spoken text or sound effect description',
        '   - "type": "speech" for dialogue, "sfx" for sound effects/music',
        "4. Example output:",
        '   [',
        '     {"speaker": "浜田", "text": "皆さんよろしくお願いします", "type": "speech"},',
        '     {"speaker": "", "text": "拍手", "type": "sfx"},',
        '     {"speaker": "小峠", "text": "バイきんぐ小峠でございます", "type": "speech"}',
        '   ]',
        "5. Do NOT add timestamps.",
        "6. Do NOT translate — keep everything in Japanese.",
        "7. Do NOT skip or summarize — transcribe every utterance, even short reactions.",
        "8. For overlapping speech, list each speaker's line separately.",
        "9. Output ONLY the JSON array, no markdown fences or other text.",
        "",
    ]

    if glossary_entries:
        lines.append("### GLOSSARY (names and terms) ###")
        for entry in glossary_entries:
            lines.append(f"- {entry}")
        lines.append("")

    if prev_context:
        lines.append("### PREVIOUS CONTEXT (last lines from previous chunk — do NOT repeat) ###")
        for u in prev_context[-8:]:
            if u["type"] == "speech":
                lines.append(f'{u["speaker"]}: {u["text"]}')
            else:
                lines.append(f'({u["text"]})')
        lines.append("")
        lines.append("Continue transcribing from where this left off. Do not repeat the context lines.")
        lines.append("")

    lines.append("Transcribe the audio now. Output the JSON array only.")
    return "\n".join(lines)


def _strip_json_fences(text: str) -> str:
    """Remove markdown code fences from JSON response."""
    out = text.strip()
    if out.startswith("```"):
        out = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", out)
        out = re.sub(r"\s*```$", "", out)
    return out.strip()


def _parse_utterances(raw: str) -> list[dict]:
    """Parse Gemini's JSON response into utterance dicts."""
    cleaned = _strip_json_fences(raw)
    # Find the JSON array
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    data = json.loads(cleaned)
    if not isinstance(data, list):
        raise ValueError("Expected JSON array")

    utterances = []
    for item in data:
        utterances.append({
            "speaker": str(item.get("speaker", "")),
            "text": str(item.get("text", "")),
            "type": str(item.get("type", "speech")),
        })
    return utterances


def _countdown(seconds: float):
    """Print a live countdown on a single line."""
    import math
    remaining = math.ceil(seconds)
    while remaining > 0:
        print(f"\r  Waiting... {remaining}s ", end="", flush=True)
        time.sleep(1)
        remaining -= 1
    print("\r" + " " * 30 + "\r", end="", flush=True)


def transcribe_chunk(
    audio_bytes: bytes,
    prompt: str,
    model: str,
    max_retries: int = 10,
) -> list[dict]:
    """Send audio chunk to Gemini via streaming, return list of utterance dicts."""
    from google import genai
    from google.genai import types

    client = genai.Client()

    audio_part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/mpeg")
    text_part = types.Part.from_text(text=prompt)

    config = types.GenerateContentConfig(
        temperature=0.1,
        response_mime_type="application/json",
        max_output_tokens=65536,
        httpOptions=types.HttpOptions(timeout=180_000),  # 180s
    )

    for attempt in range(max_retries):
        attempt_label = f"[attempt {attempt + 1}/{max_retries}]"
        try:
            print(f"  {attempt_label} Requesting...", end="", flush=True)
            t0 = time.time()

            # Stream response for faster feedback and fewer timeouts
            chunks: list[str] = []
            char_count = 0
            first_chunk = True
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=[audio_part, text_part],
                config=config,
            ):
                if first_chunk:
                    ttfb = time.time() - t0
                    print(f" first token in {ttfb:.1f}s, streaming",
                          end="", flush=True)
                    first_chunk = False
                text = chunk.text or ""
                chunks.append(text)
                char_count += len(text)
                # Dot every ~1k chars
                if char_count % 1000 < len(text):
                    print(".", end="", flush=True)

            elapsed = time.time() - t0
            raw = "".join(chunks)
            print(f" {len(raw)} chars in {elapsed:.1f}s", flush=True)
            return _parse_utterances(raw)
        except json.JSONDecodeError as e:
            elapsed = time.time() - t0
            print(flush=True)  # newline after partial output
            if attempt < max_retries - 1:
                delay = _backoff_delay(attempt + 1)
                print(f"  {attempt_label} JSON parse failed after {elapsed:.0f}s: {e}",
                      flush=True)
                _countdown(delay)
                continue
            raise
        except Exception as e:
            elapsed = time.time() - t0
            print(flush=True)  # newline after partial output
            if attempt < max_retries - 1:
                msg = str(e).strip().splitlines()[0] if str(e).strip() else repr(e)
                delay = _backoff_delay(attempt + 1)
                # Classify error for clearer display
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    err_type = "RATE LIMITED"
                elif "499" in msg or "CANCELLED" in msg:
                    err_type = "CANCELLED"
                elif "500" in msg or "INTERNAL" in msg:
                    err_type = "SERVER ERROR"
                else:
                    err_type = "ERROR"
                print(f"  {attempt_label} {err_type} after {elapsed:.0f}s: {msg}",
                      flush=True)
                _countdown(delay)
                continue
            raise

    raise RuntimeError("Transcription request failed with no response.")


def transcribe_full(
    video_path: str,
    glossary_entries: list[str],
    model: str,
    chunk_minutes: float,
) -> list[dict]:
    """Transcribe full video in chunks, return all utterances."""
    duration = get_duration(video_path)
    chunk_s = chunk_minutes * 60
    overlap_s = 5.0  # 5s audio overlap for continuity

    n_chunks = max(1, int((duration - overlap_s) / (chunk_s - overlap_s)) + 1)
    print(f"Duration: {duration:.0f}s ({duration/60:.1f} min)")
    print(f"Chunks: {n_chunks} x {chunk_minutes:.0f} min (with {overlap_s:.0f}s overlap)")

    all_utterances: list[dict] = []
    prev_context: list[dict] | None = None

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(n_chunks):
            start = i * (chunk_s - overlap_s)
            chunk_dur = min(chunk_s, duration - start)
            if start >= duration or chunk_dur <= 0:
                break

            chunk_path = os.path.join(tmpdir, f"chunk_{i}.mp3")
            extract_audio_chunk(video_path, chunk_path, start_s=start, duration_s=chunk_dur)
            chunk_bytes = Path(chunk_path).read_bytes()
            chunk_mb = len(chunk_bytes) / (1024 * 1024)

            label = f"Chunk {i + 1}/{n_chunks}"
            m_start = start / 60
            m_end = (start + chunk_dur) / 60
            print(f"\n[{label}] {m_start:.1f}-{m_end:.1f} min ({chunk_mb:.1f} MB)")

            prompt = build_prompt(glossary_entries, prev_context)
            utterances = transcribe_chunk(chunk_bytes, prompt, model)

            speech_count = sum(1 for u in utterances if u["type"] == "speech")
            sfx_count = sum(1 for u in utterances if u["type"] == "sfx")
            print(f"  -> {speech_count} speech + {sfx_count} sfx utterances")

            # Tag utterances with chunk info for debugging
            for u in utterances:
                u["chunk"] = i
                u["chunk_start_s"] = start

            all_utterances.extend(utterances)
            prev_context = utterances

    return all_utterances


def write_outputs(utterances: list[dict], output_path: str):
    """Write JSON and plain text transcript."""
    # JSON output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(utterances, f, ensure_ascii=False, indent=2)

    # Also write plain text for alignment (speech only, no speaker labels)
    txt_path = output_path.replace(".json", "_speech.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for u in utterances:
            if u["type"] == "speech" and u["text"].strip():
                f.write(u["text"] + "\n")

    # And a labeled version for reference
    labeled_path = output_path.replace(".json", "_labeled.txt")
    with open(labeled_path, "w", encoding="utf-8") as f:
        for u in utterances:
            if u["type"] == "sfx":
                f.write(f"({u['text']})\n")
            elif u["speaker"]:
                f.write(f"{u['speaker']}: {u['text']}\n")
            else:
                f.write(f"{u['text']}\n")

    return txt_path, labeled_path


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio with Gemini (structured JSON, chunked)."
    )
    parser.add_argument("--video", required=True, help="Input video file.")
    parser.add_argument(
        "--output", default="",
        help="Output JSON path. Defaults to episode transcription dir.",
    )
    parser.add_argument("--glossary", default="", help="Glossary TSV for names/terms.")
    parser.add_argument(
        "--model",
        default=os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview"),
        help="Gemini model (default: gemini-3.1-pro-preview).",
    )
    parser.add_argument(
        "--chunk-minutes", type=float, default=10,
        help="Chunk duration in minutes (default: 10).",
    )
    args = parser.parse_args()

    video_path = args.video
    if not os.path.exists(video_path):
        print(f"Error: Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    if not args.output:
        stem = Path(video_path).stem
        out_dir = Path(video_path).parent.parent / "transcription"
        args.output = str(out_dir / f"{stem}_gemini_transcript.json")

    # Load glossary
    glossary_entries = []
    if args.glossary and os.path.exists(args.glossary):
        glossary_entries = load_glossary_names(args.glossary)
        print(f"Loaded {len(glossary_entries)} glossary entries")

    # Transcribe
    utterances = transcribe_full(
        video_path=video_path,
        glossary_entries=glossary_entries,
        model=args.model,
        chunk_minutes=args.chunk_minutes,
    )

    # Write outputs
    txt_path, labeled_path = write_outputs(utterances, args.output)

    speech = [u for u in utterances if u["type"] == "speech"]
    sfx = [u for u in utterances if u["type"] == "sfx"]
    speakers = set(u["speaker"] for u in speech if u["speaker"])

    print(f"\nWrote {len(utterances)} utterances to {args.output}")
    print(f"  Speech: {len(speech)}, SFX: {len(sfx)}")
    print(f"  Speakers: {', '.join(sorted(speakers)) or '(none)'}")
    print(f"  Speech text: {txt_path}")
    print(f"  Labeled text: {labeled_path}")


if __name__ == "__main__":
    main()
