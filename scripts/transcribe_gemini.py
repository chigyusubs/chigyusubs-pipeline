#!/usr/bin/env python3
"""Transcribe audio using Gemini's multimodal audio understanding.

The maintained path returns plain Japanese transcript text only.
Speaker turns are indicated with a leading `-- ` marker so they can be stripped
before alignment and restored later.
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
    episode_memory: list[str] | None = None,
    ocr_chunk_terms: list[str] | None = None,
    prev_context: str | None = None,
    include_visual_brackets: bool = False,
) -> str:
    """Build the plain-text transcription prompt."""
    lines = [
        "You are transcribing a Japanese variety/comedy show.",
        "",
        "Instructions:",
        "1. Transcribe ALL spoken Japanese dialogue faithfully.",
        "2. Output ONLY plain text. Do NOT use JSON, markdown, or commentary.",
        "3. Do NOT add speaker names or speaker labels.",
        "4. Indicate a speaker turn by starting the line with exactly `-- `.",
        "5. Output one utterance per line.",
        "6. Keep normal Japanese punctuation (、。！？).",
        "7. Do NOT translate.",
        "8. Do NOT summarize or skip short reactions.",
        "9. If audio is only silence, music, or ambience, output nothing for that moment.",
        "10. Do NOT get stuck repeating laughter or a short phrase indefinitely.",
        "",
    ]

    if include_visual_brackets:
        lines.extend([
            "11. Transcribe only what is actually spoken as `-- ...` lines.",
            "12. If important on-screen text is relevant but not spoken aloud, put it on its own line as `[画面: ...]`.",
            "13. Keep `[画面: ...]` lines selective and short. Do not dump every visible word.",
            "14. Do NOT merge on-screen text into spoken lines unless it is clearly read aloud.",
            "",
        ])

    if glossary_entries:
        lines.append("### GLOSSARY ###")
        for entry in glossary_entries:
            lines.append(f"- {entry}")
        lines.append("")

    if episode_memory:
        lines.append("### EPISODE MEMORY ###")
        for entry in episode_memory[:30]:
            lines.append(f"- {entry}")
        lines.append("")

    if ocr_chunk_terms:
        lines.append("### LOCAL ON-SCREEN TEXT CONTEXT ###")
        lines.append("Use these only as spelling or topic hints for this chunk.")
        lines.append("Do not assume every line was spoken aloud.")
        for entry in ocr_chunk_terms[:20]:
            lines.append(f"- {entry}")
        lines.append("")

    if prev_context:
        lines.append("### PREVIOUS CONTEXT (for continuity only; do not repeat) ###")
        lines.append(prev_context[-1200:])
        lines.append("")

    lines.append("Transcribe the audio now.")
    return "\n".join(lines)


def _strip_text_fences(text: str) -> str:
    out = text.strip()
    if out.startswith("```"):
        out = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", out)
        out = re.sub(r"\s*```$", "", out)
    return out.strip()


def _normalize_transcript_text(raw: str) -> str:
    text = _strip_text_fences(raw)
    lines = []
    for line in text.splitlines():
        t = line.strip()
        if not t:
            continue
        if t.startswith("- ") and not t.startswith("-- "):
            t = "-- " + t[2:].lstrip()
        if re.match(r"^speaker\s+[a-z0-9]+:\s*", t, flags=re.I):
            t = "-- " + re.sub(r"^speaker\s+[a-z0-9]+:\s*", "", t, flags=re.I)
        if t.startswith("[画面"):
            m = re.match(r"^\[画面[:：]\s*(.*?)\s*\]?$", t)
            if m:
                t = f"[画面: {m.group(1).strip()}]"
        lines.append(t)
    return "\n".join(lines).strip()


def _countdown(seconds: float):
    """Print a live countdown on a single line."""
    import math
    remaining = math.ceil(seconds)
    while remaining > 0:
        print(f"\r  Waiting... {remaining}s ", end="", flush=True)
        time.sleep(1)
        remaining -= 1
    print("\r" + " " * 30 + "\r", end="", flush=True)


def _make_client(location: str, api_key: str = ""):
    """Build a genai Client for either Gemini API (free tier) or Vertex AI."""
    from google import genai

    if api_key:
        # The SDK checks GOOGLE_GENAI_USE_VERTEXAI and forces Vertex routing
        # even when api_key is explicitly provided. Override it.
        os.environ.pop("GOOGLE_GENAI_USE_VERTEXAI", None)
        return genai.Client(api_key=api_key)
    return genai.Client(vertexai=True, location=location)


def transcribe_chunk(
    audio_bytes: bytes,
    prompt: str,
    model: str,
    location: str,
    max_retries: int = 10,
    temperature: float = 0.1,
    mime_type: str = "audio/mpeg",
    api_key: str = "",
) -> str:
    """Send a media chunk to Gemini via streaming, return plain transcript text.

    When api_key is provided, uses the Gemini API (Google AI Studio free tier)
    instead of Vertex AI.
    """
    from google.genai import types

    client = _make_client(location, api_key)

    audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
    text_part = types.Part.from_text(text=prompt)

    config = types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="text/plain",
        max_output_tokens=65536,
        httpOptions=types.HttpOptions(timeout=180_000),
    )

    for attempt in range(max_retries):
        attempt_label = f"[attempt {attempt + 1}/{max_retries}]"
        try:
            print(f"  {attempt_label} Requesting...", end="", flush=True)
            t0 = time.time()

            chunks: list[str] = []
            char_count = 0
            first_chunk = True
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=[audio_part, text_part],
                config=config,
            ):
                text = chunk.text or ""
                if first_chunk and text:
                    ttfb = time.time() - t0
                    print(f" first token in {ttfb:.1f}s, streaming", end="", flush=True)
                    first_chunk = False
                chunks.append(text)
                char_count += len(text)
                if char_count % 1000 < len(text):
                    print(".", end="", flush=True)

            elapsed = time.time() - t0
            raw = "".join(chunks)
            normalized = _normalize_transcript_text(raw)
            print(f" {len(normalized)} chars in {elapsed:.1f}s", flush=True)
            return normalized
        except Exception as e:
            elapsed = time.time() - t0
            print(flush=True)
            if attempt < max_retries - 1:
                msg = str(e).strip().splitlines()[0] if str(e).strip() else repr(e)
                delay = _backoff_delay(attempt + 1)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    err_type = "RATE LIMITED"
                elif "499" in msg or "CANCELLED" in msg:
                    err_type = "CANCELLED"
                elif "500" in msg or "INTERNAL" in msg:
                    err_type = "SERVER ERROR"
                else:
                    err_type = "ERROR"
                print(f"  {attempt_label} {err_type} after {elapsed:.0f}s: {msg}", flush=True)
                _countdown(delay)
                continue
            raise

    raise RuntimeError("Transcription request failed with no response.")


def transcribe_full(
    video_path: str,
    glossary_entries: list[str],
    model: str,
    chunk_minutes: float,
    api_key: str = "",
) -> list[dict]:
    """Transcribe full video in chunks, return chunk texts."""
    duration = get_duration(video_path)
    chunk_s = chunk_minutes * 60
    overlap_s = 5.0

    n_chunks = max(1, int((duration - overlap_s) / (chunk_s - overlap_s)) + 1)
    print(f"Duration: {duration:.0f}s ({duration/60:.1f} min)")
    print(f"Chunks: {n_chunks} x {chunk_minutes:.0f} min (with {overlap_s:.0f}s overlap)")

    all_chunks: list[dict] = []
    prev_context: str | None = None

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

            prompt = build_prompt(glossary_entries, prev_context=prev_context)
            text = transcribe_chunk(
                chunk_bytes, prompt, model,
                os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
                api_key=api_key,
            )
            all_chunks.append({"chunk": i, "chunk_start_s": start, "text": text})
            prev_context = text

    return all_chunks


def write_outputs(chunks: list[dict], output_path: str):
    """Write chunk JSON and plain text transcript."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    txt_path = output_path.replace(".json", "_text.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            text = chunk.get("text", "").strip()
            if text:
                f.write(text + "\n\n")
    return txt_path


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio with Gemini (plain text, turn markers)."
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
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GEMINI_API_KEY", ""),
        help="Gemini API key for Google AI Studio free tier. "
        "Defaults to GEMINI_API_KEY env var. When set, uses the Gemini API instead of Vertex AI.",
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

    glossary_entries = []
    if args.glossary and os.path.exists(args.glossary):
        glossary_entries = load_glossary_names(args.glossary)
        print(f"Loaded {len(glossary_entries)} glossary entries")

    if args.api_key:
        print("Using Gemini API (Google AI Studio)")
    else:
        print("Using Vertex AI")

    chunks = transcribe_full(
        video_path=video_path,
        glossary_entries=glossary_entries,
        model=args.model,
        chunk_minutes=args.chunk_minutes,
        api_key=args.api_key,
    )

    txt_path = write_outputs(chunks, args.output)
    print(f"\nWrote {len(chunks)} chunks to {args.output}")
    print(f"  Plain text: {txt_path}")


if __name__ == "__main__":
    main()
