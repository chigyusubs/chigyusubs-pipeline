#!/usr/bin/env python3
"""Translate VTT/SRT subtitles using structured JSON via any OpenAI-compatible API.

Designed for Whisper-segmented Japanese subtitle output.  Sends the full
transcript in a single pass by default (frontier models handle 45+ min
episodes easily).  Falls back to chunked translation via --chunk-seconds
if you hit output token limits.

Usage:
  # Vertex Gemini
  python scripts/translate_vtt.py --backend vertex \
    --input subs.vtt --output subs_en.vtt

  # OpenAI-compatible (local or remote)
  python scripts/translate_vtt.py --backend openai \
    --url http://127.0.0.1:8787 --model qwen3-30b \
    --input subs.vtt --output subs_en.vtt

  # With glossary and summary
  python scripts/translate_vtt.py --backend vertex \
    --input subs.vtt --output subs_en.vtt \
    --glossary glossary.tsv --summary "A comedy variety show featuring..."

  # Chunked mode for very long episodes or smaller models
  python scripts/translate_vtt.py --backend openai \
    --input subs.vtt --chunk-seconds 600
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# VTT / SRT parsing
# ---------------------------------------------------------------------------

_TIME_RE = re.compile(
    r"(?:(\d{2}):)?(\d{2}):(\d{2})[.,](\d{3})"
)


class Cue:
    __slots__ = ("start", "end", "text")

    def __init__(self, start: float, end: float, text: str):
        self.start = start
        self.end = end
        self.text = text


def _time_to_seconds(tc: str) -> float:
    m = _TIME_RE.match(tc.strip())
    if not m:
        raise ValueError(f"Invalid timecode: {tc}")
    h = int(m.group(1) or 0)
    mi = int(m.group(2))
    s = int(m.group(3))
    ms = int(m.group(4))
    return h * 3600 + mi * 60 + s + ms / 1000


def _seconds_to_time(seconds: float) -> str:
    total_ms = round(seconds * 1000)
    h = total_ms // 3600000
    mi = (total_ms % 3600000) // 60000
    s = (total_ms % 60000) // 1000
    ms = total_ms % 1000
    return f"{h:02d}:{mi:02d}:{s:02d}.{ms:03d}"


def parse_vtt(text: str) -> list[Cue]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = [l.replace("\ufeff", "") for l in lines]
    cues: list[Cue] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not line or line.upper().startswith("WEBVTT") or "-->" not in line:
            idx += 1
            continue
        parts = line.split("-->")
        start = _time_to_seconds(parts[0].strip())
        end = _time_to_seconds(parts[1].strip().split(" ")[0])
        idx += 1
        text_lines: list[str] = []
        while idx < len(lines) and lines[idx].strip():
            text_lines.append(lines[idx])
            idx += 1
        cues.append(Cue(start, end, "\n".join(text_lines)))
        while idx < len(lines) and not lines[idx].strip():
            idx += 1
    return cues


def parse_srt(text: str) -> list[Cue]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = [l.replace("\ufeff", "") for l in lines]
    cues: list[Cue] = []
    idx = 0
    while idx < len(lines):
        while idx < len(lines) and not lines[idx].strip():
            idx += 1
        if idx >= len(lines):
            break
        if re.fullmatch(r"\d+", lines[idx].strip()):
            idx += 1
        if idx >= len(lines) or "-->" not in lines[idx]:
            idx += 1
            continue
        parts = lines[idx].split("-->")
        start = _time_to_seconds(parts[0].strip())
        end = _time_to_seconds(parts[1].strip().split(" ")[0])
        idx += 1
        text_lines: list[str] = []
        while idx < len(lines) and lines[idx].strip():
            text_lines.append(lines[idx])
            idx += 1
        cues.append(Cue(start, end, "\n".join(text_lines)))
    return cues


def serialize_vtt(cues: list[Cue]) -> str:
    parts = ["WEBVTT", ""]
    for cue in cues:
        parts.append(f"{_seconds_to_time(cue.start)} --> {_seconds_to_time(cue.end)}")
        parts.append(cue.text)
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def serialize_srt(cues: list[Cue]) -> str:
    parts: list[str] = []
    for i, cue in enumerate(cues, 1):
        start = _seconds_to_time(cue.start).replace(".", ",")
        end = _seconds_to_time(cue.end).replace(".", ",")
        parts.append(str(i))
        parts.append(f"{start} --> {end}")
        parts.append(cue.text)
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Chunking (optional, for large episodes or small-context models)
# ---------------------------------------------------------------------------

class Chunk:
    def __init__(self, idx: int, cues: list[Cue], prev_context: list[Cue]):
        self.idx = idx
        self.cues = cues
        self.prev_context = prev_context


def chunk_cues(
    cues: list[Cue],
    target_seconds: float = 600,
    overlap_cues: int = 2,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    current: list[Cue] = []
    prev_tail: list[Cue] = []

    for cue in cues:
        if not current:
            current.append(cue)
            continue
        tentative = current + [cue]
        duration = tentative[-1].end - tentative[0].start
        if duration > target_seconds:
            chunks.append(Chunk(len(chunks), current, prev_tail))
            prev_tail = current[-overlap_cues:] if overlap_cues > 0 else []
            current = [cue]
        else:
            current.append(cue)

    if current:
        chunks.append(Chunk(len(chunks), current, prev_tail))
    return chunks


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a subtitle translator for Japanese variety and comedy shows.\n"
    "Translate subtitles into {target}.\n"
    "\n"
    "Rules:\n"
    '1. Output JSON: {{ "translations": [ {{ "id": <number>, "text": "<translated>" }} ] }}\n'
    "2. One output item per input cue. Keep IDs in order. Do not drop or duplicate IDs.\n"
    "3. Do not merge, split, or reorder cues.\n"
    "4. Keep comedic timing; prefer punchy phrasing over literal filler.\n"
    "5. Do not sanitize slang or humor; translate faithfully.\n"
    "6. Sound effects and annotations in parentheses: translate to English equivalents\n"
    "   e.g. (拍手) -> (applause), (笑い) -> (laughter), (音楽) -> (music).\n"
    "{speaker_instruction}"
)

SPEAKER_INSTRUCTION = (
    "7. Speaker labels (Name: text) are provided. Preserve them in the output as-is.\n"
    "   Maintain consistent voice characterization per speaker."
)

# Pattern: Japanese/English name followed by colon at start of cue text.
_SPEAKER_RE = re.compile(r"^[\w\u3000-\u9fff\uff00-\uffef]+:\s")


def _has_speaker_labels(cues: list[Cue]) -> bool:
    """Check if cues contain speaker prefix labels (e.g. 'フジモン: ...')."""
    if not cues:
        return False
    labeled = sum(1 for c in cues if _SPEAKER_RE.match(c.text))
    return labeled / len(cues) > 0.2


def _build_user_prompt(
    cues: list[Cue],
    target_lang: str,
    glossary: Optional[str],
    context_cues: list[Cue],
    summary: Optional[str],
) -> str:
    lines: list[str] = []

    if glossary and glossary.strip():
        lines.append("### GLOSSARY ###")
        lines.append(glossary.strip())
        lines.append("")

    if summary and summary.strip():
        lines.append("### GLOBAL SUMMARY ###")
        lines.append(summary.strip())
        lines.append("")

    if context_cues:
        lines.append("### PREVIOUS CONTEXT (REFERENCE ONLY, DO NOT TRANSLATE) ###")
        for cue in context_cues:
            safe = cue.text.replace("\n", " ")
            lines.append(f"{_seconds_to_time(cue.start)} --> {_seconds_to_time(cue.end)} {safe}")
        lines.append("")

    lines.append("### CUES TO TRANSLATE ###")
    for i, cue in enumerate(cues):
        safe = cue.text.replace("\n", " ")
        lines.append(f"[{i + 1}] {safe}")

    lines.append("")
    lines.append(f"Translate all cues into {target_lang}. Output JSON only.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Structured output validation & VTT reconstruction
# ---------------------------------------------------------------------------

def _validate_structured_output(data: dict) -> list[dict]:
    if not isinstance(data, dict):
        raise ValueError("Output is not a valid JSON object")
    translations = data.get("translations")
    if not isinstance(translations, list):
        print(f"RAW PARSED JSON: {data}"); raise ValueError("Missing or invalid 'translations' array")

    validated = []
    for i, item in enumerate(translations):
        if not isinstance(item, dict):
            raise ValueError(f"Item at index {i} is not an object")
        cue_id = item.get("id")
        if not isinstance(cue_id, (int, float)) or not (cue_id == int(cue_id)):
            raise ValueError(f"Item at index {i} has invalid 'id'")
        text = item.get("text")
        if text is None:
            text = ""
        if not isinstance(text, str):
            raise ValueError(f"Item at index {i} has invalid 'text'")
        validated.append({"id": int(cue_id), "text": text})
    return validated


def _reconstruct_vtt(
    translations: list[dict], original_cues: list[Cue]
) -> tuple[list[Cue], list[str]]:
    warnings: list[str] = []
    n = len(original_cues)

    seen: set[int] = set()
    items: dict[int, dict] = {}
    for item in translations:
        cid = item["id"]
        if cid < 1 or cid > n:
            warnings.append(f"Skipping invalid ID {cid} (out of bounds)")
            continue
        if cid in seen:
            warnings.append(f"Duplicate entry for ID {cid} (keeping first)")
            continue
        seen.add(cid)
        items[cid] = item

    new_cues: list[Cue] = []
    for i in range(1, n + 1):
        item = items.get(i)
        if not item:
            warnings.append(f"Missing translation for cue {i}; dropping cue.")
            continue
        text = (item["text"] or "").strip()
        orig = original_cues[i - 1]
        new_cues.append(Cue(orig.start, orig.end, text))

    missing = [cid for cid in range(1, n + 1) if cid not in items]
    if missing:
        warnings.append(f"Input cues not returned: {', '.join(map(str, missing))}")

    return new_cues, warnings


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    out = text.strip()
    if out.startswith("```"):
        out = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", out)
        out = re.sub(r"\s*```$", "", out)
    return out.strip()


def _parse_json_response(raw: str) -> dict:
    cleaned = _strip_code_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        return json.loads(cleaned[start : end + 1])
    if not raw.strip(): return {"translations": []}
    raise ValueError(f"Failed to parse JSON from model response: {raw}")


def _call_openai_compatible(
    url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    api_key: Optional[str] = None,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def _call_vertex(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client()
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=temperature,
        response_mime_type="application/json",
        max_output_tokens=65536,
    )
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=config,
    )
    return response.text or ""


def _backoff_delay(attempt: int, cap: float = 30.0, k: float = 2.0) -> float:
    """Hyperbolic backoff: ramps fast, asymptotes to cap.

    attempt 1 -> 10s, attempt 2 -> 15s, attempt 3 -> 18s,
    attempt 5 -> 21s, attempt 10 -> 25s  (with cap=30, k=2)
    """
    return cap * attempt / (attempt + k)


def _call_with_retry(
    call_fn,
    max_retries: int = 10,
    **kwargs,
) -> str:
    for attempt in range(max_retries):
        try:
            return call_fn(**kwargs)
        except Exception as e:
            if attempt < max_retries - 1:
                msg = str(e).strip().splitlines()[0] if str(e).strip() else repr(e)
                delay = _backoff_delay(attempt + 1)
                print(f"  API call failed (attempt {attempt + 1}/{max_retries}): {msg}")
                print(f"  Retrying in {delay:.0f}s...")
                time.sleep(delay)
                continue
            raise
    raise RuntimeError("API request failed with no response.")


# ---------------------------------------------------------------------------
# Translation pipeline
# ---------------------------------------------------------------------------

def _translate_cues(
    cues: list[Cue],
    context_cues: list[Cue],
    target_lang: str,
    glossary: Optional[str],
    summary: Optional[str],
    backend: str,
    model: str,
    temperature: float,
    url: Optional[str],
    api_key: Optional[str],
) -> tuple[list[Cue], list[str]]:
    """Translate a list of cues. Returns (translated_cues, warnings)."""

    speaker_instruction = ""
    if _has_speaker_labels(cues):
        speaker_instruction = SPEAKER_INSTRUCTION
    system_prompt = SYSTEM_PROMPT.format(
        target=target_lang, speaker_instruction=speaker_instruction
    )
    user_prompt = _build_user_prompt(
        cues, target_lang, glossary, context_cues, summary
    )

    if backend == "vertex":
        raw = _call_with_retry(
            _call_vertex,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
        )
    else:
        raw = _call_with_retry(
            _call_openai_compatible,
            url=url or "http://127.0.0.1:8787",
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            api_key=api_key,
        )

    parsed = _parse_json_response(raw)
    translations = _validate_structured_output(parsed)
    return _reconstruct_vtt(translations, cues)


def translate_subtitles(
    input_path: str,
    output_path: str,
    target_lang: str,
    backend: str,
    model: str,
    temperature: float,
    url: Optional[str],
    api_key: Optional[str],
    glossary_path: Optional[str],
    summary: Optional[str],
    chunk_seconds: float,
    overlap_cues: int,
    output_format: str,
):
    # Load input
    raw = Path(input_path).read_text(encoding="utf-8")
    if input_path.lower().endswith(".srt"):
        cues = parse_srt(raw)
    else:
        cues = parse_vtt(raw)

    if not cues:
        print("No cues found in input file.")
        sys.exit(1)

    print(f"Parsed {len(cues)} cues from {input_path}")

    # Load glossary
    glossary: Optional[str] = None
    if glossary_path:
        glossary = Path(glossary_path).read_text(encoding="utf-8").strip()
        print(f"Loaded glossary from {glossary_path}")

    # Decide: single-pass or chunked
    total_duration = cues[-1].end - cues[0].start
    use_chunking = chunk_seconds > 0 and total_duration > chunk_seconds

    if use_chunking:
        chunks = chunk_cues(cues, target_seconds=chunk_seconds, overlap_cues=overlap_cues)
        print(f"Chunked mode: {len(chunks)} chunks ({chunk_seconds:.0f}s target)")

        all_translated: list[Cue] = []
        all_warnings: list[str] = []

        for chunk in chunks:
            label = f"Chunk {chunk.idx + 1}/{len(chunks)}"
            cue_range = (
                f"{_seconds_to_time(chunk.cues[0].start)} - "
                f"{_seconds_to_time(chunk.cues[-1].end)}"
            )
            print(f"[{label}] Translating {len(chunk.cues)} cues ({cue_range})...")

            translated, warnings = _translate_cues(
                cues=chunk.cues,
                context_cues=chunk.prev_context,
                target_lang=target_lang,
                glossary=glossary,
                summary=summary,
                backend=backend,
                model=model,
                temperature=temperature,
                url=url,
                api_key=api_key,
            )
            all_translated.extend(translated)
            if warnings:
                for w in warnings:
                    print(f"  Warning: {w}")
                all_warnings.extend(warnings)
            print(f"  -> {len(translated)} output cues")

        all_translated.sort(key=lambda c: c.start)
        translated_cues = all_translated
        warnings = all_warnings
    else:
        print(f"Single-pass mode ({len(cues)} cues, {total_duration:.0f}s)")
        translated_cues, warnings = _translate_cues(
            cues=cues,
            context_cues=[],
            target_lang=target_lang,
            glossary=glossary,
            summary=summary,
            backend=backend,
            model=model,
            temperature=temperature,
            url=url,
            api_key=api_key,
        )
        if warnings:
            for w in warnings:
                print(f"  Warning: {w}")

    # Write output
    if output_format == "srt":
        out_text = serialize_srt(translated_cues)
    else:
        out_text = serialize_vtt(translated_cues)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(out_text, encoding="utf-8")

    print(f"\nWrote {len(translated_cues)} translated cues to {output_path}")
    if warnings:
        print(f"Total warnings: {len(warnings)}")


def main():
    parser = argparse.ArgumentParser(
        description="Translate VTT/SRT subtitles using structured JSON via LLM."
    )
    parser.add_argument("--input", required=True, help="Input VTT or SRT file.")
    parser.add_argument(
        "--output", default="", help="Output file. Defaults to <input>_<lang>.<ext>"
    )
    parser.add_argument(
        "--target-lang", default="English", help="Target language (default: English)."
    )
    parser.add_argument(
        "--backend",
        default="openai",
        choices=["openai", "vertex"],
        help="LLM backend: 'openai' (any OpenAI-compatible API) or 'vertex' (Gemini).",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Model name. Defaults: vertex=gemini-2.5-pro, openai=gpt-4o",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("OPENAI_BASE_URL", ""),
        help="Base URL for OpenAI-compatible backend (env: OPENAI_BASE_URL).",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="API key for OpenAI-compatible backend (env: OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--glossary",
        default="",
        help="Path to glossary file (CSV or TSV: source,target,context).",
    )
    parser.add_argument(
        "--summary",
        default="",
        help="Optional summary to provide context (e.g. show premise, segment theme).",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=0,
        help="Enable chunked mode with this target duration (0 = single-pass, default).",
    )
    parser.add_argument(
        "--overlap-cues",
        type=int,
        default=2,
        help="Context cues from previous chunk in chunked mode (default: 2).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM temperature (default: 0.2).",
    )
    parser.add_argument(
        "--format",
        default="",
        choices=["vtt", "srt", ""],
        help="Output format. Defaults to match input file extension.",
    )
    args = parser.parse_args()

    # Resolve defaults
    if not args.model:
        args.model = (
            os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")
            if args.backend == "vertex"
            else "gpt-4o"
        )

    if not args.output:
        stem = Path(args.input).stem
        ext = Path(args.input).suffix or ".vtt"
        lang_slug = args.target_lang.lower().replace(" ", "_")
        args.output = str(Path(args.input).parent / f"{stem}_{lang_slug}{ext}")

    output_format = args.format
    if not output_format:
        output_format = "srt" if args.output.lower().endswith(".srt") else "vtt"

    translate_subtitles(
        input_path=args.input,
        output_path=args.output,
        target_lang=args.target_lang,
        backend=args.backend,
        model=args.model,
        temperature=args.temperature,
        url=args.url or None,
        api_key=args.api_key or None,
        glossary_path=args.glossary or None,
        summary=args.summary or None,
        chunk_seconds=args.chunk_seconds,
        overlap_cues=args.overlap_cues,
        output_format=output_format,
    )


if __name__ == "__main__":
    main()
