#!/usr/bin/env python3
"""Translate VTT/SRT subtitles using structured JSON via any OpenAI-compatible API.

Mirrors the structured JSON translation approach from chigyusubs (browser app)
but runs locally as a CLI tool.  Supports Vertex/Gemini, OpenAI, Anthropic,
or any OpenAI-compatible endpoint (llama.cpp, vLLM, OpenRouter, etc.).

Usage:
  # Vertex Gemini (uses GOOGLE_CLOUD_PROJECT env)
  python scripts/translate_vtt.py --backend vertex \\
    --input subs.vtt --output subs_en.vtt

  # OpenAI-compatible (local or remote)
  python scripts/translate_vtt.py --backend openai \\
    --url http://127.0.0.1:8787 --model qwen3-30b \\
    --input subs.vtt --output subs_en.vtt

  # With glossary and summary
  python scripts/translate_vtt.py --backend vertex \\
    --input subs.vtt --output subs_en.vtt \\
    --glossary glossary.tsv --summary "A comedy variety show featuring..."
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# VTT / SRT parsing (matches chigyusubs src/lib/vtt.ts)
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
# Chunking (matches chigyusubs src/lib/chunker.ts)
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
# Prompt building (matches chigyusubs src/lib/structured/StructuredPrompt.ts)
# ---------------------------------------------------------------------------

SHORT_CUE_SECONDS = 1.5

SYSTEM_PROMPT = (
    "You are a subtitle translator tool. Your task is to translate subtitles into {target}.\n"
    "\n"
    "Rules:\n"
    '1. Output JSON with a "translations" array.\n'
    '2. Each item: {{ "id": number, "text": string, "merge_with_next"?: boolean }}.\n'
    "3. Keep IDs in order, one item per input cue. Do not drop or duplicate IDs.\n"
    "4. You MAY merge very short adjacent cues: set merge_with_next=true on the first "
    "cue of the merged group; later cues in that group can have empty text.\n"
    "5. Never split a single cue across multiple outputs.\n"
    "\n"
    "Annotation handling (from Whisper/Gemini transcription):\n"
    "- (--) speaker change: Remove or keep as -- (dash). Don't translate.\n"
    "- (テロップ: ...) on-screen text: Translate content, keep format: (Caption: ...).\n"
    "- (拍手), (笑い), (音楽): Translate to English: (applause), (laughter), (music).\n"
    "- Keep comedic timing; prefer punchy phrasing over literal filler.\n"
    "- Do not sanitize slang/humor; translate faithfully."
)


def _build_user_prompt(
    cues: list[Cue],
    target_lang: str,
    glossary: Optional[str],
    context_cues: list[Cue],
    summary: Optional[str],
    hint_mode: str = "duration",
) -> str:
    lines: list[str] = []

    # 1. Glossary
    if glossary and glossary.strip():
        lines.append("### GLOSSARY ###")
        lines.append(glossary.strip())
        lines.append("")

    # 2. Summary
    if summary and summary.strip():
        lines.append("### GLOBAL SUMMARY ###")
        lines.append(summary.strip())
        lines.append("")

    # 3. Previous context
    if context_cues:
        lines.append("### PREVIOUS CONTEXT (REFERENCE ONLY) ###")
        for cue in context_cues:
            safe = cue.text.replace("\n", " ")
            lines.append(
                f"{_seconds_to_time(cue.start)} --> {_seconds_to_time(cue.end)} {safe}"
            )
        lines.append("")

    # 4. Cues to translate
    lines.append("### CUES TO TRANSLATE ###")
    if hint_mode == "short-tag":
        lines.append("Format: [ID <id>][SHORT?] <start> --> <end> <source_text>")
    else:
        lines.append(
            "Format: [ID <id>] <start> --> <end> (duration <seconds>s) <source_text>"
        )
    lines.append("---")

    for i, cue in enumerate(cues):
        cue_id = i + 1
        dur = cue.end - cue.start
        start = _seconds_to_time(cue.start)
        end = _seconds_to_time(cue.end)
        safe = cue.text.replace("\n", " ")
        if hint_mode == "short-tag":
            short = " [SHORT]" if dur < SHORT_CUE_SECONDS else ""
            lines.append(f"[ID {cue_id}]{short} {start} --> {end} {safe}")
        else:
            lines.append(
                f"[ID {cue_id}] {start} --> {end} (duration {dur:.2f}s) {safe}"
            )

    lines.append("---")
    lines.append(f"Task: Translate the above cues into {target_lang}.")
    lines.append("")
    lines.append("Rules (keep it minimal):")
    lines.append('1. Output a JSON object with a "translations" array.')
    lines.append(
        '2. Each item must have "id" (number) and "text" (string). '
        'Optional: "merge_with_next": true to merge with the next cue.'
    )
    if hint_mode == "short-tag":
        lines.append(
            "3. Only merge when a cue is marked [SHORT] and clearly flows into the "
            'next cue; set merge_with_next: true on that cue and return "" for the '
            "next cue's text."
        )
    else:
        lines.append(
            "3. You MAY merge very short adjacent cues; set merge_with_next: true on "
            "the first cue of the merged group; later cues in that group can have "
            "empty text."
        )
    lines.append(
        "4. Keep cues in order. Do NOT drop or duplicate any IDs. "
        "One output item per input ID."
    )
    lines.append("5. Do NOT split a single cue across multiple outputs.")
    lines.append("")
    lines.append("Example Output:")
    lines.append("{")
    lines.append('  "translations": [')
    lines.append(
        '    { "id": 1, "text": "Translated text...", "merge_with_next": false },'
    )
    lines.append(
        '    { "id": 2, "text": "Combined with next", "merge_with_next": true },'
    )
    lines.append('    { "id": 3, "text": "" }')
    lines.append("  ]")
    lines.append("}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Structured output validation & VTT reconstruction
# (matches chigyusubs StructuredOutput.ts + VttReconstructor.ts)
# ---------------------------------------------------------------------------


def _validate_structured_output(data: dict) -> list[dict]:
    if not isinstance(data, dict):
        raise ValueError("Output is not a valid JSON object")
    translations = data.get("translations")
    if not isinstance(translations, list):
        raise ValueError("Missing or invalid 'translations' array")

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
        merge = item.get("merge_with_next", False)
        if not isinstance(merge, bool):
            raise ValueError(f"Item at index {i} has invalid 'merge_with_next'")
        validated.append({"id": int(cue_id), "text": text, "merge_with_next": merge})
    return validated


def _reconstruct_vtt(translations: list[dict], original_cues: list[Cue]) -> tuple[list[Cue], list[str]]:
    warnings: list[str] = []
    n = len(original_cues)

    # Build lookup, detect dupes/invalid
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
    i = 1
    while i <= n:
        item = items.get(i)
        if not item:
            warnings.append(f"Missing translation for cue {i}; dropping cue.")
            i += 1
            continue

        group_text = (item["text"] or "").strip()
        group_start_idx = i
        group_end_idx = i
        merge_next = item["merge_with_next"]

        while merge_next and group_end_idx < n:
            next_id = group_end_idx + 1
            next_item = items.get(next_id)
            if not next_item:
                warnings.append(
                    f"merge_with_next=true on cue {group_end_idx} but cue {next_id} is missing; stopping merge."
                )
                break
            next_text = (next_item["text"] or "").strip()
            if next_text:
                group_text = f"{group_text} {next_text}" if group_text else next_text
            group_end_idx = next_id
            merge_next = next_item["merge_with_next"]

        start = original_cues[group_start_idx - 1].start
        end = original_cues[group_end_idx - 1].end
        new_cues.append(Cue(start, end, group_text))
        i = group_end_idx + 1

    # Warn about missing IDs
    missing = [cid for cid in range(1, n + 1) if cid not in items]
    if missing:
        warnings.append(f"Input cues not returned: {', '.join(map(str, missing))}")

    new_cues.sort(key=lambda c: c.start)
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
    """Parse JSON from LLM response, handling code fences."""
    cleaned = _strip_code_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Fallback: extract first JSON object
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        return json.loads(cleaned[start : end + 1])
    raise ValueError("Failed to parse JSON from model response")


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
        max_output_tokens=8192,
    )
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=config,
    )
    return response.text or ""


def _call_with_retry(
    call_fn,
    max_retries: int = 6,
    **kwargs,
) -> str:
    for attempt in range(max_retries):
        try:
            return call_fn(**kwargs)
        except Exception as e:
            if attempt < max_retries - 1:
                msg = str(e).strip().splitlines()[0] if str(e).strip() else repr(e)
                print(f"  API call failed (attempt {attempt + 1}/{max_retries}): {msg}")
                print("  Retrying...")
                continue
            raise
    raise RuntimeError("API request failed with no response.")


# ---------------------------------------------------------------------------
# Translation pipeline
# ---------------------------------------------------------------------------

def translate_chunk(
    cues: list[Cue],
    context_cues: list[Cue],
    target_lang: str,
    glossary: Optional[str],
    summary: Optional[str],
    hint_mode: str,
    backend: str,
    model: str,
    temperature: float,
    url: Optional[str],
    api_key: Optional[str],
) -> tuple[list[Cue], list[str]]:
    """Translate a single chunk of cues. Returns (translated_cues, warnings)."""

    system_prompt = SYSTEM_PROMPT.format(target=target_lang)
    user_prompt = _build_user_prompt(
        cues, target_lang, glossary, context_cues, summary, hint_mode
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
    hint_mode: str,
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

    # Chunk
    chunks = chunk_cues(cues, target_seconds=chunk_seconds, overlap_cues=overlap_cues)
    print(f"Split into {len(chunks)} chunk(s) ({chunk_seconds}s target)")

    # Translate each chunk
    all_translated: list[Cue] = []
    all_warnings: list[str] = []

    for chunk in chunks:
        label = f"Chunk {chunk.idx + 1}/{len(chunks)}"
        cue_range = (
            f"{_seconds_to_time(chunk.cues[0].start)} - "
            f"{_seconds_to_time(chunk.cues[-1].end)}"
        )
        print(f"[{label}] Translating {len(chunk.cues)} cues ({cue_range})...")

        translated, warnings = translate_chunk(
            cues=chunk.cues,
            context_cues=chunk.prev_context,
            target_lang=target_lang,
            glossary=glossary,
            summary=summary,
            hint_mode=hint_mode,
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

    # Sort by start time (chunks are sequential but just in case)
    all_translated.sort(key=lambda c: c.start)

    # Write output
    if output_format == "srt":
        out_text = serialize_srt(all_translated)
    else:
        out_text = serialize_vtt(all_translated)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(out_text, encoding="utf-8")

    print(f"\nWrote {len(all_translated)} translated cues to {output_path}")
    if all_warnings:
        print(f"Total warnings: {len(all_warnings)}")


def main():
    parser = argparse.ArgumentParser(
        description="Translate VTT/SRT subtitles using structured JSON via LLM."
    )
    parser.add_argument("--input", required=True, help="Input VTT or SRT file.")
    parser.add_argument(
        "--output", default="", help="Output file. Defaults to <input>_<lang>.vtt"
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
        help="Global summary text to provide context for translation.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=600,
        help="Target chunk duration in seconds (default: 600).",
    )
    parser.add_argument(
        "--overlap-cues",
        type=int,
        default=2,
        help="Number of context cues from previous chunk (default: 2).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM temperature (default: 0.2).",
    )
    parser.add_argument(
        "--hint-mode",
        default="duration",
        choices=["duration", "short-tag"],
        help="Cue hint mode: 'duration' (show seconds) or 'short-tag' (mark short cues).",
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
        hint_mode=args.hint_mode,
        output_format=output_format,
    )


if __name__ == "__main__":
    main()
