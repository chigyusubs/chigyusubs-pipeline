#!/usr/bin/env python3
"""Translate VTT/SRT subtitles into readable English subtitle batches via Mistral.

Designed for reflowed Japanese subtitle output. The script translates in
small local batches, preserves cue timings/count, and retries batches that
produce overlong English subtitles.

Usage:
  # Default Mistral endpoint + model
  python scripts/translate_vtt_mistral.py \
    --input subs.vtt --output subs_en_mistral.vtt

  # With glossary and summary
  python scripts/translate_vtt_mistral.py \
    --input subs.vtt --output subs_en_mistral.vtt \
    --glossary glossary.tsv --summary "A comedy variety show featuring..."

  # Subtitle-editing batch controls
  python scripts/translate_vtt_mistral.py \
    --input subs.vtt --batch-cues 12 --batch-seconds 45
"""

import argparse
import json
import os
import re
import sys
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional


DEFAULT_MISTRAL_API_BASE = "https://api.mistral.ai"


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

class Batch:
    def __init__(
        self,
        idx: int,
        start_index: int,
        end_index: int,
        cues: list[Cue],
        prev_context: list[Cue],
        next_context: list[Cue],
    ):
        self.idx = idx
        self.start_index = start_index
        self.end_index = end_index
        self.cues = cues
        self.prev_context = prev_context
        self.next_context = next_context


def batch_cues(
    cues: list[Cue],
    max_cues: int = 12,
    max_seconds: float = 45,
    context_cues: int = 2,
) -> list[Batch]:
    batches: list[Batch] = []
    current_start = 0

    while current_start < len(cues):
        current_end = current_start + 1
        while current_end < len(cues):
            duration = cues[current_end].end - cues[current_start].start
            if current_end - current_start + 1 > max_cues or duration > max_seconds:
                break
            current_end += 1

        target = cues[current_start:current_end]
        prev_context = cues[max(0, current_start - context_cues):current_start]
        next_context = cues[current_end:min(len(cues), current_end + context_cues)]
        batches.append(
            Batch(
                idx=len(batches),
                start_index=current_start,
                end_index=current_end,
                cues=target,
                prev_context=prev_context,
                next_context=next_context,
            )
        )
        current_start = current_end

    return batches


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert subtitle translator and subtitle editor for Japanese variety and comedy shows.\n"
    "Translate the target cues into {target} as readable subtitles, not literal glosses.\n"
    "\n"
    "Rules:\n"
    '1. Output JSON only: {{ "translations": [ {{ "id": <number>, "text": "<translated>" }} ] }}\n'
    "2. Return exactly one output item per target cue ID, in the same order. Do not drop, duplicate, or reorder IDs.\n"
    "3. Preserve cue count and timings exactly; you may only rewrite the text.\n"
    "4. Treat the target batch as one subtitle-editing problem. You may redistribute meaning across adjacent target cues if it improves English readability, but every target cue must remain non-empty.\n"
    "5. Prioritize natural, concise subtitle English over literal correspondence.\n"
    "6. Keep punchlines sharp. Compress filler, repetition, and hesitation when needed for readable subtitles.\n"
    "7. Sound effects and annotations in parentheses should become short English subtitle equivalents, e.g. (拍手) -> (applause).\n"
    "8. Aim for about {target_cps:.0f} characters per second. Never exceed {hard_cps:.0f} characters per second if you can reasonably compress the line.\n"
    "9. Keep each cue to at most two lines, with balanced short lines suitable for subtitles.\n"
    "10. Previous and next context are reference only. Translate only the target cues.\n"
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


def _cue_duration(cue: Cue) -> float:
    return max(0.001, cue.end - cue.start)


def _text_char_count(text: str) -> int:
    return len(text.replace("\n", " ").strip())


def _text_cps(text: str, duration: float) -> float:
    return _text_char_count(text) / max(duration, 0.001)


def _normalize_text(text: str) -> str:
    lines = [" ".join(part.strip().split()) for part in text.replace("\r", "\n").split("\n")]
    lines = [line for line in lines if line]
    if not lines:
        return ""
    return "\n".join(lines)


def _wrap_english_text(text: str, max_line_length: int, max_lines: int = 2) -> str:
    normalized = _normalize_text(text)
    if not normalized:
        return ""

    raw_lines = normalized.split("\n")
    if len(raw_lines) <= max_lines and all(len(line) <= max_line_length for line in raw_lines):
        return normalized

    wrapper = textwrap.TextWrapper(
        width=max_line_length,
        break_long_words=False,
        break_on_hyphens=False,
        replace_whitespace=True,
        drop_whitespace=True,
    )
    wrapped = wrapper.wrap(" ".join(raw_lines))
    if not wrapped:
        return normalized
    if len(wrapped) <= max_lines:
        return "\n".join(wrapped)

    first = wrapped[0]
    second = " ".join(wrapped[1:])
    return "\n".join([first, second])


def _checkpoint_path(output_path: str) -> str:
    return f"{output_path}.checkpoint.json"


def _write_json_atomic(path: str, data: dict) -> None:
    tmp_path = f"{path}.tmp"
    Path(tmp_path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def _build_manifest(
    input_path: str,
    output_path: str,
    target_lang: str,
    model: str,
    api_base: str,
    batch_size: int,
    batch_seconds: float,
    context_cues: int,
    target_cps: float,
    hard_cps: float,
    max_line_length: int,
    total_cues: int,
) -> dict:
    return {
        "input": input_path,
        "output": output_path,
        "target_language": target_lang,
        "backend": "mistral",
        "model": model,
        "api_base": api_base,
        "batch_size": batch_size,
        "batch_seconds": batch_seconds,
        "context_cues": context_cues,
        "target_cps": target_cps,
        "hard_cps": hard_cps,
        "max_line_length": max_line_length,
        "total_cues": total_cues,
    }


def _load_checkpoint(path: str, manifest: dict) -> tuple[dict[int, str], set[int], dict[int, dict]]:
    if not Path(path).exists():
        return {}, set(), {}

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    saved_manifest = data.get("manifest", {})
    comparable_keys = [
        "input",
        "output",
        "target_language",
        "backend",
        "model",
        "api_base",
        "batch_size",
        "batch_seconds",
        "context_cues",
        "target_cps",
        "hard_cps",
        "max_line_length",
        "total_cues",
    ]
    mismatches = [
        key
        for key in comparable_keys
        if saved_manifest.get(key) != manifest.get(key)
    ]
    if mismatches:
        raise ValueError(
            "Checkpoint manifest does not match current run: "
            + ", ".join(mismatches)
        )

    translations = {
        int(cue_id): text
        for cue_id, text in data.get("translations", {}).items()
        if isinstance(text, str)
    }
    completed_batches = {int(idx) for idx in data.get("completed_batches", [])}
    batch_diagnostics = {
        int(idx): diag
        for idx, diag in data.get("batch_diagnostics", {}).items()
        if isinstance(diag, dict)
    }
    return translations, completed_batches, batch_diagnostics


def _save_checkpoint(
    checkpoint_path: str,
    manifest: dict,
    translations: dict[int, str],
    completed_batches: set[int],
    batch_diagnostics: dict[int, dict],
) -> None:
    payload = {
        "manifest": manifest,
        "translations": {str(k): v for k, v in sorted(translations.items())},
        "completed_batches": sorted(completed_batches),
        "batch_diagnostics": {
            str(k): batch_diagnostics[k] for k in sorted(batch_diagnostics)
        },
    }
    _write_json_atomic(checkpoint_path, payload)


def _cue_payload(cue_id: int, cue: Cue) -> str:
    safe = cue.text.replace("\n", " / ")
    duration = _cue_duration(cue)
    return (
        f"[{cue_id}] {_seconds_to_time(cue.start)} --> {_seconds_to_time(cue.end)} "
        f"(duration={duration:.3f}s, source_cps={_text_cps(cue.text, duration):.1f}) {safe}"
    )


def _build_user_prompt(
    batch: Batch,
    target_lang: str,
    glossary: Optional[str],
    summary: Optional[str],
    target_cps: float,
    hard_cps: float,
    retry_instruction: Optional[str],
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

    if batch.prev_context:
        lines.append("### PREVIOUS CONTEXT (REFERENCE ONLY, DO NOT TRANSLATE) ###")
        for idx, cue in enumerate(batch.prev_context, 1):
            lines.append(_cue_payload(-idx, cue))
        lines.append("")

    lines.append("### TARGET CUES (TRANSLATE THESE ONLY) ###")
    for i, cue in enumerate(batch.cues, 1):
        lines.append(_cue_payload(i, cue))
    lines.append("")

    if batch.next_context:
        lines.append("### NEXT CONTEXT (REFERENCE ONLY, DO NOT TRANSLATE) ###")
        for idx, cue in enumerate(batch.next_context, 1):
            lines.append(_cue_payload(1000 + idx, cue))
        lines.append("")

    lines.append("### OUTPUT REQUIREMENTS ###")
    lines.append(f"- Translate into {target_lang}.")
    lines.append(f"- Target about {target_cps:.0f} CPS, hard limit {hard_cps:.0f} CPS.")
    lines.append("- Every target cue must remain non-empty.")
    lines.append("- Keep a maximum of 2 subtitle lines per cue.")
    lines.append("- Use short, idiomatic subtitle English.")
    if retry_instruction:
        lines.append(f"- Retry instruction: {retry_instruction}")
    lines.append("")
    lines.append("Output JSON only.")

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
    translations: list[dict],
    original_cues: list[Cue],
    max_line_length: int,
) -> tuple[list[Cue], list[str], set[int]]:
    warnings: list[str] = []
    fallback_ids: set[int] = set()
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
        text = ""
        if item:
            text = _wrap_english_text(item["text"], max_line_length=max_line_length)
        if not text:
            warnings.append(f"Missing or empty translation for cue {i}; using source text as fallback.")
            fallback_ids.add(i)
            text = _wrap_english_text(original_cues[i - 1].text, max_line_length=max_line_length)
        orig = original_cues[i - 1]
        new_cues.append(Cue(orig.start, orig.end, text))

    missing = [cid for cid in range(1, n + 1) if cid not in items]
    if missing:
        warnings.append(f"Input cues not returned: {', '.join(map(str, missing))}")

    return new_cues, warnings, fallback_ids


# ---------------------------------------------------------------------------
# Mistral backend
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


def _extract_message_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
                continue
            if item.get("type") == "text" and isinstance(item.get("content"), str):
                parts.append(item["content"])
        return "".join(parts)
    return ""


def _call_mistral(
    api_base: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    api_key: str,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    req = urllib.request.Request(
        f"{api_base.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace").strip()
        detail = f"{e.code} {e.reason}"
        if body:
            detail = f"{detail}: {body}"
        raise RuntimeError(detail) from e

    message = data["choices"][0]["message"]
    text = _extract_message_text(message.get("content"))
    if text:
        return text
    raise ValueError(f"Mistral returned empty message content: {message}")


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

def _cue_diag(cue_id: int, source: Cue, translated: Cue, used_fallback: bool) -> dict:
    duration = _cue_duration(source)
    return {
        "id": cue_id,
        "start": source.start,
        "end": source.end,
        "duration": duration,
        "source_chars": _text_char_count(source.text),
        "source_cps": round(_text_cps(source.text, duration), 3),
        "translated_chars": _text_char_count(translated.text),
        "translated_cps": round(_text_cps(translated.text, duration), 3),
        "line_count": len(translated.text.splitlines()) or 1,
        "used_fallback": used_fallback,
    }


def _translate_batch(
    batch: Batch,
    target_lang: str,
    glossary: Optional[str],
    summary: Optional[str],
    model: str,
    temperature: float,
    target_cps: float,
    hard_cps: float,
    max_line_length: int,
    api_base: str,
    api_key: str,
) -> tuple[list[Cue], list[str], dict]:
    speaker_instruction = ""
    if _has_speaker_labels(batch.cues):
        speaker_instruction = SPEAKER_INSTRUCTION
    system_prompt = SYSTEM_PROMPT.format(
        target=target_lang,
        speaker_instruction=speaker_instruction,
        target_cps=target_cps,
        hard_cps=hard_cps,
    )
    warnings: list[str] = []
    retries = 0
    retry_instruction: Optional[str] = None
    translated_cues: list[Cue] = []
    fallback_ids: set[int] = set()
    cue_diags: list[dict] = []

    for attempt in range(2):
        if attempt > 0:
            retries += 1
            retry_instruction = (
                "Compress the batch further. Keep every cue non-empty, but shorten wording "
                f"so all target cues stay within {hard_cps:.0f} CPS."
            )
        user_prompt = _build_user_prompt(
            batch=batch,
            target_lang=target_lang,
            glossary=glossary,
            summary=summary,
            target_cps=target_cps,
            hard_cps=hard_cps,
            retry_instruction=retry_instruction,
        )

        raw = _call_with_retry(
            _call_mistral,
            api_base=api_base,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            api_key=api_key,
        )

        parsed = _parse_json_response(raw)
        translations = _validate_structured_output(parsed)
        translated_cues, attempt_warnings, fallback_ids = _reconstruct_vtt(
            translations,
            batch.cues,
            max_line_length=max_line_length,
        )
        warnings = attempt_warnings
        cue_diags = [
            _cue_diag(i + 1, src, dst, (i + 1) in fallback_ids)
            for i, (src, dst) in enumerate(zip(batch.cues, translated_cues))
        ]
        hard_violations = [c for c in cue_diags if c["translated_cps"] > hard_cps or c["line_count"] > 2]
        if not hard_violations:
            break
        if attempt == 0:
            warnings.append(
                "Hard CPS or line-count violations detected; retrying batch with stronger compression."
            )
            continue
    else:
        hard_violations = []

    batch_diag = {
        "batch_index": batch.idx,
        "start": batch.cues[0].start,
        "end": batch.cues[-1].end,
        "cue_count": len(batch.cues),
        "source_chars": sum(_text_char_count(c.text) for c in batch.cues),
        "translated_chars": sum(_text_char_count(c.text) for c in translated_cues),
        "retry_count": retries,
        "warnings": warnings,
        "hard_cps_violations": sum(1 for c in cue_diags if c["translated_cps"] > hard_cps),
        "needs_review": any(
            c["translated_cps"] > hard_cps or c["line_count"] > 2 or c["used_fallback"]
            for c in cue_diags
        ),
        "cues": cue_diags,
    }
    return translated_cues, warnings, batch_diag


def translate_subtitles(
    input_path: str,
    output_path: str,
    target_lang: str,
    model: str,
    temperature: float,
    api_base: str,
    api_key: str,
    glossary_path: Optional[str],
    summary: Optional[str],
    batch_seconds: float,
    batch_size: int,
    context_cues: int,
    target_cps: float,
    hard_cps: float,
    max_line_length: int,
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

    total_duration = cues[-1].end - cues[0].start
    batches = batch_cues(
        cues,
        max_cues=batch_size,
        max_seconds=batch_seconds,
        context_cues=context_cues,
    )
    print(
        f"Batch mode: {len(batches)} batches "
        f"({batch_size} cues / {batch_seconds:.0f}s target, {context_cues} context cues)"
    )

    manifest = _build_manifest(
        input_path=input_path,
        output_path=output_path,
        target_lang=target_lang,
        model=model,
        api_base=api_base,
        batch_size=batch_size,
        batch_seconds=batch_seconds,
        context_cues=context_cues,
        target_cps=target_cps,
        hard_cps=hard_cps,
        max_line_length=max_line_length,
        total_cues=len(cues),
    )
    checkpoint_path = _checkpoint_path(output_path)
    translated_texts, completed_batches, batch_diagnostics = _load_checkpoint(
        checkpoint_path,
        manifest,
    )
    if completed_batches:
        print(
            f"Resuming translation checkpoint: "
            f"{len(completed_batches)}/{len(batches)} batches complete, "
            f"{len(translated_texts)}/{len(cues)} cues cached"
        )

    warnings: list[str] = []

    for batch in batches:
        if batch.idx in completed_batches:
            print(f"[Batch {batch.idx + 1}/{len(batches)}] Skipping completed batch")
            continue
        label = f"Batch {batch.idx + 1}/{len(batches)}"
        cue_range = (
            f"{_seconds_to_time(batch.cues[0].start)} - "
            f"{_seconds_to_time(batch.cues[-1].end)}"
        )
        print(f"[{label}] Translating {len(batch.cues)} cues ({cue_range})...")
        translated, batch_warnings, batch_diag = _translate_batch(
            batch=batch,
            target_lang=target_lang,
            glossary=glossary,
            summary=summary,
            model=model,
            temperature=temperature,
            target_cps=target_cps,
            hard_cps=hard_cps,
            max_line_length=max_line_length,
            api_base=api_base,
            api_key=api_key,
        )
        for offset, cue in enumerate(translated, start=batch.start_index + 1):
            translated_texts[offset] = cue.text
        batch_diagnostics[batch.idx] = batch_diag
        completed_batches.add(batch.idx)
        _save_checkpoint(
            checkpoint_path=checkpoint_path,
            manifest=manifest,
            translations=translated_texts,
            completed_batches=completed_batches,
            batch_diagnostics=batch_diagnostics,
        )
        if batch_warnings:
            for warning in batch_warnings:
                print(f"  Warning: {warning}")
            warnings.extend(batch_warnings)
        print(
            f"  -> {len(translated)} output cues, "
            f"{batch_diag['hard_cps_violations']} hard CPS violations"
        )

    translated_cues: list[Cue] = []
    final_missing: list[int] = []
    for cue_id, cue in enumerate(cues, 1):
        text = translated_texts.get(cue_id, "").strip()
        if not text:
            final_missing.append(cue_id)
            text = _wrap_english_text(cue.text, max_line_length=max_line_length)
        translated_cues.append(Cue(cue.start, cue.end, text))

    if final_missing:
        warnings.append(
            "Missing translated cues after batch processing; using source-text fallback for "
            + ", ".join(map(str, final_missing))
        )

    diagnostics = {
        **manifest,
        "temperature": temperature,
        "total_duration": total_duration,
        "batches": [batch_diagnostics[idx] for idx in sorted(batch_diagnostics)],
    }
    diagnostics["review_batches"] = sum(1 for batch in diagnostics["batches"] if batch["needs_review"])
    diagnostics["total_retries"] = sum(batch["retry_count"] for batch in diagnostics["batches"])
    diagnostics["hard_cps_violations"] = sum(
        batch["hard_cps_violations"] for batch in diagnostics["batches"]
    )

    # Write output
    if output_format == "srt":
        out_text = serialize_srt(translated_cues)
    else:
        out_text = serialize_vtt(translated_cues)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(out_text, encoding="utf-8")
    diagnostics_path = f"{output_path}.diagnostics.json"
    Path(diagnostics_path).write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()

    print(f"\nWrote {len(translated_cues)} translated cues to {output_path}")
    print(f"Wrote translation diagnostics to {diagnostics_path}")
    if warnings:
        print(f"Total warnings: {len(warnings)}")


def main():
    parser = argparse.ArgumentParser(
        description="Translate VTT/SRT subtitles via the Mistral API."
    )
    parser.add_argument("--input", required=True, help="Input VTT or SRT file.")
    parser.add_argument(
        "--output",
        default="",
        help="Output file. Defaults to <input>_<lang>_mistral.<ext>",
    )
    parser.add_argument(
        "--target-lang", default="English", help="Target language (default: English)."
    )
    parser.add_argument(
        "--model",
        default="",
        help="Mistral model name (default: MISTRAL_MODEL or mistral-small-latest).",
    )
    parser.add_argument(
        "--api-base",
        default=os.environ.get("MISTRAL_API_BASE", DEFAULT_MISTRAL_API_BASE),
        help="Base URL for the Mistral API (env: MISTRAL_API_BASE).",
    )
    parser.add_argument(
        "--url",
        default="",
        help="Deprecated alias for --api-base.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("MISTRAL_API_KEY", ""),
        help="API key for the Mistral API (env: MISTRAL_API_KEY).",
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
        "--batch-seconds",
        type=float,
        default=45.0,
        help="Target subtitle-editing batch duration in seconds (default: 45).",
    )
    parser.add_argument(
        "--batch-cues",
        type=int,
        default=12,
        help="Maximum number of cues per translation batch (default: 12).",
    )
    parser.add_argument(
        "--context-cues",
        type=int,
        default=2,
        help="Read-only reference cues before and after each batch (default: 2).",
    )
    parser.add_argument(
        "--target-cps",
        type=float,
        default=17.0,
        help="Target English CPS for subtitle editing (default: 17).",
    )
    parser.add_argument(
        "--hard-cps",
        type=float,
        default=20.0,
        help="Hard English CPS threshold that triggers retry/review (default: 20).",
    )
    parser.add_argument(
        "--max-line-length",
        type=int,
        default=42,
        help="Preferred maximum characters per subtitle line (default: 42).",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=0.0,
        help="Deprecated alias for --batch-seconds.",
    )
    parser.add_argument(
        "--overlap-cues",
        type=int,
        default=-1,
        help="Deprecated alias for --context-cues.",
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
        args.model = os.environ.get("MISTRAL_MODEL", "mistral-small-latest")
    if not args.api_key:
        raise SystemExit("Missing Mistral API key. Set --api-key or MISTRAL_API_KEY.")

    if not args.output:
        stem = Path(args.input).stem
        ext = Path(args.input).suffix or ".vtt"
        lang_slug = args.target_lang.lower().replace(" ", "_")
        args.output = str(Path(args.input).parent / f"{stem}_{lang_slug}_mistral{ext}")

    output_format = args.format
    if not output_format:
        output_format = "srt" if args.output.lower().endswith(".srt") else "vtt"

    batch_seconds = args.batch_seconds
    if args.chunk_seconds > 0:
        batch_seconds = args.chunk_seconds
    context_cues = args.context_cues
    if args.overlap_cues >= 0:
        context_cues = args.overlap_cues
    api_base = args.url or args.api_base

    translate_subtitles(
        input_path=args.input,
        output_path=args.output,
        target_lang=args.target_lang,
        model=args.model,
        temperature=args.temperature,
        api_base=api_base,
        api_key=args.api_key,
        glossary_path=args.glossary or None,
        summary=args.summary or None,
        batch_seconds=batch_seconds,
        batch_size=args.batch_cues,
        context_cues=context_cues,
        target_cps=args.target_cps,
        hard_cps=args.hard_cps,
        max_line_length=args.max_line_length,
        output_format=output_format,
    )


if __name__ == "__main__":
    main()
