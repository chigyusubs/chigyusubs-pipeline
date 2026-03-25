"""Shared subtitle translation utilities."""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


_TIME_RE = re.compile(r"(?:(\d{2}):)?(\d{2}):(\d{2})[.,](\d{3})")


@dataclass
class Cue:
    start: float
    end: float
    text: str


@dataclass
class Batch:
    idx: int
    start_index: int
    end_index: int
    cues: list[Cue]
    prev_context: list[Cue]
    next_context: list[Cue]


def time_to_seconds(tc: str) -> float:
    m = _TIME_RE.match(tc.strip())
    if not m:
        raise ValueError(f"Invalid timecode: {tc}")
    h = int(m.group(1) or 0)
    mi = int(m.group(2))
    s = int(m.group(3))
    ms = int(m.group(4))
    return h * 3600 + mi * 60 + s + ms / 1000


def seconds_to_time(seconds: float) -> str:
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
        start = time_to_seconds(parts[0].strip())
        end = time_to_seconds(parts[1].strip().split(" ")[0])
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
        start = time_to_seconds(parts[0].strip())
        end = time_to_seconds(parts[1].strip().split(" ")[0])
        idx += 1
        text_lines: list[str] = []
        while idx < len(lines) and lines[idx].strip():
            text_lines.append(lines[idx])
            idx += 1
        cues.append(Cue(start, end, "\n".join(text_lines)))
    return cues


def serialize_vtt(cues: list[Cue], note_lines: list[str] | None = None) -> str:
    parts = ["WEBVTT", ""]
    if note_lines:
        parts.append("NOTE")
        parts.extend(note_lines)
        parts.append("")
    for cue in cues:
        parts.append(f"{seconds_to_time(cue.start)} --> {seconds_to_time(cue.end)}")
        parts.append(cue.text)
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def serialize_srt(cues: list[Cue]) -> str:
    parts: list[str] = []
    for i, cue in enumerate(cues, 1):
        start = seconds_to_time(cue.start).replace(".", ",")
        end = seconds_to_time(cue.end).replace(".", ",")
        parts.append(str(i))
        parts.append(f"{start} --> {end}")
        parts.append(cue.text)
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def batch_cues(
    cues: list[Cue],
    max_cues: int = 12,
    max_seconds: float = 45.0,
    context_cues: int = 2,
) -> list[Batch]:
    batches: list[Batch] = []
    current_start = 0
    while current_start < len(cues):
        current_end = current_start + 1
        while current_end < len(cues):
            duration = cues[current_end].end - cues[current_start].start
            if current_end - current_start + 1 > max_cues:
                break
            if max_seconds > 0 and duration > max_seconds:
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


def cue_duration(cue: Cue) -> float:
    return max(0.001, cue.end - cue.start)


def text_char_count(text: str) -> int:
    return len(text.replace("\n", " ").strip())


def text_cps(text: str, duration: float) -> float:
    return text_char_count(text) / max(duration, 0.001)


def normalize_text(text: str) -> str:
    lines = [" ".join(part.strip().split()) for part in text.replace("\r", "\n").split("\n")]
    lines = [line for line in lines if line]
    if not lines:
        return ""
    return "\n".join(lines)


def wrap_english_text(text: str, max_line_length: int, max_lines: int = 2) -> str:
    normalized = normalize_text(text)
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


def checkpoint_path(output_path: str) -> str:
    return f"{output_path}.checkpoint.json"


def write_json_atomic(path: str | Path, data: dict) -> None:
    path = str(path)
    tmp_path = f"{path}.tmp"
    Path(tmp_path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def build_manifest(
    input_path: str,
    output_path: str,
    target_lang: str,
    backend: str,
    model: str,
    location: str,
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
        "backend": backend,
        "model": model,
        "location": location,
        "batch_size": batch_size,
        "batch_seconds": batch_seconds,
        "context_cues": context_cues,
        "target_cps": target_cps,
        "hard_cps": hard_cps,
        "max_line_length": max_line_length,
        "total_cues": total_cues,
    }


def load_checkpoint(path: str, manifest: dict) -> tuple[dict[int, str], set[int], dict[int, dict]]:
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
        "location",
        "batch_size",
        "batch_seconds",
        "context_cues",
        "target_cps",
        "hard_cps",
        "max_line_length",
        "total_cues",
    ]
    mismatches = [
        key for key in comparable_keys
        if saved_manifest.get(key) != manifest.get(key)
    ]
    if mismatches:
        raise ValueError(
            "Checkpoint manifest does not match current run: " + ", ".join(mismatches)
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


def save_checkpoint(
    checkpoint_path_value: str,
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
    write_json_atomic(checkpoint_path_value, payload)


def cue_diag(cue_id: int, source: Cue, translated: Cue, used_fallback: bool) -> dict:
    duration = cue_duration(source)
    return {
        "id": cue_id,
        "start": source.start,
        "end": source.end,
        "duration": duration,
        "source_chars": text_char_count(source.text),
        "source_cps": round(text_cps(source.text, duration), 3),
        "translated_chars": text_char_count(translated.text),
        "translated_cps": round(text_cps(translated.text, duration), 3),
        "line_count": len(translated.text.splitlines()) or 1,
        "used_fallback": used_fallback,
    }


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

_SPEAKER_RE = re.compile(r"^[\w\u3000-\u9fff\uff00-\uffef]+:\s")


def has_speaker_labels(cues: list[Cue]) -> bool:
    if not cues:
        return False
    labeled = sum(1 for c in cues if _SPEAKER_RE.match(c.text))
    return labeled / len(cues) > 0.2


def cue_payload_text(cue_id: int, cue: Cue) -> str:
    safe = cue.text.replace("\n", " / ")
    duration = cue_duration(cue)
    return (
        f"[{cue_id}] {seconds_to_time(cue.start)} --> {seconds_to_time(cue.end)} "
        f"(duration={duration:.3f}s, source_cps={text_cps(cue.text, duration):.1f}) {safe}"
    )


def build_user_prompt(
    batch: Batch,
    target_lang: str,
    glossary: Optional[str],
    summary: Optional[str],
    target_cps: float,
    hard_cps: float,
    retry_instruction: Optional[str] = None,
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
            lines.append(cue_payload_text(-idx, cue))
        lines.append("")

    lines.append("### TARGET CUES (TRANSLATE THESE ONLY) ###")
    for i, cue in enumerate(batch.cues, 1):
        lines.append(cue_payload_text(i, cue))
    lines.append("")

    if batch.next_context:
        lines.append("### NEXT CONTEXT (REFERENCE ONLY, DO NOT TRANSLATE) ###")
        for idx, cue in enumerate(batch.next_context, 1):
            lines.append(cue_payload_text(1000 + idx, cue))
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
# Response parsing and validation
# ---------------------------------------------------------------------------

def strip_code_fences(text: str) -> str:
    out = text.strip()
    if out.startswith("```"):
        out = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", out)
        out = re.sub(r"\s*```$", "", out)
    return out.strip()


def parse_json_response(raw: str) -> dict:
    cleaned = strip_code_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        return json.loads(cleaned[start : end + 1])
    if not raw.strip():
        return {"translations": []}
    raise ValueError(f"Failed to parse JSON from model response: {raw}")


def validate_structured_output(data: dict) -> list[dict]:
    if not isinstance(data, dict):
        raise ValueError("Output is not a valid JSON object")
    translations = data.get("translations")
    if not isinstance(translations, list):
        raise ValueError(f"Missing or invalid 'translations' array in: {data}")

    validated = []
    for i, item in enumerate(translations):
        if not isinstance(item, dict):
            raise ValueError(f"Item at index {i} is not an object")
        cue_id = item.get("id")
        # Accept string IDs (e.g. "3") — some models return strings in JSON mode
        if isinstance(cue_id, str):
            try:
                cue_id = int(cue_id)
            except ValueError:
                cue_id = None
        # Fall back to positional ID (1-based) when id is missing or invalid
        if cue_id is None or not isinstance(cue_id, (int, float)) or cue_id != int(cue_id):
            cue_id = i + 1
        text = item.get("text")
        if text is None:
            text = ""
        if not isinstance(text, str):
            raise ValueError(f"Item at index {i} has invalid 'text'")
        validated.append({"id": int(cue_id), "text": text})
    return validated


def reconstruct_vtt(
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
            text = wrap_english_text(item["text"], max_line_length=max_line_length)
        if not text:
            warnings.append(f"Missing or empty translation for cue {i}; using source text as fallback.")
            fallback_ids.add(i)
            text = wrap_english_text(original_cues[i - 1].text, max_line_length=max_line_length)
        orig = original_cues[i - 1]
        new_cues.append(Cue(orig.start, orig.end, text))

    missing = [cid for cid in range(1, n + 1) if cid not in items]
    if missing:
        warnings.append(f"Input cues not returned: {', '.join(map(str, missing))}")

    return new_cues, warnings, fallback_ids


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

def backoff_delay(attempt: int, cap: float = 30.0, k: float = 2.0) -> float:
    """Hyperbolic backoff: ramps fast, asymptotes to cap."""
    return cap * attempt / (attempt + k)


def call_with_retry(
    call_fn: Callable[..., str],
    max_retries: int = 10,
    **kwargs,
) -> str:
    for attempt in range(max_retries):
        try:
            return call_fn(**kwargs)
        except Exception as e:
            if attempt < max_retries - 1:
                msg = str(e).strip().splitlines()[0] if str(e).strip() else repr(e)
                delay = backoff_delay(attempt + 1)
                print(f"  API call failed (attempt {attempt + 1}/{max_retries}): {msg}")
                print(f"  Retrying in {delay:.0f}s...")
                time.sleep(delay)
                continue
            raise
    raise RuntimeError("API request failed with no response.")


# ---------------------------------------------------------------------------
# Main batch translation loop
# ---------------------------------------------------------------------------

# Type: call_fn(system_prompt, user_prompt, temperature) -> raw response text
BackendCallFn = Callable[[str, str, float], str]


def _translate_batch(
    batch: Batch,
    target_lang: str,
    glossary: Optional[str],
    summary: Optional[str],
    call_fn: BackendCallFn,
    temperature: float,
    target_cps: float,
    hard_cps: float,
    max_line_length: int,
) -> tuple[list[Cue], list[str], dict]:
    speaker_instruction = SPEAKER_INSTRUCTION if has_speaker_labels(batch.cues) else ""
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
        user_prompt = build_user_prompt(
            batch=batch,
            target_lang=target_lang,
            glossary=glossary,
            summary=summary,
            target_cps=target_cps,
            hard_cps=hard_cps,
            retry_instruction=retry_instruction,
        )

        raw = call_fn(system_prompt, user_prompt, temperature)
        parsed = parse_json_response(raw)
        translations = validate_structured_output(parsed)
        translated_cues, attempt_warnings, fallback_ids = reconstruct_vtt(
            translations, batch.cues, max_line_length=max_line_length,
        )
        warnings = attempt_warnings
        cue_diags = [
            cue_diag(i + 1, src, dst, (i + 1) in fallback_ids)
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

    batch_diag = {
        "batch_index": batch.idx,
        "start": batch.cues[0].start,
        "end": batch.cues[-1].end,
        "cue_count": len(batch.cues),
        "source_chars": sum(text_char_count(c.text) for c in batch.cues),
        "translated_chars": sum(text_char_count(c.text) for c in translated_cues),
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
    call_fn: BackendCallFn,
    manifest: dict,
    glossary_path: Optional[str] = None,
    summary: Optional[str] = None,
    batch_seconds: float = 45.0,
    batch_size: int = 12,
    context_cues: int = 2,
    target_cps: float = 17.0,
    hard_cps: float = 20.0,
    max_line_length: int = 42,
    temperature: float = 0.2,
    output_format: str = "vtt",
) -> None:
    """Run the full batch translation pipeline.

    call_fn(system_prompt, user_prompt, temperature) -> raw response text.
    manifest is a dict used for checkpoint validation (caller builds it).
    """
    raw = Path(input_path).read_text(encoding="utf-8")
    if input_path.lower().endswith(".srt"):
        cues = parse_srt(raw)
    else:
        cues = parse_vtt(raw)

    if not cues:
        print("No cues found in input file.")
        sys.exit(1)

    # Update total_cues in manifest (callers may pass 0 as placeholder)
    manifest["total_cues"] = len(cues)

    print(f"Parsed {len(cues)} cues from {input_path}")

    glossary: Optional[str] = None
    if glossary_path:
        glossary = Path(glossary_path).read_text(encoding="utf-8").strip()
        print(f"Loaded glossary from {glossary_path}")

    total_duration = cues[-1].end - cues[0].start
    batches = batch_cues(
        cues, max_cues=batch_size, max_seconds=batch_seconds, context_cues=context_cues,
    )
    print(
        f"Batch mode: {len(batches)} batches "
        f"({batch_size} cues / {batch_seconds:.0f}s target, {context_cues} context cues)"
    )

    cp = checkpoint_path(output_path)
    translated_texts, completed_batches, batch_diagnostics = load_checkpoint(cp, manifest)
    if completed_batches:
        print(
            f"Resuming translation checkpoint: "
            f"{len(completed_batches)}/{len(batches)} batches complete, "
            f"{len(translated_texts)}/{len(cues)} cues cached"
        )

    all_warnings: list[str] = []

    for batch in batches:
        if batch.idx in completed_batches:
            print(f"[Batch {batch.idx + 1}/{len(batches)}] Skipping completed batch")
            continue
        label = f"Batch {batch.idx + 1}/{len(batches)}"
        cue_range = (
            f"{seconds_to_time(batch.cues[0].start)} - "
            f"{seconds_to_time(batch.cues[-1].end)}"
        )
        print(f"[{label}] Translating {len(batch.cues)} cues ({cue_range})...")
        translated, batch_warnings, batch_diag = _translate_batch(
            batch=batch,
            target_lang=target_lang,
            glossary=glossary,
            summary=summary,
            call_fn=call_fn,
            temperature=temperature,
            target_cps=target_cps,
            hard_cps=hard_cps,
            max_line_length=max_line_length,
        )
        for offset, cue in enumerate(translated, start=batch.start_index + 1):
            translated_texts[offset] = cue.text
        batch_diagnostics[batch.idx] = batch_diag
        completed_batches.add(batch.idx)
        save_checkpoint(
            checkpoint_path_value=cp,
            manifest=manifest,
            translations=translated_texts,
            completed_batches=completed_batches,
            batch_diagnostics=batch_diagnostics,
        )
        if batch_warnings:
            for warning in batch_warnings:
                print(f"  Warning: {warning}")
            all_warnings.extend(batch_warnings)
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
            text = wrap_english_text(cue.text, max_line_length=max_line_length)
        translated_cues.append(Cue(cue.start, cue.end, text))

    if final_missing:
        all_warnings.append(
            "Missing translated cues after batch processing; using source-text fallback for "
            + ", ".join(map(str, final_missing))
        )

    diagnostics = {
        **manifest,
        "temperature": temperature,
        "total_duration": total_duration,
        "batches": [batch_diagnostics[idx] for idx in sorted(batch_diagnostics)],
    }
    diagnostics["review_batches"] = sum(1 for b in diagnostics["batches"] if b["needs_review"])
    diagnostics["total_retries"] = sum(b["retry_count"] for b in diagnostics["batches"])
    diagnostics["hard_cps_violations"] = sum(b["hard_cps_violations"] for b in diagnostics["batches"])

    if output_format == "srt":
        out_text = serialize_srt(translated_cues)
    else:
        out_text = serialize_vtt(translated_cues)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(out_text, encoding="utf-8")
    diagnostics_path = f"{output_path}.diagnostics.json"
    Path(diagnostics_path).write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    if Path(cp).exists():
        Path(cp).unlink()

    print(f"\nWrote {len(translated_cues)} translated cues to {output_path}")
    print(f"Wrote translation diagnostics to {diagnostics_path}")
    if all_warnings:
        print(f"Total warnings: {len(all_warnings)}")
