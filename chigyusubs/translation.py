"""Shared subtitle translation utilities."""

from __future__ import annotations

import json
import os
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path


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


def serialize_vtt(cues: list[Cue]) -> str:
    parts = ["WEBVTT", ""]
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

