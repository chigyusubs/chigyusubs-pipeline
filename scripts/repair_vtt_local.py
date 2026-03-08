#!/usr/bin/env python3
"""Repair a reflowed VTT using aligned words and a local candidate-choosing LLM.

This is a post-reflow repair pass. It preserves transcript content and cue order,
but repairs short fragment cues, selectively extends short complete cues into
silence, and splits oversized cues at legal aligned-word boundaries.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chigyusubs.metadata import finish_run, metadata_path, start_run, write_metadata
from chigyusubs.vtt import write_vtt


JSON_RESPONSE_SCHEMA = {"type": "json_object"}

SYSTEM_PROMPT = """You are a Japanese subtitle boundary editor.

You are NOT rewriting subtitles. You must choose the best candidate action from a
precomputed list.

Goals:
- eliminate obviously broken split-word / split-phrase cues
- prefer merging fragmentary short cues
- only extend timings when the cue is already semantically complete and there is
  explicit silent slack
- split oversized cues at natural phrase boundaries
- keep subtitle timing monotonic and readable

Rules:
- never invent a new action id
- do not ask for text rewrites
- prefer the smallest repair that fixes the boundary problem
- if multiple candidates are acceptable, prefer the one that preserves stronger
  phrase integrity

Return strict JSON only:
{"chosen_action":"candidate_id","reason_short":"..."}
"""

PUNCT_RE = re.compile(r"[。！？!?」』）】]$")
JP_CHAR_RE = re.compile(r"[ぁ-んァ-ン一-龯ー]")
TIMESTAMP_RE = re.compile(
    r"^(?:(\d{2}):)?(\d{2}):(\d{2}\.\d{3})\s+-->\s+(?:(\d{2}):)?(\d{2}):(\d{2}\.\d{3})$"
)


def log(msg: str = ""):
    print(msg, flush=True)


def _flatten_text(text: str) -> str:
    return text.replace("\n", "").strip()


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _cue_duration(cue: dict) -> float:
    return max(0.0, float(cue["end"]) - float(cue["start"]))


def _cue_char_len(cue: dict) -> int:
    return len(_flatten_text(cue["text"]))


def _parse_timestamp(ts_line: str) -> tuple[float, float]:
    m = TIMESTAMP_RE.match(ts_line.strip())
    if not m:
        raise ValueError(f"Invalid VTT timestamp line: {ts_line}")

    sh, sm, ss, eh, em, es = m.groups()
    start = (int(sh) * 3600 if sh else 0) + int(sm) * 60 + float(ss)
    end = (int(eh) * 3600 if eh else 0) + int(em) * 60 + float(es)
    return start, end


def load_vtt(path: str) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8")
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    cues: list[dict] = []

    for block in blocks:
        if block == "WEBVTT":
            continue
        lines = block.splitlines()
        if not lines:
            continue
        if "-->" in lines[0]:
            ts_line = lines[0]
            text_lines = lines[1:]
        elif len(lines) >= 2 and "-->" in lines[1]:
            ts_line = lines[1]
            text_lines = lines[2:]
        else:
            continue
        start, end = _parse_timestamp(ts_line)
        cue_text = "\n".join(text_lines).strip()
        if not cue_text:
            continue
        cues.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "text": cue_text,
            }
        )

    return cues


def load_aligned_words(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    words: list[dict] = []
    for segment in data:
        for raw_word in segment.get("words", []):
            text = str(raw_word.get("word", ""))
            if not text:
                continue
            words.append(
                {
                    "index": len(words),
                    "text": text,
                    "start": round(float(raw_word["start"]), 3),
                    "end": round(float(raw_word["end"]), 3),
                }
            )
    words.sort(key=lambda w: (w["start"], w["end"], w["index"]))
    for idx, word in enumerate(words):
        word["index"] = idx
    return words


def _cue_from_word_span(
    words: list[dict],
    start_idx: int,
    end_idx: int,
    *,
    text_override: str | None = None,
    start_override: float | None = None,
    end_override: float | None = None,
) -> dict:
    if start_idx > end_idx:
        raise ValueError(f"Invalid word span: {start_idx}>{end_idx}")
    return {
        "start": round(words[start_idx]["start"] if start_override is None else start_override, 3),
        "end": round(words[end_idx]["end"] if end_override is None else end_override, 3),
        "text": text_override if text_override is not None else "".join(
            words[i]["text"] for i in range(start_idx, end_idx + 1)
        ),
        "word_start": start_idx,
        "word_end": end_idx,
    }


def _time_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _fallback_match_by_time(cue: dict, words: list[dict], pos: int) -> tuple[int, int] | None:
    candidates = []
    for word in words[pos:]:
        if word["start"] > cue["end"] + 0.8:
            break
        overlaps = _time_overlap(cue["start"], cue["end"], word["start"], word["end"]) > 0
        inside_zero = cue["start"] <= word["start"] <= cue["end"] and word["start"] == word["end"]
        near_edge = cue["start"] - 0.3 <= word["start"] <= cue["end"] + 0.3
        if overlaps or inside_zero or near_edge:
            candidates.append(word)
    if not candidates:
        return None
    start_idx = candidates[0]["index"]
    end_idx = candidates[-1]["index"]
    return start_idx, end_idx


def _find_local_exact_match(
    *,
    cue: dict,
    words: list[dict],
    pos: int,
    lookback: int = 120,
    lookahead: int = 120,
    max_span_words: int = 180,
) -> tuple[int, int] | None:
    target = _normalize_for_match(_flatten_text(cue["text"]))
    if not target:
        return None

    start_min = max(0, pos - lookback)
    start_max = min(len(words) - 1, pos + lookahead)
    best: tuple[float, int, int] | None = None

    for start_idx in range(start_min, start_max + 1):
        acc = ""
        for end_idx in range(start_idx, min(len(words), start_idx + max_span_words)):
            acc += words[end_idx]["text"]
            norm_acc = _normalize_for_match(acc)
            if norm_acc == target:
                time_penalty = abs(words[start_idx]["start"] - cue["start"]) + abs(words[end_idx]["end"] - cue["end"])
                positional_penalty = 0 if start_idx >= pos else 0.25
                score = time_penalty + positional_penalty
                if best is None or score < best[0]:
                    best = (score, start_idx, end_idx)
                break
            if len(norm_acc) > len(target) + 16:
                break

    if best is None:
        return None
    return best[1], best[2]


def _find_time_window_exact_match(
    *,
    cue: dict,
    words: list[dict],
    max_span_words: int = 220,
) -> tuple[int, int] | None:
    target = _normalize_for_match(_flatten_text(cue["text"]))
    if not target:
        return None

    start_window = cue["start"] - 1.0
    end_window = cue["end"] + 0.8
    best: tuple[float, int, int] | None = None

    for start_idx, word in enumerate(words):
        if word["start"] > end_window:
            break
        if word["end"] < start_window:
            continue

        acc = ""
        for end_idx in range(start_idx, min(len(words), start_idx + max_span_words)):
            acc += words[end_idx]["text"]
            norm_acc = _normalize_for_match(acc)
            if norm_acc == target:
                time_penalty = abs(words[start_idx]["start"] - cue["start"]) + abs(words[end_idx]["end"] - cue["end"])
                if best is None or time_penalty < best[0]:
                    best = (time_penalty, start_idx, end_idx)
                break
            if len(norm_acc) > len(target) + 16:
                break

    if best is None:
        return None
    return best[1], best[2]


def map_cues_to_words(cues: list[dict], words: list[dict]) -> list[dict]:
    mapped: list[dict] = []
    pos = 0
    for cue in cues:
        target = _flatten_text(cue["text"])
        if not target:
            continue
        matched = _find_local_exact_match(cue=cue, words=words, pos=pos)
        if matched is None:
            matched = _find_time_window_exact_match(cue=cue, words=words)
        if matched is None:
            matched = _fallback_match_by_time(cue, words, pos)
        if matched is None:
            raise ValueError(
                f"Failed to map cue text to aligned words near {cue['start']:.3f}-{cue['end']:.3f}: {target!r}"
            )
        start_idx, end_idx = matched
        pos = end_idx + 1

        mapped.append(
            {
                "start": cue["start"],
                "end": cue["end"],
                "text": cue["text"],
                "word_start": start_idx,
                "word_end": end_idx,
            }
        )
    return mapped


def _serialize_cue(cue: dict) -> dict:
    return {
        "start": round(float(cue["start"]), 3),
        "end": round(float(cue["end"]), 3),
        "duration": round(_cue_duration(cue), 3),
        "text": cue["text"],
        "flat_text": _flatten_text(cue["text"]),
        "word_start": int(cue["word_start"]),
        "word_end": int(cue["word_end"]),
    }


def _serialize_cues(cues: list[dict]) -> list[dict]:
    return [_serialize_cue(c) for c in cues]


def _is_complete(text: str) -> bool:
    return bool(PUNCT_RE.search(_flatten_text(text)))


def _starts_jp(text: str) -> bool:
    flat = _flatten_text(text)
    return bool(flat and JP_CHAR_RE.search(flat[0]))


def _ends_jp(text: str) -> bool:
    flat = _flatten_text(text)
    return bool(flat and JP_CHAR_RE.search(flat[-1]))


def _boundary_looks_split(left_text: str, right_text: str) -> bool:
    left = _flatten_text(left_text)
    right = _flatten_text(right_text)
    if not left or not right:
        return False
    if _is_complete(left):
        return False
    return _ends_jp(left) and _starts_jp(right)


def _cue_is_fragment(cues: list[dict], idx: int) -> bool:
    cue = cues[idx]
    text = _flatten_text(cue["text"])
    if not text:
        return False
    if len(text) <= 2:
        return True
    if len(text) <= 4 and not _is_complete(text):
        return True

    prev_cue = cues[idx - 1] if idx > 0 else None
    next_cue = cues[idx + 1] if idx + 1 < len(cues) else None

    if next_cue and _boundary_looks_split(text, next_cue["text"]) and (len(text) <= 10 or _cue_char_len(next_cue) <= 10):
        return True
    if prev_cue and _boundary_looks_split(prev_cue["text"], text) and len(text) <= 10:
        return True
    return False


def _cue_is_short(cues: list[dict], idx: int) -> bool:
    cue = cues[idx]
    return _cue_duration(cue) < 0.8 or _cue_char_len(cue) <= 4 or _cue_is_fragment(cues, idx)


def _cue_is_long(cue: dict) -> bool:
    return _cue_duration(cue) > 6.5 or _cue_char_len(cue) > 42


def _available_gap_before(cues: list[dict], idx: int) -> float:
    if idx <= 0:
        return max(0.0, cues[idx]["start"])
    return max(0.0, cues[idx]["start"] - cues[idx - 1]["end"])


def _available_gap_after(cues: list[dict], idx: int) -> float:
    if idx + 1 >= len(cues):
        return 0.0
    return max(0.0, cues[idx + 1]["start"] - cues[idx]["end"])


def _collect_fragment_run(cues: list[dict], idx: int, max_run: int = 4) -> int:
    end = idx
    while end + 1 < len(cues) and end - idx + 1 < max_run and _cue_is_short(cues, end + 1):
        end += 1
    return end


def _can_merge_span(
    cues: list[dict],
    start_idx: int,
    end_idx: int,
    *,
    max_internal_gap: float = 0.6,
    max_duration: float = 6.5,
    max_chars: int = 48,
) -> bool:
    if start_idx >= end_idx:
        return True
    merged_duration = cues[end_idx]["end"] - cues[start_idx]["start"]
    if merged_duration > max_duration:
        return False
    merged_chars = sum(_cue_char_len(cues[i]) for i in range(start_idx, end_idx + 1))
    if merged_chars > max_chars:
        return False
    for idx in range(start_idx, end_idx):
        if cues[idx + 1]["start"] - cues[idx]["end"] > max_internal_gap:
            return False
    return True


def _build_extension_candidate(
    cues: list[dict],
    words: list[dict],
    idx: int,
    *,
    extend_before: float,
    extend_after: float,
    candidate_id: str,
    label: str,
    fallback_rank: int,
) -> dict | None:
    cue = cues[idx]
    before_gap = _available_gap_before(cues, idx)
    after_gap = _available_gap_after(cues, idx)
    before = min(extend_before, before_gap)
    after = min(extend_after, after_gap)
    new_start = cue["start"] - before
    new_end = cue["end"] + after
    if new_end - new_start < max(1.2, _cue_duration(cue) + 0.2):
        return None
    repaired = _cue_from_word_span(
        words,
        cue["word_start"],
        cue["word_end"],
        text_override=cue["text"],
        start_override=new_start,
        end_override=new_end,
    )
    return {
        "id": candidate_id,
        "label": label,
        "reason": "extend short complete cue into neighboring silence",
        "replace_start": idx,
        "replace_end": idx,
        "result_cues": [repaired],
        "fallback_rank": fallback_rank,
    }


def _rank_split_points(cue: dict, words: list[dict], max_points: int = 5) -> list[tuple[int, int, float]]:
    points: list[tuple[int, int, float]] = []
    for split_after in range(cue["word_start"], cue["word_end"]):
        left = _cue_from_word_span(words, cue["word_start"], split_after)
        right = _cue_from_word_span(words, split_after + 1, cue["word_end"])
        if _cue_duration(left) < 0.3 or _cue_duration(right) < 0.3:
            continue
        left_chars = _cue_char_len(left)
        right_chars = _cue_char_len(right)
        if left_chars < 4 or right_chars < 4:
            continue
        score = 0
        gap = max(0.0, words[split_after + 1]["start"] - words[split_after]["end"])
        if PUNCT_RE.search(words[split_after]["text"]):
            score += 3
        if gap >= 0.18:
            score += 2
        elif gap >= 0.1:
            score += 1
        if 6 <= left_chars <= 28 and 6 <= right_chars <= 28:
            score += 1
        if score <= 0:
            continue
        points.append((split_after, score, gap))
    points.sort(key=lambda item: (-item[1], -item[2], item[0]))
    return points[:max_points]


def _apply_candidate(cues: list[dict], candidate: dict) -> list[dict]:
    return cues[: candidate["replace_start"]] + candidate["result_cues"] + cues[candidate["replace_end"] + 1 :]


def _window_slice(cues: list[dict], center_start: int, center_end: int, radius: int = 2) -> tuple[int, int]:
    return max(0, center_start - radius), min(len(cues), center_end + radius + 1)


def _make_candidate_summary(candidate: dict, cues: list[dict], words: list[dict]) -> dict:
    applied = _apply_candidate(copy.deepcopy(cues), candidate)
    w_start, w_end = _window_slice(applied, candidate["replace_start"], candidate["replace_start"] + len(candidate["result_cues"]) - 1)
    return {
        "id": candidate["id"],
        "label": candidate["label"],
        "reason": candidate["reason"],
        "after_window": _serialize_cues(applied[w_start:w_end]),
    }


def build_candidates(cues: list[dict], words: list[dict], idx: int) -> tuple[str, list[dict]]:
    cue = cues[idx]
    candidates = [
        {
            "id": "keep",
            "label": "keep",
            "reason": "leave the current cue boundaries unchanged",
            "replace_start": idx,
            "replace_end": idx,
            "result_cues": [copy.deepcopy(cue)],
            "fallback_rank": 999,
        }
    ]

    if _cue_is_long(cue):
        for n, (split_after, _score, _gap) in enumerate(_rank_split_points(cue, words), start=1):
            left = _cue_from_word_span(words, cue["word_start"], split_after)
            right = _cue_from_word_span(words, split_after + 1, cue["word_end"])
            candidates.append(
                {
                    "id": f"split_{n}",
                    "label": f"split after word {split_after}",
                    "reason": "split long cue at a high-confidence timing boundary",
                    "replace_start": idx,
                    "replace_end": idx,
                    "result_cues": [left, right],
                    "fallback_rank": n,
                }
            )
        return "long", candidates

    run_end = _collect_fragment_run(cues, idx)
    if idx > 0 and _can_merge_span(cues, idx - 1, idx):
        candidates.append(
            {
                "id": "merge_left",
                "label": "merge with left cue",
                "reason": "join the short cue to the previous cue",
                "replace_start": idx - 1,
                "replace_end": idx,
                "result_cues": [
                    _cue_from_word_span(words, cues[idx - 1]["word_start"], cue["word_end"])
                ],
                "fallback_rank": 4,
            }
        )
    if idx + 1 < len(cues) and _can_merge_span(cues, idx, idx + 1):
        candidates.append(
            {
                "id": "merge_right",
                "label": "merge with right cue",
                "reason": "join the short cue to the next cue",
                "replace_start": idx,
                "replace_end": idx + 1,
                "result_cues": [
                    _cue_from_word_span(words, cue["word_start"], cues[idx + 1]["word_end"])
                ],
                "fallback_rank": 2,
            }
        )
    if run_end > idx and _can_merge_span(cues, idx, run_end):
        candidates.append(
            {
                "id": "merge_chain",
                "label": f"merge short run {idx}-{run_end}",
                "reason": "collapse a local run of fragmentary short cues into one cue",
                "replace_start": idx,
                "replace_end": run_end,
                "result_cues": [
                    _cue_from_word_span(words, cue["word_start"], cues[run_end]["word_end"])
                ],
                "fallback_rank": 1,
            }
        )

    if _is_complete(cue["text"]):
        for spec in (
            ("extend_right", "extend right", 0.0, 0.6, 5),
            ("extend_both", "extend both", 0.2, 0.6, 6),
        ):
            candidate = _build_extension_candidate(
                cues,
                words,
                idx,
                extend_before=spec[2],
                extend_after=spec[3],
                candidate_id=spec[0],
                label=spec[1],
                fallback_rank=spec[4],
            )
            if candidate is not None:
                candidates.append(candidate)

    return "short", candidates


def _extract_json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(cleaned[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Model response did not contain a valid JSON object.")


def _call_local_llm(
    *,
    url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
) -> dict:
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "response_format": JSON_RESPONSE_SCHEMA,
        },
        ensure_ascii=False,
    ).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=payload,
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return _extract_json_object(body["choices"][0]["message"]["content"].strip())


def choose_candidate(
    *,
    cues: list[dict],
    idx: int,
    issue_type: str,
    candidate_summaries: list[dict],
    candidate_ids: set[str],
    url: str,
    api_key: str,
    model: str,
    temperature: float,
    deterministic_only: bool,
) -> tuple[str, str, bool]:
    if deterministic_only:
        return "", "deterministic-only mode", True

    window_start, window_end = _window_slice(cues, idx, idx)
    payload = {
        "issue_type": issue_type,
        "target_index": idx,
        "window_before": _serialize_cues(cues[window_start:window_end]),
        "candidates": candidate_summaries,
    }
    user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)

    max_retries = 6
    for attempt in range(max_retries):
        try:
            parsed = _call_local_llm(
                url=url,
                api_key=api_key,
                model=model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=temperature,
            )
            chosen = str(parsed.get("chosen_action", "")).strip()
            reason = str(parsed.get("reason_short", "")).strip()
            if chosen in candidate_ids:
                return chosen, reason, False
            raise ValueError(f"Model chose unknown action: {chosen!r}")
        except Exception as e:
            if attempt >= max_retries - 1:
                break
            delay = (attempt + 1) * 4
            log(f"  LLM decision failed (attempt {attempt + 1}/{max_retries}): {e}")
            log(f"  Retrying in {delay}s...")
            time.sleep(delay)

    return "", "fallback after LLM failure", True


def fallback_choice(candidates: list[dict]) -> str:
    ordered = sorted(candidates, key=lambda c: (c["fallback_rank"], c["id"]))
    return ordered[0]["id"]


def _save_checkpoint(
    path: Path,
    *,
    input_vtt: str,
    input_words: str,
    output_vtt: str,
    settings: dict,
    current_pass: int,
    cursor: int,
    cues: list[dict],
    decisions: list[dict],
    completed: bool,
):
    payload = {
        "step": "repair_vtt_local",
        "input_vtt": input_vtt,
        "input_words": input_words,
        "output_vtt": output_vtt,
        "settings": settings,
        "current_pass": current_pass,
        "cursor": cursor,
        "completed": completed,
        "current_cues": _serialize_cues(cues),
        "decisions": decisions,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_checkpoint(path: Path, *, input_vtt: str, input_words: str) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("input_vtt") != input_vtt or payload.get("input_words") != input_words:
        return None
    return payload


def _restore_cues(serialized: list[dict]) -> list[dict]:
    return [
        {
            "start": round(float(c["start"]), 3),
            "end": round(float(c["end"]), 3),
            "text": str(c["text"]),
            "word_start": int(c["word_start"]),
            "word_end": int(c["word_end"]),
        }
        for c in serialized
    ]


def _default_output_paths(input_vtt: Path) -> tuple[Path, Path]:
    base = input_vtt.with_suffix("")
    return (
        base.with_name(base.name + "_gemma_repair.vtt"),
        base.with_name(base.name + "_gemma_repair.decisions.json"),
    )


def _state_signature(cues: list[dict], idx: int, radius: int = 2) -> tuple:
    start, end = _window_slice(cues, idx, idx, radius=radius)
    return (
        idx,
        tuple(
            (
                round(c["start"], 3),
                round(c["end"], 3),
                _flatten_text(c["text"]),
                c["word_start"],
                c["word_end"],
            )
            for c in cues[start:end]
        ),
    )


def main():
    run = start_run("repair_vtt_local")
    parser = argparse.ArgumentParser(description="Repair a reflowed VTT using aligned words and a local LLM.")
    parser.add_argument("--input-vtt", required=True, help="Existing reflowed VTT to repair.")
    parser.add_argument("--input-words", required=True, help="Aligned words JSON from align_ctc.py.")
    parser.add_argument("--output-vtt", default="", help="Output repaired VTT path.")
    parser.add_argument("--output-decisions", default="", help="Output JSON checkpoint/diagnostics path.")
    parser.add_argument(
        "--url",
        default=os.environ.get("CUE_REPAIR_BASE_URL", os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8082")),
        help="OpenAI-compatible base URL.",
    )
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""), help="Optional OpenAI-compatible API key.")
    parser.add_argument("--model", default=os.environ.get("CUE_REPAIR_MODEL", "gemma3-27b"), help="Model alias exposed by llama.cpp.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Local LLM sampling temperature.")
    parser.add_argument("--max-passes", type=int, default=2, help="Maximum left-to-right repair passes.")
    parser.add_argument("--force", action="store_true", help="Ignore any existing checkpoint and restart from source inputs.")
    parser.add_argument("--deterministic-only", action="store_true", help="Skip LLM calls and use deterministic fallback choices only.")
    args = parser.parse_args()

    input_vtt = Path(args.input_vtt)
    input_words = Path(args.input_words)
    if not input_vtt.exists():
        raise SystemExit(f"Input VTT not found: {input_vtt}")
    if not input_words.exists():
        raise SystemExit(f"Input words JSON not found: {input_words}")

    default_vtt, default_decisions = _default_output_paths(input_vtt)
    output_vtt = Path(args.output_vtt) if args.output_vtt else default_vtt
    output_decisions = Path(args.output_decisions) if args.output_decisions else default_decisions
    output_vtt.parent.mkdir(parents=True, exist_ok=True)
    output_decisions.parent.mkdir(parents=True, exist_ok=True)

    words = load_aligned_words(str(input_words))
    log(f"Aligned words: {len(words)}")

    checkpoint = None if args.force else _load_checkpoint(output_decisions, input_vtt=str(input_vtt), input_words=str(input_words))
    if checkpoint:
        cues = _restore_cues(checkpoint.get("current_cues", []))
        decisions = checkpoint.get("decisions", [])
        current_pass = int(checkpoint.get("current_pass", 0))
        cursor = int(checkpoint.get("cursor", 0))
        log(f"Resuming from pass {current_pass + 1}, cue {cursor + 1}")
    else:
        source_cues = load_vtt(str(input_vtt))
        log(f"Input cues: {len(source_cues)}")
        cues = map_cues_to_words(source_cues, words)
        decisions = []
        current_pass = 0
        cursor = 0
        _save_checkpoint(
            output_decisions,
            input_vtt=str(input_vtt),
            input_words=str(input_words),
            output_vtt=str(output_vtt),
            settings={
                "url": args.url,
                "model": args.model,
                "temperature": args.temperature,
                "max_passes": args.max_passes,
                "deterministic_only": args.deterministic_only,
            },
            current_pass=current_pass,
            cursor=cursor,
            cues=cues,
            decisions=decisions,
            completed=False,
        )

    if checkpoint and checkpoint.get("completed"):
        log("Checkpoint already completed; rewriting VTT from checkpoint state.")
        write_vtt(cues, str(output_vtt))
        return

    while current_pass < args.max_passes:
        pass_changed = False
        seen_states: set[tuple] = set()
        log(f"\n=== Pass {current_pass + 1}/{args.max_passes} ===")
        while cursor < len(cues):
            target_position = cursor + 1
            issue_type = ""
            if _cue_is_long(cues[cursor]):
                issue_type = "long"
            elif _cue_is_short(cues, cursor):
                issue_type = "short"

            if not issue_type:
                cursor += 1
                continue

            # Guard against local oscillation in the same pass.
            signature = _state_signature(cues, cursor)
            if signature in seen_states:
                log(f"  cue {target_position}: repeated local state, skipping")
                cursor += 1
                continue
            seen_states.add(signature)

            issue_type, candidates = build_candidates(cues, words, cursor)
            if len(candidates) == 1:
                cursor += 1
                continue

            candidate_summaries = [_make_candidate_summary(c, cues, words) for c in candidates]
            chosen_id, llm_reason, used_fallback = choose_candidate(
                cues=cues,
                idx=cursor,
                issue_type=issue_type,
                candidate_summaries=candidate_summaries,
                candidate_ids={c["id"] for c in candidates},
                url=args.url,
                api_key=args.api_key,
                model=args.model,
                temperature=args.temperature,
                deterministic_only=args.deterministic_only,
            )
            if not chosen_id:
                chosen_id = fallback_choice(candidates)

            chosen = next(c for c in candidates if c["id"] == chosen_id)
            replace_start = chosen["replace_start"]
            replace_end = chosen["replace_end"]
            window_before_start, window_before_end = _window_slice(cues, replace_start, replace_end)
            before_window = _serialize_cues(cues[window_before_start:window_before_end])
            cues = _apply_candidate(cues, chosen)
            new_replace_end = replace_start + len(chosen["result_cues"]) - 1
            window_after_start, window_after_end = _window_slice(cues, replace_start, new_replace_end)
            after_window = _serialize_cues(cues[window_after_start:window_after_end])

            decision = {
                "pass": current_pass + 1,
                "cursor_before": cursor,
                "issue_type": issue_type,
                "chosen_action": chosen_id,
                "llm_reason": llm_reason,
                "used_fallback": used_fallback,
                "before_window": before_window,
                "after_window": after_window,
                "candidates": candidate_summaries,
            }
            decisions.append(decision)
            pass_changed = pass_changed or chosen_id != "keep"

            cursor = max(0, replace_start - 1) if chosen_id != "keep" else cursor + 1
            log(
                f"  cue {target_position}: {issue_type} -> {chosen_id}"
                f"{' (fallback)' if used_fallback else ''}"
            )

            _save_checkpoint(
                output_decisions,
                input_vtt=str(input_vtt),
                input_words=str(input_words),
                output_vtt=str(output_vtt),
                settings={
                    "url": args.url,
                    "model": args.model,
                    "temperature": args.temperature,
                    "max_passes": args.max_passes,
                    "deterministic_only": args.deterministic_only,
                },
                current_pass=current_pass,
                cursor=cursor,
                cues=cues,
                decisions=decisions,
                completed=False,
            )

        if not pass_changed:
            break
        current_pass += 1
        cursor = 0
        _save_checkpoint(
            output_decisions,
            input_vtt=str(input_vtt),
            input_words=str(input_words),
            output_vtt=str(output_vtt),
            settings={
                "url": args.url,
                "model": args.model,
                "temperature": args.temperature,
                "max_passes": args.max_passes,
                "deterministic_only": args.deterministic_only,
            },
            current_pass=current_pass,
            cursor=cursor,
            cues=cues,
            decisions=decisions,
            completed=False,
        )

    final_cues = [
        {
            "start": cue["start"],
            "end": cue["end"],
            "text": cue["text"],
        }
        for cue in cues
    ]
    write_vtt(final_cues, str(output_vtt))
    _save_checkpoint(
        output_decisions,
        input_vtt=str(input_vtt),
        input_words=str(input_words),
        output_vtt=str(output_vtt),
        settings={
            "url": args.url,
            "model": args.model,
            "temperature": args.temperature,
            "max_passes": args.max_passes,
            "deterministic_only": args.deterministic_only,
        },
        current_pass=current_pass,
        cursor=len(cues),
        cues=cues,
        decisions=decisions,
        completed=True,
    )

    metadata = finish_run(
        run,
        inputs={"input_vtt": str(input_vtt), "input_words": str(input_words)},
        outputs={"repaired_vtt": str(output_vtt), "decisions_json": str(output_decisions)},
        settings={
            "url": args.url,
            "model": args.model,
            "temperature": args.temperature,
            "max_passes": args.max_passes,
            "deterministic_only": args.deterministic_only,
        },
        stats={
            "input_cues": len(load_vtt(str(input_vtt))),
            "output_cues": len(final_cues),
            "aligned_words": len(words),
            "decision_steps": len(decisions),
        },
    )
    write_metadata(output_decisions, metadata)
    log(f"\nWritten VTT: {output_vtt}")
    log(f"Written decisions: {output_decisions}")
    log(f"Metadata written: {metadata_path(output_decisions)}")


if __name__ == "__main__":
    main()
