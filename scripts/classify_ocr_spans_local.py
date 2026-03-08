#!/usr/bin/env python3
"""Classify OCR spans with a local OpenAI-compatible LLM and derive curated context."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chigyusubs.metadata import finish_run, metadata_path, start_run, write_metadata
from chigyusubs.paths import find_episode_dir_from_path, find_latest_episode_dir


SYSTEM_PROMPT = """You are classifying OCR spans from Japanese TV/variety show video.

Your job is to turn noisy OCR span summaries into transcription-safe context.

Return a JSON object with these fields:
- class: one of ["name_or_title_card","caption_or_quote","dense_board_or_list","ui_or_credit_noise","mixed"]
- transcription_usefulness: one of ["high","medium","low"]
- glossary_usefulness: one of ["high","medium","low"]
- anchor_terms: array of up to 5 high-confidence names, entities, places, or titles
- aux_terms: array of up to 5 weaker local topic hints
- episode_memory_terms: array of up to 5 recurring names/titles worth keeping episode-wide
- notes: short string

Rules:
- Prefer names, performer names, group names, place names, segment titles, and branded titles.
- `anchor_terms` should be conservative and safe to inject into transcription prompts.
- `aux_terms` can include weaker local hints, but still exclude long instructions, full sentence captions, puzzle questions, equations, and numeric codes.
- For dense boards/lists, keep only the few terms likely to help identify what is being discussed.
- Do not assume on-screen text was spoken aloud.
- Keep terms concise and deduplicated.
- Output strict JSON only. No markdown.
"""


def _resolve_episode_dir(args: argparse.Namespace, *path_attrs: str) -> Path | None:
    if getattr(args, "episode_dir", ""):
        return Path(args.episode_dir)
    for attr in path_attrs:
        raw = getattr(args, attr, "")
        if not raw:
            continue
        found = find_episode_dir_from_path(Path(raw))
        if found:
            return found
    return find_latest_episode_dir()


def _pick_spans_json(episode_dir: Path) -> Path:
    preferred = episode_dir / "ocr" / "qwen_ocr_spans.json"
    if preferred.exists():
        return preferred
    matches = sorted((episode_dir / "ocr").glob("*spans*.json"), key=lambda p: p.stat().st_mtime)
    if matches:
        return matches[-1]
    return preferred


def _pick_context_json(episode_dir: Path) -> Path:
    preferred = episode_dir / "ocr" / "qwen_ocr_context_gemma.json"
    return preferred


def _strip_code_fences(text: str) -> str:
    out = text.strip()
    if out.startswith("```"):
        out = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", out)
        out = re.sub(r"\s*```$", "", out)
    return out.strip()


def _extract_json_object(text: str) -> dict:
    cleaned = _strip_code_fences(text)
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


def _call_chat(url: str, model: str, prompt: str, temperature: float) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"].strip()


def _backoff_delay(attempt: int, cap: float = 30.0, k: float = 2.0) -> float:
    return cap * attempt / (attempt + k)


def _generate_with_retry(url: str, model: str, prompt: str, temperature: float) -> dict:
    max_retries = 8
    for attempt in range(max_retries):
        try:
            raw = _call_chat(url, model, prompt, temperature)
            return _extract_json_object(raw)
        except Exception as e:
            if attempt >= max_retries - 1:
                raise
            delay = _backoff_delay(attempt + 1)
            print(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
            print(f"Retrying in {delay:.0f}s...")
            time.sleep(delay)
    raise RuntimeError("Request failed with no response.")


def _normalize_terms(values, limit: int) -> list[str]:
    out = []
    seen = set()
    if not isinstance(values, list):
        return out
    for item in values:
        term = str(item).strip()
        if not term or term in seen:
            continue
        seen.add(term)
        out.append(term)
        if len(out) >= limit:
            break
    return out


def _build_span_prompt(span: dict) -> str:
    stats = span.get("stats", {})
    rep = "\n".join(f"- {line}" for line in span.get("representative_lines", [])[:12])
    term_counts = "\n".join(
        f"- {term} ({count})" for term, count in list(span.get("term_counts", {}).items())[:12]
    )
    heur = "\n".join(f"- {term}" for term in span.get("prompt_safe_terms", [])[:8])
    return (
        f"Span metadata:\n"
        f"- span_id: {span.get('span_id')}\n"
        f"- time: {span.get('start_sec')} to {span.get('end_sec')}\n"
        f"- current_class: {span.get('class')}\n"
        f"- unique_terms: {stats.get('unique_terms')}\n"
        f"- nonempty_frame_count: {stats.get('nonempty_frame_count')}\n"
        f"- dense_flag: {stats.get('dense_flag')}\n\n"
        f"Representative lines:\n{rep or '- (none)'}\n\n"
        f"Top term counts:\n{term_counts or '- (none)'}\n\n"
        f"Current heuristic prompt-safe terms:\n{heur or '- (none)'}\n"
    )


def _count_completed_spans(spans: list[dict]) -> int:
    return sum(1 for span in spans if isinstance(span.get("llm"), dict))


def _load_resume_spans(out_json: Path, source_spans: list[dict], source_path: Path) -> list[dict]:
    if not out_json.exists():
        return source_spans

    try:
        existing = json.loads(out_json.read_text(encoding="utf-8"))
    except Exception:
        return source_spans

    meta = existing.get("meta", {})
    if meta.get("source") and meta.get("source") != str(source_path):
        return source_spans

    existing_by_id = {
        int(span["span_id"]): span
        for span in existing.get("spans", [])
        if isinstance(span, dict) and "span_id" in span
    }

    merged = []
    for span in source_spans:
        restored = existing_by_id.get(int(span["span_id"]))
        if restored and isinstance(restored.get("llm"), dict):
            merged.append(restored)
        else:
            merged.append(span)
    return merged


def _build_output_payload(
    *,
    in_json: Path,
    url: str,
    model: str,
    temperature: float,
    chunk_sec: float,
    pad_sec: float,
    max_terms: int,
    episode_memory_limit: int,
    spans: list[dict],
    total_spans: int,
) -> dict:
    classified_spans = [span for span in spans if isinstance(span.get("llm"), dict)]
    episode_memory = _build_episode_memory(classified_spans, limit=episode_memory_limit)
    chunk_contexts = _build_chunk_context(classified_spans, chunk_sec=chunk_sec, pad_sec=pad_sec, max_terms=max_terms)
    completed_spans = len(classified_spans)
    return {
        "meta": {
            "source": str(in_json),
            "url": url,
            "model": model,
            "temperature": temperature,
            "chunk_sec": chunk_sec,
            "pad_sec": pad_sec,
            "max_terms": max_terms,
            "episode_memory_terms": len(episode_memory),
            "spans": total_spans,
            "completed_spans": completed_spans,
            "chunk_contexts": len(chunk_contexts),
            "status": "complete" if completed_spans == total_spans else "partial",
        },
        "episode_memory": episode_memory,
        "spans": spans,
        "chunk_contexts": chunk_contexts,
    }


def _write_progress(
    out_json: Path,
    *,
    in_json: Path,
    url: str,
    model: str,
    temperature: float,
    chunk_sec: float,
    pad_sec: float,
    max_terms: int,
    episode_memory_limit: int,
    spans: list[dict],
):
    payload = _build_output_payload(
        in_json=in_json,
        url=url,
        model=model,
        temperature=temperature,
        chunk_sec=chunk_sec,
        pad_sec=pad_sec,
        max_terms=max_terms,
        episode_memory_limit=episode_memory_limit,
        spans=spans,
        total_spans=len(spans),
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _classify_spans(
    spans: list[dict],
    url: str,
    model: str,
    temperature: float,
    *,
    out_json: Path,
    in_json: Path,
    chunk_sec: float,
    pad_sec: float,
    max_terms: int,
    episode_memory_limit: int,
    checkpoint_every: int,
) -> list[dict]:
    annotated = list(spans)
    total = len(annotated)
    completed = _count_completed_spans(annotated)
    if completed:
        print(f"Resuming with {completed}/{total} spans already classified.", flush=True)

    for idx, span in enumerate(annotated, start=1):
        if isinstance(span.get("llm"), dict):
            continue
        print(f"[{idx}/{total}] span {span['span_id']} {span['start_sec']:.1f}-{span['end_sec']:.1f}s", flush=True)
        result = _generate_with_retry(url, model, _build_span_prompt(span), temperature)
        annotated[idx - 1] = {
            **span,
            "llm": {
                "class": str(result.get("class", span.get("class", "mixed"))),
                "transcription_usefulness": str(result.get("transcription_usefulness", "medium")),
                "glossary_usefulness": str(result.get("glossary_usefulness", "medium")),
                "anchor_terms": _normalize_terms(result.get("anchor_terms", []), limit=5),
                "aux_terms": _normalize_terms(result.get("aux_terms", []), limit=5),
                "safe_prompt_terms": _normalize_terms(
                    list(result.get("anchor_terms", [])) + list(result.get("aux_terms", [])),
                    limit=8,
                ),
                "episode_memory_terms": _normalize_terms(result.get("episode_memory_terms", []), limit=5),
                "notes": str(result.get("notes", "")).strip(),
            },
        }
        completed += 1
        if checkpoint_every > 0 and (completed % checkpoint_every == 0 or completed == total):
            _write_progress(
                out_json,
                in_json=in_json,
                url=url,
                model=model,
                temperature=temperature,
                chunk_sec=chunk_sec,
                pad_sec=pad_sec,
                max_terms=max_terms,
                episode_memory_limit=episode_memory_limit,
                spans=annotated,
            )
            print(f"  checkpoint: {completed}/{total}", flush=True)
    return annotated


def _build_episode_memory(spans: list[dict], limit: int) -> list[str]:
    counts = Counter()
    span_presence: dict[str, set[int]] = defaultdict(set)
    time_buckets: dict[str, set[int]] = defaultdict(set)
    for span in spans:
        llm = span.get("llm", {})
        if llm.get("transcription_usefulness") == "low":
            continue
        for term in llm.get("episode_memory_terms", []):
            counts[term] += 1
            span_presence[term].add(span["span_id"])
            time_buckets[term].add(int(float(span.get("start_sec", 0.0)) // 300))
        anchor_terms = llm.get("anchor_terms", llm.get("safe_prompt_terms", []))
        for term in anchor_terms:
            if llm.get("glossary_usefulness") == "high":
                counts[term] += 1
                span_presence[term].add(span["span_id"])
                time_buckets[term].add(int(float(span.get("start_sec", 0.0)) // 300))
    ranked = sorted(
        [
            (
                (
                    len(span_presence[t]) * 3
                    + counts[t]
                    + len(time_buckets[t]) * 2
                    + (1 if len(time_buckets[t]) >= 2 else 0)
                ),
                t,
            )
            for t in counts
        ],
        reverse=True,
    )
    return [term for _, term in ranked[:limit]]


def _is_dense_llm_span(span: dict) -> bool:
    llm_class = span.get("llm", {}).get("class")
    if llm_class:
        return llm_class == "dense_board_or_list"
    return span.get("class") == "dense_board_or_list"


def _collect_non_dense_anchor_terms(spans: list[dict]) -> set[str]:
    seen = set()
    for span in spans:
        if _is_dense_llm_span(span):
            continue
        llm = span.get("llm", {})
        for term in llm.get("anchor_terms", llm.get("safe_prompt_terms", [])):
            seen.add(term)
    return seen


def _build_chunk_context(spans: list[dict], chunk_sec: float, pad_sec: float, max_terms: int) -> list[dict]:
    max_end_sec = max(float(span.get("end_sec", 0.0)) for span in spans) if spans else 0.0
    chunk_count = int(max_end_sec // chunk_sec) + 1 if max_end_sec > 0 else 0
    episode_memory_set = set(_build_episode_memory(spans, limit=1000))
    non_dense_anchor_terms = _collect_non_dense_anchor_terms(spans)
    chunk_contexts = []
    for chunk_id in range(chunk_count):
        start_sec = chunk_id * chunk_sec
        end_sec = start_sec + chunk_sec
        scores: dict[str, float] = defaultdict(float)
        overlaps = []
        guaranteed_terms: list[str] = []
        for span in spans:
            if float(span["end_sec"]) < start_sec - pad_sec or float(span["start_sec"]) > end_sec + pad_sec:
                continue
            llm = span.get("llm", {})
            if not llm:
                continue
            usefulness = llm.get("transcription_usefulness", "medium")
            anchor_terms = list(llm.get("anchor_terms", llm.get("safe_prompt_terms", [])))
            aux_terms = list(llm.get("aux_terms", []))
            if usefulness == "low" and not anchor_terms:
                continue
            overlaps.append({"span_id": span["span_id"], "class": llm.get("class", span.get("class", "mixed"))})
            weight = {"high": 2.0, "medium": 1.0, "low": 0.5}.get(usefulness, 1.0)
            dense = _is_dense_llm_span(span)
            if dense:
                aux_terms = [
                    term for term in aux_terms
                    if term in non_dense_anchor_terms or term in episode_memory_set
                ]
            if usefulness == "low":
                aux_terms = []
            if dense:
                for term in anchor_terms[:3]:
                    if term not in guaranteed_terms:
                        guaranteed_terms.append(term)
            else:
                for term in anchor_terms[:2]:
                    if term not in guaranteed_terms:
                        guaranteed_terms.append(term)
            for idx, term in enumerate(anchor_terms):
                scores[term] += weight * 3.0 * max(0.5, max_terms - idx)
            for idx, term in enumerate(aux_terms):
                scores[term] += weight * 1.0 * max(0.5, (max_terms / 2) - idx)
        ranked_terms = [term for _, term in sorted((score, term) for term, score in scores.items())[::-1]]
        final_terms = list(guaranteed_terms)
        for term in ranked_terms:
            if term not in final_terms:
                final_terms.append(term)
        chunk_contexts.append(
            {
                "chunk_id": chunk_id,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "terms": final_terms[:max_terms],
                "span_ids": [item["span_id"] for item in overlaps],
            }
        )
    return chunk_contexts


def main():
    parser = argparse.ArgumentParser(description="Classify OCR spans with a local OpenAI-compatible LLM.")
    parser.add_argument("--episode-dir", default="", help="Episode root, e.g. samples/episodes/<episode_slug>")
    parser.add_argument("--in-json", default="", help="Defaults to <episode>/ocr/qwen_ocr_spans.json")
    parser.add_argument("--out-json", default="", help="Defaults to <episode>/ocr/qwen_ocr_context_gemma.json")
    parser.add_argument("--url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8080"), help="OpenAI-compatible base URL.")
    parser.add_argument("--model", default=os.environ.get("OCR_FILTER_MODEL", "gemma3-27b"), help="Local model/alias for span classification.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--chunk-sec", type=float, default=30.0)
    parser.add_argument("--pad-sec", type=float, default=2.0)
    parser.add_argument("--max-terms", type=int, default=12)
    parser.add_argument("--episode-memory-limit", type=int, default=30)
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Write partial progress every N newly classified spans.")
    args = parser.parse_args()

    run = start_run("ocr_span_classify_local")
    episode_dir = _resolve_episode_dir(args, "in_json", "out_json")
    if args.in_json:
        in_json = Path(args.in_json)
    elif episode_dir:
        in_json = _pick_spans_json(episode_dir)
    else:
        raise SystemExit("Could not infer spans JSON. Pass --in-json or --episode-dir.")

    if args.out_json:
        out_json = Path(args.out_json)
    elif episode_dir:
        out_json = _pick_context_json(episode_dir)
    else:
        raise SystemExit("Could not infer output JSON. Pass --out-json or --episode-dir.")

    payload = json.loads(in_json.read_text(encoding="utf-8"))
    source_spans = payload.get("spans", [])
    if not source_spans:
        raise SystemExit(f"No spans found in {in_json}")

    spans = _load_resume_spans(out_json, source_spans, in_json)
    annotated_spans = _classify_spans(
        spans,
        url=args.url,
        model=args.model,
        temperature=args.temperature,
        out_json=out_json,
        in_json=in_json,
        chunk_sec=args.chunk_sec,
        pad_sec=args.pad_sec,
        max_terms=args.max_terms,
        episode_memory_limit=args.episode_memory_limit,
        checkpoint_every=args.checkpoint_every,
    )
    out_payload = _build_output_payload(
        in_json=in_json,
        url=args.url,
        model=args.model,
        temperature=args.temperature,
        chunk_sec=args.chunk_sec,
        pad_sec=args.pad_sec,
        max_terms=args.max_terms,
        episode_memory_limit=args.episode_memory_limit,
        spans=annotated_spans,
        total_spans=len(source_spans),
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Gemma OCR context written: {out_json}")

    metadata = finish_run(
        run,
        episode_dir=str(episode_dir) if episode_dir else None,
        inputs={"spans_json": str(in_json)},
        outputs={"context_json": str(out_json)},
        settings={
            "url": args.url,
            "model": args.model,
            "temperature": args.temperature,
            "chunk_sec": args.chunk_sec,
            "pad_sec": args.pad_sec,
            "max_terms": args.max_terms,
            "episode_memory_limit": args.episode_memory_limit,
            "checkpoint_every": args.checkpoint_every,
        },
        stats={
            "spans": len(annotated_spans),
            "chunk_contexts": len(out_payload["chunk_contexts"]),
            "episode_memory_terms": len(out_payload["episode_memory"]),
        },
    )
    write_metadata(out_json, metadata)
    print(f"Metadata written: {metadata_path(out_json)}")


if __name__ == "__main__":
    main()
