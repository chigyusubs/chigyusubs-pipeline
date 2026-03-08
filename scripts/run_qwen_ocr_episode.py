#!/usr/bin/env python3
"""Run full-episode OCR with llama.cpp Qwen-VL, then derive OCR artifacts.

Usage:
  # Stage 1: OCR all frames (resume-safe)
  python scripts/run_qwen_ocr_episode.py ocr \
    --episode-dir samples/episodes/wednesday_downtown_2025-02-05_406 \
    --url http://127.0.0.1:8787 \
    --model qwen3.5-9b

  # Stage 2: Build OCR spans from OCR results
  python scripts/run_qwen_ocr_episode.py spans \
    --episode-dir samples/episodes/wednesday_downtown_2025-02-05_406

  # Stage 3: Build glossary from OCR results
  python scripts/run_qwen_ocr_episode.py filter \
    --episode-dir samples/episodes/wednesday_downtown_2025-02-05_406 \
    --chunk-sec 30 \
    --top-global 80 \
    --top-chunk 20
"""

import argparse
import base64
import json
import os
import re
import sys
import unicodedata
import urllib.error
import urllib.request
from collections import defaultdict
from collections import Counter
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chigyusubs.metadata import finish_run, metadata_path, start_run, write_metadata
from chigyusubs.paths import find_episode_dir_from_path, find_latest_episode_dir


OCR_SYSTEM_PROMPT = """\
You are an OCR system for Japanese TV and streaming video.
Read ALL visible text exactly as shown.
Output plain text only, one line per distinct text region.
Preserve numbers, symbols, and punctuation when they are visible text.
If there is no readable text, output exactly [[NO_TEXT]].
Do not describe the image.
Do not translate. Do not summarize. Do not explain.
Do not infer missing text.
Do not continue patterns.
Do not repeat text that is not visibly present.
"""


# Frequent non-lexical TV overlay/UI terms.
NOISE_EXACT = {
    "提供",
    "字幕",
    "次回予告",
    "番組",
    "再生",
    "停止",
}


def _pick_ocr_frames_dir(episode_dir: Path) -> Path:
    candidates = [
        episode_dir / "frames" / "workdir",
        episode_dir / "frames" / "raw_2s",
        episode_dir / "frames",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _pick_filter_in_jsonl(episode_dir: Path) -> Path:
    preferred = episode_dir / "ocr" / "qwen_ocr_results.jsonl"
    if preferred.exists():
        return preferred
    existing = sorted((episode_dir / "ocr").glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    non_failed = [p for p in existing if "fail" not in p.name.lower()]
    if non_failed:
        return non_failed[-1]
    if existing:
        return existing[-1]
    return preferred


def _pick_spans_json(episode_dir: Path) -> Path:
    preferred = episode_dir / "ocr" / "qwen_ocr_spans.json"
    if preferred.exists():
        return preferred
    existing = sorted((episode_dir / "ocr").glob("*spans*.json"), key=lambda p: p.stat().st_mtime)
    if existing:
        return existing[-1]
    return preferred


def _resolve_episode_dir(args: argparse.Namespace, *path_attrs: str) -> Optional[Path]:
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


def image_to_data_url(path: Path) -> str:
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    return f"data:{mime};base64,{b64}"


def parse_frame_number(frame_name: str) -> int:
    m = re.search(r"(\d+)", frame_name)
    if not m:
        return 0
    return int(m.group(1))


def load_processed_frames(jsonl_path: Path) -> set[str]:
    done = set()
    if not jsonl_path.exists():
        return done
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                frame = row.get("frame")
                raw = row.get("raw")
                lines = row.get("lines")
                if frame and raw is not None and lines is not None:
                    done.add(frame)
            except json.JSONDecodeError:
                continue
    return done


def clean_ocr_lines(raw_text: str) -> list[str]:
    lines = []
    for ln in raw_text.splitlines():
        t = ln.strip()
        if not t:
            continue
        # Strip obvious bullet markers, but preserve leading digits and symbols
        # because they are often part of real puzzle boards, counts, or codes.
        t = re.sub(r"^[\-\*\u2022]+", "", t).strip()
        lower = t.lower().strip()
        if t == "[[NO_TEXT]]":
            continue
        if re.fullmatch(
            r"(there is )?no (visible |readable )?text( found)?( in( this)? image)?[.! ]*",
            lower,
        ):
            continue
        if t:
            lines.append(t)
    return lines


def call_vision(url: str, model: str, image_path: Path, max_tokens: int) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": OCR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                    {
                        "type": "text",
                        "text": (
                            "Extract all visible text exactly as shown. "
                            "Return plain text only, one line per text region. "
                            "Preserve numbers, symbols, and punctuation when visibly present. "
                            "Do not infer missing text. Do not continue patterns. "
                            "If there is no readable text, return exactly [[NO_TEXT]] and nothing else."
                        ),
                    },
                ],
            },
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def call_vision_with_retry(
    url: str,
    model: str,
    image_path: Path,
    max_tokens: int,
    retries: int,
) -> str:
    last_err = None
    for attempt in range(retries + 1):
        try:
            return call_vision(url, model, image_path, max_tokens)
        except urllib.error.HTTPError as e:
            last_err = e
            # Transient server-side errors.
            if e.code in {500, 503} and attempt < retries:
                continue
            raise
        except urllib.error.URLError as e:
            last_err = e
            if attempt < retries:
                continue
            raise
    raise RuntimeError(f"OCR failed after retries: {last_err}")


def run_ocr(args: argparse.Namespace):
    run = start_run("ocr")
    episode_dir = _resolve_episode_dir(args, "frames_dir", "out_jsonl")
    if args.frames_dir:
        frames_dir = Path(args.frames_dir)
    elif episode_dir:
        frames_dir = _pick_ocr_frames_dir(episode_dir)
    else:
        raise SystemExit("Could not infer frames dir. Pass --frames-dir or --episode-dir.")

    if args.out_jsonl:
        out_jsonl = Path(args.out_jsonl)
    elif episode_dir:
        out_jsonl = episode_dir / "ocr" / "qwen_ocr_results.jsonl"
    else:
        raise SystemExit("Could not infer OCR output path. Pass --out-jsonl or --episode-dir.")

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if args.limit > 0:
        frame_paths = frame_paths[: args.limit]

    if not frame_paths:
        raise SystemExit(f"No frames found in {frames_dir}")

    done = load_processed_frames(out_jsonl)
    pending = [p for p in frame_paths if p.name not in done]
    print(f"Frames total: {len(frame_paths)}")
    print(f"Already done: {len(done)}")
    print(f"Pending: {len(pending)}")

    if not pending:
        print("Nothing to do.")
        return

    ok = 0
    fail = 0
    stopped_due_to_max_fail = False
    with out_jsonl.open("a", encoding="utf-8") as fout:
        for i, frame_path in enumerate(pending, start=1):
            frame_no = parse_frame_number(frame_path.name)
            time_sec = (frame_no - 1) * args.frame_step_sec if frame_no > 0 else None
            try:
                raw = call_vision_with_retry(
                    args.url,
                    args.model,
                    frame_path,
                    args.max_tokens,
                    args.retries,
                )
                lines = clean_ocr_lines(raw)
                record = {
                    "frame": frame_path.name,
                    "frame_no": frame_no,
                    "time_sec": time_sec,
                    "raw": raw,
                    "lines": lines,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                ok += 1
                if i % args.progress_every == 0:
                    print(f"[{i}/{len(pending)}] ok={ok} fail={fail} last={frame_path.name} lines={len(lines)}")
            except Exception as e:  # noqa: BLE001
                fail += 1
                err = {
                    "frame": frame_path.name,
                    "frame_no": frame_no,
                    "time_sec": time_sec,
                    "error": str(e),
                }
                fout.write(json.dumps(err, ensure_ascii=False) + "\n")
                fout.flush()
                print(f"[{i}/{len(pending)}] FAIL {frame_path.name}: {e}")
                if fail >= args.max_fail:
                    print("Stopping due to max_fail.")
                    stopped_due_to_max_fail = True
                    break

    print(f"Done OCR. ok={ok}, fail={fail}, output={out_jsonl}")
    metadata = finish_run(
        run,
        episode_dir=str(episode_dir) if episode_dir else None,
        inputs={
            "frames_dir": str(frames_dir),
        },
        outputs={
            "ocr_jsonl": str(out_jsonl),
        },
        ocr_prompt=OCR_SYSTEM_PROMPT,
        client_settings={
            "url": args.url,
            "model": args.model,
            "max_tokens": args.max_tokens,
            "frame_step_sec": args.frame_step_sec,
            "progress_every": args.progress_every,
            "max_fail": args.max_fail,
            "retries": args.retries,
            "limit": args.limit,
        },
        server_settings=args.server_settings or None,
        stats={
            "frames_total_considered": len(frame_paths),
            "already_done_before_run": len(done),
            "pending_at_start": len(pending),
            "ok_this_run": ok,
            "fail_this_run": fail,
            "stopped_due_to_max_fail": stopped_due_to_max_fail,
        },
    )
    write_metadata(out_jsonl, metadata)
    print(f"Metadata written: {metadata_path(out_jsonl)}")


def normalize_text(s: str) -> str:
    t = unicodedata.normalize("NFKC", s)
    t = t.replace("\u3000", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def is_noise_term(term: str) -> bool:
    if not term:
        return True
    if term in NOISE_EXACT:
        return True
    if len(term) <= 1:
        return True
    # Pure punctuation/symbols.
    if re.fullmatch(r"[^\w\u3040-\u30ff\u3400-\u9fff]+", term):
        return True
    # Very short pure ASCII fragments.
    if len(term) <= 2 and re.fullmatch(r"[A-Za-z0-9]+", term):
        return True
    # Pathological OCR garbage like "111111111..." or "ーーーーーー".
    if re.fullmatch(r"(.)\1{7,}", term):
        return True
    # Production credits are not useful transcription hints.
    if "©" in term or re.search(r"\bProduced by\b", term, flags=re.IGNORECASE):
        return True
    return False


def read_ocr_terms(in_jsonl: Path) -> list[tuple[int, float, str]]:
    rows = []
    with in_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "lines" not in obj:
                continue
            frame_no = int(obj.get("frame_no", 0))
            time_sec = float(obj.get("time_sec", 0.0))
            seen_in_frame = set()
            for raw in obj["lines"]:
                term = normalize_text(raw)
                if not term or term in seen_in_frame:
                    continue
                seen_in_frame.add(term)
                rows.append((frame_no, time_sec, term))
    return rows


def read_ocr_frames(in_jsonl: Path) -> list[dict]:
    frames = []
    with in_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "lines" not in obj:
                continue
            frame = str(obj.get("frame", ""))
            frame_no = int(obj.get("frame_no", 0) or parse_frame_number(frame))
            time_sec = float(obj.get("time_sec", 0.0))
            normalized_terms = []
            seen_in_frame = set()
            for raw in obj["lines"]:
                term = normalize_text(raw)
                if not term or term in seen_in_frame:
                    continue
                seen_in_frame.add(term)
                normalized_terms.append(term)
            sanitized_terms, frame_stats = sanitize_frame_terms(normalized_terms)
            kept_terms = [term for term in sanitized_terms if not is_noise_term(term)]
            frames.append(
                {
                    "frame": frame,
                    "frame_no": frame_no,
                    "time_sec": time_sec,
                    "lines": kept_terms,
                    "raw_unique_lines": len(normalized_terms),
                    "frame_stats": frame_stats,
                }
            )
    frames.sort(key=lambda row: row["frame_no"])
    return frames


def _is_short_spam_token(term: str) -> bool:
    if len(term) <= 2:
        return True
    if term in {"答え", "解答"}:
        return True
    if re.fullmatch(r"[?？!！↓↑→←]+", term):
        return True
    return False


def sanitize_frame_terms(terms: list[str]) -> tuple[list[str], dict]:
    """Suppress repetitive dense-frame OCR spam while keeping informative terms.

    This is intentionally conservative: it only deduplicates repeated short/symbolic
    lines within a single frame and flags obviously dense frames for diagnostics.
    """
    if not terms:
        return [], {
            "raw_count": 0,
            "kept_count": 0,
            "unique_count": 0,
            "dominant_term": "",
            "dominant_count": 0,
            "dense_flag": False,
            "repetition_flag": False,
        }

    counts = Counter(terms)
    dominant_term, dominant_count = counts.most_common(1)[0]
    dense_flag = len(terms) >= 40
    repetition_flag = dominant_count >= 5 and _is_short_spam_token(dominant_term)

    kept = []
    seen_short_spam = set()
    for term in terms:
        if counts[term] >= 5 and _is_short_spam_token(term):
            if term in seen_short_spam:
                continue
            seen_short_spam.add(term)
        kept.append(term)

    return kept, {
        "raw_count": len(terms),
        "kept_count": len(kept),
        "unique_count": len(counts),
        "dominant_term": dominant_term,
        "dominant_count": dominant_count,
        "dense_flag": dense_flag,
        "repetition_flag": repetition_flag,
    }


def overlap_coeff(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def is_sentence_like_term(term: str) -> bool:
    if len(term) >= 24:
        return True
    if term.endswith(("。", "！", "？", "!", "?")):
        return True
    if "、" in term and len(term) >= 12:
        return True
    return False


def is_instruction_like_term(term: str) -> bool:
    if any(marker in term for marker in {"答え", "解答", "方法", "してください", "できません", "可能", "掛け算", "動かして"}):
        return True
    if "?" in term or "？" in term:
        return True
    if re.search(r"[↓↑→←]", term):
        return True
    if re.search(r"\d{3,}", term):
        return True
    if re.fullmatch(r".*[はをにへでとのがの]$", term):
        return True
    if re.fullmatch(r".*(している|されている|できる|できない|できません|する|した|して|なる|なった|ある|いる|せよ|とは)$", term):
        return True
    return False


def is_entity_like_term(term: str) -> bool:
    if is_noise_term(term) or is_sentence_like_term(term) or is_instruction_like_term(term):
        return False
    if len(term) < 2 or len(term) > 20:
        return False
    if re.search(r"\d{3,}", term):
        return False
    return True


def is_prompt_safe_term(term: str) -> bool:
    if is_noise_term(term):
        return False
    if is_sentence_like_term(term):
        return False
    if len(term) > 24:
        return False
    if is_instruction_like_term(term):
        return False
    if re.fullmatch(r"[0-9A-Za-z?？!！↓↑→←\-=(){}./:~〜]+", term):
        return False
    if re.search(r"\b(?:AM|PM)\b", term):
        return False
    if re.search(r"\d{1,2}:\d{2}", term):
        return False
    if re.search(r"[(月火水木金土日)]", term) and re.search(r"\d{1,2}:\d{2}", term):
        return False
    if "パーソナリティ" in term or "番組HP" in term:
        return False
    return True


def _rank_span_terms(term_counts: Counter, *, prompt_safe_only: bool, limit: int) -> list[str]:
    pairs = []
    for term, count in term_counts.items():
        if prompt_safe_only and not is_prompt_safe_term(term):
            continue
        score = (count * 2.0) - (0.03 * max(0, len(term) - 12))
        if not prompt_safe_only and is_sentence_like_term(term):
            score -= 0.5
        if prompt_safe_only:
            if is_entity_like_term(term):
                score += 2.0
            if re.search(r"[ァ-ヶ・]", term):
                score += 0.5
            if " " in term:
                score += 0.5
            if re.search(r"[A-Za-z]", term) and not re.fullmatch(r"[A-Za-z0-9 ]+", term):
                score += 0.25
        pairs.append((score, count, -len(term), term))
    pairs.sort(reverse=True)
    return [term for _, _, _, term in pairs[:limit]]


def _span_core_terms(term_counts: Counter) -> set[str]:
    persistent = {term for term, count in term_counts.items() if count >= 2}
    if persistent:
        return persistent
    return set(_rank_span_terms(term_counts, prompt_safe_only=False, limit=8))


def _classify_span(term_counts: Counter, stats: dict) -> str:
    if not term_counts:
        return "empty"
    if stats["dense_flag"]:
        return "dense_board_or_list"
    if stats["sentence_like_ratio"] >= 0.6 and stats["avg_line_length"] >= 12:
        return "caption_or_quote"
    if stats["prompt_safe_term_count"] >= 1 and stats["avg_line_length"] <= 14:
        return "name_or_title_card"
    return "mixed"


def build_episode_memory_from_spans(spans: list[dict], limit: int) -> list[str]:
    counts = Counter()
    span_presence: dict[str, set[int]] = defaultdict(set)
    for span in spans:
        if span["class"] == "dense_board_or_list":
            continue
        for term in span["prompt_safe_terms"]:
            counts[term] += span["term_counts"].get(term, 0)
            span_presence[term].add(span["span_id"])

    ranked = []
    for term, total_count in counts.items():
        span_count = len(span_presence[term])
        score = (span_count * 3.0) + total_count
        if is_entity_like_term(term):
            score += 2.0
        ranked.append((score, span_count, total_count, -len(term), term))
    ranked.sort(reverse=True)
    return [term for _, _, _, _, term in ranked[:limit]]


def build_chunk_context(args: argparse.Namespace):
    run = start_run("context")
    episode_dir = _resolve_episode_dir(args, "in_json", "out_json")
    if args.in_json:
        in_json = Path(args.in_json)
    elif episode_dir:
        in_json = _pick_spans_json(episode_dir)
    else:
        raise SystemExit("Could not infer input spans JSON. Pass --in-json or --episode-dir.")

    if args.out_json:
        out_json = Path(args.out_json)
    elif episode_dir:
        out_json = episode_dir / "ocr" / "qwen_ocr_context.json"
    else:
        raise SystemExit("Could not infer context output path. Pass --out-json or --episode-dir.")

    out_json.parent.mkdir(parents=True, exist_ok=True)

    payload = json.loads(in_json.read_text(encoding="utf-8"))
    spans = payload.get("spans", [])
    if not spans:
        raise SystemExit(f"No spans found in {in_json}")

    episode_memory = payload.get("episode_memory", [])
    max_end_sec = max(float(span.get("end_sec", 0.0)) for span in spans)
    chunk_count = int(max_end_sec // args.chunk_sec) + 1

    chunk_contexts = []
    for chunk_id in range(chunk_count):
        chunk_start = chunk_id * args.chunk_sec
        chunk_end = chunk_start + args.chunk_sec
        overlaps = []
        term_scores: dict[str, float] = defaultdict(float)
        for span in spans:
            span_start = float(span["start_sec"])
            span_end = float(span["end_sec"])
            if span_end < chunk_start - args.pad_sec or span_start > chunk_end + args.pad_sec:
                continue
            overlaps.append(
                {
                    "span_id": span["span_id"],
                    "class": span["class"],
                    "start_sec": span_start,
                    "end_sec": span_end,
                }
            )
            class_weight = 1.5 if span["class"] == "name_or_title_card" else 1.0
            if span["class"] == "dense_board_or_list":
                class_weight = 0.8
            for idx, term in enumerate(span.get("prompt_safe_terms", [])):
                term_scores[term] += class_weight * max(0.5, args.max_terms - idx)

        ranked_terms = [term for _, term in sorted((score, term) for term, score in term_scores.items())[::-1]]
        chunk_contexts.append(
            {
                "chunk_id": chunk_id,
                "start_sec": chunk_start,
                "end_sec": chunk_end,
                "terms": ranked_terms[: args.max_terms],
                "span_ids": [item["span_id"] for item in overlaps],
                "spans": overlaps[: args.max_spans],
            }
        )

    context_payload = {
        "meta": {
            "source": str(in_json),
            "chunk_sec": args.chunk_sec,
            "pad_sec": args.pad_sec,
            "max_terms": args.max_terms,
            "max_spans": args.max_spans,
            "chunk_count": len(chunk_contexts),
        },
        "episode_memory": episode_memory[: args.episode_memory_limit],
        "chunk_contexts": chunk_contexts,
    }

    out_json.write_text(json.dumps(context_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OCR context written: {out_json}")
    print(f"Chunks: {len(chunk_contexts)}")
    metadata = finish_run(
        run,
        episode_dir=str(episode_dir) if episode_dir else None,
        inputs={
            "spans_json": str(in_json),
        },
        outputs={
            "context_json": str(out_json),
        },
        context_settings={
            "chunk_sec": args.chunk_sec,
            "pad_sec": args.pad_sec,
            "max_terms": args.max_terms,
            "max_spans": args.max_spans,
            "episode_memory_limit": args.episode_memory_limit,
        },
        stats={
            "chunks": len(chunk_contexts),
            "episode_memory_terms": len(context_payload["episode_memory"]),
        },
    )
    write_metadata(out_json, metadata)
    print(f"Metadata written: {metadata_path(out_json)}")


def build_spans(args: argparse.Namespace):
    run = start_run("spans")
    episode_dir = _resolve_episode_dir(args, "in_jsonl", "out_json")
    if args.in_jsonl:
        in_jsonl = Path(args.in_jsonl)
    elif episode_dir:
        in_jsonl = _pick_filter_in_jsonl(episode_dir)
    else:
        raise SystemExit("Could not infer input OCR JSONL. Pass --in-jsonl or --episode-dir.")

    if args.out_json:
        out_json = Path(args.out_json)
    elif episode_dir:
        out_json = episode_dir / "ocr" / "qwen_ocr_spans.json"
    else:
        raise SystemExit("Could not infer spans output path. Pass --out-json or --episode-dir.")

    out_json.parent.mkdir(parents=True, exist_ok=True)

    frames = read_ocr_frames(in_jsonl)
    if not frames:
        raise SystemExit(f"No OCR rows found in {in_jsonl}")

    nonempty_frames = [frame for frame in frames if frame["lines"]]
    if not nonempty_frames:
        raise SystemExit(f"No non-empty OCR frames found in {in_jsonl}")

    def start_span(frame: dict) -> dict:
        return {
            "start_frame": frame["frame_no"],
            "end_frame": frame["frame_no"],
            "start_sec": frame["time_sec"],
            "end_sec": frame["time_sec"],
            "nonempty_frame_count": 1,
            "term_counts": Counter(frame["lines"]),
            "dense_frame_count": int(frame["frame_stats"]["dense_flag"]),
            "repetition_frame_count": int(frame["frame_stats"]["repetition_flag"]),
            "raw_line_total": frame["raw_unique_lines"],
            "kept_line_total": len(frame["lines"]),
            "last_terms": set(frame["lines"]),
            "last_nonempty_frame_no": frame["frame_no"],
        }

    spans = []
    current = None

    for frame in nonempty_frames:
        frame_terms = set(frame["lines"])
        if current is None:
            current = start_span(frame)
            continue

        gap_frames = frame["frame_no"] - current["last_nonempty_frame_no"] - 1
        last_overlap = overlap_coeff(current["last_terms"], frame_terms)
        core_overlap = overlap_coeff(_span_core_terms(current["term_counts"]), frame_terms)
        merge_ok = (
            gap_frames <= args.max_gap_frames
            and (
                last_overlap >= args.merge_overlap
                or core_overlap >= args.merge_overlap
                or frame_terms.issubset(current["last_terms"])
                or current["last_terms"].issubset(frame_terms)
            )
        )

        if not merge_ok:
            spans.append(current)
            current = start_span(frame)
            continue

        current["end_frame"] = frame["frame_no"]
        current["end_sec"] = frame["time_sec"]
        current["nonempty_frame_count"] += 1
        current["term_counts"].update(frame["lines"])
        current["dense_frame_count"] += int(frame["frame_stats"]["dense_flag"])
        current["repetition_frame_count"] += int(frame["frame_stats"]["repetition_flag"])
        current["raw_line_total"] += frame["raw_unique_lines"]
        current["kept_line_total"] += len(frame["lines"])
        current["last_terms"] = frame_terms
        current["last_nonempty_frame_no"] = frame["frame_no"]

    if current is not None:
        spans.append(current)

    payload_spans = []
    dense_spans = 0
    for span_id, span in enumerate(spans):
        term_counts = span["term_counts"]
        representative_lines = _rank_span_terms(term_counts, prompt_safe_only=False, limit=args.max_lines)
        prompt_safe_terms = _rank_span_terms(term_counts, prompt_safe_only=True, limit=args.max_prompt_terms)
        unique_terms = len(term_counts)
        line_lengths = [len(term) for term in term_counts]
        sentence_like_count = sum(1 for term in term_counts if is_sentence_like_term(term))
        digit_heavy_count = sum(1 for term in term_counts if sum(ch.isdigit() for ch in term) >= 2)
        dense_flag = (
            span["dense_frame_count"] > 0
            or unique_terms >= args.dense_unique_terms
            or span["kept_line_total"] >= args.dense_total_lines
        )
        if dense_flag:
            dense_spans += 1
        stats = {
            "frame_span": span["end_frame"] - span["start_frame"] + 1,
            "nonempty_frame_count": span["nonempty_frame_count"],
            "unique_terms": unique_terms,
            "raw_line_total": span["raw_line_total"],
            "kept_line_total": span["kept_line_total"],
            "avg_line_length": round(sum(line_lengths) / len(line_lengths), 2) if line_lengths else 0.0,
            "max_line_length": max(line_lengths) if line_lengths else 0,
            "digit_heavy_ratio": round(digit_heavy_count / unique_terms, 3) if unique_terms else 0.0,
            "sentence_like_ratio": round(sentence_like_count / unique_terms, 3) if unique_terms else 0.0,
            "dense_flag": dense_flag,
            "dense_frame_count": span["dense_frame_count"],
            "repetition_frame_count": span["repetition_frame_count"],
            "prompt_safe_term_count": len(prompt_safe_terms),
        }
        payload_spans.append(
            {
                "span_id": span_id,
                "start_frame": span["start_frame"],
                "end_frame": span["end_frame"],
                "start_sec": span["start_sec"],
                "end_sec": span["end_sec"],
                "class": _classify_span(term_counts, stats),
                "stats": stats,
                "representative_lines": representative_lines,
                "prompt_safe_terms": prompt_safe_terms,
                "term_counts": dict(term_counts.most_common(args.max_term_counts)),
            }
        )

    episode_memory = build_episode_memory_from_spans(payload_spans, limit=args.episode_memory_limit)

    payload = {
        "meta": {
            "source": str(in_jsonl),
            "merge_overlap": args.merge_overlap,
            "max_gap_frames": args.max_gap_frames,
            "max_lines": args.max_lines,
            "max_prompt_terms": args.max_prompt_terms,
            "max_term_counts": args.max_term_counts,
            "frames_total": len(frames),
            "nonempty_frames": len(nonempty_frames),
            "spans": len(payload_spans),
            "dense_spans": dense_spans,
            "episode_memory_terms": len(episode_memory),
        },
        "episode_memory": episode_memory,
        "spans": payload_spans,
    }

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OCR spans written: {out_json}")
    print(f"Non-empty frames: {len(nonempty_frames)} -> Spans: {len(payload_spans)}")
    metadata = finish_run(
        run,
        episode_dir=str(episode_dir) if episode_dir else None,
        inputs={
            "ocr_jsonl": str(in_jsonl),
        },
        outputs={
            "spans_json": str(out_json),
        },
        span_settings={
            "merge_overlap": args.merge_overlap,
            "max_gap_frames": args.max_gap_frames,
            "max_lines": args.max_lines,
            "max_prompt_terms": args.max_prompt_terms,
            "max_term_counts": args.max_term_counts,
            "episode_memory_limit": args.episode_memory_limit,
            "dense_unique_terms": args.dense_unique_terms,
            "dense_total_lines": args.dense_total_lines,
        },
        stats={
            "frames_total": len(frames),
            "nonempty_frames": len(nonempty_frames),
            "spans": len(payload_spans),
            "dense_spans": dense_spans,
            "episode_memory_terms": len(episode_memory),
        },
    )
    write_metadata(out_json, metadata)
    print(f"Metadata written: {metadata_path(out_json)}")


def longest_run(frame_list: list[int]) -> int:
    if not frame_list:
        return 0
    frames = sorted(set(frame_list))
    best = 1
    run = 1
    for i in range(1, len(frames)):
        if frames[i] == frames[i - 1] + 1:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best


def substring_collapse(terms: list[str], counts: dict[str, int]) -> list[str]:
    ordered = sorted(terms, key=lambda t: (len(t), counts[t]), reverse=True)
    kept: list[str] = []
    for t in ordered:
        if any(t != k and t in k for k in kept):
            continue
        kept.append(t)
    return kept


def build_glossary(args: argparse.Namespace):
    run = start_run("filter")
    episode_dir = _resolve_episode_dir(args, "in_jsonl", "out_json")
    if args.in_jsonl:
        in_jsonl = Path(args.in_jsonl)
    elif episode_dir:
        in_jsonl = _pick_filter_in_jsonl(episode_dir)
    else:
        raise SystemExit("Could not infer input OCR JSONL. Pass --in-jsonl or --episode-dir.")

    if args.out_json:
        out_json = Path(args.out_json)
    elif episode_dir:
        out_json = episode_dir / "glossary" / "qwen_glossary.json"
    else:
        raise SystemExit("Could not infer glossary output path. Pass --out-json or --episode-dir.")

    out_json.parent.mkdir(parents=True, exist_ok=True)

    terms_rows = read_ocr_terms(in_jsonl)
    if not terms_rows:
        raise SystemExit(f"No OCR rows found in {in_jsonl}")

    terms_by_frame: dict[int, list[tuple[float, str]]] = defaultdict(list)
    for frame_no, time_sec, term in terms_rows:
        terms_by_frame[frame_no].append((time_sec, term))

    frames_by_term: dict[str, list[int]] = defaultdict(list)
    chunks_by_term: dict[str, set[int]] = defaultdict(set)
    dense_frames = 0
    repetition_frames = 0
    dropped_terms = 0

    for frame_no, pairs in terms_by_frame.items():
        time_sec = pairs[0][0]
        frame_terms = [term for _, term in pairs]
        sanitized_terms, frame_stats = sanitize_frame_terms(frame_terms)
        if frame_stats["dense_flag"]:
            dense_frames += 1
        if frame_stats["repetition_flag"]:
            repetition_frames += 1
        dropped_terms += frame_stats["raw_count"] - frame_stats["kept_count"]

        for term in sanitized_terms:
            if is_noise_term(term):
                continue
            chunk_id = int(time_sec // args.chunk_sec)
            frames_by_term[term].append(frame_no)
            chunks_by_term[term].add(chunk_id)

    counts = {t: len(set(frames)) for t, frames in frames_by_term.items()}
    runs = {t: longest_run(frames) for t, frames in frames_by_term.items()}

    scored_terms = []
    for term in counts:
        score = (counts[term] * 1.0) + (runs[term] * 1.5) + (len(term) * 0.2)
        scored_terms.append((score, term))
    scored_terms.sort(reverse=True)

    ranked_terms = [t for _, t in scored_terms]
    collapsed = substring_collapse(ranked_terms, counts)
    global_terms = collapsed[: args.top_global]

    chunk_terms: dict[int, list[str]] = {}
    terms_per_chunk: dict[int, list[tuple[float, str]]] = defaultdict(list)
    for term in global_terms:
        for chunk_id in chunks_by_term[term]:
            local_score = counts[term] + runs[term] + len(term) * 0.1
            terms_per_chunk[chunk_id].append((local_score, term))

    for chunk_id, pairs in terms_per_chunk.items():
        pairs.sort(reverse=True)
        chunk_terms[chunk_id] = [t for _, t in pairs[: args.top_chunk]]

    payload = {
        "meta": {
            "source": str(in_jsonl),
            "chunk_sec": args.chunk_sec,
            "top_global": args.top_global,
            "top_chunk": args.top_chunk,
            "candidate_terms": len(counts),
            "global_terms": len(global_terms),
            "dense_frames": dense_frames,
            "repetition_frames": repetition_frames,
            "dropped_terms": dropped_terms,
        },
        "global_glossary": global_terms,
        "per_chunk_terms": chunk_terms,
        "term_stats": {
            t: {"frame_count": counts[t], "max_consecutive_frames": runs[t]}
            for t in global_terms
        },
    }

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Glossary written: {out_json}")
    print(f"Candidates: {len(counts)} -> Global: {len(global_terms)}")
    metadata = finish_run(
        run,
        episode_dir=str(episode_dir) if episode_dir else None,
        inputs={
            "ocr_jsonl": str(in_jsonl),
        },
        outputs={
            "glossary_json": str(out_json),
        },
        filter_settings={
            "chunk_sec": args.chunk_sec,
            "top_global": args.top_global,
            "top_chunk": args.top_chunk,
            "noise_exact": sorted(NOISE_EXACT),
        },
        stats={
            "candidate_terms": len(counts),
            "global_terms": len(global_terms),
            "chunks_with_terms": len(chunk_terms),
            "dense_frames": dense_frames,
            "repetition_frames": repetition_frames,
            "dropped_terms": dropped_terms,
        },
    )
    write_metadata(out_json, metadata)
    print(f"Metadata written: {metadata_path(out_json)}")


def main():
    parser = argparse.ArgumentParser(description="Qwen OCR full episode + span/filter artifacts")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ocr = sub.add_parser("ocr", help="Run OCR over frames and write JSONL")
    p_ocr.add_argument("--episode-dir", default="", help="Episode root, e.g. samples/episodes/<episode_slug>")
    p_ocr.add_argument("--frames-dir", default="", help="Defaults to <episode>/frames/workdir or <episode>/frames/raw_2s")
    p_ocr.add_argument("--out-jsonl", default="", help="Defaults to <episode>/ocr/qwen_ocr_results.jsonl")
    p_ocr.add_argument("--url", default="http://127.0.0.1:8787")
    p_ocr.add_argument(
        "--model",
        default=os.environ.get("QWEN_VISION_MODEL", "qwen3.5-9b"),
        help="Model/alias served by llama.cpp (env: QWEN_VISION_MODEL)",
    )
    p_ocr.add_argument("--max-tokens", type=int, default=768)
    p_ocr.add_argument("--frame-step-sec", type=float, default=2.0, help="0.5 fps => 2.0 sec/frame")
    p_ocr.add_argument("--progress-every", type=int, default=25)
    p_ocr.add_argument("--max-fail", type=int, default=50)
    p_ocr.add_argument("--retries", type=int, default=2, help="Retries per frame on 500/503/connection errors")
    p_ocr.add_argument("--limit", type=int, default=0)
    p_ocr.add_argument(
        "--server-settings",
        default="",
        help="Optional freeform note or JSON blob describing llama-server settings used for this OCR run.",
    )

    p_filter = sub.add_parser("filter", help="Build filtered glossary JSON from OCR JSONL")
    p_filter.add_argument("--episode-dir", default="", help="Episode root, e.g. samples/episodes/<episode_slug>")
    p_filter.add_argument("--in-jsonl", default="", help="Defaults to <episode>/ocr/qwen_ocr_results.jsonl (or newest JSONL in ocr/)")
    p_filter.add_argument("--out-json", default="", help="Defaults to <episode>/glossary/qwen_glossary.json")
    p_filter.add_argument("--chunk-sec", type=float, default=30.0)
    p_filter.add_argument("--top-global", type=int, default=80)
    p_filter.add_argument("--top-chunk", type=int, default=20)

    p_spans = sub.add_parser("spans", help="Build OCR span JSON from OCR JSONL")
    p_spans.add_argument("--episode-dir", default="", help="Episode root, e.g. samples/episodes/<episode_slug>")
    p_spans.add_argument("--in-jsonl", default="", help="Defaults to <episode>/ocr/qwen_ocr_results.jsonl (or newest JSONL in ocr/)")
    p_spans.add_argument("--out-json", default="", help="Defaults to <episode>/ocr/qwen_ocr_spans.json")
    p_spans.add_argument("--merge-overlap", type=float, default=0.4, help="Minimum overlap coefficient to merge adjacent OCR frames into one span")
    p_spans.add_argument("--max-gap-frames", type=int, default=2, help="Allow this many empty/mismatched frames between merged OCR spans")
    p_spans.add_argument("--max-lines", type=int, default=12, help="Representative lines to keep per span")
    p_spans.add_argument("--max-prompt-terms", type=int, default=8, help="Prompt-safe terms to keep per span")
    p_spans.add_argument("--max-term-counts", type=int, default=20, help="Term frequency entries to keep per span")
    p_spans.add_argument("--episode-memory-limit", type=int, default=40, help="Episode-level prompt-safe memory terms to keep")
    p_spans.add_argument("--dense-unique-terms", type=int, default=18, help="Mark a span dense if it has at least this many unique terms")
    p_spans.add_argument("--dense-total-lines", type=int, default=40, help="Mark a span dense if kept lines across frames reach this count")

    p_context = sub.add_parser("context", help="Build chunk-level OCR context from OCR spans")
    p_context.add_argument("--episode-dir", default="", help="Episode root, e.g. samples/episodes/<episode_slug>")
    p_context.add_argument("--in-json", default="", help="Defaults to <episode>/ocr/qwen_ocr_spans.json")
    p_context.add_argument("--out-json", default="", help="Defaults to <episode>/ocr/qwen_ocr_context.json")
    p_context.add_argument("--chunk-sec", type=float, default=30.0)
    p_context.add_argument("--pad-sec", type=float, default=2.0)
    p_context.add_argument("--max-terms", type=int, default=12)
    p_context.add_argument("--max-spans", type=int, default=6)
    p_context.add_argument("--episode-memory-limit", type=int, default=30)

    args = parser.parse_args()
    if args.cmd == "ocr":
        run_ocr(args)
    elif args.cmd == "spans":
        build_spans(args)
    elif args.cmd == "context":
        build_chunk_context(args)
    elif args.cmd == "filter":
        build_glossary(args)


if __name__ == "__main__":
    main()
