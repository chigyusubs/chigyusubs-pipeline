#!/usr/bin/env python3
"""Run full-episode OCR with llama.cpp Qwen-VL, then dedupe/filter terms.

Usage:
  # Stage 1: OCR all frames (resume-safe)
  python pipeline/scripts/run_qwen_ocr_episode.py ocr \
    --episode-dir samples/episodes/wednesday_downtown_2025-02-05_406 \
    --url http://127.0.0.1:8787 \
    --model qwen3-vl

  # Stage 2: Build glossary from OCR results
  python pipeline/scripts/run_qwen_ocr_episode.py filter \
    --episode-dir samples/episodes/wednesday_downtown_2025-02-05_406 \
    --chunk-sec 30 \
    --top-global 80 \
    --top-chunk 20
"""

import argparse
import base64
import json
import re
import unicodedata
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Optional

from episode_paths import find_episode_dir_from_path, find_latest_episode_dir


OCR_SYSTEM_PROMPT = """\
You are an OCR system for Japanese TV variety shows.
Read ALL visible text exactly as shown.
Output plain text only, one line per distinct text region.
Do not translate. Do not summarize. Do not explain.
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
                if frame:
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
        t = re.sub(r"^[\-\*\u2022\d\.\)\(]+", "", t).strip()
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
                    {"type": "text", "text": "Extract all text visible in this image."},
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
                    break

    print(f"Done OCR. ok={ok}, fail={fail}, output={out_jsonl}")


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

    frames_by_term: dict[str, list[int]] = defaultdict(list)
    chunks_by_term: dict[str, set[int]] = defaultdict(set)

    for frame_no, time_sec, term in terms_rows:
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


def main():
    parser = argparse.ArgumentParser(description="Qwen OCR full episode + dedupe/filter")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ocr = sub.add_parser("ocr", help="Run OCR over frames and write JSONL")
    p_ocr.add_argument("--episode-dir", default="", help="Episode root, e.g. samples/episodes/<episode_slug>")
    p_ocr.add_argument("--frames-dir", default="", help="Defaults to <episode>/frames/workdir or <episode>/frames/raw_2s")
    p_ocr.add_argument("--out-jsonl", default="", help="Defaults to <episode>/ocr/qwen_ocr_results.jsonl")
    p_ocr.add_argument("--url", default="http://127.0.0.1:8787")
    p_ocr.add_argument("--model", default="qwen3-vl")
    p_ocr.add_argument("--max-tokens", type=int, default=768)
    p_ocr.add_argument("--frame-step-sec", type=float, default=2.0, help="0.5 fps => 2.0 sec/frame")
    p_ocr.add_argument("--progress-every", type=int, default=25)
    p_ocr.add_argument("--max-fail", type=int, default=50)
    p_ocr.add_argument("--retries", type=int, default=2, help="Retries per frame on 500/503/connection errors")
    p_ocr.add_argument("--limit", type=int, default=0)

    p_filter = sub.add_parser("filter", help="Build filtered glossary JSON from OCR JSONL")
    p_filter.add_argument("--episode-dir", default="", help="Episode root, e.g. samples/episodes/<episode_slug>")
    p_filter.add_argument("--in-jsonl", default="", help="Defaults to <episode>/ocr/qwen_ocr_results.jsonl (or newest JSONL in ocr/)")
    p_filter.add_argument("--out-json", default="", help="Defaults to <episode>/glossary/qwen_glossary.json")
    p_filter.add_argument("--chunk-sec", type=float, default=30.0)
    p_filter.add_argument("--top-global", type=int, default=80)
    p_filter.add_argument("--top-chunk", type=int, default=20)

    args = parser.parse_args()
    if args.cmd == "ocr":
        run_ocr(args)
    elif args.cmd == "filter":
        build_glossary(args)


if __name__ == "__main__":
    main()
