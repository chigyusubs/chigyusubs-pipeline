#!/usr/bin/env python3
"""Full-episode E4B ASR driver — Phase A of the harness-to-pipeline path.

Takes an episode (silero VAD segments + audio), packs the segments into
ASR-sized chunks (~18 s, capped at 25 s), and runs each chunk through
the vLLM Gemma 4 E4B server using a config from the harness's
``configs.py``. Output mirrors the schema downstream alignment expects
(``[{chunk, chunk_start_s, chunk_end_s, text, ...}]``), so the existing
CTC alignment / reflow path can pick this up the same way it picks up
``*_gemini_raw.json``.

This is intentionally minimal — no retries, no fallback to whisper, no
per-clip oracle. Validation flags bad chunks but doesn't auto-recover.
The point is to produce a single full-episode artifact we can compare
head-to-head against whisper-large-v3 and the existing Gemini pre-pass.

Usage:
    python3 scripts/transcribe_episode_e4b.py \\
        --episode killah_kuts_s01e01 \\
        --video samples/episodes/killah_kuts_s01e01/source/foo.mkv \\
        --out  samples/episodes/killah_kuts_s01e01/transcription/\\
                killah_kuts_s01e01_e4b_raw.json
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(
    0, str(REPO / "scripts" / "experiments" / "vllm_gemma4_harness")
)

from configs import CONFIGS  # noqa: E402
from build_per_clip_oracle import (  # noqa: E402
    index_ocr_batches, load_glossary_cast, oracle_for_window,
)

DEFAULT_MODEL = "google/gemma-4-E4B-it"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1/chat/completions"
DEFAULT_CONFIG = "P_audio_sysrole_minimal"

CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿]")
REFUSAL_HINTS = (
    "I cannot transcribe", "I'm sorry", "I cannot", "申し訳",
    "transcribe", "transcription",  # output should never echo prompt vocab
)


# --- chunk packing ------------------------------------------------------

def pack_vad_segments(
    vad: list[dict], target_dur: float = 18.0,
    max_dur: float = 25.0, max_gap: float = 1.5,
    min_dur: float = 10.0,
) -> list[dict]:
    """Greedy-pack silero VAD segments into ASR-sized chunks.

    Rules:
    - Merge adjacent VAD segments while combined span < target_dur or
      until extending would push past max_dur.
    - Don't bridge gaps longer than ``max_gap`` — a long silence is
      almost certainly a scene boundary, packing across it wastes
      audio-tower attention on dead air. EXCEPTION: if the in-progress
      chunk is still shorter than ``min_dur``, bridge anyway. The merge
      doesn't drop audio — the slice covers cur_start..end so the
      silence in the middle just travels with the chunk; the model
      transcribes nothing for that stretch. The only cost is a slightly
      wider time window per call, which is much cheaper than spending a
      whole vLLM call on a 0.3 s "あっ".
    - Single VAD segments longer than max_dur get split into
      ceil(dur/target_dur) equal-time pieces. CTC alignment downstream
      re-snaps to word boundaries so a mid-word split is recoverable.
    """
    chunks: list[tuple[float, float]] = []
    cur_start: float | None = None
    cur_end: float | None = None
    for seg in vad:
        s, e = float(seg["start"]), float(seg["end"])
        if e - s > max_dur:
            # flush whatever we were building, then split this segment
            if cur_start is not None:
                chunks.append((cur_start, cur_end))
                cur_start = cur_end = None
            n_pieces = max(1, int((e - s) // target_dur) + 1)
            piece = (e - s) / n_pieces
            for i in range(n_pieces):
                chunks.append((s + i * piece, s + (i + 1) * piece))
            continue
        if cur_start is None:
            cur_start, cur_end = s, e
            continue
        gap = s - cur_end
        new_dur = e - cur_start
        too_short = (cur_end - cur_start) < min_dur
        if not too_short and (
            gap > max_gap or new_dur > max_dur or (
                (cur_end - cur_start) >= target_dur and gap > 0.3
            )
        ):
            chunks.append((cur_start, cur_end))
            cur_start, cur_end = s, e
        else:
            cur_end = e
    if cur_start is not None:
        chunks.append((cur_start, cur_end))
    return [
        {"chunk": i, "chunk_start_s": round(s, 3), "chunk_end_s": round(e, 3),
         "duration_s": round(e - s, 3)}
        for i, (s, e) in enumerate(chunks)
    ]


# --- WAV slice ----------------------------------------------------------

def extract_wav_slice(
    video: Path, out_wav: Path, start_s: float, dur_s: float,
) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    if out_wav.exists():
        return
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{start_s:.3f}", "-i", str(video),
         "-t", f"{dur_s:.3f}", "-vn", "-ac", "1", "-ar", "16000",
         str(out_wav)],
        capture_output=True, check=True,
    )


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


# --- request / validation ----------------------------------------------

def build_payload(model: str, wav: Path, cfg: dict) -> dict:
    content: list[dict] = [
        {"type": "input_audio",
         "input_audio": {"data": b64(wav), "format": "wav"}},
        {"type": "text", "text": cfg["prompt"]},
    ]
    messages: list[dict] = []
    sys_msg = cfg.get("system")
    if sys_msg:
        messages.append({"role": "system", "content": sys_msg})
    messages.append({"role": "user", "content": content})
    payload: dict = {
        "model": model,
        "max_tokens": 768,
        "temperature": 0.0,
        "messages": messages,
    }
    if cfg.get("mst") is not None:
        payload["mm_processor_kwargs"] = {"max_soft_tokens": cfg["mst"]}
    return payload


def post(base_url: str, payload: dict, timeout: int = 300) -> tuple[dict, float]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url, data=data,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read()), time.monotonic() - t0
    except urllib.error.HTTPError as e:
        return ({"error": f"HTTP {e.code}",
                 "body": e.read().decode("utf-8", "replace")[:500]},
                time.monotonic() - t0)
    except urllib.error.URLError as e:
        return ({"error": f"URLError {e.reason}"},
                time.monotonic() - t0)


def validate(text: str) -> dict:
    """Surface obvious garbage. Doesn't reject — caller decides."""
    flags: list[str] = []
    s = text.strip()
    if not s:
        flags.append("empty")
        return {"flags": flags, "cjk_ratio": 0.0}
    cjk = len(CJK_RE.findall(s))
    cjk_ratio = cjk / max(len(s), 1)
    if cjk_ratio < 0.3:
        flags.append(f"low_cjk({cjk_ratio:.2f})")
    low = s.lower()
    for hint in REFUSAL_HINTS:
        if hint.lower() in low:
            flags.append(f"refusal_or_echo({hint!r})")
            break
    # crude repetition detector: same 4-gram appearing >5×
    counts: dict[str, int] = {}
    for i in range(len(s) - 3):
        g = s[i:i + 4]
        counts[g] = counts.get(g, 0) + 1
    top = max(counts.values()) if counts else 0
    if top > 6:
        flags.append(f"repetition(top4gram={top})")
    return {"flags": flags, "cjk_ratio": round(cjk_ratio, 3)}


# --- VTT output ---------------------------------------------------------

def fmt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ".")


def write_vtt(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for r in records:
            text = (r.get("text") or "").strip()
            if not text:
                continue
            f.write(f"{fmt_ts(r['chunk_start_s'])} --> "
                    f"{fmt_ts(r['chunk_end_s'])}\n{text}\n\n")


# --- main ---------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True,
                    help="episode slug under samples/episodes/")
    ap.add_argument("--video", required=True, help="source video path")
    ap.add_argument("--out", required=True, help="output transcript JSON")
    ap.add_argument("--vtt", default="",
                    help="optional VTT output (default: derive from --out)")
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help=f"harness config name (default {DEFAULT_CONFIG})")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--target-dur", type=float, default=18.0)
    ap.add_argument("--max-dur", type=float, default=25.0)
    ap.add_argument("--max-gap", type=float, default=1.5)
    ap.add_argument("--min-dur", type=float, default=10.0,
                    help="force-merge across long gaps until chunk is at "
                         "least this long. Prevents 0.3s VAD blips from "
                         "burning a vLLM call each.")
    ap.add_argument("--start-s", type=float, default=0.0,
                    help="only transcribe chunks starting at or after this "
                         "time (smoke-test knob)")
    ap.add_argument("--end-s", type=float, default=0.0,
                    help="only transcribe chunks starting before this time "
                         "(0 = no limit)")
    ap.add_argument("--cache-dir", default="",
                    help="dir for per-chunk WAV cache "
                         "(default: alongside --out)")
    ap.add_argument("--dump-chunks", default="",
                    help="if set, write the packed chunk plan here and exit "
                         "without calling the model")
    args = ap.parse_args()

    cfg = next((c for c in CONFIGS if c["name"] == args.config), None)
    if cfg is None:
        raise SystemExit(
            f"unknown config {args.config!r}; "
            f"options: {[c['name'] for c in CONFIGS]}"
        )
    if cfg.get("frames") or cfg.get("video"):
        raise SystemExit(
            f"config {args.config!r} requires frames/video — "
            "this driver is audio-only for now (use D_audio_sysrole_primed)"
        )

    episode_dir = REPO / "samples" / "episodes" / args.episode
    vad_path = episode_dir / "transcription" / "silero_vad_segments.json"
    if not vad_path.exists():
        raise SystemExit(f"silero VAD not found at {vad_path}")
    vad = json.loads(vad_path.read_text(encoding="utf-8"))
    chunks = pack_vad_segments(
        vad, target_dur=args.target_dur,
        max_dur=args.max_dur, max_gap=args.max_gap,
        min_dur=args.min_dur,
    )
    print(f"VAD segments:    {len(vad)}")
    print(f"Packed chunks:   {len(chunks)}  "
          f"(target={args.target_dur}s max={args.max_dur}s)")
    durs = [c["duration_s"] for c in chunks]
    print(f"  duration min/mean/max: "
          f"{min(durs):.1f}s / {sum(durs)/len(durs):.1f}s / {max(durs):.1f}s")

    if args.dump_chunks:
        Path(args.dump_chunks).write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote chunks to {args.dump_chunks}")
        return

    out_path = Path(args.out).resolve()
    cache_dir = (Path(args.cache_dir).resolve() if args.cache_dir
                 else out_path.parent / f"{args.episode}_e4b_chunks_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    video = Path(args.video).resolve()
    if not video.exists():
        raise SystemExit(f"video not found: {video}")

    skipped_before = skipped_after = 0
    selected: list[dict] = []
    for c in chunks:
        if c["chunk_start_s"] < args.start_s:
            skipped_before += 1; continue
        if args.end_s and c["chunk_start_s"] >= args.end_s:
            skipped_after += 1; continue
        selected.append(c)
    if skipped_before or skipped_after:
        print(f"  selected {len(selected)} chunks "
              f"(skipped {skipped_before} before, {skipped_after} after)")

    print(f"Config:   {cfg['name']}")
    print(f"Model:    {args.model}")
    print(f"Base URL: {args.base_url}")

    records: list[dict] = []
    n_flagged = 0
    t_total = time.monotonic()
    for c in selected:
        idx = c["chunk"]
        start, end = c["chunk_start_s"], c["chunk_end_s"]
        wav = cache_dir / f"chunk_{idx:04d}_{start:.0f}s.wav"
        extract_wav_slice(video, wav, start, end - start)
        body, wall = post(args.base_url, build_payload(args.model, wav, cfg))
        if "error" in body:
            print(f"  chunk {idx:>4} [{start:7.1f}-{end:7.1f}] "
                  f"ERROR {body['error']}  wall={wall:.1f}s")
            records.append({**c, "text": "", "wall": round(wall, 2),
                            "error": body["error"], "flags": ["error"]})
            n_flagged += 1
            continue
        text = body["choices"][0]["message"]["content"]
        v = validate(text)
        usage = body.get("usage", {})
        rec = {
            **c,
            "text": text,
            "wall": round(wall, 2),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            **v,
        }
        records.append(rec)
        if v["flags"]:
            n_flagged += 1
        flag_str = f"  ⚠ {','.join(v['flags'])}" if v["flags"] else ""
        print(f"  chunk {idx:>4} [{start:7.1f}-{end:7.1f}]  "
              f"wall={wall:.1f}s  cjk={v['cjk_ratio']}  "
              f"chars={len(text.strip())}{flag_str}")
        if text.strip():
            print(f"    | {text.strip()[:120]}")

    total_wall = time.monotonic() - t_total
    total_audio = sum(c["duration_s"] for c in selected)
    print(f"\n==== done in {total_wall:.1f}s ({total_wall/60:.1f}m) ====")
    print(f"  audio:           {total_audio:.1f}s ({total_audio/60:.1f}m)")
    print(f"  realtime ratio:  {total_wall/max(total_audio,1):.2f}x")
    print(f"  chunks:          {len(records)}")
    print(f"  flagged:         {n_flagged}")
    print(f"  total chars:     {sum(len((r.get('text') or '').strip()) for r in records)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({
            "episode": args.episode,
            "video": str(video),
            "model": args.model,
            "config": args.config,
            "vad_path": str(vad_path),
            "n_chunks": len(records),
            "n_flagged": n_flagged,
            "total_audio_s": round(total_audio, 1),
            "total_wall_s": round(total_wall, 1),
            "chunks": records,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {out_path}")

    vtt_path = Path(args.vtt) if args.vtt else out_path.with_suffix(".vtt")
    write_vtt(records, vtt_path)
    print(f"Wrote {vtt_path}")


if __name__ == "__main__":
    main()
