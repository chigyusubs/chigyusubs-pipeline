#!/usr/bin/env python3
"""Run all harness CONFIGS across an eval spec, score, and save results.

Usage:
    python3 run_bench.py \\
        --spec eval_specs/killah_kuts_s01e01.json \\
        --out  results/killah_kuts_s01e01__$(date +%Y%m%d_%H%M%S).json

Assumes a vLLM OpenAI-compatible server is running at --base-url. See
server/README.md for the docker-run recipe.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE,
)


def extract_json_field(text: str, field: str) -> tuple[str, bool]:
    """Parse a guided_json response and return (value, parsed_ok).

    vLLM sometimes wraps JSON output in a markdown code fence even with
    guided_json, so strip fences before json.loads. On any failure, fall
    back to the raw text so the scorer still sees something.
    """
    candidate = text.strip()
    m = _JSON_FENCE_RE.search(candidate)
    if m:
        candidate = m.group(1)
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict) and field in obj:
            return str(obj[field]), True
    except json.JSONDecodeError:
        pass
    return text, False

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from configs import CONFIGS  # noqa: E402
from scoring import aggregate, score_segment  # noqa: E402


DEFAULT_MODEL = "google/gemma-4-E2B-it"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1/chat/completions"


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def resolve_system(cfg: dict, seg: dict) -> str | None:
    """Return the system message for this (cfg, seg), or None.

    Configs may provide either a static ``system`` string or a
    ``system_template`` with ``{oracle_names}`` style placeholders. The
    template path is how the "oracle OCR pre-pass" configs (H, I) get
    per-segment cast-name injections from CTC gold.
    """
    template = cfg.get("system_template")
    if template:
        oracle_names = ", ".join(seg.get("names") or []) or "(none)"
        return template.format(oracle_names=oracle_names)
    return cfg.get("system") or None


def build_request(
    *, model: str, wav: Path, frames: list[Path],
    video: Path | None, cfg: dict, seg: dict,
) -> dict:
    content: list[dict] = []
    if cfg.get("video") and video and video.exists():
        content.append({
            "type": "video_url",
            "video_url": {"url": f"data:video/mp4;base64,{b64(video)}"},
        })
    elif cfg.get("frames") and frames:
        stride = max(1, int(cfg.get("frame_stride", 1)))
        picked = frames[::stride]
        for f in picked:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64(f)}"},
            })
    content.append({
        "type": "input_audio",
        "input_audio": {"data": b64(wav), "format": "wav"},
    })
    content.append({"type": "text", "text": cfg["prompt"]})
    messages: list[dict] = []
    sys_msg = resolve_system(cfg, seg)
    if sys_msg:
        messages.append({"role": "system", "content": sys_msg})
    messages.append({"role": "user", "content": content})
    payload = {
        "model": model,
        "max_tokens": 512,
        "temperature": 0.0,
        "messages": messages,
    }
    if cfg.get("mst") is not None:
        payload["mm_processor_kwargs"] = {"max_soft_tokens": cfg["mst"]}
    if cfg.get("guided_json") is not None:
        payload["guided_json"] = cfg["guided_json"]
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
            body = json.loads(resp.read())
        return body, time.monotonic() - t0
    except urllib.error.HTTPError as e:
        return ({"error": f"HTTP {e.code}",
                 "body": e.read().decode("utf-8", "replace")[:500]},
                time.monotonic() - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="eval spec JSON")
    ap.add_argument("--out", required=True, help="output results JSON")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--configs", default="",
                    help="comma-separated subset of config names; default = all")
    args = ap.parse_args()

    spec_path = Path(args.spec).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec_dir = spec_path.parent

    wanted_names = set(args.configs.split(",")) if args.configs else None
    configs = [c for c in CONFIGS
               if wanted_names is None or c["name"] in wanted_names]
    if not configs:
        raise SystemExit(f"no configs match {args.configs!r}")

    print(f"Spec: {spec_path.name}  ({spec['n_segments']} segments, "
          f"episode={spec['episode']})")
    print(f"Model: {args.model}  Base: {args.base_url}")
    print(f"Configs: {[c['name'] for c in configs]}")

    results: list[dict] = []
    for seg in spec["segments"]:
        name = seg["name"]
        wav = (spec_dir / seg["wav_rel"]).resolve()
        video = wav.with_suffix(".mp4")  # co-located mp4 clip
        frames = sorted(
            (spec_dir / seg["frames_dir_rel"]).resolve().glob("*.jpg")
        )
        print(f"\n### {name}  [{seg['seg_start']:.1f}-{seg['seg_end']:.1f}]")
        print(f"    target: {seg['text'][:80]}")
        row: dict = {
            "seg": name,
            "target": seg["text"],
            "kata": seg["kata"],
            "names": seg["names"],
        }
        for cfg in configs:
            payload = build_request(
                model=args.model, wav=wav, frames=frames, video=video,
                cfg=cfg, seg=seg,
            )
            body, wall = post(args.base_url, payload)
            if "error" in body:
                print(f"  {cfg['name']}: ERROR {body['error']}")
                row[cfg["name"]] = {"error": body["error"], "wall": wall}
                continue
            raw_text = body["choices"][0]["message"]["content"]
            json_field = cfg.get("json_field")
            if json_field:
                text, parsed_ok = extract_json_field(raw_text, json_field)
            else:
                text, parsed_ok = raw_text, None
            s = score_segment(text, seg["text"], seg["kata"], seg["names"])
            usage = body.get("usage", {})
            entry = {
                "text": text,
                "wall": round(wall, 3),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                **s.as_dict(),
            }
            if json_field:
                entry["raw"] = raw_text
                entry["json_parsed"] = parsed_ok
            row[cfg["name"]] = entry
            print(f"  {cfg['name']}: kata={row[cfg['name']]['kata_recall']} "
                  f"name={row[cfg['name']]['name_recall']} "
                  f"lcs={row[cfg['name']]['lcs_ratio']}  "
                  f"wall={wall:.2f}s")
            print(f"    out: {text.strip()[:150]}")
        results.append(row)

    summary = [aggregate(results, c["name"]) for c in configs]
    print("\n\n=========== SUMMARY ===========")
    for s in summary:
        print(f"  {s['config']}:  kata {s['kata_num']}/{s['kata_den']} "
              f"({s['kata_pct']}%)  name {s['name_num']}/{s['name_den']} "
              f"({s['name_pct']}%)  mean lcs {s['mean_lcs']}  "
              f"(n={s['n_segments']})")

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({
            "spec": str(spec_path),
            "model": args.model,
            "base_url": args.base_url,
            "configs": [c["name"] for c in configs],
            "summary": summary,
            "results": results,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
