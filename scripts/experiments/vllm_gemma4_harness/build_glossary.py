#!/usr/bin/env python3
"""Stage 2 of the OCR → glossary pipeline.

Takes the deduped telop list from ``ocr_sweep.py`` and asks E4B to turn
it into a structured glossary JSON. One shot, guided_json, temp=0. The
output is the per-episode cast/term file that feeds the ASR prime in
config K (video + oracle names).
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_MODEL = "google/gemma-4-E4B-it"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1/chat/completions"

SYSTEM = (
    "You build reference glossaries for Japanese variety shows from "
    "raw OCR output. The input is a deduped list of on-screen text "
    "blocks pulled from an entire episode: name plates, scoreboards, "
    "section headers, title cards, and some noise.\n"
    "\n"
    "Your job:\n"
    "1. Identify the canonical Japanese strings for each entity "
    "(fighters, MC, announcer, referee, creator, show title, section "
    "markers, rule/show terms).\n"
    "2. Collapse OCR misreads of the same entity into a single entry "
    "whose `variants` field lists the misread strings seen in the "
    "input. This is used downstream to tell the ASR model what the "
    "correct transcription of a garbled chyron should be.\n"
    "3. Drop lines that are clearly spoken dialogue leaks, decorative "
    "gag captions, or OCR hallucinations — keep only durable reference "
    "information about the episode.\n"
    "\n"
    "Japanese names must be written exactly as the show displays them "
    "(preserve kana vs kanji as shown on the canonical title card). "
    "Team affiliations are the parenthesized string next to a fighter "
    "name, e.g. `お見送り芸人しんいち` → team `お見送り芸人`."
)

USER_PROMPT_TEMPLATE = (
    "Here are the {n_lines} unique on-screen text blocks extracted from "
    "episode `{episode}`:\n\n"
    "<telops>\n{telops}\n</telops>\n\n"
    "Build the glossary JSON. Fill every field you have evidence for; "
    "use null / empty list if the telops do not support a value. Do not "
    "invent fighters or terms that are not backed by at least one line "
    "in the input."
)


GLOSSARY_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "show_title": {
            "type": "object",
            "properties": {
                "canonical": {"type": "string"},
                "variants": {"type": "array", "items": {"type": "string"}},
                "romaji": {"type": ["string", "null"]},
            },
            "required": ["canonical", "variants"],
            "additionalProperties": False,
        },
        "episode_marker": {"type": ["string", "null"]},
        "creator": {"type": ["string", "null"]},
        "mc": {
            "type": "object",
            "properties": {
                "name": {"type": ["string", "null"]},
                "team": {"type": ["string", "null"]},
                "variants": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name", "variants"],
            "additionalProperties": False,
        },
        "announcer": {
            "type": "object",
            "properties": {
                "name": {"type": ["string", "null"]},
                "variants": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name", "variants"],
            "additionalProperties": False,
        },
        "referee": {
            "type": "object",
            "properties": {
                "name": {"type": ["string", "null"]},
                "variants": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name", "variants"],
            "additionalProperties": False,
        },
        "fighters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "team": {"type": ["string", "null"]},
                    "age": {"type": ["integer", "null"]},
                    "height_cm": {"type": ["integer", "null"]},
                    "weight_kg": {"type": ["integer", "null"]},
                    "birthdate": {"type": ["string", "null"]},
                    "variants": {
                        "type": "array", "items": {"type": "string"},
                    },
                },
                "required": ["name", "variants"],
                "additionalProperties": False,
            },
        },
        "section_markers": {"type": "array", "items": {"type": "string"}},
        "rule_terms": {"type": "array", "items": {"type": "string"}},
        "show_terms": {"type": "array", "items": {"type": "string"}},
        "dropped_noise_examples": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Up to 10 example lines you discarded as dialogue leaks "
                "or hallucinations — for auditability only."
            ),
        },
    },
    "required": [
        "show_title", "episode_marker", "creator", "mc", "announcer",
        "referee", "fighters", "section_markers", "rule_terms",
        "show_terms", "dropped_noise_examples",
    ],
    "additionalProperties": False,
}


_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE,
)


def extract_json(text: str) -> dict:
    s = text.strip()
    m = _JSON_FENCE_RE.search(s)
    if m:
        s = m.group(1)
    return json.loads(s)


def post(base_url: str, payload: dict, timeout: int = 600) -> tuple[dict, float]:
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
                 "body": e.read().decode("utf-8", "replace")[:1000]},
                time.monotonic() - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="OCR sweep JSON (output of ocr_sweep.py)")
    ap.add_argument("--out", required=True, help="glossary JSON path")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()

    ocr = json.loads(Path(args.input).read_text(encoding="utf-8"))
    episode = ocr.get("episode", "unknown")
    lines = ocr["unique_lines"]
    telops = "\n".join(lines)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        n_lines=len(lines), episode=episode, telops=telops,
    )

    print(f"Episode:     {episode}")
    print(f"Telop lines: {len(lines)}")
    print(f"Model:       {args.model}")
    print(f"Posting to:  {args.base_url}")

    payload = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        "guided_json": GLOSSARY_SCHEMA,
    }
    body, wall = post(args.base_url, payload)
    if "error" in body:
        raise SystemExit(
            f"server error: {body['error']}\n{body.get('body', '')}"
        )
    raw = body["choices"][0]["message"]["content"]
    usage = body.get("usage", {})
    print(f"  wall={wall:.1f}s  prompt_tok={usage.get('prompt_tokens')}"
          f"  completion_tok={usage.get('completion_tokens')}")

    try:
        glossary = extract_json(raw)
    except json.JSONDecodeError as e:
        dump = Path(args.out).with_suffix(".raw.txt")
        dump.write_text(raw, encoding="utf-8")
        raise SystemExit(
            f"failed to parse glossary JSON ({e}); raw saved to {dump}"
        )

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload_out = {
        "episode": episode,
        "model": args.model,
        "source_ocr": str(Path(args.input).resolve()),
        "n_input_lines": len(lines),
        "wall": round(wall, 2),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "glossary": glossary,
    }
    out_path.write_text(
        json.dumps(payload_out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {out_path}")

    g = glossary
    empty: dict = {}
    print("\n--- glossary summary ---")
    print(f"  show_title:   {g.get('show_title', empty).get('canonical')}")
    print(f"  episode:      {g.get('episode_marker')}")
    print(f"  creator:      {g.get('creator')}")
    print(f"  mc:           {g.get('mc', empty).get('name')}  "
          f"({g.get('mc', empty).get('team')})")
    print(f"  announcer:    {g.get('announcer', empty).get('name')}")
    print(f"  referee:      {g.get('referee', empty).get('name')}")
    print(f"  fighters:     {len(g.get('fighters', []))}")
    for f in g.get("fighters", []):
        print(f"    - {f['name']}  team={f.get('team')}  "
              f"age={f.get('age')}  {f.get('height_cm')}cm "
              f"{f.get('weight_kg')}kg")
    print(f"  sections:     {g.get('section_markers')}")
    print(f"  rule_terms:   {len(g.get('rule_terms', []))}")
    print(f"  show_terms:   {len(g.get('show_terms', []))}")


if __name__ == "__main__":
    main()
