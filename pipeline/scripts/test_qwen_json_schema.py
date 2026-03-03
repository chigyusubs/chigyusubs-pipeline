#!/usr/bin/env python3
"""Compare llama.cpp/Qwen output reliability: json_schema vs prompt JSON vs comma list.

Usage:
    python pipeline/scripts/test_qwen_json_schema.py
    python pipeline/scripts/test_qwen_json_schema.py --runs 12 --url http://127.0.0.1:8787
    python pipeline/scripts/test_qwen_json_schema.py --input samples/whisper_prompt.txt
"""

import argparse
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


DEFAULT_OCR_LINES = [
    "MC 浜田 雅功 (ダウンタウン)",
    "ダウンタウン",
    "ガキの使い",
    "次回予告",
    "提供",
    "渋谷",
    "スーパー マリオ",
    "テロップ",
]


JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "glossary",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "global_glossary": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "noise_terms": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["global_glossary", "noise_terms"],
            "additionalProperties": False,
        },
    },
}


@dataclass
class ModeStats:
    mode: str
    parse_ok: int = 0
    parse_fail: int = 0
    fallback_used: int = 0
    first_failure: str = ""
    first_success: str = ""


def load_ocr_lines(path: str | None) -> list[str]:
    if not path:
        return DEFAULT_OCR_LINES
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    text = p.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines or DEFAULT_OCR_LINES


def call_chat(url: str, payload: dict) -> str:
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def parse_json_object(text: str) -> dict:
    # Try direct parse first.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Retry after stripping markdown fences.
    cleaned = strip_code_fences(text)
    obj = json.loads(cleaned)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object, got {type(obj).__name__}")
    return obj


def parse_comma_terms(text: str) -> list[str]:
    raw = strip_code_fences(text).replace("、", ",").replace("\n", ",")
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]


def build_payload(mode: str, model: str, ocr_lines: list[str], max_tokens: int) -> dict:
    ocr_blob = "\n".join(ocr_lines)

    if mode == "json_schema":
        return {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Extract important proper nouns and show-specific jargon from OCR lines. "
                        "Do not invent new terms."
                    ),
                },
                {"role": "user", "content": f"OCR lines:\n{ocr_blob}"},
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": JSON_SCHEMA,
        }

    if mode == "prompt_json":
        return {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Output STRICT JSON only with keys global_glossary and noise_terms. "
                        "Each key must map to an array of strings. No markdown."
                    ),
                },
                {"role": "user", "content": f"OCR lines:\n{ocr_blob}"},
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
        }

    if mode == "comma":
        return {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Output exactly one line of comma-separated terms like A, B, C. "
                        "Use ASCII commas only. No markdown."
                    ),
                },
                {"role": "user", "content": f"OCR lines:\n{ocr_blob}"},
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
        }

    raise ValueError(f"Unknown mode: {mode}")


def run_mode(
    mode: str,
    runs: int,
    url: str,
    model: str,
    ocr_lines: list[str],
    max_tokens: int,
) -> ModeStats:
    stats = ModeStats(mode=mode)
    for _ in range(runs):
        payload = build_payload(mode, model, ocr_lines, max_tokens)
        raw = call_chat(url, payload)

        try:
            if mode in {"json_schema", "prompt_json"}:
                fallback = False
                try:
                    parsed = parse_json_object(raw)
                except Exception:
                    terms = parse_comma_terms(raw)
                    parsed = {"global_glossary": terms, "noise_terms": []}
                    fallback = True

                if "global_glossary" not in parsed or "noise_terms" not in parsed:
                    raise ValueError("Missing required keys")
                if not isinstance(parsed["global_glossary"], list) or not isinstance(parsed["noise_terms"], list):
                    raise ValueError("Required keys are not arrays")
                if fallback:
                    stats.fallback_used += 1

            else:
                terms = parse_comma_terms(raw)
                if not terms:
                    raise ValueError("No comma terms parsed")

            stats.parse_ok += 1
            if not stats.first_success:
                stats.first_success = raw
        except Exception:
            stats.parse_fail += 1
            if not stats.first_failure:
                stats.first_failure = raw
    return stats


def print_stats(stats: ModeStats, runs: int):
    print(f"- {stats.mode}: ok={stats.parse_ok}/{runs}, fail={stats.parse_fail}, fallback={stats.fallback_used}")
    if stats.first_success:
        print("  sample success:")
        print(f"    {stats.first_success.strip()[:240]}")
    if stats.first_failure:
        print("  sample failure:")
        print(f"    {stats.first_failure.strip()[:240]}")


def main():
    parser = argparse.ArgumentParser(description="Test Qwen output shape reliability on llama.cpp")
    parser.add_argument("--url", default="http://127.0.0.1:8787", help="llama.cpp server URL")
    parser.add_argument("--model", default="qwen3.5-9b", help="Model name passed to /chat/completions")
    parser.add_argument("--input", default=None, help="Optional text file of OCR lines (one line per item)")
    parser.add_argument("--runs", type=int, default=8, help="Number of calls per mode")
    parser.add_argument("--max-tokens", type=int, default=220, help="Max tokens per request")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["json_schema", "prompt_json", "comma"],
        choices=["json_schema", "prompt_json", "comma"],
        help="Modes to run",
    )
    args = parser.parse_args()

    ocr_lines = load_ocr_lines(args.input)
    print(f"URL: {args.url}")
    print(f"Model: {args.model}")
    print(f"Runs per mode: {args.runs}")
    print(f"OCR lines: {len(ocr_lines)}")
    print()

    try:
        for mode in args.modes:
            stats = run_mode(
                mode=mode,
                runs=args.runs,
                url=args.url,
                model=args.model,
                ocr_lines=ocr_lines,
                max_tokens=args.max_tokens,
            )
            print_stats(stats, args.runs)
            print()
    except urllib.error.URLError as e:
        print(f"Connection error: {e}")
        print("Start llama-server first, then re-run this script.")
        raise SystemExit(1) from e
    except urllib.error.HTTPError as e:
        print(f"HTTP error: {e.code} {e.reason}")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
