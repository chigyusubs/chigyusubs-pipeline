#!/usr/bin/env python3
"""Translate VTT/SRT subtitles via Vertex Gemini or any OpenAI-compatible API.

Usage:
  # Vertex Gemini
  python scripts/translate_vtt_api.py --backend vertex \
    --input subs.vtt --output subs_en.vtt

  # OpenAI-compatible (local or remote)
  python scripts/translate_vtt_api.py --backend openai \
    --url http://127.0.0.1:8787 --model qwen3-30b \
    --input subs.vtt --output subs_en.vtt

  # With glossary and summary
  python scripts/translate_vtt_api.py --backend vertex \
    --input subs.vtt --output subs_en.vtt \
    --glossary glossary.tsv --summary "A comedy variety show featuring..."
"""

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chigyusubs.translation import (
    build_manifest,
    call_with_retry,
    translate_subtitles,
)


def _call_openai_compatible(
    url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    api_key: str = "",
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def _call_vertex(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    location: str,
) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(vertexai=True, location=location)
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=temperature,
        response_mime_type="application/json",
        max_output_tokens=65536,
    )
    response = client.models.generate_content(
        model=model, contents=user_prompt, config=config,
    )
    return response.text or ""


def main():
    parser = argparse.ArgumentParser(
        description="Translate VTT/SRT subtitles using structured JSON via LLM."
    )
    parser.add_argument("--input", required=True, help="Input VTT or SRT file.")
    parser.add_argument("--output", default="", help="Output file. Defaults to <input>_<lang>.<ext>")
    parser.add_argument("--target-lang", default="English", help="Target language (default: English).")
    parser.add_argument(
        "--backend", default="openai", choices=["openai", "vertex"],
        help="LLM backend: 'openai' (any OpenAI-compatible API) or 'vertex' (Gemini).",
    )
    parser.add_argument("--model", default="", help="Model name. Defaults: vertex=gemini-2.5-pro, openai=gpt-4o")
    parser.add_argument(
        "--location", default=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
        help="Vertex location for Gemini requests (default: GOOGLE_CLOUD_LOCATION or global).",
    )
    parser.add_argument(
        "--url", default=os.environ.get("OPENAI_BASE_URL", ""),
        help="Base URL for OpenAI-compatible backend (env: OPENAI_BASE_URL).",
    )
    parser.add_argument(
        "--api-key", default=os.environ.get("OPENAI_API_KEY", ""),
        help="API key for OpenAI-compatible backend (env: OPENAI_API_KEY).",
    )
    parser.add_argument("--glossary", default="", help="Path to glossary file.")
    parser.add_argument("--summary", default="", help="Optional summary to provide context.")
    parser.add_argument("--batch-seconds", type=float, default=45.0, help="Batch duration in seconds (default: 45).")
    parser.add_argument("--batch-cues", type=int, default=12, help="Max cues per batch (default: 12).")
    parser.add_argument("--context-cues", type=int, default=2, help="Context cues before/after each batch (default: 2).")
    parser.add_argument("--target-cps", type=float, default=17.0, help="Target CPS (default: 17).")
    parser.add_argument("--hard-cps", type=float, default=20.0, help="Hard CPS limit (default: 20).")
    parser.add_argument("--max-line-length", type=int, default=42, help="Max chars per subtitle line (default: 42).")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature (default: 0.2).")
    parser.add_argument("--format", default="", choices=["vtt", "srt", ""], help="Output format. Defaults to match input.")
    args = parser.parse_args()

    if not args.model:
        args.model = (
            os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")
            if args.backend == "vertex"
            else "gpt-4o"
        )

    if not args.output:
        stem = Path(args.input).stem
        ext = Path(args.input).suffix or ".vtt"
        lang_slug = args.target_lang.lower().replace(" ", "_")
        args.output = str(Path(args.input).parent / f"{stem}_{lang_slug}{ext}")

    output_format = args.format or ("srt" if args.output.lower().endswith(".srt") else "vtt")

    # Build backend-specific call function
    if args.backend == "vertex":
        model, location = args.model, args.location

        def call_fn(system_prompt: str, user_prompt: str, temperature: float) -> str:
            return call_with_retry(
                _call_vertex, model=model, system_prompt=system_prompt,
                user_prompt=user_prompt, temperature=temperature, location=location,
            )
    else:
        model = args.model
        url = args.url or "http://127.0.0.1:8787"
        api_key = args.api_key

        def call_fn(system_prompt: str, user_prompt: str, temperature: float) -> str:
            return call_with_retry(
                _call_openai_compatible, url=url, model=model,
                system_prompt=system_prompt, user_prompt=user_prompt,
                temperature=temperature, api_key=api_key,
            )

    manifest = build_manifest(
        input_path=args.input, output_path=args.output, target_lang=args.target_lang,
        backend=args.backend, model=args.model,
        location=args.location if args.backend == "vertex" else (args.url or ""),
        batch_size=args.batch_cues, batch_seconds=args.batch_seconds,
        context_cues=args.context_cues, target_cps=args.target_cps,
        hard_cps=args.hard_cps, max_line_length=args.max_line_length,
        total_cues=0,  # filled by translate_subtitles via checkpoint
    )

    translate_subtitles(
        input_path=args.input, output_path=args.output, target_lang=args.target_lang,
        call_fn=call_fn, manifest=manifest,
        glossary_path=args.glossary or None, summary=args.summary or None,
        batch_seconds=args.batch_seconds, batch_size=args.batch_cues,
        context_cues=args.context_cues, target_cps=args.target_cps,
        hard_cps=args.hard_cps, max_line_length=args.max_line_length,
        temperature=args.temperature, output_format=output_format,
    )


if __name__ == "__main__":
    main()
