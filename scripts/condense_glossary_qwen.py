#!/usr/bin/env python3
"""Build a structured glossary from OCR candidates via local OpenAI-compatible endpoint."""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_candidates import clean_candidates


RESPONSE_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "glossary",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "entries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "target": {"type": "string"},
                            "context": {"type": "string"},
                            "is_hotword": {"type": "boolean"},
                        },
                        "required": ["source", "target", "context", "is_hotword"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["entries"],
            "additionalProperties": False,
        },
    },
}


def _load_raw_text(input_path: str) -> str:
    try:
        return Path(input_path).read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"Error: Could not find {input_path}")
        sys.exit(1)


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


def _normalize_entries(raw_entries: list, top_n: int) -> list[dict]:
    normalized = []
    seen_sources = set()

    for item in raw_entries:
        if not isinstance(item, dict):
            continue

        source = str(item.get("source", "")).strip()
        target = str(item.get("target", "")).strip()
        context = str(item.get("context", "")).strip()
        is_hotword = bool(item.get("is_hotword", False))

        if not source or len(source) <= 1:
            continue

        source_key = source.lower()
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)

        if not target:
            target = source
        if not context:
            context = "show-specific term"

        normalized.append({
            "source": source,
            "target": target,
            "context": context,
            "is_hotword": is_hotword,
        })
        if len(normalized) >= top_n:
            break

    return normalized


def _build_asr_terms(entries: list[dict]) -> list[str]:
    seen = set()
    terms = []
    for row in entries:
        src = row["source"].strip()
        if src and src not in seen:
            terms.append(src)
            seen.add(src)
    return terms


def _build_hotword_terms(entries: list[dict]) -> list[str]:
    seen = set()
    terms = []
    for row in entries:
        if not row.get("is_hotword"):
            continue
        src = row["source"].strip()
        if src and src not in seen:
            terms.append(src)
            seen.add(src)
    return terms


def _cap_terms_by_budget(terms: list[str], max_terms: int, max_chars: int) -> list[str]:
    capped = terms[: max(0, max_terms)]
    if max_chars <= 0:
        return capped
    out = []
    total = 0
    for term in capped:
        add = len(term) if not out else len(term) + 2  # account for ", "
        if total + add > max_chars:
            break
        out.append(term)
        total += add
    return out


def _call_chat_completions(
    url: str,
    model: str,
    system_instruction: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    use_schema: bool,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if use_schema:
        payload["response_format"] = RESPONSE_JSON_SCHEMA

    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def _backoff_delay(attempt: int, cap: float = 30.0, k: float = 2.0) -> float:
    """Hyperbolic backoff: ramps fast, asymptotes to cap."""
    return cap * attempt / (attempt + k)


def _generate_with_retry(
    url: str,
    model: str,
    system_instruction: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    use_schema: bool,
) -> str:
    max_retries = 10
    for attempt in range(max_retries):
        try:
            return _call_chat_completions(
                url=url,
                model=model,
                system_instruction=system_instruction,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                use_schema=use_schema,
            )
        except urllib.error.URLError as e:
            if attempt < max_retries - 1:
                delay = _backoff_delay(attempt + 1)
                print(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"Retrying in {delay:.0f}s...")
                time.sleep(delay)
                continue
            raise
        except Exception:
            if attempt < max_retries - 1:
                delay = _backoff_delay(attempt + 1)
                print(f"Request failed (attempt {attempt + 1}/{max_retries}). Retrying in {delay:.0f}s...")
                time.sleep(delay)
                continue
            raise
    raise RuntimeError("Request failed with no response.")


def _build_system_instruction(top_n: int, hotwords_n: int, asr_max_term_len: int) -> str:
    return (
        "You are an expert Japanese TV subtitle glossary editor.\n"
        "You will receive noisy OCR candidates extracted from a Japanese variety/owarai show.\n"
        "Build a clean bilingual glossary for ASR transcription and subtitle translation.\n\n"
        "Your output has two tiers of terms. Understand what each is for:\n\n"
        f"## initial_prompt terms (up to {top_n} entries)\n"
        "These go into Whisper's initial_prompt to set vocabulary context for the whole episode.\n"
        "The 224-token decoder budget means roughly 80 terms max in practice.\n"
        "Include: talent/comedian names, show segment titles, place names, brands,\n"
        "menu items, and domain jargon specific to this episode.\n"
        "Prioritize terms with unusual or ambiguous kanji readings.\n"
        "Drop: generic Japanese words, common verbs/adjectives, anything Whisper\n"
        "already transcribes correctly without help.\n\n"
        f"## hotwords terms (mark up to ~{hotwords_n} most critical with is_hotword: true)\n"
        "These get logit bias during Whisper decoding — much stronger than initial_prompt.\n"
        "Use sparingly. Only flag terms where Whisper is likely to produce the WRONG\n"
        "transcription without explicit bias:\n"
        "- Names with non-standard readings (e.g. unusual kanji for a person's name)\n"
        "- Coined show-specific terms that aren't real dictionary words\n"
        "- Foreign loanwords rendered in unusual katakana\n"
        "Do NOT hotword common names, well-known places, or standard Japanese vocabulary.\n\n"
        "## Output rules\n"
        "1) source: canonical Japanese form (or original script if non-Japanese).\n"
        "2) target: preferred English subtitle rendering (romanization acceptable for names).\n"
        "3) context: short disambiguating note (<= 12 words).\n"
        "4) is_hotword: true only for terms meeting the strict hotwords criteria above.\n"
        "5) No duplicates by source.\n"
        "6) Name handling: prefer full canonical name; include surname-only separately\n"
        "   only when frequently used alone on the show.\n"
        f"7) Keep source terms concise for ASR prompts (aim for <= {asr_max_term_len} chars);\n"
        "   if OCR text is longer, split into canonical sub-terms.\n"
        "8) Fewer high-confidence entries is better than padding the list.\n\n"
        "Return strict JSON only. No markdown.\n"
        'Output schema: {"entries":[{"source":"...","target":"...","context":"...","is_hotword":true/false}]}'
    )


def _generate_structured_glossary(
    url: str,
    model: str,
    raw_glossary: str,
    top_n: int,
    hotwords_n: int,
    asr_max_term_len: int,
) -> list[dict]:
    system_instruction = _build_system_instruction(top_n, hotwords_n, asr_max_term_len)
    user_prompt = (
        f"Extract up to {top_n} entries from this noisy OCR candidate dump:\n\n{raw_glossary}"
    )

    raw = _generate_with_retry(
        url=url,
        model=model,
        system_instruction=system_instruction,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=max(3000, top_n * 65),
        use_schema=True,
    )

    try:
        payload = _extract_json_object(raw)
    except Exception:
        repair_system = (
            "You repair malformed JSON. Return strict JSON only, no markdown."
        )
        repair_user = (
            f"Convert this malformed output into valid JSON schema "
            f'{{"entries":[{{"source":"...","target":"...","context":"...","is_hotword":true/false}}]}}. '
            f"Keep up to {top_n} entries.\n\n{raw}"
        )
        repaired = _generate_with_retry(
            url=url,
            model=model,
            system_instruction=repair_system,
            user_prompt=repair_user,
            temperature=0.0,
            max_tokens=max(3000, top_n * 65),
            use_schema=True,
        )
        payload = _extract_json_object(repaired)

    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError("Response JSON missing entries[]")
    return _normalize_entries(entries, top_n=top_n)


def _write_outputs(
    entries: list[dict],
    output_prompt_path: str,
    output_structured_path: str,
    output_translation_tsv_path: str,
    output_hotwords_path: str,
    model_name: str,
    source_candidates: int,
    asr_max_terms: int,
    asr_max_chars: int,
):
    out_prompt = Path(output_prompt_path)
    out_structured = Path(output_structured_path)
    out_tsv = Path(output_translation_tsv_path)
    out_hotwords = Path(output_hotwords_path)

    out_prompt.parent.mkdir(parents=True, exist_ok=True)
    out_structured.parent.mkdir(parents=True, exist_ok=True)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    out_hotwords.parent.mkdir(parents=True, exist_ok=True)

    asr_terms = _build_asr_terms(entries)
    asr_terms = _cap_terms_by_budget(
        asr_terms,
        max_terms=asr_max_terms,
        max_chars=asr_max_chars,
    )
    hotwords_terms = _build_hotword_terms(entries)

    out_prompt.write_text(", ".join(asr_terms), encoding="utf-8")
    out_hotwords.write_text(", ".join(hotwords_terms), encoding="utf-8")

    with out_tsv.open("w", encoding="utf-8") as f:
        f.write("source\ttarget\tcontext\n")
        for row in entries:
            src = row["source"].replace("\t", " ").strip()
            tgt = row["target"].replace("\t", " ").strip()
            ctx = row["context"].replace("\t", " ").strip()
            f.write(f"{src}\t{tgt}\t{ctx}\n")

    payload = {
        "source_candidates": source_candidates,
        "model": model_name,
        "count": len(entries),
        "entries": entries,
        "asr_terms": asr_terms,
        "hotwords_terms": hotwords_terms,
    }
    out_structured.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_glossary(
    input_path: str,
    output_prompt_path: str,
    output_structured_path: str,
    output_translation_tsv_path: str,
    output_hotwords_path: str,
    url: str,
    model_name: str,
    top_n: int,
    hotwords_n: int,
    asr_max_terms: int,
    asr_max_chars: int,
    asr_max_term_len: int,
):
    print(f"Reading raw OCR dump from {input_path}...")
    raw_glossary = _load_raw_text(input_path)

    print("Running deterministic cleanup...")
    candidates = clean_candidates(raw_glossary)
    source_candidates = len(candidates)

    max_input_candidates = 900
    prompt_candidates = candidates[:max_input_candidates]
    prompt_blob = "\n".join(prompt_candidates)

    print(f"Cleaned to {source_candidates} candidates.")
    if source_candidates > max_input_candidates:
        print(f"Using first {max_input_candidates} candidates for prompting.")

    print(f"Sending structured glossary request to {model_name} at {url}...")
    entries = _generate_structured_glossary(
        url=url,
        model=model_name,
        raw_glossary=prompt_blob,
        top_n=top_n,
        hotwords_n=hotwords_n,
        asr_max_term_len=asr_max_term_len,
    )
    print(f"Received {len(entries)} structured glossary entries.")

    hotword_count = sum(1 for e in entries if e.get("is_hotword"))
    print(f"  ({hotword_count} flagged as hotwords)")

    _write_outputs(
        entries=entries,
        output_prompt_path=output_prompt_path,
        output_structured_path=output_structured_path,
        output_translation_tsv_path=output_translation_tsv_path,
        output_hotwords_path=output_hotwords_path,
        model_name=model_name,
        source_candidates=source_candidates,
        asr_max_terms=asr_max_terms,
        asr_max_chars=asr_max_chars,
    )

    print(f"ASR prompt written to {output_prompt_path}")
    print(f"Structured glossary written to {output_structured_path}")
    print(f"Translation glossary written to {output_translation_tsv_path}")
    print(f"ASR hotwords written to {output_hotwords_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build a structured source/target/context glossary via local OpenAI-compatible endpoint."
    )
    parser.add_argument(
        "--input",
        default="samples/whisper_prompt.txt",
        help="Input OCR candidate file (newline and/or comma separated).",
    )
    parser.add_argument(
        "--out-prompt",
        default="samples/whisper_prompt_condensed_qwen.txt",
        help="Comma-separated ASR prompt output.",
    )
    parser.add_argument(
        "--out-structured",
        default="samples/whisper_prompt_condensed_qwen_structured.json",
        help="Structured glossary JSON output.",
    )
    parser.add_argument(
        "--out-translation",
        default="samples/translation_glossary_qwen.tsv",
        help="Translation glossary TSV output (source,target,context).",
    )
    parser.add_argument(
        "--out-hotwords",
        default="samples/whisper_hotwords_qwen.txt",
        help="ASR hotwords output (comma-separated, LLM-selected terms only).",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("QWEN_URL", "http://127.0.0.1:8787"),
        help="OpenAI-compatible base URL for llama.cpp server.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("QWEN_VISION_MODEL", "qwen3-vl"),
        help="Model alias exposed by llama.cpp server.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=80,
        help="Max glossary entries to request from the LLM.",
    )
    parser.add_argument(
        "--hotwords-n",
        type=int,
        default=15,
        help="Suggested hotword count (soft guidance to LLM).",
    )
    parser.add_argument(
        "--asr-max-terms",
        type=int,
        default=200,
        help="Hard cap for number of terms in Whisper initial_prompt output.",
    )
    parser.add_argument(
        "--asr-max-chars",
        type=int,
        default=1000,
        help="Hard cap for total chars in Whisper initial_prompt output (0 disables char cap).",
    )
    parser.add_argument(
        "--asr-max-term-len",
        type=int,
        default=20,
        help="Instruct the model to keep source terms at or below this length (soft guidance).",
    )
    args = parser.parse_args()

    build_glossary(
        input_path=args.input,
        output_prompt_path=args.out_prompt,
        output_structured_path=args.out_structured,
        output_translation_tsv_path=args.out_translation,
        output_hotwords_path=args.out_hotwords,
        url=args.url,
        model_name=args.model,
        top_n=args.top_n,
        hotwords_n=args.hotwords_n,
        asr_max_terms=args.asr_max_terms,
        asr_max_chars=args.asr_max_chars,
        asr_max_term_len=args.asr_max_term_len,
    )


if __name__ == "__main__":
    main()
