import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import TypedDict

from google import genai
from google.genai import types


class GlossaryEntry(TypedDict):
    source: str
    target: str
    context: str


class GlossaryResponse(TypedDict):
    entries: list[GlossaryEntry]


def _load_raw_text(input_path: str) -> str:
    try:
        return Path(input_path).read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"Error: Could not find {input_path}")
        sys.exit(1)


def _extract_candidates(raw_text: str) -> list[str]:
    # Supports both newline-delimited and comma-delimited candidate files.
    normalized = raw_text.replace("、", ",")
    parts = []
    seen = set()
    for line in normalized.splitlines():
        line = line.strip()
        if not line:
            continue
        if "," in line:
            items = [x.strip() for x in line.split(",") if x.strip()]
        else:
            items = [line]
        for item in items:
            if len(item) <= 1:
                continue
            # Drop obvious long sentence-like noise.
            if len(item) > 48:
                continue
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            parts.append(item)
    return parts


def _strip_code_fences(text: str) -> str:
    out = text.strip()
    if out.startswith("```"):
        out = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", out)
        out = re.sub(r"\s*```$", "", out)
    return out.strip()


def _extract_json(text: str) -> dict:
    cleaned = _strip_code_fences(text)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Fallback: pull the first object-like block.
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

        normalized.append(
            {
                "source": source,
                "target": target,
                "context": context,
            }
        )
        if len(normalized) >= top_n:
            break

    return normalized


def _is_asr_friendly_term(term: str) -> bool:
    t = term.strip()
    if not t:
        return False
    # Avoid feeding sentence-like or very long phrases into Whisper prompts/hotwords.
    if len(t) > 28:
        return False
    if len(t) > 18 and re.search(r"[。.!！?？]", t):
        return False
    if len(t) > 20 and t.count(" ") >= 2:
        return False
    return True


def _build_asr_terms(entries: list[dict]) -> list[str]:
    preferred = []
    seen = set()

    for row in entries:
        src = row["source"].strip()
        if src in seen:
            continue
        if _is_asr_friendly_term(src):
            preferred.append(src)
            seen.add(src)

    # Keep fallback terms so count remains stable even with strict filtering.
    for row in entries:
        src = row["source"].strip()
        if src in seen:
            continue
        preferred.append(src)
        seen.add(src)

    return preferred


def _generate_with_retry(
    client: genai.Client,
    model_name: str,
    contents: str,
    config: types.GenerateContentConfig,
):
    max_retries = 10
    response = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                msg = str(e).strip().splitlines()[0]
                print(f"Vertex call failed (attempt {attempt + 1}/{max_retries}): {msg}")
                print("Retrying immediately...")
                continue
            raise
    if response is None:
        raise RuntimeError("Vertex request failed with no response.")
    return response


def _parse_response_payload(response) -> dict:
    parsed_payload = getattr(response, "parsed", None)
    if isinstance(parsed_payload, dict):
        return parsed_payload
    if parsed_payload is not None and hasattr(parsed_payload, "model_dump"):
        return parsed_payload.model_dump()
    return _extract_json(response.text or "")


def _repair_json_payload(
    client: genai.Client,
    model_name: str,
    raw_text: str,
    top_n: int,
) -> dict:
    repair_prompt = (
        f"Convert this malformed model output into valid JSON.\n"
        f"Required schema: {{\"entries\":[{{\"source\":\"...\",\"target\":\"...\",\"context\":\"...\"}}]}}\n"
        f"Keep up to {top_n} entries.\n"
        f"Output JSON only, no markdown.\n\n"
        f"{raw_text}"
    )
    repair_config = types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type="application/json",
        response_schema=GlossaryResponse,
        max_output_tokens=8192,
    )
    repair_response = _generate_with_retry(
        client=client,
        model_name=model_name,
        contents=repair_prompt,
        config=repair_config,
    )
    return _parse_response_payload(repair_response)


def _generate_structured_glossary(
    client: genai.Client,
    model_name: str,
    raw_glossary: str,
    top_n: int,
) -> list[dict]:
    system_instruction = (
        "You are an expert Japanese TV subtitle glossary editor.\n"
        "Convert noisy OCR candidates into a clean bilingual glossary for both ASR and subtitle translation.\n\n"
        "Rules:\n"
        "1) Keep only proper nouns, places, show segments, menu items, brands, and unique jargon.\n"
        "2) Drop full sentences, conversational fragments, UI noise, and generic words.\n"
        "3) Produce exactly the most important terms first (names and core show terms at the top).\n"
        "4) source must be canonical Japanese term (or original script seen in OCR if non-Japanese).\n"
        "5) target must be preferred English subtitle rendering (romanization is acceptable for names).\n"
        "6) context must be short and disambiguating (<= 12 words).\n"
        "7) No duplicates by source.\n\n"
        "Return strict JSON only, with this shape:\n"
        '{"entries":[{"source":"...","target":"...","context":"..."}]}'
    )

    prompt = (
        f"Extract the top {top_n} entries from this noisy OCR candidate dump:\n\n"
        f"{raw_glossary}"
    )

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.1,
        response_mime_type="application/json",
        response_schema=GlossaryResponse,
        max_output_tokens=8192,
    )
    response = _generate_with_retry(
        client=client,
        model_name=model_name,
        contents=prompt,
        config=config,
    )

    try:
        payload = _parse_response_payload(response)
    except Exception:
        print("Primary JSON parse failed. Running repair pass...")
        payload = _repair_json_payload(
            client=client,
            model_name=model_name,
            raw_text=response.text or "",
            top_n=top_n,
        )

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
    hotwords_n: int,
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
    hotwords_terms = asr_terms[: max(0, hotwords_n)]

    # Legacy-compatible output: comma-separated ASR prompt terms.
    out_prompt.write_text(", ".join(asr_terms), encoding="utf-8")

    # Hotwords output: same style, but typically smaller.
    out_hotwords.write_text(", ".join(hotwords_terms), encoding="utf-8")

    # Translation glossary output.
    with out_tsv.open("w", encoding="utf-8") as f:
        f.write("source\ttarget\tcontext\n")
        for row in entries:
            src = row["source"].replace("\t", " ").strip()
            tgt = row["target"].replace("\t", " ").strip()
            ctx = row["context"].replace("\t", " ").strip()
            f.write(f"{src}\t{tgt}\t{ctx}\n")

    # Master structured output.
    payload = {
        "source_candidates": source_candidates,
        "model": model_name,
        "count": len(entries),
        "entries": entries,
        "asr_terms": asr_terms,
        "hotwords_terms": hotwords_terms,
    }
    out_structured.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_glossary(
    input_path: str,
    output_prompt_path: str,
    output_structured_path: str,
    output_translation_tsv_path: str,
    output_hotwords_path: str,
    model_name: str,
    top_n: int,
    hotwords_n: int,
):
    print(f"Reading raw OCR dump from {input_path}...")
    raw_glossary = _load_raw_text(input_path)
    candidates = _extract_candidates(raw_glossary)
    source_candidates = len(candidates)
    # Keep prompt size bounded for more stable structured generation.
    max_input_candidates = 900
    prompt_candidates = candidates[:max_input_candidates]
    prompt_blob = "\n".join(prompt_candidates)
    print(f"Loaded {source_candidates} source candidates.")
    if source_candidates > max_input_candidates:
        print(f"Using first {max_input_candidates} cleaned candidates for prompting.")

    print("Initializing Vertex AI Client...")
    client = genai.Client()

    print(f"Sending structured glossary request to {model_name}...")
    entries = _generate_structured_glossary(
        client=client,
        model_name=model_name,
        raw_glossary=prompt_blob,
        top_n=top_n,
    )
    print(f"Received {len(entries)} structured glossary entries.")

    if len(entries) < top_n:
        seen = {x["source"].strip().lower() for x in entries}
        remaining_candidates = [c for c in candidates if c.lower() not in seen]
        fill_attempts = 0
        max_fill_attempts = 5

        while len(entries) < top_n and remaining_candidates and fill_attempts < max_fill_attempts:
            fill_attempts += 1
            need = top_n - len(entries)
            fill_blob = "\n".join(remaining_candidates[:max_input_candidates])
            print(f"Supplement pass {fill_attempts}: requesting {need} additional entries...")

            extra = _generate_structured_glossary(
                client=client,
                model_name=model_name,
                raw_glossary=fill_blob,
                top_n=need,
            )

            added = 0
            for row in extra:
                key = row["source"].strip().lower()
                if key in seen:
                    continue
                entries.append(row)
                seen.add(key)
                added += 1
                if len(entries) >= top_n:
                    break

            remaining_candidates = [c for c in remaining_candidates if c.lower() not in seen]
            print(f"Supplement pass {fill_attempts}: added {added}, total now {len(entries)}.")

        if len(entries) < top_n:
            print(f"Warning: Requested {top_n} entries but received {len(entries)} after supplements.")

    _write_outputs(
        entries=entries,
        output_prompt_path=output_prompt_path,
        output_structured_path=output_structured_path,
        output_translation_tsv_path=output_translation_tsv_path,
        output_hotwords_path=output_hotwords_path,
        model_name=model_name,
        source_candidates=source_candidates,
        hotwords_n=hotwords_n,
    )

    print(f"ASR prompt written to {output_prompt_path}")
    print(f"Structured glossary written to {output_structured_path}")
    print(f"Translation glossary written to {output_translation_tsv_path}")
    print(f"ASR hotwords written to {output_hotwords_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build a structured source/target/context glossary from OCR candidates via Vertex Gemini."
    )
    parser.add_argument(
        "--input",
        default="samples/whisper_prompt.txt",
        help="Input OCR candidate file (newline and/or comma separated).",
    )
    parser.add_argument(
        "--out-prompt",
        default="samples/whisper_prompt_condensed.txt",
        help="Legacy-compatible comma-separated ASR prompt output.",
    )
    parser.add_argument(
        "--out-structured",
        default="samples/whisper_prompt_condensed_structured.json",
        help="Structured glossary JSON output.",
    )
    parser.add_argument(
        "--out-translation",
        default="samples/translation_glossary.tsv",
        help="Translation glossary TSV output (source,target,context).",
    )
    parser.add_argument(
        "--out-hotwords",
        default="samples/whisper_hotwords.txt",
        help="ASR hotwords output (comma-separated subset of source terms).",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview"),
        help="Gemini model name.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Number of glossary entries to keep.",
    )
    parser.add_argument(
        "--hotwords-n",
        type=int,
        default=40,
        help="Number of source terms to export to hotwords.",
    )
    args = parser.parse_args()

    build_glossary(
        input_path=args.input,
        output_prompt_path=args.out_prompt,
        output_structured_path=args.out_structured,
        output_translation_tsv_path=args.out_translation,
        output_hotwords_path=args.out_hotwords,
        model_name=args.model,
        top_n=args.top_n,
        hotwords_n=args.hotwords_n,
    )


if __name__ == "__main__":
    main()
