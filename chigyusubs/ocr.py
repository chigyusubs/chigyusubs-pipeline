"""OCR data loading and LLM-based filtering for chunk-wise context."""

import json
import re
import urllib.request


def load_ocr_data(jsonl_path: str) -> list[dict]:
    """Load OCR JSONL file into a list of dictionaries."""
    ocr_frames = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                ocr_frames.append(json.loads(line))
    return ocr_frames


def get_ocr_context_for_chunk(ocr_frames: list[dict], start_s: float, end_s: float) -> list[str]:
    """Extract unique, relevant Japanese text lines that appeared during this chunk."""
    chunk_lines = set()
    for frame in ocr_frames:
        if start_s <= frame.get("time_sec", 0) <= end_s:
            for line in frame.get("lines", []):
                if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', line):
                    chunk_lines.add(line)
    return sorted(list(chunk_lines))


_OCR_FILTER_PROMPT = """Here is raw OCR text extracted from a Japanese TV show:

{ocr_lines}

Please extract and clean up ONLY the most important proper nouns:
1. Comedian/Celebrity names (e.g. 浜田雅功)
2. Nicknames or titles
3. Specific locations or unique terms

RULES:
- Exclude general UI text, subtitles of normal dialogue, "WEDNESDAY DOWNTOWN", "字幕", etc.
- Combine split text if obvious (e.g. "浜田" and "雅功" -> "浜田雅功").
- Deduplicate OCR variants (e.g. keep 小籔千豊 not 小藪千豊, keep the longest correct form not truncated versions).
- Output ONLY a JSON array of strings. Do not use markdown code blocks or add explanations. Example: ["浜田雅功", "松本人志"]
"""


def filter_ocr_terms_with_llm(
    raw_ocr_lines: list[str],
    ocr_filter_url: str | None = None,
    ocr_filter_model: str | None = None,
) -> list[str]:
    """Extract proper nouns from raw OCR via local OpenAI-compatible endpoint or Vertex."""
    if not raw_ocr_lines:
        return []

    ocr_block = "\n".join(f"- {line}" for line in raw_ocr_lines)
    prompt = _OCR_FILTER_PROMPT.format(ocr_lines=ocr_block)

    try:
        print("  -> Asking LLM to filter OCR terms...", flush=True)
        if ocr_filter_url:
            text = _filter_ocr_local(prompt, ocr_filter_url, ocr_filter_model or "default")
        else:
            text = _filter_ocr_vertex(prompt)

        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    return v
            return list(parsed.keys())
    except Exception as e:
        print(f"  -> OCR Filter LLM failed: {e}", flush=True)

    return raw_ocr_lines


def _filter_ocr_vertex(prompt: str) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(vertexai=True, location="europe-west4")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2, response_mime_type="application/json"),
    )
    return response.text.strip()


def _filter_ocr_local(prompt: str, url: str, model: str) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }).encode()
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read())
    return body["choices"][0]["message"]["content"].strip()
