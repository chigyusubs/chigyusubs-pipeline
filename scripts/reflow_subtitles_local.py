#!/usr/bin/env python3
"""Reflow word-level timestamps into subtitle cues using a local LLM (llama.cpp server).

Usage:
    python scripts/reflow_subtitles_local.py \
        --input samples/transcription_output/WEDNESDAY_DOWNTOWN_faster_word_timestamps_words.json \
        --out samples/transcription_output/WEDNESDAY_DOWNTOWN_local_reflowed.vtt \
        --url http://localhost:8787
"""

import json
import argparse
import time
import urllib.request
import urllib.error

SYSTEM_PROMPT = """\
You are a professional Japanese subtitle editor working on a fast-paced variety show.
I am giving you a JSON array of words/tokens with their exact `start` and `end` times in seconds.
Your job is to group these tokens into sensible, readable subtitle cues.

Rules:
1. ONLY use the exact "word" text from the input. Concatenate them exactly as given. NEVER add, remove, or change characters. NEVER invent labels like （静寂）.
2. Semantics: Group complete thoughts, phrases, or clauses. Do not split a noun from its particle (e.g. keep 私は together).
3. Readability: Aim for roughly 10-30 characters per cue. Break long rapid speech into smaller chunks.
4. Pacing: If there is a gap >0.4s between consecutive words, split the subtitle there.
5. Punctuation: You may add punctuation marks (、。！？) between words but NEVER change the words themselves.
6. Output: Return a JSON object with a single key "cues" containing an array. Each element must have:
   - "start": float — start time of the FIRST token in the group
   - "end": float — end time of the LAST token in the group
   - "text": string — the concatenated word texts with optional punctuation

Example: {"cues": [{"start": 1.0, "end": 2.5, "text": "こんにちは、皆さん。"}]}

/no_think"""


def format_timestamp(seconds: float) -> str:
    """Convert seconds to VTT timestamp HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def chunk_words(words, max_words=80):
    """Split word list into chunks for LLM processing."""
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(words[i:i + max_words])
    return chunks


def call_llm(system: str, user: str, url: str, max_retries: int = 3) -> str:
    """Call the llama.cpp OpenAI-compatible chat endpoint."""
    payload = json.dumps({
        "model": "qwen3",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
        "max_tokens": 4096,
        "response_format": {"type": "json_object"},
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code == 503:
                wait = (attempt + 1) * 5
                print(f"  Server busy (503), retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  HTTP error {e.code}: {e.reason}")
                raise
        except urllib.error.URLError as e:
            print(f"  Connection error: {e.reason}")
            raise

    raise RuntimeError("Max retries exceeded")


def extract_json_array(text: str) -> list:
    """Extract a JSON array from LLM output, handling markdown fences and wrapper objects."""
    text = text.strip()
    # Strip markdown fences
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    parsed = json.loads(text)

    # If the model wrapped it in an object like {"cues": [...]}, extract the array
    if isinstance(parsed, dict):
        for v in parsed.values():
            if isinstance(v, list):
                return v
        raise ValueError(f"JSON object has no array value: {list(parsed.keys())}")

    if isinstance(parsed, list):
        return parsed

    raise ValueError(f"Expected JSON array, got {type(parsed).__name__}")


def reflow_chunk(words: list, url: str) -> list:
    """Send a chunk of words to the local LLM for reflowing."""
    # Build compact input — only send start, end, word (drop probability to save tokens)
    compact = [{"start": w["start"], "end": w["end"], "word": w["word"]} for w in words]

    user_prompt = (
        'Group these word tokens into subtitle cues. Return {"cues": [...]} with {start, end, text} objects.\n\n'
        f"{json.dumps(compact, ensure_ascii=False)}"
    )

    raw = call_llm(SYSTEM_PROMPT, user_prompt, url)
    cues = extract_json_array(raw)

    # Validate: each cue needs start, end, text
    valid = []
    for cue in cues:
        if "start" in cue and "end" in cue and "text" in cue:
            text = str(cue["text"]).strip()
            if text:
                valid.append({"start": float(cue["start"]), "end": float(cue["end"]), "text": text})

    return valid


def write_vtt(cues: list, output_path: str):
    """Write cues to a VTT file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for cue in cues:
            start = format_timestamp(cue["start"])
            end = format_timestamp(cue["end"])
            f.write(f"{start} --> {end}\n")
            f.write(f"{cue['text']}\n\n")


def main():
    parser = argparse.ArgumentParser(description="Reflow word timestamps into VTT using local LLM")
    parser.add_argument("--input", default="samples/transcription_output/WEDNESDAY_DOWNTOWN_faster_word_timestamps_words.json")
    parser.add_argument("--out", default="samples/transcription_output/WEDNESDAY_DOWNTOWN_local_reflowed.vtt")
    parser.add_argument("--url", default="http://localhost:8787", help="llama.cpp server URL")
    parser.add_argument("--chunk-size", type=int, default=80, help="Words per LLM chunk")
    parser.add_argument("--limit", type=int, default=0, help="Limit to first N words (0=all, for testing)")
    args = parser.parse_args()

    print(f"Loading word timestamps from {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_words = []
    for segment in data:
        if "words" in segment:
            all_words.extend(segment["words"])

    if args.limit > 0:
        all_words = all_words[:args.limit]

    print(f"Total words: {len(all_words)}")

    word_chunks = chunk_words(all_words, max_words=args.chunk_size)
    print(f"Processing {len(word_chunks)} chunks with local LLM at {args.url}...")

    all_cues = []
    t0 = time.time()

    for i, chunk in enumerate(word_chunks):
        chunk_t0 = time.time()
        print(f"  Chunk {i+1}/{len(word_chunks)} ({len(chunk)} words)...", end=" ", flush=True)
        try:
            cues = reflow_chunk(chunk, args.url)
            elapsed = time.time() - chunk_t0
            print(f"-> {len(cues)} cues ({elapsed:.1f}s)")
            all_cues.extend(cues)
        except Exception as e:
            print(f"FAILED: {e}")

    elapsed_total = time.time() - t0
    print(f"\nDone: {len(all_cues)} cues in {elapsed_total:.1f}s")

    write_vtt(all_cues, args.out)
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
