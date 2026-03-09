#!/usr/bin/env python3
"""Extract visual cues ([画面: ...]) from Gemini raw transcription JSON."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chigyusubs.metadata import finish_run, start_run, write_metadata

VISUAL_CUE_RE = re.compile(r"^\[画面:\s*(.+)\]$")


def extract_visual_cues(chunks: list[dict]) -> list[dict]:
    cues: list[dict] = []
    for chunk in chunks:
        chunk_num = chunk["chunk"]
        start_s = chunk["chunk_start_s"]
        end_s = chunk["chunk_end_s"]
        for line in chunk.get("text", "").split("\n"):
            m = VISUAL_CUE_RE.match(line.strip())
            if m:
                cues.append({
                    "chunk": chunk_num,
                    "chunk_start_s": start_s,
                    "chunk_end_s": end_s,
                    "text": m.group(1),
                })
    return cues


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract visual cues from Gemini raw JSON.")
    parser.add_argument("--input", required=True, help="Gemini raw JSON file.")
    parser.add_argument("--output", default="", help="Output visual cues JSON (default: *_visual_cues.json).")
    args = parser.parse_args()

    run = start_run("extract_visual_cues")

    input_path = Path(args.input)
    chunks = json.loads(input_path.read_text(encoding="utf-8"))
    cues = extract_visual_cues(chunks)

    if args.output:
        output_path = Path(args.output)
    else:
        stem = input_path.stem.replace("_gemini_raw", "")
        output_path = input_path.with_name(f"{stem}_visual_cues.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(cues, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    meta = finish_run(run, input=str(input_path), output=str(output_path), visual_cues_count=len(cues))
    write_metadata(output_path, meta)

    print(f"Extracted {len(cues)} visual cues → {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
