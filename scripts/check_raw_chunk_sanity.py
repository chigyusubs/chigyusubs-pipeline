#!/usr/bin/env python3
"""Deterministic chunk-level sanity gate for raw transcript JSON artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chigyusubs.metadata import finish_run, inherit_run_id, lineage_output_path, metadata_path, start_run, write_metadata
from chigyusubs.raw_chunk_sanity import inspect_chunk, summarize_chunk_inspections


def main() -> None:
    run = start_run("raw_chunk_sanity")
    parser = argparse.ArgumentParser(description="Check raw transcript chunks for obvious upstream failures before alignment/reflow.")
    parser.add_argument("--input", required=True, help="Raw chunk transcript JSON, usually *_gemini_raw.json.")
    parser.add_argument("--output", default="", help="Output JSON path. Defaults to transcription/diagnostics/<run>_raw_chunk_sanity.json.")
    parser.add_argument("--fail-on-red", action="store_true", help="Exit non-zero when any chunk is red.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    run = inherit_run_id(run, input_path)
    chunks = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(chunks, list):
        raise SystemExit("Expected a list of chunk objects.")

    if not args.output:
        if input_path.parent.name == "transcription":
            output_dir = input_path.parent / "diagnostics"
        else:
            output_dir = input_path.parent
        output_path = lineage_output_path(output_dir, artifact_type="raw_chunk_sanity", run=run, suffix=".json")
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    inspected = [inspect_chunk(chunk) for chunk in chunks]
    summary = summarize_chunk_inspections(inspected)

    payload = {
        "input_raw_json": str(input_path),
        "summary": summary,
        "policy": {
            "red": [
                "thought / meta transcription leakage",
                "spoken chunks that lost all `-- ` turn markers",
                "visual-only chunk substitution",
                "pathological repetition loops",
            ],
            "yellow": [
                "mixed marked/unmarked spoken lines",
                "visual-heavy mixed chunk",
                "dominant repeated short line",
            ],
        },
        "chunks": inspected,
    }

    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    metadata = finish_run(
        run,
        inputs={"raw_json": str(input_path)},
        outputs={"sanity_report_json": str(output_path)},
        settings={"fail_on_red": args.fail_on_red},
        stats={
            "chunks": len(inspected),
            "green_chunks": summary["green_chunks"],
            "yellow_chunks": summary["yellow_chunks"],
            "red_chunks": summary["red_chunks"],
        },
    )
    write_metadata(output_path, metadata)

    print(f"Wrote raw chunk sanity report: {output_path}")
    print(
        "  "
        f"green={summary['green_chunks']} "
        f"yellow={summary['yellow_chunks']} "
        f"red={summary['red_chunks']}"
    )
    if summary["red_chunk_ids"]:
        print(f"  Red chunks: {summary['red_chunk_ids']}")
    if summary["yellow_chunk_ids"]:
        print(f"  Yellow chunks: {summary['yellow_chunk_ids']}")
    print(f"Metadata written: {metadata_path(output_path)}")

    if args.fail_on_red and summary["red_chunk_ids"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
