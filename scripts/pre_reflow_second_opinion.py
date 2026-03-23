#!/usr/bin/env python3
"""Always-on Whisper second-opinion coverage check with optional glossary and VAD filtering."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chigyusubs.alignment_diagnostics import discover_alignment_diagnostics_path, load_alignment_diagnostics
from chigyusubs.metadata import finish_run, metadata_path, start_run, write_metadata
from chigyusubs.paths import find_episode_dir_from_path
from chigyusubs.rocm import ensure_rocm_env


def discover_video_for_words(words_path: Path) -> Path | None:
    episode_dir = find_episode_dir_from_path(words_path)
    if not episode_dir:
        return None
    source_dir = episode_dir / "source"
    candidates = sorted(
        p for ext in ("*.mp4", "*.mkv", "*.webm", "*.mov") for p in source_dir.glob(ext)
    )
    return candidates[0] if candidates else None


def default_output_paths(words_path: Path, model_name: str) -> tuple[Path, Path, Path]:
    diagnostics_dir = words_path.parent / "diagnostics"
    model_slug = model_name.replace("/", "_").replace(".", "_")
    stem = words_path.stem
    faster_vtt = diagnostics_dir / f"{stem}_{model_slug}.vtt"
    coverage = diagnostics_dir / f"{stem}_vs_{model_slug}_coverage.json"
    summary = diagnostics_dir / f"{stem}_{model_slug}_pre_reflow_second_opinion.json"
    return faster_vtt, coverage, summary


def default_omission_report_path(words_path: Path, model_name: str) -> Path:
    diagnostics_dir = words_path.parent / "diagnostics"
    model_slug = model_name.replace("/", "_").replace(".", "_")
    return diagnostics_dir / f"{words_path.stem}_vs_{model_slug}_raw_omissions.json"


def default_short_disagreement_report_path(words_path: Path, model_name: str) -> Path:
    diagnostics_dir = words_path.parent / "diagnostics"
    model_slug = model_name.replace("/", "_").replace(".", "_")
    return diagnostics_dir / f"{words_path.stem}_vs_{model_slug}_short_disagreements.json"


def discover_existing_secondary(words_path: Path, model_name: str) -> tuple[Path, Path] | None:
    diagnostics_dir = words_path.parent / "diagnostics"
    if not diagnostics_dir.exists():
        return None
    model_slug = model_name.replace("/", "_").replace(".", "_")
    candidates = sorted(diagnostics_dir.glob("*_words.json"))
    scored: list[tuple[int, Path]] = []
    for candidate in candidates:
        name = candidate.name.lower()
        score = 0
        if "faster" in name:
            score += 3
        if model_slug.lower() in name:
            score += 2
        if words_path.stem.lower().replace("_ctc_words", "") in name:
            score += 1
        if score > 0:
            scored.append((score, candidate))
    if not scored:
        return None
    secondary_words = sorted(scored, key=lambda item: (item[0], item[1].name))[-1][1]
    secondary_vtt = Path(str(secondary_words).replace("_words.json", ".vtt"))
    if not secondary_vtt.exists():
        return None
    return secondary_vtt, secondary_words


def discover_existing_secondary_chunk_json(words_path: Path, model_name: str) -> Path | None:
    episode_dir = find_episode_dir_from_path(words_path)
    if not episode_dir:
        return None
    probes_dir = episode_dir / "transcription" / "probes"
    if not probes_dir.exists():
        return None
    model_slug = model_name.replace("/", "_").replace(".", "_").lower()
    candidates = sorted(probes_dir.rglob("*.json"))
    scored: list[tuple[int, Path]] = []
    for candidate in candidates:
        name = candidate.name.lower()
        if "summary" in name or "compare" in name or "selected" in name:
            continue
        score = 0
        if "whisper" in name or "faster" in name:
            score += 3
        if model_slug in name:
            score += 2
        if "semantic180" in name:
            score += 2
        if score > 0:
            scored.append((score, candidate))
    if not scored:
        return None
    return sorted(scored, key=lambda item: (item[0], item[1].name))[-1][1]


def materialize_secondary_words_from_chunk_json(source_json: Path, dest_words_json: Path) -> Path | None:
    try:
        data = json.loads(source_json.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, list) or not data:
        return None
    flattened: list[dict] = []
    for chunk in data:
        if not isinstance(chunk, dict):
            return None
        chunk_start = float(chunk.get("chunk_start_s", 0.0))
        chunk_end = float(chunk.get("chunk_end_s", chunk_start))
        segments = chunk.get("segments")
        if not isinstance(segments, list):
            return None
        for seg in segments:
            if not isinstance(seg, dict):
                return None
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            start = round(chunk_start + float(seg.get("start", 0.0)), 3)
            end = round(chunk_start + float(seg.get("end", 0.0)), 3)
            if chunk_end > chunk_start:
                start = max(start, round(chunk_start, 3))
                end = min(end, round(chunk_end, 3))
            if end <= start:
                continue
            flattened.append({"start": start, "end": end, "text": text})
    if not flattened:
        return None
    dest_words_json.parent.mkdir(parents=True, exist_ok=True)
    dest_words_json.write_text(json.dumps(flattened, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return dest_words_json


def discover_gemini_raw_for_words(words_path: Path) -> Path | None:
    candidates = [words_path.stem]
    if words_path.stem.endswith("_ctc_words"):
        candidates.append(words_path.stem[: -len("_ctc_words")])
    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        path = words_path.parent / f"{candidate}_gemini_raw.json"
        if path.exists():
            return path
    return None
def _build_whisper_prompt_from_glossary(glossary_path: Path) -> str:
    """Read glossary.json and concatenate source terms into a comma-separated prompt string."""
    data = json.loads(glossary_path.read_text(encoding="utf-8"))
    entries = data.get("entries", [])
    sources = [entry["source"] for entry in entries if entry.get("source")]
    return "、".join(sources)


def _discover_glossary_file(
    explicit_glossary: str, episode_dir: Optional[Path], diagnostics_dir: Path
) -> Optional[Path]:
    """Resolve the glossary file to pass as --glossary to run_faster_whisper.py.

    Discovery order:
    1. Explicit --glossary arg
    2. <episode>/glossary/whisper_prompt_condensed.txt (legacy)
    3. Auto-generate from <episode>/glossary/glossary.json
    """
    if explicit_glossary:
        p = Path(explicit_glossary)
        return p if p.exists() else None

    if not episode_dir:
        return None

    legacy = episode_dir / "glossary" / "whisper_prompt_condensed.txt"
    if legacy.exists():
        return legacy

    glossary_json = episode_dir / "glossary" / "glossary.json"
    if glossary_json.exists():
        prompt_text = _build_whisper_prompt_from_glossary(glossary_json)
        if prompt_text:
            diagnostics_dir.mkdir(parents=True, exist_ok=True)
            generated = diagnostics_dir / "whisper_prompt_from_glossary.txt"
            generated.write_text(prompt_text, encoding="utf-8")
            return generated

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Always-on Whisper second-opinion coverage check. Runs faster-whisper on every episode and compares against the primary transcript.")
    parser.add_argument("--words", required=True, help="Primary aligned words JSON, usually *_ctc_words.json")
    parser.add_argument("--alignment-diagnostics", default="", help="Optional alignment diagnostics sidecar override")
    parser.add_argument("--video", default="", help="Optional source video override")
    parser.add_argument("--faster-vtt", default="", help="Optional faster-whisper VTT output path")
    parser.add_argument("--coverage-json", default="", help="Optional coverage report output path")
    parser.add_argument("--omission-report-json", default="", help="Optional raw-chunk omission report output path")
    parser.add_argument("--short-disagreement-report-json", default="", help="Optional short-line disagreement report output path")
    parser.add_argument("--summary-json", default="", help="Optional helper summary output path")
    parser.add_argument("--glossary", default="", help="Optional glossary file for Whisper initial_prompt. Auto-discovered from episode glossary if not given.")
    parser.add_argument("--model", default="large-v3", help="Whisper model name passed to run_faster_whisper.py")
    parser.add_argument("--compute-type", default="float16", help="Compute type passed to run_faster_whisper.py")
    parser.add_argument("--reflow-pause-ms", type=int, default=250, help="Pause threshold for the helper's faster-whisper VTT")
    parser.add_argument("--reflow-min-cue-s", type=float, default=0.25, help="Minimum cue duration for the helper's faster-whisper VTT")
    parser.add_argument("--window-s", type=float, default=2.0, help="Coverage diff window size")
    parser.add_argument("--step-s", type=float, default=1.0, help="Coverage diff step size")
    parser.add_argument("--top", type=int, default=20, help="How many top coverage-gap regions to keep in the report summary")
    parser.add_argument("--force", action="store_true", help="Rerun whisper even if artifacts already exist")
    parser.add_argument("--rerun-whisper", action="store_true", help="Ignore existing faster-whisper artifacts and rerun the secondary ASR pass")
    args = parser.parse_args()

    run = start_run("pre_reflow_second_opinion")
    words_path = Path(args.words)
    if not words_path.exists():
        raise SystemExit(f"Words JSON not found: {words_path}")

    alignment_path = discover_alignment_diagnostics_path(
        explicit_path=args.alignment_diagnostics,
        words_path=words_path,
    )
    diagnostics = load_alignment_diagnostics(alignment_path) if alignment_path else None

    faster_vtt_path, coverage_path, summary_path = default_output_paths(words_path, args.model)
    if args.faster_vtt:
        faster_vtt_path = Path(args.faster_vtt)
    if args.coverage_json:
        coverage_path = Path(args.coverage_json)
    omission_report_path = default_omission_report_path(words_path, args.model)
    if args.omission_report_json:
        omission_report_path = Path(args.omission_report_json)
    short_disagreement_report_path = default_short_disagreement_report_path(words_path, args.model)
    if args.short_disagreement_report_json:
        short_disagreement_report_path = Path(args.short_disagreement_report_json)
    if args.summary_json:
        summary_path = Path(args.summary_json)

    visual_risk_chunks = [] if not diagnostics else diagnostics.get("visual_narration_risk_chunks", [])
    secondary_words_path = Path(str(faster_vtt_path).replace(".vtt", "_words.json"))
    summary = {
        "words_json": str(words_path),
        "alignment_diagnostics_json": str(alignment_path) if alignment_path else "",
        "visual_risk_chunk_count": len(visual_risk_chunks),
        "visual_risk_chunks": visual_risk_chunks[:8],
        "secondary_vtt": str(faster_vtt_path),
        "secondary_words_json": str(secondary_words_path),
        "coverage_report_json": str(coverage_path),
        "raw_omission_report_json": str(omission_report_path),
        "short_disagreement_report_json": str(short_disagreement_report_path),
    }

    episode_dir = find_episode_dir_from_path(words_path)
    diagnostics_dir = faster_vtt_path.parent
    glossary_file = _discover_glossary_file(args.glossary, episode_dir, diagnostics_dir)
    if glossary_file:
        summary["glossary_file"] = str(glossary_file)

    video_path = Path(args.video) if args.video else discover_video_for_words(words_path)
    if not video_path or not video_path.exists():
        raise SystemExit("Could not discover source video; pass --video explicitly.")

    rerun = args.rerun_whisper or args.force
    discovered_secondary = None if rerun else discover_existing_secondary(words_path, args.model)
    discovered_secondary_json = None if rerun else discover_existing_secondary_chunk_json(words_path, args.model)
    if discovered_secondary and not (faster_vtt_path.exists() and secondary_words_path.exists()):
        faster_vtt_path, secondary_words_path = discovered_secondary
        summary["secondary_vtt"] = str(faster_vtt_path)
        summary["secondary_words_json"] = str(secondary_words_path)
    elif discovered_secondary_json and not secondary_words_path.exists():
        materialized = materialize_secondary_words_from_chunk_json(discovered_secondary_json, secondary_words_path)
        if materialized is not None:
            summary["secondary_source_json"] = str(discovered_secondary_json)
            summary["secondary_words_json"] = str(materialized)
    reused_existing = secondary_words_path.exists() and not rerun
    if not reused_existing:
        cmd = [
            sys.executable,
            "scripts/run_faster_whisper.py",
            "--video",
            str(video_path),
            "--out",
            str(faster_vtt_path),
            "--model",
            args.model,
            "--compute-type",
            args.compute_type,
            "--reflow",
            "--reflow-pause-ms",
            str(args.reflow_pause_ms),
            "--reflow-min-cue-s",
            str(args.reflow_min_cue_s),
        ]
        if glossary_file:
            cmd.extend(["--glossary", str(glossary_file)])
        print("Running faster-whisper second opinion...")
        subprocess.run(cmd, check=True, env=ensure_rocm_env(os.environ))
    else:
        print(f"Reusing existing second-opinion transcript: {secondary_words_path}")

    compare_cmd = [
        sys.executable,
        "scripts/compare_transcript_coverage.py",
        "--primary",
        str(words_path),
        "--secondary",
        str(secondary_words_path),
        "--output",
        str(coverage_path),
        "--window-s",
        str(args.window_s),
        "--step-s",
        str(args.step_s),
        "--top",
        str(args.top),
    ]
    if episode_dir:
        vad_json = episode_dir / "transcription" / "silero_vad_segments.json"
        if vad_json.exists():
            compare_cmd.extend(["--vad-json", str(vad_json)])

    subprocess.run(compare_cmd, check=True)

    omission_summary = None
    short_disagreement_summary = None
    gemini_raw_path = discover_gemini_raw_for_words(words_path)
    if gemini_raw_path:
        omission_cmd = [
            sys.executable,
            "scripts/report_raw_chunk_omissions.py",
            "--gemini-raw",
            str(gemini_raw_path),
            "--primary",
            str(words_path),
            "--secondary",
            str(secondary_words_path),
            "--output",
            str(omission_report_path),
            "--window-s",
            str(args.window_s),
            "--step-s",
            str(args.step_s),
            "--top",
            str(args.top),
        ]
        subprocess.run(omission_cmd, check=True)
        omission_report = json.loads(omission_report_path.read_text(encoding="utf-8"))
        omission_summary = omission_report.get("summary", {})
        summary["gemini_raw_json"] = str(gemini_raw_path)

        short_disagreement_cmd = [
            sys.executable,
            "scripts/report_short_line_disagreements.py",
            "--gemini-raw",
            str(gemini_raw_path),
            "--primary",
            str(words_path),
            "--secondary",
            str(secondary_words_path),
            "--output",
            str(short_disagreement_report_path),
            "--top",
            str(args.top),
        ]
        subprocess.run(short_disagreement_cmd, check=True)
        short_disagreement_report = json.loads(short_disagreement_report_path.read_text(encoding="utf-8"))
        short_disagreement_summary = short_disagreement_report.get("summary", {})

    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "status": "completed",
            "reused_existing_secondary": reused_existing,
            "video": str(video_path),
            "coverage_summary": coverage.get("summary", {}),
            "top_regions": coverage.get("top_regions", [])[: min(args.top, 8)],
        }
    )
    if omission_summary is not None:
        summary["raw_omission_summary"] = omission_summary
    if short_disagreement_summary is not None:
        summary["short_disagreement_summary"] = short_disagreement_summary
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    metadata = finish_run(
        run,
        inputs={
            "words_json": str(words_path),
            "alignment_diagnostics_json": str(alignment_path) if alignment_path else None,
            "video": str(video_path),
        },
        outputs={
            "summary_json": str(summary_path),
            "coverage_json": str(coverage_path),
            "raw_omission_report_json": str(omission_report_path) if omission_summary is not None else None,
            "short_disagreement_report_json": str(short_disagreement_report_path) if short_disagreement_summary is not None else None,
            "secondary_vtt": str(faster_vtt_path),
            "secondary_words_json": str(secondary_words_path),
        },
        settings={
            "model": args.model,
            "compute_type": args.compute_type,
            "window_s": args.window_s,
            "step_s": args.step_s,
            "reflow_pause_ms": args.reflow_pause_ms,
            "reflow_min_cue_s": args.reflow_min_cue_s,
            "rerun": bool(rerun),
        },
        stats={
            "visual_risk_chunks": len(visual_risk_chunks),
            "ran_second_opinion": 1,
            "reused_existing_secondary": int(reused_existing),
            "flagged_regions": int(coverage.get("summary", {}).get("flagged_regions", 0)),
            "raw_omission_candidates": int(omission_summary.get("classified_candidates", 0)) if omission_summary else 0,
            "short_disagreement_candidates": int(short_disagreement_summary.get("classified_candidates", 0)) if short_disagreement_summary else 0,
        },
    )
    write_metadata(summary_path, metadata)

    print(f"Summary written: {summary_path}")
    print(f"Coverage report: {coverage_path}")


if __name__ == "__main__":
    main()
