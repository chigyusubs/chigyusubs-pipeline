#!/usr/bin/env python3
"""Silero VAD + Gemini + stable-ts transcription pipeline.

Phase 1: Silero VAD → speech/silence boundaries (for chunk splitting)
Phase 2: Gemini transcription (chunked at VAD silence gaps) → JSON with speaker + text
Phase 3: stable-ts forced alignment (Whisper cross-attention) → word-level timestamps
Phase 4: Reflow + speaker labels → final VTT

Speaker identification is done by Gemini in Phase 2 — it returns a "speaker" field
per utterance. These labels are carried through to the final VTT by estimating each
utterance's temporal position within its chunk.

Usage:
  python3.12 scripts/transcribe_pipeline.py \
    --video samples/episodes/.../source/video.mp4 \
    --glossary samples/episodes/.../glossary/translation_glossary_v2.tsv \
    --output samples/episodes/.../transcription/406_pipeline.json \
    --model gemini-3.1-pro-preview \
    --whisper-model large-v3 \
    --chunk-minutes 10
"""

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

from chigyusubs.audio import extract_audio_chunk, get_duration
from chigyusubs.chunking import find_chunk_boundaries
from chigyusubs.glossary import load_glossary_names
from chigyusubs.reflow import reflow_words
from chigyusubs.vad import run_silero_vad

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transcribe_gemini import build_prompt, transcribe_chunk


def log(msg: str = ""):
    print(msg, flush=True)


def phase_header(n: int, title: str):
    log()
    log("=" * 60)
    log(f"PHASE {n}: {title}")
    log("=" * 60)


# ---------------------------------------------------------------------------
# Phase 4: Attach Gemini speaker labels to reflowed cues
# ---------------------------------------------------------------------------

def estimate_utterance_times(
    utterances: list[dict],
    chunk_boundaries: list[tuple[float, float]],
) -> list[dict]:
    """Estimate temporal position of each Gemini utterance within its chunk.

    Distributes utterances evenly across chunk duration.
    Returns list of {time, speaker} for speech utterances with speakers.
    """
    timed = []
    by_chunk: dict[int, list[dict]] = {}
    for u in utterances:
        if u["type"] != "speech" or not u.get("speaker"):
            continue
        by_chunk.setdefault(u.get("chunk", 0), []).append(u)

    for ci, utts in by_chunk.items():
        if ci < len(chunk_boundaries):
            c_start, c_end = chunk_boundaries[ci]
        else:
            c_start = utts[0].get("chunk_start_s", 0)
            c_end = c_start + 600
        c_dur = c_end - c_start
        n = len(utts)
        for i, u in enumerate(utts):
            t = c_start + c_dur * (i + 0.5) / n
            timed.append({"time": t, "speaker": u["speaker"]})

    return timed


def attach_speakers(cues: list[dict], gemini_timed: list[dict]) -> list[dict]:
    """Attach Gemini speaker names to reflowed cues by nearest temporal match."""
    if not gemini_timed:
        return cues

    for cue in cues:
        cue_mid = (cue["start"] + cue["end"]) / 2
        best_speaker = ""
        best_dist = float("inf")
        for gt in gemini_timed:
            dist = abs(gt["time"] - cue_mid)
            if dist < best_dist:
                best_dist = dist
                best_speaker = gt["speaker"]
        cue["speaker"] = best_speaker

    return cues


from chigyusubs.vtt import write_vtt as _write_vtt_basic


def write_vtt(cues: list[dict], output_path: str):
    _write_vtt_basic(cues, output_path, include_speaker=True)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    video_path: str,
    output_path: str,
    glossary_path: str = "",
    gemini_model: str = "gemini-3.1-pro-preview",
    whisper_model: str = "large-v3",
    chunk_minutes: float = 10,
    only_phases: set[int] | None = None,
    work_dir: str | None = None,
):
    run_phase = (lambda p: p in only_phases) if only_phases else (lambda p: True)
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(output_path).stem

    # Intermediate file paths (persist in output dir for resumability)
    vad_json_path = str(out_dir / f"{stem}_vad.json")
    gemini_json_path = str(out_dir / f"{stem}_gemini_raw.json")
    words_json_path = str(out_dir / f"{stem}_words.json")
    vtt_path = output_path.replace(".json", ".vtt")

    log(f"Video: {video_path}")
    log(f"Output: {output_path}")
    duration = get_duration(video_path)
    log(f"Duration: {duration:.0f}s ({duration / 60:.1f} min)")

    # ==================================================================
    # Phase 1: Silero VAD
    # ==================================================================
    if run_phase(1):
        phase_header(1, "Silero VAD")
        vad_segments = run_silero_vad(video_path, work_dir=str(out_dir))

        total_speech = sum(s["end"] - s["start"] for s in vad_segments)
        log(f"  Speech: {total_speech:.0f}s / {duration:.0f}s "
            f"({total_speech / duration * 100:.0f}%)")

        log(f"  Writing {vad_json_path}")
        with open(vad_json_path, "w", encoding="utf-8") as f:
            json.dump(vad_segments, f, ensure_ascii=False, indent=2)
        log("  Phase 1 done.")
    else:
        log("\nPhase 1: SKIPPED (using existing VAD)")

    # Load VAD
    if not os.path.exists(vad_json_path):
        log(f"ERROR: VAD output not found: {vad_json_path}")
        log("Run Phase 1 first.")
        sys.exit(1)
    with open(vad_json_path, "r", encoding="utf-8") as f:
        vad_segments = json.load(f)
    log(f"VAD loaded: {len(vad_segments)} speech segments")

    # ==================================================================
    # Phase 2: Gemini transcription (VAD-chunked)
    # ==================================================================
    chunk_bounds: list[tuple[float, float]] = []
    all_utterances: list[dict] = []

    if run_phase(2):
        phase_header(2, "Gemini Transcription")

        target_chunk_s = chunk_minutes * 60
        log(f"  Finding chunk boundaries (target {chunk_minutes:.0f} min, "
            f"min gap 2.0s)...")
        chunk_bounds = find_chunk_boundaries(
            vad_segments, duration,
            target_chunk_s=target_chunk_s,
            min_gap_s=2.0,
        )
        log(f"  {len(chunk_bounds)} chunks:")
        for i, (cs, ce) in enumerate(chunk_bounds):
            log(f"    [{i + 1}] {cs / 60:.1f} - {ce / 60:.1f} min ({ce - cs:.0f}s)")

        # Load glossary
        glossary_entries = []
        if glossary_path and os.path.exists(glossary_path):
            glossary_entries = load_glossary_names(glossary_path)
            log(f"  Glossary: {len(glossary_entries)} entries")

        # Resume from partial output if available
        start_chunk = 0
        if os.path.exists(gemini_json_path):
            with open(gemini_json_path, "r", encoding="utf-8") as f:
                all_utterances = json.load(f)
            done_chunks = set(u.get("chunk", 0) for u in all_utterances)
            start_chunk = max(done_chunks) + 1 if done_chunks else 0
            if start_chunk > 0:
                log(f"  Resuming from chunk {start_chunk + 1} "
                    f"({len(all_utterances)} utterances from previous run)")

        prev_context: list[dict] | None = None
        if all_utterances:
            last_ci = max(u.get("chunk", 0) for u in all_utterances)
            prev_context = [u for u in all_utterances if u.get("chunk") == last_ci]

        with tempfile.TemporaryDirectory() as audio_tmpdir:
            for i, (c_start, c_end) in enumerate(chunk_bounds):
                if i < start_chunk:
                    continue
                c_dur = c_end - c_start

                log(f"\n  --- Chunk {i + 1}/{len(chunk_bounds)} ---")
                log(f"  Time: {c_start / 60:.1f} - {c_end / 60:.1f} min ({c_dur:.0f}s)")

                chunk_path = os.path.join(audio_tmpdir, f"chunk_{i}.mp3")
                log(f"  Extracting audio to {chunk_path}...")
                extract_audio_chunk(
                    video_path, chunk_path,
                    start_s=c_start, duration_s=c_dur,
                )
                chunk_bytes = Path(chunk_path).read_bytes()
                chunk_mb = len(chunk_bytes) / (1024 * 1024)
                log(f"  Audio size: {chunk_mb:.1f} MB")

                log("  Sending to Gemini (streaming)...")
                prompt = build_prompt(glossary_entries, prev_context)
                utterances = transcribe_chunk(
                    chunk_bytes, prompt, gemini_model, max_retries=20,
                )

                speech_count = sum(1 for u in utterances if u["type"] == "speech")
                sfx_count = sum(1 for u in utterances if u["type"] == "sfx")
                speakers = sorted(set(
                    u["speaker"] for u in utterances
                    if u["type"] == "speech" and u["speaker"]
                ))
                log(f"  Result: {speech_count} speech + {sfx_count} sfx")
                log(f"  Speakers: {', '.join(speakers) or '(none detected)'}")

                for u in utterances:
                    u["chunk"] = i
                    u["chunk_start_s"] = c_start

                all_utterances.extend(utterances)
                prev_context = utterances

                # Save after each chunk (resumable)
                log(f"  Saving progress ({len(all_utterances)} total utterances)...")
                with open(gemini_json_path, "w", encoding="utf-8") as f:
                    json.dump(all_utterances, f, ensure_ascii=False, indent=2)

        # Summary
        all_speakers = Counter(
            u["speaker"] for u in all_utterances
            if u["type"] == "speech" and u["speaker"]
        )
        log(f"\n  Gemini transcript complete: {len(all_utterances)} utterances")
        log(f"  Speakers: {dict(all_speakers.most_common())}")
        log(f"  Saved: {gemini_json_path}")
        log("  Phase 2 done.")
    else:
        log("\nPhase 2: SKIPPED (using existing Gemini transcript)")

    # Load Gemini transcript (required for phases 3-4)
    if os.path.exists(gemini_json_path):
        with open(gemini_json_path, "r", encoding="utf-8") as f:
            all_utterances = json.load(f)
        if not chunk_bounds:
            chunk_indices = sorted(set(u.get("chunk", 0) for u in all_utterances))
            for ci in chunk_indices:
                ci_utts = [u for u in all_utterances if u.get("chunk") == ci]
                c_start = ci_utts[0].get("chunk_start_s", 0)
                chunk_bounds.append((c_start, c_start + chunk_minutes * 60))
        log(f"Gemini transcript: {len(all_utterances)} utterances loaded")
    elif any(run_phase(p) for p in [3, 4]):
        log(f"ERROR: Gemini transcript not found: {gemini_json_path}")
        log("Run Phase 2 first.")
        sys.exit(1)
    else:
        log("No Gemini transcript yet — stopping here.")
        return

    # ==================================================================
    # Phase 3: stable-ts forced alignment
    # ==================================================================
    if run_phase(3):
        phase_header(3, "stable-ts Forced Alignment")

        speech_lines = [
            u["text"].strip()
            for u in all_utterances
            if u["type"] == "speech" and u.get("text", "").strip()
        ]
        plain_text = "\n".join(speech_lines)
        log(f"  {len(speech_lines)} speech utterances to align")
        log(f"  Text length: {len(plain_text)} chars")

        log(f"  Loading Whisper model: {whisper_model}...")
        import stable_whisper
        model = stable_whisper.load_model(whisper_model, device="cuda")
        log("  Model loaded.")

        log("  Running alignment (this takes a while)...")
        result = model.align(video_path, plain_text, language="ja")

        log("  Extracting word-level timestamps...")
        segments_data = []
        for seg in result.segments:
            words_data = [{
                "start": w.start,
                "end": w.end,
                "word": w.word,
                "probability": getattr(w, "probability", 0.0),
            } for w in seg.words]
            segments_data.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "words": words_data,
            })

        total_words = sum(len(s["words"]) for s in segments_data)
        log(f"  Aligned: {total_words} words in {len(segments_data)} segments")

        log(f"  Writing {words_json_path}")
        with open(words_json_path, "w", encoding="utf-8") as f:
            json.dump(segments_data, f, ensure_ascii=False, indent=2)

        log("  Freeing GPU memory...")
        del model
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        log("  Phase 3 done.")
    else:
        log("\nPhase 3: SKIPPED (using existing word timestamps)")

    # ==================================================================
    # Phase 4: Reflow + speaker labels → final VTT
    # ==================================================================
    if run_phase(4):
        phase_header(4, "Reflow + Speaker Labels")

        if not os.path.exists(words_json_path):
            log(f"ERROR: Word timestamps not found: {words_json_path}")
            log("Run Phase 3 first.")
            sys.exit(1)
        log(f"  Loading word timestamps from {words_json_path}...")
        with open(words_json_path, "r", encoding="utf-8") as f:
            segments_data = json.load(f)
        total_words = sum(len(s["words"]) for s in segments_data)
        log(f"  {total_words} words in {len(segments_data)} segments")

        log("  Reflowing words into cues (300ms pause threshold)...")
        cues = reflow_words(
            segments_data,
            pause_threshold=0.3,
            max_cue_s=10.0,
            min_cue_s=0.3,
        )
        log(f"  Reflowed into {len(cues)} cues")

        log("  Estimating utterance timestamps for speaker mapping...")
        gemini_timed = estimate_utterance_times(all_utterances, chunk_bounds)
        log(f"  {len(gemini_timed)} timed speaker labels")

        log("  Attaching speaker labels to cues...")
        cues = attach_speakers(cues, gemini_timed)

        speakers_in_output = Counter(
            c.get("speaker", "") for c in cues if c.get("speaker")
        )
        log(f"  Speakers: {dict(speakers_in_output.most_common())}")

        log(f"  Writing VTT: {vtt_path}")
        write_vtt(cues, vtt_path)

        log(f"  Writing JSON: {output_path}")
        output_data = {
            "video": video_path,
            "glossary": glossary_path,
            "gemini_model": gemini_model,
            "whisper_model": whisper_model,
            "n_vad_segments": len(vad_segments),
            "n_gemini_utterances": len(all_utterances),
            "n_cues": len(cues),
            "cues": cues,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        # Summary
        durations = [c["end"] - c["start"] for c in cues]
        if durations:
            avg_dur = sum(durations) / len(durations)
            log(f"\n  Cue stats: {len(cues)} cues, "
                f"avg {avg_dur:.1f}s, min {min(durations):.1f}s, "
                f"max {max(durations):.1f}s")
        log("  Phase 4 done.")
    else:
        log("\nPhase 4: SKIPPED")

    log("\nPipeline complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Silero VAD + Gemini + stable-ts transcription pipeline."
    )
    parser.add_argument("--video", required=True, help="Input video file.")
    parser.add_argument(
        "--output", default="",
        help="Output JSON path. Defaults to episode transcription dir.",
    )
    parser.add_argument("--glossary", default="", help="Glossary TSV for names/terms.")
    parser.add_argument(
        "--model",
        default=os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview"),
        help="Gemini model (default: gemini-3.1-pro-preview).",
    )
    parser.add_argument(
        "--whisper-model", default="large-v3",
        help="Whisper model for stable-ts alignment (default: large-v3).",
    )
    parser.add_argument(
        "--chunk-minutes", type=float, default=10,
        help="Target chunk duration in minutes (default: 10).",
    )
    parser.add_argument(
        "--phase", type=str, default="",
        help="Run only these phases (e.g. '2' or '2,3'). Default: all.",
    )
    parser.add_argument(
        "--work-dir", default="",
        help="Persistent work directory (default: temp dir, cleaned up).",
    )
    args = parser.parse_args()

    video_path = args.video
    if not os.path.exists(video_path):
        print(f"Error: Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    if not args.output:
        stem = Path(video_path).stem
        out_dir = Path(video_path).parent.parent / "transcription"
        args.output = str(out_dir / f"{stem}_pipeline.json")

    only_phases = None
    if args.phase:
        only_phases = {int(x.strip()) for x in args.phase.split(",")}

    run_pipeline(
        video_path=video_path,
        output_path=args.output,
        glossary_path=args.glossary,
        gemini_model=args.model,
        whisper_model=args.whisper_model,
        chunk_minutes=args.chunk_minutes,
        only_phases=only_phases,
        work_dir=args.work_dir or None,
    )


if __name__ == "__main__":
    main()
