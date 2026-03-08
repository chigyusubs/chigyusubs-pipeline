#!/usr/bin/env python3
"""Silero VAD + Gemini + stable-ts transcription pipeline.

Phase 1: Silero VAD → speech/silence boundaries (for chunk splitting)
Phase 2: Gemini transcription (chunked at VAD silence gaps) → plain text with `-- ` turn markers
Phase 3: stable-ts forced alignment (Whisper cross-attention) → word-level timestamps
Phase 4: Reflow → final VTT

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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chigyusubs.audio import extract_audio_chunk, get_duration
from chigyusubs.chunking import find_chunk_boundaries
from chigyusubs.glossary import load_glossary_names
from chigyusubs.metadata import finish_run, metadata_path, start_run, write_metadata
from chigyusubs.paths import infer_episode_dir_from_video
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


from chigyusubs.vtt import write_vtt as _write_vtt_basic


def write_vtt(cues: list[dict], output_path: str):
    _write_vtt_basic(cues, output_path, include_speaker=False)


def _load_chunk_bounds(chunk_json_path: str) -> list[tuple[float, float]]:
    with open(chunk_json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return [(float(c["start_sec"]), float(c["end_sec"])) for c in chunks]


def _load_ocr_context(context_json_path: str) -> tuple[list[str], list[dict]]:
    with open(context_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("episode_memory", []), payload.get("chunk_contexts", [])


def _default_ocr_context_path(episode_dir: Path) -> str:
    preferred = episode_dir / "ocr" / "qwen_ocr_context_gemma.json"
    if preferred.exists():
        return str(preferred)
    fallback = episode_dir / "ocr" / "qwen_ocr_context.json"
    return str(fallback)


def _ocr_terms_for_window(
    ocr_chunk_contexts: list[dict],
    start_sec: float,
    end_sec: float,
    *,
    pad_sec: float = 2.0,
    limit: int = 20,
) -> list[str]:
    scores: dict[str, float] = {}
    for item in ocr_chunk_contexts:
        item_start = float(item["start_sec"])
        item_end = float(item["end_sec"])
        if item_end < start_sec - pad_sec or item_start > end_sec + pad_sec:
            continue
        for idx, term in enumerate(item.get("terms", [])):
            scores[term] = scores.get(term, 0.0) + max(0.5, 8 - idx)
    ranked = [term for _, term in sorted((score, term) for term, score in scores.items())[::-1]]
    return ranked[:limit]


def _rolling_prev_context(all_chunks: list[dict], upto_chunk: int, keep_chunks: int) -> str | None:
    if keep_chunks <= 0 or not all_chunks:
        return None
    start_chunk = max(0, upto_chunk - keep_chunks)
    kept = [c.get("text", "").strip() for c in all_chunks if start_chunk <= c.get("chunk", -1) < upto_chunk]
    kept = [c for c in kept if c]
    if not kept:
        return None
    return "\n".join(kept)


def _strip_turn_markers(text: str) -> str:
    out_lines = []
    for line in text.splitlines():
        t = line.strip()
        if not t:
            continue
        if t.startswith("[画面:") or t.startswith("[画面："):
            continue
        if t.startswith("-- "):
            t = t[3:].lstrip()
        out_lines.append(t)
    return "\n".join(out_lines)


def _speech_lines_from_chunks(chunks: list[dict]) -> list[str]:
    lines: list[str] = []
    for chunk in chunks:
        raw = chunk.get("text", "") or ""
        cleaned = _strip_turn_markers(raw)
        for line in cleaned.splitlines():
            t = line.strip()
            if t:
                lines.append(t)
    return lines


def _aligned_segments_with_offset(result, start_offset_s: float) -> list[dict]:
    segments_data = []
    for seg in result.segments:
        words_data = [{
            "start": w.start + start_offset_s,
            "end": w.end + start_offset_s,
            "word": w.word,
            "probability": getattr(w, "probability", 0.0),
        } for w in seg.words]
        segments_data.append({
            "start": seg.start + start_offset_s,
            "end": seg.end + start_offset_s,
            "text": seg.text,
            "words": words_data,
        })
    return segments_data


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    video_path: str,
    output_path: str,
    glossary_path: str = "",
    gemini_model: str = "gemini-3.1-pro-preview",
    gemini_location: str = "",
    whisper_model: str = "large-v3",
    chunk_minutes: float = 10,
    only_phases: set[int] | None = None,
    work_dir: str | None = None,
    vad_json: str = "",
    chunk_json: str = "",
    ocr_context_json: str = "",
    rolling_context_chunks: int = 2,
):
    run = start_run("transcribe_pipeline")
    run_phase = (lambda p: p in only_phases) if only_phases else (lambda p: True)
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(output_path).stem
    episode_dir = infer_episode_dir_from_video(Path(video_path))

    # Intermediate file paths (persist in output dir for resumability)
    vad_json_path = vad_json or str(episode_dir / "transcription" / "silero_vad_segments.json")
    chunk_json_path = chunk_json or str(episode_dir / "transcription" / "vad_chunks.json")
    ocr_context_path = ocr_context_json or _default_ocr_context_path(episode_dir)
    gemini_json_path = str(out_dir / f"{stem}_gemini_raw.json")
    words_json_path = str(out_dir / f"{stem}_words.json")
    align_diag_json_path = str(out_dir / f"{stem}_alignment_chunks.json")
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

        Path(vad_json_path).parent.mkdir(parents=True, exist_ok=True)
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
    all_chunk_texts: list[dict] = []

    if run_phase(2):
        phase_header(2, "Gemini Transcription")

        if os.path.exists(chunk_json_path):
            log(f"  Loading chunk boundaries: {chunk_json_path}")
            chunk_bounds = _load_chunk_bounds(chunk_json_path)
        else:
            target_chunk_s = chunk_minutes * 60
            log(f"  Finding chunk boundaries (target {chunk_minutes:.0f} min, "
                f"min gap 2.0s)...")
            chunk_bounds = find_chunk_boundaries(
                vad_segments, duration,
                target_chunk_s=target_chunk_s,
                min_gap_s=2.0,
            )
            Path(chunk_json_path).parent.mkdir(parents=True, exist_ok=True)
            with open(chunk_json_path, "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {
                            "chunk_id": idx,
                            "start_sec": cs,
                            "end_sec": ce,
                            "duration_sec": round(ce - cs, 3),
                        }
                        for idx, (cs, ce) in enumerate(chunk_bounds)
                    ],
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        log(f"  {len(chunk_bounds)} chunks:")
        for i, (cs, ce) in enumerate(chunk_bounds):
            log(f"    [{i + 1}] {cs / 60:.1f} - {ce / 60:.1f} min ({ce - cs:.0f}s)")

        # Load glossary
        glossary_entries = []
        if glossary_path and os.path.exists(glossary_path):
            glossary_entries = load_glossary_names(glossary_path)
            log(f"  Glossary: {len(glossary_entries)} entries")

        episode_memory: list[str] = []
        ocr_chunk_contexts: list[dict] = []
        if ocr_context_path and os.path.exists(ocr_context_path):
            episode_memory, ocr_chunk_contexts = _load_ocr_context(ocr_context_path)
            log(f"  OCR context: {len(episode_memory)} episode memory terms, "
                f"{len(ocr_chunk_contexts)} chunk contexts")

        # Resume from partial output if available
        start_chunk = 0
        if os.path.exists(gemini_json_path):
            with open(gemini_json_path, "r", encoding="utf-8") as f:
                all_chunk_texts = json.load(f)
            done_chunks = set(c.get("chunk", 0) for c in all_chunk_texts)
            start_chunk = max(done_chunks) + 1 if done_chunks else 0
            if start_chunk > 0:
                log(f"  Resuming from chunk {start_chunk + 1} "
                    f"({len(all_chunk_texts)} chunk texts from previous run)")

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
                prev_context = _rolling_prev_context(all_chunk_texts, i, rolling_context_chunks)
                ocr_terms = _ocr_terms_for_window(ocr_chunk_contexts, c_start, c_end)
                prompt = build_prompt(
                    glossary_entries,
                    episode_memory=episode_memory,
                    ocr_chunk_terms=ocr_terms,
                    prev_context=prev_context,
                )
                chunk_text = transcribe_chunk(
                    chunk_bytes, prompt, gemini_model, gemini_location or os.environ.get("GOOGLE_CLOUD_LOCATION", "global"), max_retries=20,
                )
                line_count = len([ln for ln in chunk_text.splitlines() if ln.strip()])
                char_count = len(chunk_text)
                log(f"  Result: {line_count} lines, {char_count} chars")

                all_chunk_texts.append({
                    "chunk": i,
                    "chunk_start_s": c_start,
                    "chunk_end_s": c_end,
                    "text": chunk_text,
                })

                # Save after each chunk (resumable)
                log(f"  Saving progress ({len(all_chunk_texts)} total chunk texts)...")
                with open(gemini_json_path, "w", encoding="utf-8") as f:
                    json.dump(all_chunk_texts, f, ensure_ascii=False, indent=2)

        # Summary
        total_lines = sum(len([ln for ln in c.get("text", "").splitlines() if ln.strip()]) for c in all_chunk_texts)
        total_chars = sum(len(c.get("text", "")) for c in all_chunk_texts)
        log(f"\n  Gemini transcript complete: {len(all_chunk_texts)} chunks")
        log(f"  Totals: {total_lines} lines, {total_chars} chars")
        log(f"  Saved: {gemini_json_path}")
        log("  Phase 2 done.")
    else:
        log("\nPhase 2: SKIPPED (using existing Gemini transcript)")

    # Load Gemini transcript (required for phases 3-4)
    if os.path.exists(gemini_json_path):
        with open(gemini_json_path, "r", encoding="utf-8") as f:
            all_chunk_texts = json.load(f)
        if not chunk_bounds:
            if os.path.exists(chunk_json_path):
                chunk_bounds = _load_chunk_bounds(chunk_json_path)
            else:
                chunk_indices = sorted(set(c.get("chunk", 0) for c in all_chunk_texts))
                for ci in chunk_indices:
                    ci_chunks = [c for c in all_chunk_texts if c.get("chunk") == ci]
                    c_start = ci_chunks[0].get("chunk_start_s", 0)
                    chunk_bounds.append((c_start, c_start + chunk_minutes * 60))
        log(f"Gemini transcript: {len(all_chunk_texts)} chunks loaded")
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

        speech_lines = _speech_lines_from_chunks(all_chunk_texts)
        log(f"  {len(speech_lines)} speech utterances to align")

        log(f"  Loading Whisper model: {whisper_model}...")
        import stable_whisper
        model = stable_whisper.load_model(whisper_model, device="cuda")
        log("  Model loaded.")

        chunk_text_map = {int(c.get("chunk", -1)): c for c in all_chunk_texts}
        failed_alignments: list[dict] = []
        alignment_diagnostics: list[dict] = []
        segments_data = []

        with tempfile.TemporaryDirectory() as align_tmpdir:
            for i, (c_start, c_end) in enumerate(chunk_bounds):
                chunk_info = chunk_text_map.get(i)
                chunk_text = chunk_info.get("text", "") if chunk_info else ""
                cleaned_text = _strip_turn_markers(chunk_text)
                if not cleaned_text.strip():
                    log(f"  Chunk {i + 1}/{len(chunk_bounds)}: no spoken text, skipping")
                    continue

                text_len = len(cleaned_text)
                line_count = len([ln for ln in cleaned_text.splitlines() if ln.strip()])
                log(
                    f"  Chunk {i + 1}/{len(chunk_bounds)} align: "
                    f"{c_start:.1f}-{c_end:.1f}s, {line_count} lines, {text_len} chars"
                )

                chunk_audio_path = os.path.join(align_tmpdir, f"align_chunk_{i}.mp3")
                extract_audio_chunk(
                    video_path,
                    chunk_audio_path,
                    start_s=c_start,
                    duration_s=c_end - c_start,
                )

                try:
                    result = model.align(chunk_audio_path, cleaned_text, language="ja")
                    chunk_segments = _aligned_segments_with_offset(result, c_start)
                    segments_data.extend(chunk_segments)
                    chunk_words = sum(len(seg["words"]) for seg in chunk_segments)
                    zero_duration_segments = sum(
                        1 for seg in chunk_segments if float(seg["end"]) <= float(seg["start"])
                    )
                    zero_duration_words = sum(
                        1
                        for seg in chunk_segments
                        for w in seg["words"]
                        if float(w["end"]) <= float(w["start"])
                    )
                    alignment_diagnostics.append({
                        "chunk": i,
                        "chunk_start_s": c_start,
                        "chunk_end_s": c_end,
                        "line_count": line_count,
                        "text_length": text_len,
                        "segments": len(chunk_segments),
                        "words": chunk_words,
                        "zero_duration_segments": zero_duration_segments,
                        "zero_duration_words": zero_duration_words,
                        "needs_review": zero_duration_segments > 5,
                        "status": "ok",
                    })
                except Exception as e:
                    msg = str(e).strip().splitlines()[0] if str(e).strip() else repr(e)
                    failure = {
                        "chunk": i,
                        "chunk_start_s": c_start,
                        "chunk_end_s": c_end,
                        "line_count": line_count,
                        "text_length": text_len,
                        "error": msg,
                        "status": "failed",
                    }
                    failed_alignments.append(failure)
                    alignment_diagnostics.append(failure)
                    log(f"  Chunk {i + 1} alignment failed: {msg}")

        segments_data.sort(key=lambda s: (float(s["start"]), float(s["end"])))
        total_words = sum(len(s["words"]) for s in segments_data)
        log(f"  Aligned: {total_words} words in {len(segments_data)} segments")
        if failed_alignments:
            log(f"  Failed chunk alignments: {len(failed_alignments)}")
            for item in failed_alignments[:10]:
                log(
                    f"    chunk {item['chunk']} "
                    f"({item['chunk_start_s']:.1f}-{item['chunk_end_s']:.1f}s): {item['error']}"
                )

        log(f"  Writing {words_json_path}")
        with open(words_json_path, "w", encoding="utf-8") as f:
            json.dump(segments_data, f, ensure_ascii=False, indent=2)
        log(f"  Writing {align_diag_json_path}")
        with open(align_diag_json_path, "w", encoding="utf-8") as f:
            json.dump(alignment_diagnostics, f, ensure_ascii=False, indent=2)

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
    # Phase 4: Reflow → final VTT
    # ==================================================================
    if run_phase(4):
        phase_header(4, "Reflow")

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

        log(f"  Writing VTT: {vtt_path}")
        write_vtt(cues, vtt_path)

        log(f"  Writing JSON: {output_path}")
        output_data = {
            "video": video_path,
            "glossary": glossary_path,
            "gemini_model": gemini_model,
            "gemini_location": gemini_location or os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
            "whisper_model": whisper_model,
            "n_vad_segments": len(vad_segments),
            "n_gemini_chunks": len(all_chunk_texts),
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

    word_segments = 0
    if os.path.exists(words_json_path):
        with open(words_json_path, "r", encoding="utf-8") as f:
            word_segments = len(json.load(f))

    metadata = finish_run(
        run,
        episode_dir=str(infer_episode_dir_from_video(Path(video_path))),
        inputs={
            "video": video_path,
            "glossary": glossary_path or None,
            "ocr_context_json": ocr_context_path if os.path.exists(ocr_context_path) else None,
        },
        outputs={
            "pipeline_json": output_path,
            "vad_json": vad_json_path,
            "chunk_json": chunk_json_path,
            "gemini_json": gemini_json_path,
            "words_json": words_json_path,
            "alignment_chunks_json": align_diag_json_path if os.path.exists(align_diag_json_path) else None,
            "vtt": vtt_path,
        },
        settings={
            "gemini_model": gemini_model,
            "whisper_model": whisper_model,
            "chunk_minutes": chunk_minutes,
            "only_phases": sorted(only_phases) if only_phases else None,
            "work_dir": work_dir,
            "rolling_context_chunks": rolling_context_chunks,
        },
        stats={
            "duration_seconds": round(duration, 3),
            "vad_segments": len(vad_segments),
            "chunks": len(chunk_bounds),
            "gemini_chunks": len(all_chunk_texts),
            "word_segments": word_segments,
            "failed_alignment_chunks": len(failed_alignments) if run_phase(3) and 'failed_alignments' in locals() else None,
            "cues_written": len(cues) if run_phase(4) and 'cues' in locals() else None,
        },
    )
    write_metadata(output_path, metadata)
    log(f"Metadata written: {metadata_path(output_path)}")
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
        "--location",
        default=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
        help="Vertex location for Gemini requests (default: GOOGLE_CLOUD_LOCATION or global).",
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
    parser.add_argument("--vad-json", default="", help="Use/save reusable VAD JSON (defaults to episode transcription/silero_vad_segments.json).")
    parser.add_argument("--chunk-json", default="", help="Use/save reusable chunk-boundary JSON (defaults to episode transcription/vad_chunks.json).")
    parser.add_argument("--ocr-context-json", default="", help="Chunk-level OCR context JSON (defaults to episode ocr/qwen_ocr_context_gemma.json if present, otherwise qwen_ocr_context.json).")
    parser.add_argument("--rolling-context-chunks", type=int, default=2, help="How many previous chunks of transcript history to include for continuity.")
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
        gemini_location=args.location,
        whisper_model=args.whisper_model,
        chunk_minutes=args.chunk_minutes,
        only_phases=only_phases,
        work_dir=args.work_dir or None,
        vad_json=args.vad_json,
        chunk_json=args.chunk_json,
        ocr_context_json=args.ocr_context_json,
        rolling_context_chunks=args.rolling_context_chunks,
    )


if __name__ == "__main__":
    main()
