# Current Architecture

This document describes the pipeline architecture currently being built in this repo.

For what has actually been validated so far, and what still remains unresolved, see:

- `docs/lessons-learned.md`

The main goal is:

- take a raw episode
- extract reusable OCR and VAD artifacts once
- derive transcription-safe context from those artifacts
- support both Gemini and local transcription paths without coupling everything to one chunking scheme

## Design Principles

- Raw extraction artifacts stay lossless.
- Derived artifacts are resumable and inspectable.
- OCR is treated as a separate evidence layer from audio.
- VAD chunking is reusable and not buried inside one transcription script.
- Episode-wide memory and chunk-local context are separate concepts.
- Chunk-level transcription can use bounded rolling history, but the source of truth remains chunkwise artifacts.

## Artifact Layers

The pipeline is now organized around stable artifacts under `samples/episodes/<slug>/...`.

### 1. Raw OCR

Produced by:

- `scripts/run_qwen_ocr_episode.py ocr`

Main output:

- `ocr/qwen_ocr_results.jsonl`

Purpose:

- frame-level OCR record
- lossless enough to rerun downstream filters without rerunning OCR

Notes:

- no-text frames use `[[NO_TEXT]]` upstream and become `lines: []`
- OCR is recall-oriented and may still include dense-board noise

### 2. OCR Spans

Produced by:

- `scripts/run_qwen_ocr_episode.py spans`

Main output:

- `ocr/qwen_ocr_spans.json`

Purpose:

- merge neighboring OCR frames into time-local spans
- isolate dense boards/lists from ordinary name cards and title cards
- provide a stable OCR representation independent of later audio chunking

Each span includes:

- `class`
- `stats`
- `representative_lines`
- heuristic `prompt_safe_terms`
- `term_counts`

### 3. OCR Context

Produced by:

- `scripts/run_qwen_ocr_episode.py context`
- `scripts/classify_ocr_spans_local.py`

Main outputs:

- heuristic context: `ocr/qwen_ocr_context.json`
- local Gemma-cleaned context: `ocr/qwen_ocr_context_gemma.json`

Purpose:

- derive transcription-safe OCR hints from spans
- separate:
  - `episode_memory`
  - chunk-local OCR terms

Current intent:

- `episode_memory` should be strict and recurring
- chunk-local OCR terms can be broader, but should still avoid poisoning transcription

### 4. Raw VAD

Produced by:

- `scripts/run_vad_episode.py`

Main output:

- `transcription/silero_vad_segments.json`

Purpose:

- reusable speech/silence segmentation
- shared by Gemini and local transcription paths

This is separate from transcription so VAD does not need to be recomputed for every ASR experiment.

### 5. VAD Chunk Boundaries

Produced by:

- `scripts/build_vad_chunks.py`

Main output:

- `transcription/vad_chunks.json`

Purpose:

- reusable transcription chunks derived from VAD silence gaps
- stable input for chunkwise transcription

Current default target:

- `240s` chunks
- `2.0s` minimum silence gap

Reason:

- short enough for retries and bounded rolling context
- long enough to preserve comedy bit continuity

### 6. Chunkwise Transcription

Main maintained path:

- `scripts/transcribe_pipeline.py`

The maintained Gemini pipeline now prefers these inputs if present:

- `transcription/silero_vad_segments.json`
- `transcription/vad_chunks.json`
- `ocr/qwen_ocr_context.json` or `ocr/qwen_ocr_context_gemma.json`

The prompt logic now uses:

- glossary TSV if available
- `episode_memory`
- OCR terms joined to each transcription chunk by time overlap
- bounded rolling transcript history from recent chunks

Important:

- rolling history is an aid, not the source of truth
- the system is still chunkwise and resumable
- this is not one long conversation thread for the whole episode

## Current End-to-End Shape

Recommended artifact flow:

```text
source video
  -> fixed-rate frames
  -> Qwen OCR JSONL
  -> OCR spans
  -> OCR context (heuristic or Gemma-cleaned)

source video
  -> Silero VAD segments
  -> VAD chunk boundaries

VAD chunks + OCR context + episode memory + bounded rolling transcript history
  -> Gemini or local transcription
  -> alignment
  -> reflow
  -> optional Codex-interactive cue repair
  -> subtitle output
  -> CPS-aware English subtitle editing/translation
```

## Why This Is Better Than The Older Flow

Older repo shape:

- OCR candidates were flattened quickly into one broad glossary
- VAD and chunking were often recomputed inside transcription scripts
- OCR filtering was tied too closely to one transcription path

Current shape:

- OCR stays reusable as spans
- VAD stays reusable as saved segments and chunk boundaries
- OCR context is attached to chunks by time overlap
- transcription can change without invalidating OCR preprocessing

## Current Weak Spots

The architecture is in better shape than the quality layer.

Current weak spots are:

- heuristic OCR context is still too noisy on dense boards
- local Gemma context is better, but still sometimes over-filters or keeps weak local terms
- dense-board spans likely need a stricter or more structured selection strategy
- `classify_ocr_spans_local.py` was initially batch-only and now checkpoints, but its output schema may still need a split between anchor terms and auxiliary local terms

## Current Scripts By Role

Episode setup:

- `scripts/init_episode_from_media.py`

OCR extraction and derivation:

- `scripts/start_qwen_ocr_server.sh`
- `scripts/run_qwen_ocr_episode.py ocr`
- `scripts/run_qwen_ocr_episode.py spans`
- `scripts/run_qwen_ocr_episode.py context`
- `scripts/start_gemma_ocr_filter_server.sh`
- `scripts/classify_ocr_spans_local.py`

Reusable audio segmentation:

- `scripts/run_vad_episode.py`
- `scripts/build_vad_chunks.py`

Transcription and alignment:

- `scripts/transcribe_pipeline.py`
- `scripts/transcribe_gemini.py`
- `scripts/transcribe_local.py`
- `scripts/align_ctc.py` — CTC forced alignment using `NTQAI/wav2vec2-large-japanese` (recommended); writes per-chunk alignment diagnostics including interpolated all-unaligned lines
- `scripts/align_chunkwise.py` — chunked stable-ts alignment (legacy)
- `scripts/align_qwen_forced.py` — Qwen forced alignment benchmark (archived)
- `scripts/align_qwen_forced_hf.py` — Qwen HF forced alignment benchmark (archived)
- `scripts/align_stable_ts.py` — global stable-ts alignment (legacy)
- `scripts/repair_vtt_codex.py` — Codex-interactive region-based reflow repair helper with session/checkpoint state, deterministic validation/final assembly, and advisory surfacing of alignment-stage interpolated-line diagnostics
- `scripts/repair_vtt_local.py` — local Gemma cue-boundary repair on top of a reflowed VTT + aligned words (alternative path, no longer the default skill fallback)
- `scripts/translate_vtt.py`
- `scripts/translate_vtt_codex.py` — Codex-interactive one-episode translation helper with session/checkpoint, deterministic batch diagnostics, source-cue signature validation during `apply-batch`, alignment-warning carry-through from CTC diagnostics, clean restart via `prepare --force`, partial VTT assembly, and automatic batch-tier fallback
- `scripts/translate_vtt_mistral.py` — experimental translation benchmark using the same batch/checkpoint flow against the Mistral API

## Recommended Operational Default

Current recommended default architecture is:

```text
Qwen OCR
  -> OCR spans
  -> local Gemma OCR cleanup/classification
  -> saved OCR context

Silero VAD
  -> saved VAD segments
  -> saved VAD chunk boundaries

Gemini transcription
  using:
  - chunk boundaries
  - episode memory
  - chunk-local OCR context
  - bounded rolling transcript history

  -> CTC forced alignment (wav2vec2-ja)
  -> reflow
  -> optional Codex-interactive cue repair
  -> batch-based CPS-aware English subtitle editing
```

This keeps the quality path strong while preserving a reusable local artifact chain.

There is now a second maintained translation mode for cases where the user wants Codex itself to do the translation rather than an API-backed model:

```text
reflowed VTT
  -> translate_vtt_codex.py prepare
  -> Codex-interactive batch translation
  -> translate_vtt_codex.py apply-batch
  -> partial English VTT in translation/
  -> finalize to full English VTT
```

This mode is one episode at a time, stores preferences like `preferred_model=gpt-5.4` as metadata only, and uses an automatic batch-tier policy of `84 -> 60 -> 48`.

When reusing an older English draft in this mode, the helper must treat cue count
and cue timeline as the safety boundary. Draft seeding is only valid when the
candidate English file matches the current Japanese source cue-for-cue on timing;
cue-index-only reuse is unsafe once reflow changes.

There is now a matching Codex-interactive repair mode for weak-but-structurally-valid Japanese reflowed VTTs:

```text
reflowed VTT
  -> repair_vtt_codex.py prepare
  -> Codex-interactive region repair
  -> repair_vtt_codex.py apply-region
  -> partial repaired VTT in transcription/
  -> finalize to repaired Japanese VTT
```

This mode is region-based, preserves source text coverage inside each repaired region, and rebuilds timings deterministically inside the original region span.
Its diagnostics now include deterministic before/after cue metrics, sampled flagged regions, advisory alignment-stage interpolation warnings, and one recommended Japanese VTT handoff path for translation.

One important boundary remains: if Gemini never transcribed a spoken section, CTC cannot recover it later. In practice this shows up most often in narrated premise/rules VO that partially overlaps with explanatory telops. The operational workaround is local and artifact-preserving: extract the suspicious clip, transcribe that clip with `faster-whisper large-v3` as a saved second-opinion artifact, patch the affected Japanese cues, and only then rerun downstream reflow/translation for that region or episode.
