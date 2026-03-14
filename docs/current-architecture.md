# Current Architecture

This document describes the maintained pipeline shape in this repo today.

For validated findings and tradeoffs from real runs, see:

- `docs/lessons-learned.md`
- `docs/gemini-transcription-playbook.md`

For older OCR-heavy architecture notes, see:

- `docs/archive/ocr-first-architecture.md`

## Maintained Default

The maintained practical path is:

```text
source video
  -> Silero VAD
  -> full-coverage chunk boundaries guided by VAD silence gaps
  -> Gemini transcription
  -> optional chunkwise Flash Lite OCR sidecar
  -> CTC forced alignment
  -> faster-whisper second opinion
  -> reflow
  -> optional Codex repair
  -> Codex or API translation
```

The main current transcription default is:

- `gemini-3-flash-preview`
- `video`
- `spoken_only`
- `thinking=low`
- `media_resolution=high`
- `temperature=0.0`
- local chunk encoding keeps source resolution by default at `1 FPS`

Named preset:

- `flash25_free_default` for `gemini-2.5-flash` production runs with `spoken_only`, `media_resolution=high`, and no thinking override
- `flash_free_default`
- `flashlite_debug_transcript` for cheap bounded debug passes

Important chunking rule:

- VAD is used to place chunk boundaries at good silence points
- Gemini still receives continuous end-to-end video coverage across chunks
- silence is not dropped from the maintained video transcription path
- chunk plans should treat `target_chunk_s + 30s` as the default hard maximum

## Design Principles

- Save reusable artifacts at each stage.
- Keep the main path chunkwise and resumable.
- Treat OCR as optional evidence, not a mandatory front door.
- Keep speech alignment speech-only.
- Use local models as verification/backstop artifacts, not hidden replacement logic.

## Canonical Artifact Layout

Canonical artifacts live under `samples/episodes/<slug>/`:

- `source/`
- `ocr/`
- `glossary/`
- `transcription/`
- `translation/`
- `logs/`

Downstream scripts should keep discovering artifacts at those stable paths.

Within those stable folders, there are two naming regimes:

- published outputs under `source/` keep stable source-video-basename names
- lineage artifacts under `transcription/` and draft `translation/` may use short run-ID filenames plus `preferred.json` manifests

Chunk-plan filenames should also stay legible because operators routinely swap them during retries:

- `vad_chunks.json` means the default full-coverage VAD plan
- `vad_chunks_semantic_<target>.json` means a reviewed semantic plan at that target size
- `*_repair*.json` means a follow-up repair plan that resplits only a failed region
- `probes/*exact_chunks_<target>s*.json` means a debug-only exact-duration probe plan

## Run Ledger

Metadata-emitting steps also mirror a run-oriented audit trail under:

- `logs/runs/<run_id>/run.json`
- `logs/runs/<run_id>/artifacts/*.meta.json`
- `logs/runs/<run_id>/README.md`

This gives two views of the same work:

- canonical artifact view for the maintained pipeline
- run ledger view for debugging, auditing, and comparison

Lineage naming is separate from the ledger naming:

- `logs/runs/<ledger_run_id>/...` keeps the readable timestamp-based audit path
- lineage filenames use a short deterministic alias such as `r3a7f01b_ctc_words.json`

## Manual Experiment Packs

Manual site-driven experiment packs live separately under:

- `samples/experiments/<pack>/`

They are intentionally non-canonical and are used for:

- fixed-clip benchmarking
- AI Studio prompt tests
- paste-back result comparison

The helper that builds them is experimental:

- `scripts/experiments/prepare_ai_studio_experiment_pack.py`

## Maintained Artifact Chain

### 1. VAD

Produced by:

- `scripts/run_vad_episode.py`

Main output:

- `transcription/silero_vad_segments.json`

Purpose:

- reusable speech/silence segmentation
- shared chunking input for both Gemini and local ASR paths

### 2. Chunk Boundaries

Produced by:

- `scripts/build_vad_chunks.py`

Main output:

- `transcription/vad_chunks.json`

Purpose:

- reusable chunk plan for transcription
- stable retry/resume boundary
- `target + 30s` hard max with short-gap fallback before mid-speech forced splits

### 3. Gemini Raw Transcript

Produced by:

- `scripts/transcribe_pipeline.py`
- `scripts/transcribe_gemini_video.py`
- `scripts/transcribe_gemini.py`

Maintained default behavior:

- spoken transcript first
- chunkwise
- resumable
- visual cues optional depending on prompt mode

OCR is not required here.

### 3b. Optional Chunkwise OCR Sidecar

Produced by:

- `scripts/extract_gemini_chunk_ocr.py`

Main output:

- `ocr/*_flash_lite_chunk_ocr.json`

Purpose:

- extract meaningful visible text as a reusable side artifact
- keep OCR separate from the main spoken transcript call
- store chunk-scoped items with a tiny classifier:
  - `title_card`
  - `name_card`
  - `info_card`
  - `label`
  - `other`

Important:

- timings are chunk-scoped only (`timing_basis=chunk_span`)
- this is for review/glossary/translation support
- it is not automatically injected back into transcription by default
- Codex glossary/translation helpers may auto-discover it later as supporting context

Named preset:

- `flashlite_ocr_sidecar`

### 4. Alignment

Produced by:

- `scripts/align_ctc.py`

Main output:

- `transcription/<run_id>_ctc_words.json`

Purpose:

- speech-only forced alignment
- preserve turn boundaries as metadata
- keep visual-only text out of the alignment surface
- update `transcription/preferred.json`

### 5. Second Opinion

Produced by:

- `scripts/pre_reflow_second_opinion.py`

Purpose:

- compare Gemini output against `faster-whisper large-v3`
- catch likely spoken omissions or suspicious regions before translation

### 6. Reflow

Produced by:

- `scripts/reflow_words.py`

Purpose:

- turn aligned words/lines into translation-ready Japanese subtitle cues
- write lineage VTTs with short run-ID names and VTT NOTE provenance

Optional repair step:

- `scripts/repair_vtt_codex.py`

### 7. Translation

Maintained interactive path:

- `scripts/translate_vtt_codex.py`

Draft English lineage artifacts in `translation/` follow the same short run-ID pattern and update `translation/preferred.json`.

Alternative unattended path:

- `scripts/translate_vtt_api.py`

Final publish step:

- `scripts/publish_vtt.py`

Published final output:

- `source/<video_stem>.vtt`
- stable basename matching the source video
- provenance copied into `source/<video_stem>.vtt.meta.json`

## OCR In The Current Architecture

OCR is no longer the required first stage of the maintained pipeline.

Current role of OCR-like artifacts:

- optional context
- glossary support
- visual prompt/rule extraction
- review support on dense text-heavy chunks

Most promising current OCR-like use:

- cheap `flash-lite` `ocr_only` side artifacts on dense visual chunks

Not the current default:

- OCR-first transcription
- mandatory OCR filtering before Gemini
- heavy OCR taxonomy as a prerequisite for transcription

## Local Path

The fully local fallback path still exists:

- `transcribe_local.py`
- local faster-whisper
- same VAD/chunk artifact chain where possible

This is useful for zero-API operation and for second-opinion work, but it is not the primary quality path.

## Current Weak Spots

Open problems in the maintained path:

- transcript quality still varies by scene type
- OCR-assisted prompting is still unresolved as a default strategy
- some Gemini settings are model- and modality-sensitive
- dense visual narration can still interfere with spoken coverage

Those are quality-layer issues, not reasons to go back to the older OCR-first architecture.
