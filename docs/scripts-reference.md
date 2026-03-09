# Pipeline Scripts Reference

This file documents every script under `scripts/`.

For the current artifact-level architecture, see `docs/current-architecture.md`.
For validated findings and remaining issues from real episode runs, see `docs/lessons-learned.md`.

## Pipelines

There are two transcription paths. Both now share the same reusable OCR and VAD artifact stages.

### Path A: Gemini (recommended)

Best quality. Requires Vertex AI access. Local LLM handles OCR filtering.

```text
Qwen OCR -> OCR spans -> OCR context
Silero VAD -> VAD chunk boundaries
  -> Gemini transcription
  -> CTC forced alignment (wav2vec2-ja)
  -> reflow
  -> optional cue repair
  -> translation
```

1. `run_qwen_ocr_episode.py ocr` — frame OCR
2. `run_qwen_ocr_episode.py spans` — OCR span building
3. `run_qwen_ocr_episode.py context` or `classify_ocr_spans_local.py` — OCR context derivation
4. `run_vad_episode.py` — reusable Silero VAD
5. `build_vad_chunks.py` — reusable chunk boundaries
6. `transcribe_pipeline.py` — Gemini transcription + alignment + reflow
7. `repair_vtt_codex.py` — optional Codex-interactive cue-boundary repair on reflowed VTT
8. `translate_vtt.py` — LLM translation to English
9. `translate_vtt_mistral.py` — experimental Mistral translation benchmark

`transcribe_pipeline.py` can now consume the saved VAD/chunk/OCR artifacts instead of recomputing them.

### Path B: Local (faster-whisper)

Fully local, no API calls. Lower quality but zero cost.

1. `run_qwen_ocr_episode.py ocr` + `spans` + `context`
2. `run_vad_episode.py` + `build_vad_chunks.py`
3. `transcribe_local.py` — local ASR path
4. `translate_vtt.py`
5. `translate_vtt_mistral.py` (experimental benchmark)

## Episode Layout

```text
samples/episodes/<episode_slug>/
  source/       # video files
  frames/       # extracted frames for OCR
  ocr/          # qwen_ocr_results.jsonl
  glossary/     # structured glossary, hotwords, translation TSV
  transcription/ # VTT, word JSON, chunks JSON
  translation/  # translated VTT
  logs/
```

## Script Catalog

### Core Pipeline

| Script | Purpose | Status |
|---|---|---|
| `episode_paths.py` | Shared path inference helpers for episode-aware defaults | Maintained |
| `transcribe_pipeline.py` | Integrated 4-phase pipeline: Silero VAD -> Gemini -> chunked stable-ts -> reflow | Maintained |
| `transcribe_gemini_raw.py` | Gemini raw text transcription with chunk-wise local OCR filtering | Maintained |
| `transcribe_gemini.py` | Gemini transcription (JSON schema mode, used by pipeline) | Maintained |
| `transcribe_local.py` | Fully local pipeline: Silero VAD + local OCR filter + faster-whisper | Maintained |
| `align_ctc.py` | CTC forced alignment using `NTQAI/wav2vec2-large-japanese` + `torchaudio.functional.forced_align`. 0.3% zero-duration words vs 13.4% with stable-ts on `dmm`. Runs on system python3.12 with ROCm GPU for wav2vec2 inference, keeps `forced_align` on CPU, and caps PyTorch CPU threads to 24 to avoid runaway thread fanout. Repairs fully unaligned lines into small local fallback slots, enforces monotonic segment timing so short answers do not jump to chunk start and disappear in reflow, preserves Gemini turn-boundary metadata per aligned line, and writes an `.diagnostics.json` sidecar with per-chunk repaired-unaligned counts, line-level details, and advisory `possible_visual_narration_substitution` flags when dense `[画面: ...]` runs may be standing in for spoken narration. | Maintained |
| `pre_reflow_second_opinion.py` | Always-on pre-reflow helper that runs a faster-whisper second-opinion pass on every episode plus `compare_transcript_coverage.py` coverage diff. Auto-discovers episode glossary for Whisper initial_prompt conditioning (glossary.json source terms or legacy whisper_prompt_condensed.txt) and passes VAD segments to the coverage comparison when available. Visual-substitution risk from alignment diagnostics is collected as informational metadata. Use `--force` to rerun whisper even when artifacts exist. Writes a reusable summary JSON under `transcription/diagnostics/`. | Maintained |
| `align_chunkwise.py` | Chunk-wise stable-ts forced alignment from `_chunks.json` | Legacy |
| `align_qwen_forced.py` | Chunk-wise Qwen forced-alignment benchmark via `py-qwen3-asr-cpp` GGUF backend | Archived |
| `align_qwen_forced_hf.py` | Chunk-wise Qwen forced-alignment benchmark via official `qwen-asr` on system Python / ROCm | Archived |
| `align_stable_ts.py` | Global stable-ts alignment (single pass) | Legacy |
| `reflow_words.py` | Reflow word/line timestamps into subtitle cues. `--line-level` (default for CTC) treats lines as atomic, preventing mid-word splits. Includes comma-fallback splitting, repeated-line-safe word-timestamp lookup, sparse-cue clamping for CTC artifacts, and display-line rewrapping during residual micro-cue merges so sub-`0.5s` cues do not survive just because the raw merged line count is temporarily >2. | Maintained |
| `repair_vtt_codex.py` | Codex-interactive reflow review/repair helper. Prepares flagged cue regions, checkpoints progress, validates region repairs, regenerates a partial repaired VTT, and finalizes a repaired Japanese VTT with deterministic before/after diagnostics and one recommended translation-input path. If a sibling CTC alignment diagnostics sidecar exists, it is loaded automatically and surfaced as advisory alignment-risk context in session status, diagnostics, and region payloads, attaching repaired lines by cue overlap first and nearest-cue fallback second. If the aligned words JSON contains Gemini turn metadata, prepare also surfaces cue-level turn-boundary context so merged multi-turn cues remain visible during review. Cues under `0.5s` are treated as structural blockers; otherwise short/tiny cue counts stay advisory and repair triggering is driven by artifact-like boundary risks. | Maintained |
| `repair_vtt_local.py` | Repair an existing reflowed VTT using aligned words + local Gemma as a constrained merge/split/extend chooser. Optional legacy/benchmark alternative to the default Codex-interactive repair path. | Alternative |
| `translate_vtt.py` | LLM translation of Japanese VTT to English (Vertex or local) | Maintained |
| `translate_vtt_codex.py` | Codex-interactive translation helper with session/checkpoint state, deterministic batch diagnostics, source-cue signature validation during `apply-batch`, clean restart via `prepare --force`, optional `--seed-from` draft reuse guarded by exact cue-count and cue-timeline matching, optional `--alignment-diagnostics` override with best-effort auto-discovery from the input VTT, partial VTT assembly, automatic `84 -> 60 -> 48` batch-tier fallback, and preflight stop on structural timing blockers including cues under `0.5s`. Interpolated alignment warnings remain advisory and are carried into batch payloads and diagnostics instead of stopping translation, using nearest-cue fallback when a repaired line sits in a reflow gap. When aligned words JSON includes Gemini turn metadata, `next-batch` also carries advisory turn-boundary context so translators can see which cues span multiple source turns without polluting subtitle text with visible markers. | Maintained |
| `translate_vtt_mistral.py` | Mistral API translation benchmark. Keeps the same batch/checkpoint/diagnostics flow as `translate_vtt.py`, but targets Mistral chat completions directly. | Experimental |
| `init_episode_from_media.py` | Create episode workspace from media, optionally extracting fixed-rate frames | Maintained |

### OCR & Glossary

| Script | Purpose | Status |
|---|---|---|
| `run_qwen_ocr_episode.py` | Qwen-VL OCR over frames + frequency/run-length glossary filter, with per-run metadata sidecars | Maintained |
| `classify_ocr_spans_local.py` | Local Gemma OCR-span cleanup/classification into reusable chunk context | Maintained |
| `condense_glossary_vertex.py` | Build structured glossary with Gemini on Vertex | Maintained |
| `condense_glossary_qwen.py` | Build structured glossary with local Qwen | Maintained |
| `condense_glossary_llm.py` | Older local LLM condensation via OpenAI-compatible endpoint | Legacy |
| `clean_candidates.py` | Clean/deduplicate OCR candidate text | Utility |

### ASR Backends

| Script | Purpose | Status |
|---|---|---|
| `run_faster_whisper.py` | faster-whisper ASR (ROCm/CUDA), initial prompt + hotwords | Maintained |
| `compare_transcript_coverage.py` | Compare a primary aligned words JSON against a secondary ASR words JSON (typically Gemini+CTC vs faster-whisper) and emit time-local coverage-gap regions where the second opinion contains substantially more speech than the primary transcript. Optional `--vad-json` cross-references flagged regions against VAD speech segments, marking each as `vad_confirmed` or `possible_hallucination`. Intended as a deterministic pre-reflow diagnostic. | Maintained |
| `run_whisper_cpp.py` | whisper.cpp ASR via `whisper-cli` | Maintained |
| `run_local_whisper.py` | OpenAI Whisper Python package ASR | Maintained |
| `start_qwen_ocr_server.sh` | Start llama-server for Qwen OCR with recommended deterministic settings | Maintained |
| `start_gemma_ocr_filter_server.sh` | Start llama-server for local Gemma OCR cleanup/classification | Maintained |
| `start_gemma_cue_repair_server.sh` | Start llama-server for local Gemma cue-boundary repair decisions. Not part of the default Codex skill path. | Alternative |
| `start_qwen_cue_repair_server.sh` | Start llama-server for Qwen3.5-35B-A3B cue-boundary repair decisions. Not part of the default Codex skill path. | Alternative |
| `run_vad_episode.py` | Standalone reusable Silero VAD artifact builder | Maintained |
| `build_vad_chunks.py` | Build reusable chunk boundaries from saved VAD | Maintained |

### Subtitle Post-processing

| Script | Purpose | Status |
|---|---|---|
| `reflow_subtitles_vertex.py` | LLM-based cue regrouping using Vertex Gemini | Experimental |
| `reflow_subtitles_local.py` | LLM-based cue regrouping using local llama.cpp | Experimental |
| `format_vtt_netflix.py` | Netflix-style VTT formatting using Gemini | Utility |
| `json_to_smart_vtt.py` | Word-timestamp JSON to pause-split VTT (no LLM) | Utility |
| `restore_speaker_turns.py` | Re-insert speaker turn markers into reflowed VTT | Utility |
| `restore_speaker_turns_ocr.py` | Speaker turn restoration using OCR data | Experimental |

### Vertex / LLM Utilities

| Script | Purpose | Status |
|---|---|---|
| `run_vertex.py` | Generic CLI wrapper for Vertex Gemini text generation | Utility |

### Experiment / Test Scripts

| Script | Purpose | Status |
|---|---|---|
| `run_reazonspeech_nemo.py` | ReazonSpeech NeMo ASR (broken on ROCm, see docs) | Archived |
| `run_reazonspeech_nemo_chunked.py` | Chunked ReazonSpeech NeMo (broken on ROCm) | Archived |
| `transcribe_nemo_vibevoice.py` | VibeVoice ASR experiment | Archived |
| `test_vibevoice_4bit.py` | VibeVoice 4-bit inference test | Archived |
| `test_vibevoice_load.py` | VibeVoice model loading test | Archived |
| `build_whisper_glossary.py` | Older EasyOCR-based glossary builder | Legacy |
| `extract_glossary.py` | Older MangaOCR-based glossary extraction | Legacy |
| `test_easyocr.py` | EasyOCR sanity test | Test |
| `test_paddleocr.py` | PaddleOCR sanity test | Test |
| `test_faster_whisper.py` | faster-whisper runtime check | Test |
| `test_qwen_ocr.py` | Qwen-VL OCR spot-check | Test |
| `test_qwen_json_schema.py` | JSON reliability mode comparison | Test |
| `test_vlm_extraction.py` | VLM extraction prompt comparison | Test |
| `test_reflow_quality.py` | Reflow quality evaluation | Test |

### `scripts/kotoba_test/`

Older Kotoba-whisper and NeMo diarization experiments.

| Script | Purpose | Status |
|---|---|---|
| `run_kotoba.py` | Kotoba-whisper transcription | Legacy |
| `run_kotoba_clean.py` | Cleaned Kotoba pipeline | Legacy |
| `run_kotoba_diarized.py` | Kotoba + diarization | Legacy |
| `run_nemo_diarized.py` | NeMo MSDD diarization + Kotoba merge | Legacy |

## CLI Cheatsheet

### Gemini Pipeline (recommended)

```bash
# 1. OCR
python scripts/run_qwen_ocr_episode.py ocr \
  --episode-dir samples/episodes/<slug> \
  --url http://127.0.0.1:8787 \
  --model qwen3.5-9b \
  --server-settings '{"quant":"Q6_K","ctx_size":8192,"seed":3407,"temp":0,"top_p":0.9,"top_k":20,"thinking":false}'

# 2. Filter OCR into glossary candidates
python scripts/run_qwen_ocr_episode.py filter \
  --episode-dir samples/episodes/<slug>

# `ocr`, `filter`, and the maintained transcription/alignment scripts emit
# `*.meta.json` sidecars recording invocation details, settings, and run timing.

# 3. Condense glossary
python scripts/condense_glossary_vertex.py \
  --input samples/episodes/<slug>/glossary/qwen_candidates.txt

# 4. Transcribe with Gemini + local OCR filter
python scripts/transcribe_gemini_raw.py \
  --video samples/episodes/<slug>/source/video.mp4 \
  --output samples/episodes/<slug>/transcription/raw.txt \
  --glossary samples/episodes/<slug>/glossary/translation_glossary.tsv \
  --ocr-jsonl samples/episodes/<slug>/ocr/qwen_ocr_results.jsonl \
  --ocr-filter-url http://127.0.0.1:8080 \
  --ocr-filter-model gemma3-27b

# 5. CTC forced alignment (recommended)
python3.12 scripts/align_ctc.py \
  --video samples/episodes/<slug>/source/video.mp4 \
  --chunks samples/episodes/<slug>/transcription/<stem>_gemini_raw.json \
  --output-words samples/episodes/<slug>/transcription/<stem>_ctc_words.json

# Uses NTQAI/wav2vec2-large-japanese + torchaudio CTC forced alignment.
# Runs on system python3.12 with ROCm GPU.
# 0.3% zero-duration words vs 13.4% with stable-ts on dmm.
# Preserves Gemini turn-boundary metadata per aligned line in the words JSON.
# Note: this only aligns text that already exists in the Gemini transcript.
# The alignment diagnostics sidecar now also flags chunks where visual-only
# `[画面: ...]` runs look narration-like and may have displaced spoken lines.
# When those flags appear, run a local faster-whisper second-opinion pass
# before reflow and compare the two transcripts.

# 6. Reflow (line-level, recommended for CTC output)
PYTHONPATH=. python3 scripts/reflow_words.py \
  --input samples/episodes/<slug>/transcription/<stem>_ctc_words.json \
  --output samples/episodes/<slug>/transcription/<stem>_reflow.vtt \
  --line-level --stats

# 7. Optional Codex-interactive repair pass between reflow and translation:
python3 scripts/repair_vtt_codex.py prepare \
  --input samples/episodes/<slug>/transcription/<stem>_reflow.vtt \
  --words samples/episodes/<slug>/transcription/<stem>_ctc_words.json \
  --output samples/episodes/<slug>/transcription/<stem>_reflow_repaired.vtt

# If <stem>_ctc_words.json.diagnostics.json exists, prepare auto-loads it and
# surfaces interpolated alignment lines as advisory review context.
# If <stem>_ctc_words.json contains turn metadata, prepare also surfaces
# cue-level turn-boundary context for merged multi-turn cues.

# Always-on pre-reflow second-opinion diagnostic:
python3.12 scripts/pre_reflow_second_opinion.py \
  --words samples/episodes/<slug>/transcription/<stem>_ctc_words.json

# Runs faster-whisper on every episode and compares coverage against the primary
# transcript. Auto-discovers episode glossary for Whisper initial_prompt and
# passes VAD segments to the coverage comparison.
# Use --force to rerun whisper even when artifacts already exist.
# Use --glossary to override glossary auto-discovery.

python3 scripts/repair_vtt_codex.py next-region \
  --session samples/episodes/<slug>/transcription/<stem>_reflow_repaired.vtt.checkpoint.json

python3 scripts/repair_vtt_codex.py apply-region \
  --session samples/episodes/<slug>/transcription/<stem>_reflow_repaired.vtt.checkpoint.json \
  --repair-json /tmp/<stem>_repair_region.json

# Optional local second-opinion ASR for a suspicious short window
ffmpeg -y -ss <start_s> -to <end_s> -i samples/episodes/<slug>/source/video.mp4 \
  -vn -ac 1 -ar 16000 -c:a pcm_s16le /tmp/<slug>_clip.wav

env LD_LIBRARY_PATH=/opt/rocm-7.2.0/lib:${LD_LIBRARY_PATH:-} \
CT2_CUDA_ALLOCATOR=cub_caching \
python3.12 scripts/run_faster_whisper.py \
  --video /tmp/<slug>_clip.wav \
  --glossary /tmp/does_not_exist_glossary.txt \
  --out samples/episodes/<slug>/transcription/diagnostics/<slug>_clip_faster_large_v3.vtt \
  --model large-v3 \
  --compute-type float16 \
  --reflow

python3 scripts/repair_vtt_codex.py finalize \
  --session samples/episodes/<slug>/transcription/<stem>_reflow_repaired.vtt.checkpoint.json

# 8. Translate
# dmm example from the current CTC + reflow path
python scripts/translate_vtt.py \
  --input samples/episodes/dmm/transcription/dmm_ctc_reflow.vtt \
  --output samples/episodes/dmm/translation/dmm_ctc_reflow_en.vtt \
  --batch-cues 12 \
  --batch-seconds 45

# great_escape1 example from the current reflow path
python scripts/translate_vtt.py \
  --input samples/episodes/great_escape1/transcription/ge1_reflow.vtt \
  --output samples/episodes/great_escape1/translation/ge1_reflow_en.vtt \
  --batch-cues 12 \
  --batch-seconds 45

# `translate_vtt.py` now translates in local cue batches, preserves cue timings,
# targets readable English subtitle CPS, retries overlong batches once, and
# writes `<output>.diagnostics.json`.

# Codex-interactive translation helper (no API call)
python scripts/translate_vtt_codex.py prepare \
  --input samples/episodes/great_escape_s01e04/transcription/great_escape_s01e04_video_only_v1_ctc_words_reflow.vtt \
  --output samples/episodes/great_escape_s01e04/translation/great_escape_s01e04_video_only_v1_ctc_words_reflow_en_codex.vtt

# Auto-discovers a sibling CTC alignment diagnostics sidecar when present.
# Use --alignment-diagnostics <path> to override discovery explicitly.
# Each target cue payload also includes source_text_hash.

python scripts/translate_vtt_codex.py next-batch \
  --session samples/episodes/great_escape_s01e04/translation/great_escape_s01e04_video_only_v1_ctc_words_reflow_en_codex.vtt.checkpoint.json

# After Codex translates the emitted batch and writes a JSON payload:
python scripts/translate_vtt_codex.py apply-batch \
  --session samples/episodes/great_escape_s01e04/translation/great_escape_s01e04_video_only_v1_ctc_words_reflow_en_codex.vtt.checkpoint.json \
  --translations-json /tmp/batch.json

# Each translation item in /tmp/batch.json must include source_text_hash
# (or source_text) copied from next-batch so apply-batch can reject
# cue-ID-only semantic drift.

# `translate_vtt_codex.py` writes a session/checkpoint JSON, a partial VTT in
# `translation/`, a deterministic diagnostics rollup, surfaces advisory
# alignment-warning context for affected batches, and automatically reduces the
# batch tier 84 -> 60 -> 48 when a batch is reviewed as yellow.
# Restart with `prepare --force` to clear stale session/output/diagnostics
# artifacts before beginning a fresh run.

# Experimental Mistral benchmark with the same cue-preserving translation flow.
# dmm
python scripts/translate_vtt_mistral.py \
  --input samples/episodes/dmm/transcription/dmm_ctc_reflow.vtt \
  --output samples/episodes/dmm/translation/dmm_ctc_reflow_en_mistral.vtt

# great_escape1
python scripts/translate_vtt_mistral.py \
  --input samples/episodes/great_escape1/transcription/ge1_reflow.vtt \
  --output samples/episodes/great_escape1/translation/ge1_reflow_en_mistral.vtt
```

### Local Pipeline

```bash
python scripts/transcribe_local.py \
  --video samples/episodes/<slug>/source/video.mp4 \
  --ocr-jsonl samples/episodes/<slug>/ocr/qwen_ocr_results.jsonl \
  --llm-url http://127.0.0.1:8080 \
  --llm-model gemma3-27b
```
