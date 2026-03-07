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
  -> optional local cue repair
  -> translation
```

1. `run_qwen_ocr_episode.py ocr` — frame OCR
2. `run_qwen_ocr_episode.py spans` — OCR span building
3. `run_qwen_ocr_episode.py context` or `classify_ocr_spans_local.py` — OCR context derivation
4. `run_vad_episode.py` — reusable Silero VAD
5. `build_vad_chunks.py` — reusable chunk boundaries
6. `transcribe_pipeline.py` — Gemini transcription + alignment + reflow
7. `repair_vtt_local.py` — optional local Gemma cue-boundary repair on reflowed VTT
8. `translate_vtt.py` — LLM translation to English

`transcribe_pipeline.py` can now consume the saved VAD/chunk/OCR artifacts instead of recomputing them.

### Path B: Local (faster-whisper)

Fully local, no API calls. Lower quality but zero cost.

1. `run_qwen_ocr_episode.py ocr` + `spans` + `context`
2. `run_vad_episode.py` + `build_vad_chunks.py`
3. `transcribe_local.py` — local ASR path
4. `translate_vtt.py`

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
| `align_ctc.py` | CTC forced alignment using `NTQAI/wav2vec2-large-japanese` + `torchaudio.functional.forced_align`. 0.3% zero-duration words vs 13.4% with stable-ts on `dmm`. Runs on system python3.12 with ROCm GPU. | Maintained |
| `align_chunkwise.py` | Chunk-wise stable-ts forced alignment from `_chunks.json` | Legacy |
| `align_qwen_forced.py` | Chunk-wise Qwen forced-alignment benchmark via `py-qwen3-asr-cpp` GGUF backend | Archived |
| `align_qwen_forced_hf.py` | Chunk-wise Qwen forced-alignment benchmark via official `qwen-asr` on system Python / ROCm | Archived |
| `align_stable_ts.py` | Global stable-ts alignment (single pass) | Legacy |
| `reflow_words.py` | Reflow word/line timestamps into subtitle cues. `--line-level` (default for CTC) treats lines as atomic, preventing mid-word splits. Includes comma-fallback splitting and sparse-cue clamping for CTC artifacts. | Maintained |
| `repair_vtt_local.py` | Repair an existing reflowed VTT using aligned words + local Gemma as a constrained merge/split/extend chooser. Writes repaired VTT plus decisions/checkpoint JSON. | Maintained |
| `translate_vtt.py` | LLM translation of Japanese VTT to English (Vertex or local) | Maintained |
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
| `run_whisper_cpp.py` | whisper.cpp ASR via `whisper-cli` | Maintained |
| `run_local_whisper.py` | OpenAI Whisper Python package ASR | Maintained |
| `start_qwen_ocr_server.sh` | Start llama-server for Qwen OCR with recommended deterministic settings | Maintained |
| `start_gemma_ocr_filter_server.sh` | Start llama-server for local Gemma OCR cleanup/classification | Maintained |
| `start_gemma_cue_repair_server.sh` | Start llama-server for local Gemma cue-boundary repair decisions. Defaults to port `8082`, larger context, and thinking disabled via `--reasoning-budget 0`. | Maintained |
| `start_qwen_cue_repair_server.sh` | Start llama-server for Qwen3.5-35B-A3B cue-boundary repair decisions. Defaults to port `8083`, larger context, and thinking disabled via `--reasoning-budget 0`. | Maintained |
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

# 6. Reflow (line-level, recommended for CTC output)
PYTHONPATH=. python3 scripts/reflow_words.py \
  --input samples/episodes/<slug>/transcription/<stem>_ctc_words.json \
  --output samples/episodes/<slug>/transcription/<stem>_reflow.vtt \
  --line-level --stats

# 7. Translate
# Optional repair pass between reflow and translation:
scripts/start_gemma_cue_repair_server.sh

python scripts/repair_vtt_local.py \
  --input-vtt samples/episodes/<slug>/transcription/<stem>_reflow.vtt \
  --input-words samples/episodes/<slug>/transcription/<stem>_ctc_words.json \
  --url http://127.0.0.1:8082 \
  --model gemma3-27b

# Qwen3.5-35B-A3B alternative:
scripts/start_qwen_cue_repair_server.sh

python scripts/repair_vtt_local.py \
  --input-vtt samples/episodes/<slug>/transcription/<stem>_reflow.vtt \
  --input-words samples/episodes/<slug>/transcription/<stem>_ctc_words.json \
  --url http://127.0.0.1:8083 \
  --model qwen3.5-35b-a3b

# 8. Translate
python scripts/translate_vtt.py \
  --input samples/episodes/<slug>/transcription/raw_aligned.vtt \
  --glossary samples/episodes/<slug>/glossary/translation_glossary.tsv \
  --batch-cues 12 \
  --batch-seconds 45

# `translate_vtt.py` now translates in local cue batches, preserves cue timings,
# targets readable English subtitle CPS, retries overlong batches once, and
# writes `<output>.diagnostics.json`.
```

### Local Pipeline

```bash
python scripts/transcribe_local.py \
  --video samples/episodes/<slug>/source/video.mp4 \
  --ocr-jsonl samples/episodes/<slug>/ocr/qwen_ocr_results.jsonl \
  --llm-url http://127.0.0.1:8080 \
  --llm-model gemma3-27b
```
