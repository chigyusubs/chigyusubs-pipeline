# Pipeline Scripts Reference

This file documents every script under `scripts/`.

## Pipelines

There are two transcription paths. Both share the OCR and glossary stages.

### Path A: Gemini (recommended)

Best quality. Requires Vertex AI access. Local LLM handles OCR filtering.

```
Silero VAD -> chunk-wise OCR filter (local Gemma 27B) -> Gemini transcription
  -> stable-ts alignment -> reflow -> translation
```

1. `run_qwen_ocr_episode.py ocr` + `filter` — OCR frames, build glossary candidates
2. `condense_glossary_vertex.py` — structure glossary with Gemini
3. `transcribe_gemini_raw.py` — VAD-chunked Gemini transcription with local OCR filter
4. `align_chunkwise.py` — chunk-wise stable-ts forced alignment
5. `reflow_words.py` (via pipeline) — pause-based reflow into subtitle cues
6. `translate_vtt.py` — LLM translation to English

Or use `transcribe_pipeline.py` which chains steps 3-5 automatically.

### Path B: Local (faster-whisper)

Fully local, no API calls. Lower quality but zero cost.

1. `run_qwen_ocr_episode.py ocr` + `filter`
2. `transcribe_local.py` — Silero VAD + local OCR filter + faster-whisper with hotwords
3. `translate_vtt.py`

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
| `transcribe_pipeline.py` | Integrated 4-phase pipeline: Silero VAD -> Gemini -> stable-ts -> reflow | Maintained |
| `transcribe_gemini_raw.py` | Gemini raw text transcription with chunk-wise local OCR filtering | Maintained |
| `transcribe_gemini.py` | Gemini transcription (JSON schema mode, used by pipeline) | Maintained |
| `transcribe_local.py` | Fully local pipeline: Silero VAD + local OCR filter + faster-whisper | Maintained |
| `align_chunkwise.py` | Chunk-wise stable-ts forced alignment from `_chunks.json` | Maintained |
| `align_stable_ts.py` | Global stable-ts alignment (single pass, simpler) | Maintained |
| `reflow_words.py` | Pause-based reflow of word timestamps into subtitle cues | Maintained |
| `translate_vtt.py` | LLM translation of Japanese VTT to English (Vertex or local) | Maintained |

### OCR & Glossary

| Script | Purpose | Status |
|---|---|---|
| `run_qwen_ocr_episode.py` | Qwen-VL OCR over frames + frequency/run-length glossary filter | Maintained |
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
  --episode-dir samples/episodes/<slug>

# 2. Filter OCR into glossary candidates
python scripts/run_qwen_ocr_episode.py filter \
  --episode-dir samples/episodes/<slug>

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

# 5. Align chunk-wise
python scripts/align_chunkwise.py \
  --chunks-json samples/episodes/<slug>/transcription/raw_chunks.json \
  --video samples/episodes/<slug>/source/video.mp4

# 6. Translate
python scripts/translate_vtt.py \
  --input samples/episodes/<slug>/transcription/raw_aligned.vtt \
  --glossary samples/episodes/<slug>/glossary/translation_glossary.tsv
```

### Local Pipeline

```bash
python scripts/transcribe_local.py \
  --video samples/episodes/<slug>/source/video.mp4 \
  --ocr-jsonl samples/episodes/<slug>/ocr/qwen_ocr_results.jsonl \
  --llm-url http://127.0.0.1:8080 \
  --llm-model gemma3-27b
```
