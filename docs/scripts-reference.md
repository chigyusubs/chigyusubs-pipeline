# Pipeline Scripts Reference

This file documents every script under `scripts`.

## Canonical Episode Workflow

Use these scripts for the normal OCR -> glossary -> ASR flow:

1. `run_qwen_ocr_episode.py ocr`
2. `run_qwen_ocr_episode.py filter`
3. `condense_glossary_vertex.py`
4. `run_faster_whisper.py` or `run_whisper_cpp.py`

Recommended episode layout:

```text
samples/episodes/<episode_slug>/
  source/
  frames/
  ocr/
  glossary/
  transcription/
  translation/
  logs/
```

## Script Catalog

### Core Scripts

| Script | Purpose | Typical Input | Typical Output | Status |
|---|---|---|---|---|
| `episode_paths.py` | Shared path inference helpers for episode-aware defaults. | Video/episode paths | Resolved `samples/episodes/<episode>/...` paths | Maintained |
| `run_qwen_ocr_episode.py` | Qwen-VL OCR over frames (`ocr`) + frequency/run-length glossary filter (`filter`). | `frames/*.jpg` or episode dir | `ocr/qwen_ocr_results.jsonl`, `glossary/qwen_glossary.json` | Maintained |
| `condense_glossary_vertex.py` | Build structured glossary (`source,target,context`) with Gemini on Vertex. | OCR candidate text file | Prompt CSV, structured JSON, translation TSV, hotwords CSV | Maintained |
| `run_faster_whisper.py` | ASR with faster-whisper (ROCm/CUDA), optional initial prompt + hotwords. | Episode `source/*.mp4`, glossary files | `transcription/*.vtt` + `_words.json` | Maintained |
| `run_whisper_cpp.py` | ASR with `whisper-cli` (whisper.cpp), VAD enabled, glossary as prompt. | Episode `source/*.mp4`, glossary file | `transcription/*_whispercpp_natural.vtt` | Maintained |
| `run_local_whisper.py` | ASR with OpenAI Whisper CLI (`whisper` python package). | Episode `source/*.mp4`, glossary file | VTT in episode `transcription/` | Maintained |
| `run_vertex.py` | Generic CLI wrapper for Vertex Gemini text generation. | `--prompt` or stdin | stdout or `--out` file | Maintained |

### Legacy / Utility Scripts

| Script | Purpose | Typical Input | Typical Output | Status |
|---|---|---|---|---|
| `build_whisper_glossary.py` | Older EasyOCR-based full pass with resume JSONL and simple dedupe. | Video file | `samples/whisper_prompt.txt` | Legacy |
| `extract_glossary.py` | Older MangaOCR-based glossary extraction experiment. | Video file | Printed candidate terms | Legacy |
| `condense_glossary_llm.py` | Older local LLM condensation via OpenAI-compatible endpoint. | `samples/whisper_prompt.txt` | `samples/whisper_prompt_condensed.txt` | Legacy |
| `format_vtt_netflix.py` | Post-format Japanese VTT cues toward Netflix-style readability using Gemini. | Raw VTT | Formatted VTT | Utility |
| `json_to_smart_vtt.py` | Convert word-timestamp JSON into pause-split VTT cues (no LLM). | `*_words.json` | VTT | Utility |
| `reflow_subtitles_local.py` | LLM-based cue regrouping using local llama.cpp API. | `*_words.json` | Reflowed VTT | Experimental |
| `reflow_subtitles_vertex.py` | LLM-based cue regrouping using Vertex Gemini. | `*_words.json` | Reflowed VTT | Experimental |

### Test / Evaluation Scripts

| Script | Purpose | Typical Input | Typical Output | Status |
|---|---|---|---|---|
| `test_easyocr.py` | Quick OCR sanity test with EasyOCR on frame folder. | `samples/frames/*.jpg` | Console dump | Test |
| `test_paddleocr.py` | Quick OCR sanity test with PaddleOCR on frame folder. | `samples/frames/*.jpg` | Console dump | Test |
| `test_faster_whisper.py` | Quick faster-whisper runtime sanity check. | Sample video | First few segments in console | Test |
| `test_qwen_ocr.py` | Qwen-VL OCR spot-check against selected frames. | Frame files | Console OCR output | Test |
| `test_qwen_json_schema.py` | Compare JSON reliability modes on local Qwen endpoint. | OCR lines file | Parse/failure stats | Test |
| `test_vlm_extraction.py` | Compare dumb OCR prompt vs noun-extraction prompt with Gemini vision. | Frame folder | Console comparison | Test |
| `test_reflow_quality.py` | Evaluate LLM reflow quality on a sample word stream. | `*_words.json` | Console sample output | Test |

## CLI Cheatsheet

### `run_qwen_ocr_episode.py`

```bash
# OCR all frames (episode-aware defaults if --episode-dir is provided)
python scripts/run_qwen_ocr_episode.py ocr \
  --episode-dir samples/episodes/wednesday_downtown_2025-02-05_406 \
  --url http://127.0.0.1:8787 \
  --model qwen3.5-9b

# Build filtered glossary JSON from OCR JSONL
python scripts/run_qwen_ocr_episode.py filter \
  --episode-dir samples/episodes/wednesday_downtown_2025-02-05_406 \
  --top-global 120 --top-chunk 30
```

### `condense_glossary_vertex.py`

```bash
python scripts/condense_glossary_vertex.py \
  --input samples/episodes/wednesday_downtown_2025-02-05_406/glossary/qwen_candidates.txt \
  --out-prompt samples/episodes/wednesday_downtown_2025-02-05_406/glossary/whisper_prompt_condensed.txt \
  --out-structured samples/episodes/wednesday_downtown_2025-02-05_406/glossary/whisper_prompt_condensed_structured.json \
  --out-translation samples/episodes/wednesday_downtown_2025-02-05_406/glossary/translation_glossary.tsv \
  --out-hotwords samples/episodes/wednesday_downtown_2025-02-05_406/glossary/whisper_hotwords.txt \
  --model gemini-3.1-pro-preview --top-n 100 --hotwords-n 40
```

### `run_faster_whisper.py`

```bash
# Uses latest episode source/glossary defaults if flags are omitted.
python scripts/run_faster_whisper.py \
  --model large-v3 --compute-type float16
```

## Qwen Model Names Used Here

`run_qwen_ocr_episode.py` sends requests to your local OpenAI-compatible endpoint and defaults to:

- `--model qwen3.5-9b` (or whatever alias your llama.cpp server exposes)

This is a server alias. In your current setup it has been mapped to the Qwen3.5-9B vision-capable model loaded in `llama.cpp` with its `mmproj` file.
