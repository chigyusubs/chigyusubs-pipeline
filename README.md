# chigyusubs-pipeline

Subtitle pipeline for Japanese variety shows. Takes a raw episode video and produces timed, translated English subtitles through a chain of reusable artifacts.

## Workflow

```text
Video file
  ├─ Frame extraction (fixed-rate 0.5 fps)
  ├─ OCR (local Qwen-VL) → spans → context
  ├─ VAD (Silero) → chunk boundaries
  │
  ├─ Transcription (Gemini or local faster-whisper)
  │     uses: chunk boundaries + optional OCR context / hotwords
  │
  ├─ Raw transcript → glossary/context extraction
  ├─ CTC forced alignment (wav2vec2-ja)
  ├─ Whisper second-opinion coverage check
  ├─ Reflow → timed Japanese VTT
  ├─ Optional cue repair (interactive)
  │
  └─ Translation → timed English VTT
        interactive (Codex) or unattended (API)
        └─ publish back to source/<video_stem>.vtt
```

Each stage writes stable artifacts under `samples/episodes/<slug>/`. Later stages consume earlier artifacts without recomputing them.

## Translation Paths

**Interactive (primary)** — `translate_vtt_codex.py`. Codex translates batch-by-batch with human review, checkpointed session state, and automatic batch-tier fallback. Best quality.

**Unattended (API)** — `translate_vtt_api.py`. Sends batches to Vertex Gemini or any OpenAI-compatible API. Good for benchmarking, testing model capability, or fully local pipelines.

After an interactive translation run, publish the finished VTT next to the source video with:

```bash
python3 scripts/publish_vtt.py samples/episodes/<slug>/translation/<final>.vtt
```

## Requirements

- Python 3.11+ (3.12 recommended)
- `ffmpeg`
- Local `llama.cpp` server (Qwen-VL for OCR, optionally Gemma for filtering)
- GPU for ASR (`faster-whisper` with ROCm or CUDA)

Optional:

- Vertex AI / Gemini API access for transcription and glossary condensation
- Gemini API key or Mistral API key for unattended translation

See `.env.example` for environment variable reference.

## Episode Layout

```text
samples/episodes/<slug>/
  source/         # video files
  frames/         # extracted frames for OCR
  ocr/            # OCR results, spans, context
  glossary/       # ASR glossary, translation glossary
  transcription/  # VTT, aligned words, chunks, diagnostics
  translation/    # translated VTT
  logs/
```

## Quick Start

```bash
# Initialize episode workspace from media
python scripts/init_episode_from_media.py --extract-frames samples/new_episode.mp4

# OCR
scripts/start_qwen_ocr_server.sh
python scripts/run_qwen_ocr_episode.py ocr --episode-dir samples/episodes/<slug>
python scripts/run_qwen_ocr_episode.py spans --episode-dir samples/episodes/<slug>
python scripts/run_qwen_ocr_episode.py context --episode-dir samples/episodes/<slug>

# Transcription + alignment + reflow (Gemini path)
python scripts/transcribe_pipeline.py --episode-dir samples/episodes/<slug>

# Or step by step — see docs/scripts-reference.md for the full CLI cheatsheet
```

## Codex Skills

Three interactive skills for use with [Codex](https://openai.com/index/introducing-codex/):

- **`subtitle-reflow`** — reflows aligned words into subtitle cues, reviews for translation-risk issues, optionally runs interactive cue repair
- **`glossary-context`** — builds `glossary/glossary.json` and `glossary/episode_context.json` from the Gemini raw transcript
- **`subtitle-translation`** — batch-by-batch English translation with checkpointed session, source-hash validation, and CPS diagnostics

```text
Use $subtitle-reflow on samples/episodes/<slug>/transcription/<stem>_ctc_words.json
Use $glossary-context on samples/episodes/<slug>/transcription/<stem>_gemini_raw.json
Use $subtitle-translation on samples/episodes/<slug>/transcription/<stem>_reflow_repaired.vtt
```

Skills are tracked in `codex/skills/` and installed with `python3 scripts/install_codex_skills.py`.

## Recommended Interactive Handoff

For the maintained Codex workflow, the intended order is:

```text
aligned words JSON
  -> subtitle-reflow
  -> recommended Japanese VTT
  -> glossary-context
  -> subtitle-translation
  -> publish_vtt.py
  -> source/<video_stem>.vtt
```

In practice this means:

1. Reflow and review `*_ctc_words.json`
2. Build glossary/context from `*_gemini_raw.json`
3. Translate the recommended Japanese VTT into `translation/`
4. Publish the final English VTT back into `source/`

## Docs

- `docs/scripts-reference.md` — full script catalog and CLI cheatsheet
- `docs/current-architecture.md` — artifact-level pipeline design
- `docs/codex-skills.md` — Codex skill scope, defaults, and handoff
- `docs/lessons-learned.md` — validated findings from real episode runs
