# chigyusubs-pipeline

Subtitle pipeline for Japanese variety shows. Takes a raw episode video and produces timed, translated English subtitles through a chain of reusable artifacts.

## Workflow

```text
Video file
  ├─ VAD (Silero) → chunk boundaries
  │
  ├─ Transcription (Gemini or local faster-whisper)
  │     uses: chunk boundaries
  │
  ├─ Optional OCR-like side artifacts
  │     ├─ chunkwise Gemini Flash-Lite OCR sidecar
  │     └─ older local Qwen-VL OCR → spans → context
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
Manual benchmark packs for site-driven experiments can live separately under `samples/experiments/<pack>/`. The helper for building them is experimental and lives under `scripts/experiments/`.

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
- local hardware fast enough to run:
  - `faster-whisper`
  - CTC alignment (`wav2vec2-large-japanese`)
  - practically, this means a decent GPU for ASR/alignment work
- ChatGPT Plus/Pro access with Codex, if you want the maintained interactive translation and review workflow
- Gemini API key for the maintained transcription path

Optional:

- Vertex AI access for alternative Gemini runs and exact `count_tokens` preflight with generation overrides
- local `llama.cpp` OCR stack (Qwen-VL / Gemma) if you want to run the older OCR-first paths
- Gemini API key or Mistral API key for unattended translation

See `.env.example` for environment variable reference.

## Current Practical Setup

The current repo is optimized around this setup:

- Gemini API for transcription
- local faster-whisper and CTC alignment
- Codex for interactive reflow/translation work

The maintained practical path is no longer “local OCR first.” It is:

- VAD
- Gemini transcription
- optional chunkwise Flash Lite OCR sidecar
- CTC alignment
- faster-whisper second opinion
- reflow
- Codex translation/review

Local OCR and manual AI Studio experiment packs still exist, but they are secondary tools.

## Episode Layout

```text
samples/episodes/<slug>/
  source/         # video files
  frames/         # extracted frames for OCR
  ocr/            # OCR sidecars, spans, context
  glossary/       # ASR glossary, translation glossary
  transcription/  # VTT, aligned words, chunks, diagnostics
  translation/    # translated VTT
  logs/

samples/experiments/<pack>/
  prompts/        # system + user prompts for manual site experiments
  scenes/         # extracted video/audio clips plus per-scene notes
  results/        # paste-back result templates
  manifest.json   # experiment metadata and model matrix
```

## Quick Start

```bash
# Initialize episode workspace from media
python scripts/init_episode_from_media.py --extract-frames samples/new_episode.mp4

# Current maintained path
python scripts/run_vad_episode.py --episode-dir samples/episodes/<slug>
python scripts/build_vad_chunks.py --episode-dir samples/episodes/<slug>
python scripts/transcribe_pipeline.py --episode-dir samples/episodes/<slug>

# Optional structured OCR sidecar for the same chunk plan
python scripts/extract_gemini_chunk_ocr.py \
  --video samples/episodes/<slug>/source/<video>.mp4 \
  --chunk-json samples/episodes/<slug>/transcription/vad_chunks.json

# Or step by step — see docs/scripts-reference.md for the full CLI cheatsheet
```

If you want the older OCR-first path or manual experiment tooling, see:

- `docs/scripts-reference.md`
- `docs/current-architecture.md`
- `docs/gemini-transcription-playbook.md`

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
- `docs/archive/ocr-first-architecture.md` — older OCR-heavy architecture framing kept for reference
- `docs/gemini-transcription-playbook.md` — current model/chunking/media-resolution policy for Gemini transcription
- `docs/gemini-pricing-notes.md` — practical model pricing/value conclusions for transcript + OCR combinations
- `docs/codex-skills.md` — Codex skill scope, defaults, and handoff
- `docs/lessons-learned.md` — validated findings from real episode runs
- `docs/transcription-research-roadmap.md` — non-default ideas and experiment directions for chunking, OCR context, and hybrid modality routing
