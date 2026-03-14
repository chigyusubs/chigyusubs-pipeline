# chigyusubs-pipeline

Subtitle pipeline for Japanese variety shows. Takes a raw episode video and produces timed, translated English subtitles through a chain of reusable artifacts.

## Workflow

```text
Video file
  ├─ VAD (Silero) → full-coverage chunk boundaries
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

Each stage writes reusable artifacts under `samples/episodes/<slug>/`. Published outputs stay stable, while lineage artifacts in `transcription/` and draft `translation/` may use short run-ID filenames plus `preferred.json` pointers.
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

Maintained Gemini/API-backed CLI scripts auto-load the repo-root `.env` file by default, so Codex and direct terminal runs do not need manual `export` steps for keys such as `GEMINI_API_KEY` or `OPENAI_API_KEY`.

## Current Practical Setup

The current repo is optimized around this setup:

- Gemini API for transcription
- local faster-whisper and CTC alignment
- Codex for interactive reflow/translation work

Recommended named Gemini presets:

- transcript production on `2.5-flash`: `--preset flash25_free_default`
- transcript experiments: `--preset flash_free_default`
- visual-rich transcript comparison: `--preset flash_visual_artifact`
- cheap resumable debug transcript probes: `--preset flashlite_debug_transcript`
- chunkwise OCR sidecar: `--preset flashlite_ocr_sidecar`

The maintained practical path is no longer “local OCR first.” It is:

- VAD
- full-coverage chunks with VAD-guided split points
- Gemini transcription
- optional chunkwise Flash Lite OCR sidecar
- CTC alignment
- faster-whisper second opinion
- reflow
- Codex translation/review

For maintained Gemini video chunking, local ffmpeg extraction keeps source resolution by default at `1 FPS`. Use width downscaling only intentionally.
Chunk plans should also treat `target_chunk_s + 30s` as the default hard maximum.

Local OCR and manual AI Studio experiment packs still exist, but they are secondary tools.

## Episode Layout

```text
samples/episodes/<slug>/
  source/         # video files
  frames/         # extracted frames for OCR
  ocr/            # OCR sidecars, spans, context
  glossary/       # ASR glossary, translation glossary
  transcription/  # chunks, raw JSON, aligned words, reflow VTT, diagnostics
  translation/    # draft translated VTT
  logs/

samples/experiments/<pack>/
  prompts/        # system + user prompts for manual site experiments
  scenes/         # extracted video/audio clips plus per-scene notes
  results/        # saved manual experiment outputs you chose to keep
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

# Cheap one-chunk Flash Lite debug smoke test
python scripts/transcribe_gemini_video.py \
  --video samples/episodes/<slug>/source/<video>.mp4 \
  --output samples/episodes/<slug>/transcription/<slug>_flashlite_debug.json \
  --preset flashlite_debug_transcript \
  --chunk-json samples/episodes/<slug>/transcription/vad_chunks_semantic_180.json \
  --stop-after-chunks 1

# Or step by step — see docs/scripts-reference.md for the full CLI cheatsheet
```

Common chunk plan names:

- `transcription/vad_chunks.json` — default full-coverage VAD plan
- `transcription/vad_chunks_semantic_180.json` — reviewed semantic plan targeting about `180s`
- `transcription/*_repair*.json` — follow-up repair plan that only resplits failed chunks
- `transcription/probes/*exact_chunks_60s*.json` — strict debug probe plan, usually for Flash Lite

Chunking defaults now treat `target + 30s` as a hard max, but they prefer a
shorter real silence gap (down to `0.75s`) before falling back to a true
mid-speech forced split.

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
3. Translate the preferred Japanese VTT into `translation/`
4. Publish the final English VTT back into `source/`

## Artifact Naming

- `source/<video_stem>.vtt` is the stable published subtitle path.
- Lineage artifacts in `transcription/` and draft `translation/` use short run-ID names such as `r3a7f01b_ctc_words.json` and `r3a7f01b_en.vtt`.
- `transcription/preferred.json` and `translation/preferred.json` point to the current preferred lineage artifacts.
- `.meta.json` sidecars and VTT `NOTE` headers carry run provenance.

## Docs

- `docs/scripts-reference.md` — full script catalog and CLI cheatsheet
- `docs/current-architecture.md` — artifact-level pipeline design
- `docs/archive/ocr-first-architecture.md` — older OCR-heavy architecture framing kept for reference
- `docs/gemini-transcription-playbook.md` — current model/chunking/media-resolution policy for Gemini transcription
- `docs/gemini-pricing-notes.md` — practical model pricing/value conclusions for transcript + OCR combinations
- `docs/codex-skills.md` — Codex skill scope, defaults, and handoff
- `docs/lessons-learned.md` — validated findings from real episode runs
- `docs/transcription-research-roadmap.md` — non-default ideas and experiment directions for chunking, OCR context, and hybrid modality routing
