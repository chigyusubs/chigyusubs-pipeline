# chigyusubs-pipeline

Local pipeline workspace for Japanese variety-show OCR, ASR, glossary building, and translation prep.

## Current Defaults

- Use fixed-rate frame sampling (`0.5 fps`) instead of `mpdecimate` by default.
- Use Whisper `initial_prompt` as the default glossary guidance.
- Treat full hotwords injection as experimental/ablation.
- Keep glossary outputs two-track:
  - ASR glossary: Japanese-only terms for Whisper.
  - Translation glossary: `source,target,context` entries for translation prompts.
- Voice isolation is optional and segment-specific; it can be a side-grade and may increase hallucination risk.

## Requirements

- Python 3.11+ (3.12 recommended)
- `ffmpeg`
- Local `llama.cpp` server (OpenAI-compatible endpoint)
- GPU stack for ASR (`faster-whisper` / ROCm or CUDA)

Optional but common:

- Vertex/Gemini access for glossary condensation
- `jq` for small JSON extraction helpers

## Environment Variables

- `QWEN_VISION_MODEL`
  - Default used by OCR scripts: `qwen3.5-9b`
  - Set this if your llama-server alias differs.
- `GEMINI_MODEL`
  - Default in glossary condensation script: `gemini-3.1-pro-preview`
- `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`
  - Needed for Vertex workflows.

## Directory Layout

```text
samples/
  episodes/
    <episode_slug>/
      source/
      frames/
      ocr/
      glossary/
      transcription/
      translation/
      logs/
  experiments/
```

## Quick Start (Episode Workflow)

Example episode: `samples/episodes/wednesday_downtown_2025-02-05_406`

1. Run OCR on extracted frames:

```bash
python scripts/run_qwen_ocr_episode.py ocr \
  --episode-dir samples/episodes/wednesday_downtown_2025-02-05_406 \
  --url http://127.0.0.1:8787 \
  --model qwen3.5-9b
```

2. Build filtered OCR glossary JSON:

```bash
python scripts/run_qwen_ocr_episode.py filter \
  --episode-dir samples/episodes/wednesday_downtown_2025-02-05_406 \
  --top-global 120 --top-chunk 30
```

3. Export a plain candidate list for condensation:

```bash
jq -r '.global_glossary[]' \
  samples/episodes/wednesday_downtown_2025-02-05_406/glossary/qwen_glossary.json \
  > samples/episodes/wednesday_downtown_2025-02-05_406/glossary/qwen_candidates.txt
```

4. Build condensed ASR + translation glossary outputs (Vertex):

```bash
python scripts/condense_glossary_vertex.py \
  --input samples/episodes/wednesday_downtown_2025-02-05_406/glossary/qwen_candidates.txt \
  --out-prompt samples/episodes/wednesday_downtown_2025-02-05_406/glossary/whisper_prompt_condensed.txt \
  --out-structured samples/episodes/wednesday_downtown_2025-02-05_406/glossary/whisper_prompt_condensed_structured.json \
  --out-translation samples/episodes/wednesday_downtown_2025-02-05_406/glossary/translation_glossary.tsv \
  --out-hotwords samples/episodes/wednesday_downtown_2025-02-05_406/glossary/whisper_hotwords.txt \
  --model gemini-3.1-pro-preview
```

5. Run ASR (faster-whisper):

```bash
python scripts/run_faster_whisper.py --model large-v3 --compute-type float16
```

## Troubleshooting

- OCR fails to connect:
  - Check llama-server URL/port.
  - Confirm model alias matches `--model` / `QWEN_VISION_MODEL`.
- Vertex errors (`429`, quota/resource exhausted):
  - Retry (scripts already include retry logic in key paths).
  - Use stronger model tier for glossary quality-sensitive passes.
- ASR quality dips:
  - Start from raw audio + `initial_prompt` first.
  - Use hotwords/voice isolation only as targeted experiments.

## Docs

- Design + architecture: `docs/local-whisper-ocr-pipeline.md`
- Full script reference: `docs/scripts-reference.md`
