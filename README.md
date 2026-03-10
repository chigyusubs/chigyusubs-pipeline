# chigyusubs-pipeline

Local pipeline workspace for Japanese variety-show OCR, ASR, glossary building, and translation prep.

Key docs:

- `docs/current-architecture.md` — current artifact-level pipeline shape
- `docs/lessons-learned.md` — validated findings and remaining issues from real episode runs
- `docs/codex-skills.md` — Codex-interactive skills for reflow review/repair and translation

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
- Mistral API key for translation experiments
- `jq` for small JSON extraction helpers

## Environment Variables

- `QWEN_VISION_MODEL`
  - Default used by OCR scripts: `qwen3.5-9b`
  - Set this if your llama-server alias differs.
- `GEMINI_MODEL`
  - Default in glossary condensation script: `gemini-3.1-pro-preview`
- `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`
  - Needed for Vertex workflows.
- `MISTRAL_API_KEY`
  - For Mistral translation via `scripts/translate_vtt_api.py --backend openai --url https://api.mistral.ai`.

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

Initialize a new episode workspace from media:

```bash
python scripts/init_episode_from_media.py \
  --extract-frames \
  samples/new_episode.mp4
```

Recommended OCR server settings for Qwen-VL:

```bash
scripts/start_qwen_ocr_server.sh
```

By default the wrapper auto-detects `mmproj*.gguf` under `~/.cache/llama.cpp`.
Set `QWEN_OCR_MMPROJ` only if you want to override that.

1. Run OCR on extracted frames:

```bash
python scripts/run_qwen_ocr_episode.py ocr \
  --episode-dir samples/episodes/wednesday_downtown_2025-02-05_406 \
  --url http://127.0.0.1:8787 \
  --model qwen3.5-9b \
  --server-settings '{"quant":"Q6_K","ctx_size":8192,"seed":3407,"temp":0,"top_p":0.9,"top_k":20,"thinking":false}'
```

This writes OCR results to `ocr/*.jsonl` and a sidecar metadata file
`*.meta.json` with prompt, client settings, optional server settings, and run timing.
The maintained transcription/alignment scripts now emit the same metadata sidecars.

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

Mistral translation via the OpenAI-compatible backend:

```bash
python scripts/translate_vtt_api.py --backend openai \
  --url https://api.mistral.ai --api-key $MISTRAL_API_KEY --model mistral-small-latest \
  --input samples/episodes/dmm/transcription/dmm_ctc_reflow.vtt \
  --output samples/episodes/dmm/translation/dmm_ctc_reflow_en_mistral.vtt
```

## Codex Skills

The repo now also has two Codex-side skills for interactive subtitle work.
These are not API-backed model runs. They are workflow guides that make Codex
use the maintained repo scripts and review logic consistently.

- `subtitle-reflow`
  - reflows aligned Japanese subtitle artifacts with the maintained line-level
    path
  - reviews the resulting VTT for translation-risk issues
  - optionally runs Codex-interactive cue repair when the VTT is structurally valid but weak
    for translation
  - uses deterministic review metrics and recommends one Japanese handoff path for translation
- `subtitle-translation`
  - translates a Japanese VTT/SRT into English with Codex itself as the subtitle
    editor
  - uses the maintained `translate_vtt_codex.py` helper for checkpointed,
    one-episode-at-a-time batch translation
  - supports clean restarts with `prepare --force`, which resets stale session and diagnostics state
  - only allows draft seeding from an older English file when cue count and cue timings match exactly

Typical use:

```text
Use $subtitle-reflow on samples/episodes/<slug>/transcription/<stem>_ctc_words.json

Use $subtitle-translation on samples/episodes/<slug>/transcription/<stem>_reflow.vtt
```

The canonical tracked skill source now lives in `codex/skills/`.
Install or refresh the live Codex copy with:

```bash
python3 scripts/install_codex_skills.py
python3 scripts/install_codex_skills.py --skill subtitle-translation
```

See `docs/codex-skills.md` for the intended scope, defaults, and handoff between
the two skills.

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

- Current architecture: `docs/current-architecture.md`
- Older design notes: `docs/local-whisper-ocr-pipeline.md`
- Full script reference: `docs/scripts-reference.md`
- Codex skills: `docs/codex-skills.md`
