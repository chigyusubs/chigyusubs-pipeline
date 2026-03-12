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
8. `translate_vtt_api.py` — unattended LLM translation to English (Vertex/OpenAI-compatible)

`transcribe_pipeline.py` can now consume the saved VAD/chunk/OCR artifacts instead of recomputing them.

### Path B: Local (faster-whisper)

Fully local, no API calls. Lower quality but zero cost.

1. `run_qwen_ocr_episode.py ocr` + `spans` + `context`
2. `run_vad_episode.py` + `build_vad_chunks.py`
3. `transcribe_local.py` — local ASR path
4. `translate_vtt_api.py`

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

samples/experiments/<pack>/
  prompts/      # manual experiment system/user prompts
  scenes/       # extracted clip media and notes
  results/      # paste-back result templates
  manifest.json # experiment metadata and model matrix
```

## Script Catalog

### Setup

| Script | Purpose | Status |
|---|---|---|
| `init_episode_from_media.py` | Create episode workspace from media, optionally extracting fixed-rate frames | Maintained |
| `install_codex_skills.py` | Install Codex skills from the repo to `~/.codex/skills/` | Maintained |

### OCR & Glossary

| Script | Purpose | Status |
|---|---|---|
| `run_qwen_ocr_episode.py` | Qwen-VL frame OCR (`ocr`), span merging (`spans`), context derivation (`context`), glossary filtering (`filter`) | Maintained |
| `classify_ocr_spans_local.py` | Local Gemma OCR-span cleanup/classification into reusable chunk context | Maintained |
| `condense_glossary_vertex.py` | Build structured glossary with Gemini on Vertex | Maintained |
| `condense_glossary_qwen.py` | Build structured glossary with local Qwen | Maintained |
| `condense_glossary_llm.py` | Older local LLM condensation via OpenAI-compatible endpoint | Legacy |
| `clean_candidates.py` | Clean/deduplicate OCR candidate text for glossary scripts | Utility |

### Audio Preprocessing

| Script | Purpose | Status |
|---|---|---|
| `run_vad_episode.py` | Silero VAD segmentation — reusable artifact | Maintained |
| `build_vad_chunks.py` | Build chunk boundaries from saved VAD segments | Maintained |

### Transcription

| Script | Purpose | Status |
|---|---|---|
| `transcribe_gemini_video.py` | Gemini video-only transcription (sends compressed video inline, no OCR needed) | Maintained |
| `transcribe_gemini_raw.py` | Gemini audio transcription with chunk-wise local OCR context | Maintained |
| `transcribe_gemini.py` | Gemini transcription (JSON schema mode, used by `transcribe_pipeline.py`) | Maintained |
| `transcribe_pipeline.py` | Integrated pipeline: VAD → Gemini → alignment → reflow | Maintained |
| `transcribe_local.py` | Fully local pipeline: Silero VAD + OCR filter + faster-whisper | Maintained |

`transcribe_gemini_video.py` supports `--spoken-only` for experiments that ask Gemini to return spoken dialogue only and not emit `[画面: ...]` visual cue lines.

`transcribe_gemini_video.py` also supports cost/token inspection:
- `--preview-cost` encodes the real inline video chunks and calls Gemini `count_tokens` without generating a transcript. This gives an exact input-token preview for the request payload and writes a `*_gemini_cost_preview.json` artifact.
- `--count-tokens` records per-chunk prompt-token previews during a real transcription run.
- streamed Gemini responses are now saved with `usage_metadata` when available, so post-run metadata can include actual prompt/output token counts and a cost summary.
- `--media-resolution {unspecified,low,medium,high}` lets Gemini choose a different multimodal processing resolution for the same inline media payload.
- `--thinking-level {unspecified,minimal,low,medium,high}` and `--thinking-budget` expose Gemini thinking controls for transcription experiments.

Important limitation:
- Gemini API `count_tokens` does not currently accept generation-config overrides such as `media_resolution` or thinking settings. Exact preflight counts for those configurations require Vertex AI, or a real generation request followed by inspection of `usage_metadata.prompt_token_count`.

### ASR Backends & Quality

| Script | Purpose | Status |
|---|---|---|
| `run_faster_whisper.py` | faster-whisper ASR (ROCm/CUDA), initial prompt + hotwords | Maintained |
| `run_whisper_cpp.py` | whisper.cpp ASR via `whisper-cli` | Maintained |
| `run_local_whisper.py` | OpenAI Whisper Python package ASR | Maintained |
| `pre_reflow_second_opinion.py` | Always-on faster-whisper second-opinion coverage check with VAD cross-reference and optional raw-chunk omission classification | Maintained |
| `compare_transcript_coverage.py` | Time-local coverage-gap comparison between primary and secondary transcripts, with optional VAD confirmation | Maintained |
| `report_raw_chunk_omissions.py` | Classify omissions from `*_gemini_raw.json` as `visual_substituted_narration`, `missing_narration_high_confidence`, or `compressed_vs_missing_unclear` | Maintained |

### Alignment

| Script | Purpose | Status |
|---|---|---|
| `align_ctc.py` | CTC forced alignment using `NTQAI/wav2vec2-large-japanese`. Default alignment path. 0.3% zero-duration words vs 13.4% with stable-ts. Writes `.diagnostics.json` sidecar with per-chunk repair details and visual-substitution risk flags. | Maintained |
| `align_chunkwise.py` | Chunk-wise stable-ts forced alignment | Legacy |
| `align_stable_ts.py` | Global stable-ts alignment (single pass) | Legacy |
| `align_qwen_forced.py` | Qwen forced-alignment benchmark (GGUF backend) | Archived |
| `align_qwen_forced_hf.py` | Qwen forced-alignment benchmark (HF backend) | Archived |

### Reflow & Repair

| Script | Purpose | Status |
|---|---|---|
| `reflow_words.py` | Reflow word/line timestamps into subtitle cues. `--line-level` (default for CTC) treats lines as atomic. Includes comma-fallback splitting, sparse-cue clamping, micro-cue merging, and a cap on backward cue expansion so subtitles do not appear far before the first aligned speech. | Maintained |
| `repair_vtt_codex.py` | Codex-interactive reflow repair with region detection, session/checkpoint, alignment and turn-boundary advisory context, and deterministic before/after diagnostics | Maintained |
| `repair_vtt_local.py` | Local Gemma cue-boundary repair. Alternative to the Codex-interactive path. | Alternative |

### Translation

| Script | Purpose | Status |
|---|---|---|
| `translate_vtt_codex.py` | Codex-interactive translation with session/checkpoint, batch-tier auto-fallback (84→60→48), source hash validation, seed draft import, alignment/turn advisory context, and CPS diagnostics | Maintained |
| `translate_vtt_api.py` | Unattended LLM translation (Vertex Gemini or any OpenAI-compatible API). For benchmarking, testing model capability, or use without Codex. | Maintained |

### Local LLM Servers

| Script | Purpose | Status |
|---|---|---|
| `start_qwen_ocr_server.sh` | llama-server for Qwen-VL OCR | Maintained |
| `start_gemma_ocr_filter_server.sh` | llama-server for Gemma OCR span classification | Maintained |
| `start_gemma_cue_repair_server.sh` | llama-server for Gemma cue-boundary repair | Alternative |
| `start_qwen_cue_repair_server.sh` | llama-server for Qwen cue-boundary repair | Alternative |

### Utilities

| Script | Purpose | Status |
|---|---|---|
| `extract_visual_cues.py` | Extract visual cue text from Gemini raw transcripts | Utility |
| `format_vtt_netflix.py` | Netflix-style VTT formatting | Utility |
| `json_to_smart_vtt.py` | Word-timestamp JSON to pause-split VTT (no LLM) | Utility |
| `restore_speaker_turns.py` | Re-insert speaker turn markers into reflowed VTT | Utility |
| `restore_speaker_turns_ocr.py` | Speaker turn restoration using OCR data | Experimental |
| `reflow_subtitles_vertex.py` | LLM-based cue regrouping using Vertex Gemini | Experimental |
| `reflow_subtitles_local.py` | LLM-based cue regrouping using local llama.cpp | Experimental |
| `run_vertex.py` | Generic CLI wrapper for Vertex Gemini text generation | Utility |

### Experiments (`scripts/experiments/`)

| Script | Purpose | Status |
|---|---|---|
| `build_whisper_glossary.py` | Early glossary-building experiment | Archived |
| `extract_glossary.py` | Early glossary extraction experiment | Archived |
| `run_reazonspeech_nemo.py` | ReazonSpeech NeMo ASR experiment | Archived |
| `run_reazonspeech_nemo_chunked.py` | ReazonSpeech NeMo chunked ASR experiment | Archived |
| `test_easyocr.py` | EasyOCR evaluation | Archived |
| `test_faster_whisper.py` | faster-whisper evaluation | Archived |
| `test_paddleocr.py` | PaddleOCR evaluation | Archived |
| `test_qwen_json_schema.py` | Qwen JSON schema mode test | Archived |
| `test_qwen_ocr.py` | Qwen OCR evaluation | Archived |
| `test_reflow_quality.py` | Reflow quality metrics experiment | Archived |
| `test_vibevoice_4bit.py` | VibeVoice 4-bit quantization test | Archived |
| `test_vibevoice_load.py` | VibeVoice model loading test | Archived |
| `test_vlm_extraction.py` | VLM text extraction evaluation | Archived |
| `transcribe_nemo_vibevoice.py` | NeMo + VibeVoice transcription experiment | Archived |
| `prepare_ai_studio_experiment_pack.py` | Build a manual AI Studio pack with fixed clips, prompts, scene notes, and result templates under `samples/experiments/<pack>/` | Experimental |

### Experiments (`scripts/experiments/kotoba_test/`)

| Script | Purpose | Status |
|---|---|---|
| `run_kotoba.py` | Kotoba Whisper ASR experiment | Archived |
| `run_kotoba_clean.py` | Kotoba Whisper with clean audio | Archived |
| `run_kotoba_diarized.py` | Kotoba Whisper with diarization | Archived |
| `run_nemo_diarized.py` | NeMo diarized ASR experiment | Archived |

## CLI Cheatsheet

### Manual AI Studio Pack

```bash
python3 scripts/experiments/prepare_ai_studio_experiment_pack.py \
  --episode-dir samples/episodes/<slug> \
  --scene-spec samples/experiments/specs/<pack>.json \
  --pack-dir samples/experiments/<pack> \
  --force
```

This prepares:

- extracted `video.mp4` and `audio.mp3` clips per scene
- `prompts/` with separate system and user prompts, including `ocr_only`
- `results/` templates with settings metadata headers ready for pasted outputs

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
# Those metadata-emitting scripts also mirror a run record under
# `samples/episodes/<slug>/logs/runs/<run_id>/`.
# Each run mirror now includes:
# - `run.json` for machine-readable metadata
# - `README.md` with a top metadata comment block and a short human summary

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
# 0.3% zero-duration words vs 13.4% with stable-ts on oni_no_dokkiri_de_namida_ep2.
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
# passes VAD segments to the coverage comparison. When a sibling
# <stem>_gemini_raw.json exists, it also writes a raw-chunk omission report that
# compares Gemini spoken lines, Gemini visual lines, and Whisper speech.
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
# oni_no_dokkiri_de_namida_ep2 example from the current CTC + reflow path
python scripts/translate_vtt_api.py \
  --input samples/episodes/oni_no_dokkiri_de_namida_ep2/transcription/oni_no_dokkiri_de_namida_ep2_ctc_reflow.vtt \
  --output samples/episodes/oni_no_dokkiri_de_namida_ep2/translation/oni_no_dokkiri_de_namida_ep2_ctc_reflow_en.vtt \
  --batch-cues 12 \
  --batch-seconds 45

# great_escape1 example from the current reflow path
python scripts/translate_vtt_api.py \
  --input samples/episodes/great_escape1/transcription/ge1_reflow.vtt \
  --output samples/episodes/great_escape1/translation/ge1_reflow_en.vtt \
  --batch-cues 12 \
  --batch-seconds 45

# `translate_vtt_api.py` translates in local cue batches, preserves cue timings,
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

# Mistral via the OpenAI-compatible backend
python scripts/translate_vtt_api.py --backend openai \
  --url https://api.mistral.ai --api-key $MISTRAL_API_KEY --model mistral-small-latest \
  --input samples/episodes/oni_no_dokkiri_de_namida_ep2/transcription/oni_no_dokkiri_de_namida_ep2_ctc_reflow.vtt \
  --output samples/episodes/oni_no_dokkiri_de_namida_ep2/translation/oni_no_dokkiri_de_namida_ep2_ctc_reflow_en_mistral.vtt
```

### Local Pipeline

```bash
python scripts/transcribe_local.py \
  --video samples/episodes/<slug>/source/video.mp4 \
  --ocr-jsonl samples/episodes/<slug>/ocr/qwen_ocr_results.jsonl \
  --llm-url http://127.0.0.1:8080 \
  --llm-model gemma3-27b
```
