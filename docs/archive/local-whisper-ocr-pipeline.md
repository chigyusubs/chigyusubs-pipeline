# Design Document: Local Whisper + OCR Fusion Pipeline

## 0. Current Operational Notes (2026-03)

This design doc captures architecture and tradeoffs. For script-by-script usage, see:

- `docs/scripts-reference.md`

Current production path in this repo is:

1. `run_qwen_ocr_episode.py ocr`
2. `run_qwen_ocr_episode.py filter`
3. `condense_glossary_vertex.py`
4. `run_faster_whisper.py` (or `run_whisper_cpp.py`)

Important distinction:

- `run_qwen_ocr_episode.py filter` does **not** call Qwen.  
  It only post-processes existing OCR JSONL.
- Qwen is used in `run_qwen_ocr_episode.py ocr`, default `--model qwen3.5-9b` against your local llama.cpp OpenAI-compatible server.

## 1. Overview
This document outlines a fully local, offline transcription architecture for `chigyusubs`. It is designed to replace or supplement cloud-based multimodal models (like Gemini) by fusing raw audio transcription with visual on-screen text (OCR). 

The primary goal is to achieve flawless Japanese transcription—specifically capturing complex Kanji, character names, and on-screen jargon ("telops")—while maintaining the strict, word-level timestamp accuracy of a dedicated audio model.

## 2. Core Architectural Concept: Global OCR Glossary + Prompted ASR
Directly forcing OCR text onto Whisper timestamps causes alignment and hallucination issues.  
The current strategy is deliberately simpler:

1. Build one **global OCR-derived glossary** for the full episode.
2. Use that glossary as Whisper `initial_prompt` context.
3. Keep translation glossary generation as a separate, later step.

We do **not** use per-chunk OCR glossary injection in the main flow.

### Practical defaults

- Prefer `initial_prompt` over hotwords as the default guidance mechanism.
- Use hotwords only as an optional experimental/ablation path.
- Keep glossary generation two-fold:
  1. **ASR glossary**: Japanese terms only (names, places, show jargon) for Whisper.
  2. **Translation glossary**: `source,target,context` entries for translation prompts.

## 3. Hardware Budgeting (Target: 16GB VRAM GPU)

### Current Recommended Stack (Updated 2026-03-02)

| Component | Technology | VRAM Cost | Purpose |
| :--- | :--- | :--- | :--- |
| **Visual Extractor** | Qwen3.5-9B (llama.cpp + mmproj) | `~8.4 GB` | VLM-based OCR, far superior to traditional OCR for Japanese TV |
| **Smart Sorter** | Same Qwen3.5-9B or Vertex AI | `(shared)` | NLP extraction of proper nouns from OCR dump |
| **Audio Transcriber** | faster-whisper (large-v3) | `~4-5 GB` | GPU transcription with word-level timestamps |

*Models run sequentially (OCR first, then transcription), so VRAM is shared not stacked.*

### Why VLM over Traditional OCR?

Tested on Wednesday Downtown frames (2026-03-02):

| Metric | EasyOCR | Qwen3.5-9B Q6_K |
| :--- | :--- | :--- |
| Name cards | Missed most | `MC 浜田 雅功 (ダウンタウン)` |
| Captions | Wrong kana (ガ→が) | Full sentences correct |
| Noise | Garbage chars (四E, 卯E) | Clean output |
| Speed | Instant | ~2.3s/frame |

The VLM understands context — it reads complete telop text rather than character fragments.

**Setup:**
```bash
# Qwen3.5-9B with vision (unsloth GGUF has mmproj files)
llama-server -hf unsloth/Qwen3.5-9B-GGUF:Q6_K \
  --mmproj /path/to/mmproj-F16.gguf \
  --port 8080 --ctx-size 8192
```

### Previous Stack (Superseded)

| Component | Technology | VRAM Cost | Purpose |
| :--- | :--- | :--- | :--- |
| **Visual Extractor** | MangaOCR | `< 1 GB` | Fast but limited accuracy on stylized fonts |
| **Smart Sorter** | Llama.cpp (8B Text LLM) | `~6-8 GB` | NLP extraction of Proper Nouns from raw OCR dump |
| **Audio Transcriber** | Whisper.cpp (Large-v3) | `~4-5 GB` | High-accuracy transcription and timestamp generation |

---

## 4. Pipeline Stages

### Stage 1: Frame Extraction (Stable Sampling)
Default extraction is fixed-rate sampling (0.5 fps, one frame every 2s), because `mpdecimate` is unreliable on Japanese variety-show overlays and motion graphics.

```bash
ffmpeg -i input.mp4 -vf "fps=1/2" -qscale:v 2 frames/frame_%05d.jpg
```

### Stage 2: OCR Extraction (Qwen Vision)
Run frame OCR with local Qwen vision (`run_qwen_ocr_episode.py ocr`) and persist JSONL for resume-safe processing.

### Stage 3: Global Candidate Filtering
Run `run_qwen_ocr_episode.py filter` to dedupe and rank OCR terms into one global glossary candidate list.  
This is episode-level filtering, not per-chunk prompt building.

### Stage 4: Dual Glossary Outputs
Generate two outputs from the global candidates:

1. **ASR glossary (Japanese-only):**
   - compact list for Whisper `initial_prompt`
   - optional smaller hotwords file for experiments
2. **Translation glossary (source/target/context):**
   - bilingual/domain-aware entries for translation prompts

### Stage 5: ASR with Prompted Context
Run ASR on full audio/video pass and inject the Japanese glossary through `initial_prompt` by default.  
Hotwords are secondary and not the default path.

## 5. Integration into `chigyusubs`
This pipeline is currently script-first and file-based, not a bundled app provider yet.

1. **Episode workspace**: each episode lives under `samples/episodes/<episode_slug>/...`.
2. **OCR/ASR execution**: Python scripts orchestrate `ffmpeg`, llama.cpp (Qwen vision), and Whisper runners.
3. **Hand-off to app**: finalized VTT and translation glossary outputs can be consumed by the main `chigyusubs` translation flow.

## 6. Known Limitations & Mitigation
* **Prompt token budget:** Large prompts can reduce ASR stability. *Mitigation: keep ASR glossary compact and Japanese-only.*
* **Hotwords vs prompt variance:** Hotwords can help in some names but are less stable as a global default. *Mitigation: default to `initial_prompt`; test hotwords as an explicit ablation.*
* **Translation needs richer context:** ASR prompt terms are not enough for downstream translation quality. *Mitigation: maintain a separate `source,target,context` translation glossary.*

## 7. Empirical Findings (2026-03)

1. **Qwen OCR + fixed-rate sampling works well**:
   - 0.5 fps extraction was stable.
   - `mpdecimate` was not reliable for Japanese variety overlays.
2. **Dumb OCR first, cleanup later**:
   - Better recall than trying to force semantic extraction in the OCR call itself.
3. **ASR default**:
   - `faster-whisper large-v3` remains the strongest baseline for this workflow.
4. **Prompting strategy**:
   - `initial_prompt` is the better default than full hotwords injection.
   - Hotwords are best treated as targeted experiments.
5. **Voice isolation/enhancement**:
   - Mostly a side-grade across full episodes.
   - Can help some overlap-heavy regions (intro applause), but may introduce artifacts and increase hallucination risk.
   - Not recommended as default full-episode preprocessing.
6. **Segmentation/reflow**:
   - Native Whisper segmentation outperformed LLM reflow for this content.

## 8. Evaluated and Rejected Approaches

### LLM Reflow of Word-Level Timestamps (Rejected)
**Idea:** Use faster-whisper's word-level timestamps, then have an LLM group words into semantically correct subtitle cues.

**Result:** Native faster-whisper segment output is superior. The LLM produced 43% more fragmented cues, 42% short cues (<5 chars), and 234 mid-sentence splits on particles. 14/116 chunks failed entirely. The acoustic signal that faster-whisper uses for segmentation is better than pure text-based grouping.

**Script:** `scripts/reflow_subtitles_local.py` (preserved for reference)

### Qwen3.5-9B via transformers + bitsandbytes (Rejected for now)
**Idea:** Run the HF model directly with NF4 quantization on ROCm.

**Result:** OOMs on 16GB VRAM during model loading (BF16 weights + quantized copy coexist briefly). The Gated DeltaNet architecture also needs `flash-linear-attention` for optimal performance. GGUF via llama.cpp (Vulkan) is the practical path for now.
