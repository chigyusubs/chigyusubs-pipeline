# Design Document: Local Whisper + OCR Fusion Pipeline

## 1. Overview
This document outlines a fully local, offline transcription architecture for `chigyusubs`. It is designed to replace or supplement cloud-based multimodal models (like Gemini) by fusing raw audio transcription with visual on-screen text (OCR). 

The primary goal is to achieve flawless Japanese transcription—specifically capturing complex Kanji, character names, and on-screen jargon ("telops")—while maintaining the strict, word-level timestamp accuracy of a dedicated audio model.

## 2. Core Architectural Concept: The "Two-Tier Prompt Fusion"
Directly merging OCR timestamps with Whisper timestamps introduces severe hallucination and alignment risks. Instead, this architecture solves the "Fusion Problem" by using visual text purely as a **dynamic vocabulary glossary** for Whisper's internal decoder.

We inject OCR data into Whisper via its `initial_prompt` parameter, split into two tiers:
1. **Global Glossary:** A curated list of proper nouns and jargon extracted from the entire video (solves the "Name Tag Problem").
2. **Local Chunk Context:** Cleaned OCR text extracted specifically from the current 30-second audio chunk being processed (solves contextual ambiguity).

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

### Stage 1: Smart Frame Extraction
Japanese video editing relies heavily on static text overlays and "typewriter" animations. Extracting frames blindly at 1fps wastes compute.

**Mechanism:** Use FFmpeg's `mpdecimate` to only capture frames when the screen changes, throttled to a maximum of 2fps to avoid capturing micro-animations.
```bash
ffmpeg -i input.mp4 -vf "fps=2,mpdecimate" -vsync vfr frames/frame_%04d.jpg
```
*Result:* A highly optimized folder of frames containing only the visual delta points.

### Stage 2: OCR Extraction (MangaOCR)
Run the extracted frames through MangaOCR. Track the timestamp of each frame to map it to the standard audio chunks (e.g., 30s intervals) used by the existing `chigyusubs` chunker.

### Stage 3: The Global Glossary (Text LLM)
Collect *all* OCR text from the entire video and pass it to a local text-based LLM (e.g., Llama-3-8B-Instruct or Qwen-2-7B) via `llama.cpp`.

**Prompt Strategy:**
> "Here is raw OCR text from a Japanese video. Extract character names, locations, and unique jargon. Output ONLY a comma-separated list of the 50 most important terms. Ignore UI elements and common verbs."

*Result:* `綾小路, 渋谷, 呪術廻戦, スプラトゥーン3` (The Global Glossary).

### Stage 4: Local Chunk Filtering (Substring Collapsing)
To handle typewriter text animations (e.g., frame 1: `ス`, frame 2: `スーパ`, frame 3: `スーパーマリオ`), we apply a temporal Substring Collapsing algorithm to the OCR text within each 30s chunk.

**Algorithm:**
1. Filter out stray single Kana characters (likely animation artifacts).
2. Sort strings by length (descending).
3. Discard any string that is a direct substring of a longer saved string within the same time window.

### Stage 5: Whisper Transcription Fusion
For each 30-second audio chunk, combine the Global Glossary and the Cleaned Local Chunk Text into a single comma-separated string (staying under Whisper's ~224 token limit).

Pass this string into Whisper's `--prompt` argument.

**Example execution:**
```bash
./main -m models/ggml-large-v3.bin -f chunk_01.wav -l ja 
  --prompt "綾小路, 呪術廻戦, メニュー, ラーメン, 美味しい"
```

## 5. Integration into `chigyusubs`
This pipeline can be integrated into the existing architecture as a new Provider within `src/lib/providers/`.

1. **`LocalFusionProvider`**: Implements the standard `TranscriptionProvider` interface.
2. **Dependencies**: Requires a local Python/Node microservice to orchestrate `ffmpeg`, `MangaOCR`, `llama.cpp`, and `whisper.cpp`, exposing a unified API endpoint to the Vite frontend.
3. **Chunking**: The frontend's `lib/chunker.ts` handles the audio splitting; the local backend handles the frame-to-chunk timestamp alignment.

## 6. Known Limitations & Mitigation
* **Prompt Token Limit:** Whisper ignores prompt tokens beyond ~224. *Mitigation: The LLM step ensures only high-signal nouns are retained.*
* **Translation:** This pipeline handles *transcription*. Translation of the resulting high-quality JSON/VTT must be handled by a secondary LLM pass (as currently structured in `chigyusubs`), referencing the same OCR context for nuance.

## 7. Evaluated and Rejected Approaches

### LLM Reflow of Word-Level Timestamps (Rejected)
**Idea:** Use faster-whisper's word-level timestamps, then have an LLM group words into semantically correct subtitle cues.

**Result:** Native faster-whisper segment output is superior. The LLM produced 43% more fragmented cues, 42% short cues (<5 chars), and 234 mid-sentence splits on particles. 14/116 chunks failed entirely. The acoustic signal that faster-whisper uses for segmentation is better than pure text-based grouping.

**Script:** `pipeline/scripts/reflow_subtitles_local.py` (preserved for reference)

### Qwen3.5-9B via transformers + bitsandbytes (Rejected for now)
**Idea:** Run the HF model directly with NF4 quantization on ROCm.

**Result:** OOMs on 16GB VRAM during model loading (BF16 weights + quantized copy coexist briefly). The Gated DeltaNet architecture also needs `flash-linear-attention` for optimal performance. GGUF via llama.cpp (Vulkan) is the practical path for now.
