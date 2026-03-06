# Experiment Notes: Decoupled Gemini Transcription Pipeline with Dynamic OCR Context

**Date:** March 2026
**Target Episode:** Wednesday Downtown 406 (46-minute episode)
**Goal:** Evaluate decoupled transcription (LLM for text, stable-ts for alignment) against traditional single-pass methods, and test the injection of chunk-specific OCR context.

## The Architecture
We tested a fully decoupled pipeline:
1. **Audio Splitting:** Silero VAD splits the 46-minute audio into ~3-minute chunks.
2. **Dynamic OCR Filtering (LLM):** For each chunk, we query pre-existing Qwen OCR JSONL data. We pass the raw on-screen text to `gemini-2.5-flash` to extract *only* the relevant proper nouns (comedian names, locations) for that specific time window.
3. **Raw Transcription (LLM):** We use `gemini-2.5-pro` (via Vertex AI) to transcribe the audio chunk. The prompt includes:
   - A global glossary (TSV).
   - The dynamically generated chunk-specific OCR terms.
   - Instructions to format speaker turns with hyphens (`- `) and include standard punctuation (`、`, `。`).
4. **Chunk-wise Alignment:** `stable-ts` (large-v3) force-aligns the raw text back to the audio *chunk-by-chunk* to prevent global time drift.
5. **Turn Restoration:** A custom script maps the hyphens back into the word-level JSON based on character indices.
6. **VTT Reflow:** Words are grouped into subtitle cues based on acoustic pauses.
7. **Translation (LLM):** `gemini-2.5-flash` translates the VTT in 5-minute chunks, utilizing the global glossary.

## Results & Findings

### 1. Dynamic OCR Context is Highly Effective
*   **Zero-Shot Accuracy:** Injecting LLM-filtered OCR terms directly into the prompt for specific chunks drastically improved the spelling of obscure names (e.g., comedy duos like *Asagaya Shimai*, *Kumamushi*, *Sanbyoshi*).
*   **Mitigating Hallucinated "Substitutions":** In early tests, if the OCR filter missed a word (e.g. dropping the duo name "Viking" but keeping "Kotoge"), the model would sometimes substitute a completely different term from the hint list (like "Electric Chair Game Tournament"). We fixed this by adding a strict prompt warning: *"Only use these terms if they MATCH THE AUDIO EXACTLY. Do not force them in."*

### 2. Punctuation & Hyphens Improve Reflowing
*   Instructing the model to return pure raw text with hyphens for speaker changes and standard Japanese punctuation resulted in significantly better VTT pacing. The `reflow_words.py` script was able to create shorter, snappier, and more natural subtitle blocks compared to unpunctuated text.

### 3. Chunk-wise Alignment Solves Drift
*   Traditional forced alignment on a 46-minute plain text file often drifts or fails at the end. 
*   By writing a script (`align_chunkwise.py`) to align the exact 3-minute text to the exact 3-minute audio slice, we completely eliminated alignment drift.

### 4. The Flash-Lite Hallucination Loop vs. 2.5-Pro Stability
*   **Issue:** In earlier experiments using `gemini-3.1-flash-lite-preview` for transcription, if the audio chunk contained mostly silence or end-credits music, the model would fall into an infinite autoregressive loop (e.g., repeating `- 待って。 待って、` 9,000 times). This generated 80,000+ characters for a 3-minute chunk, crashing the pipeline.
*   **Attempted Fixes (Flash-Lite):** We implemented a Python circuit breaker to truncate strings over a certain length, raised the temperature to `0.2`, added prompt warnings, and passed `[END_OF_AUDIO]` to the API `stop_sequences`. While this caught the loop, it still resulted in dropped context at the end of the video.
*   **The Ultimate Fix (Switch to Pro):** Switching the transcription model to `gemini-2.5-pro` completely eliminated the hallucination loop. The Pro model successfully recognized the end-credits panel discussion without getting confused by the silence or the OCR credits list. It generated natural text lengths without needing the aggressive stop-sequences.

## Conclusion
The decoupled approach (`gemini-2.5-pro` -> `stable-ts` -> `gemini-2.5-flash`) combined with **Dynamic LLM-Filtered OCR Context** produces the highest quality transcription and translation tested to date. It outperforms single-pass Whisper models in conversational nuance and zero-shot proper noun recognition, while avoiding the catastrophic hallucination loops common in smaller Flash-Lite models.
