# Experiment Notes: ReazonSpeech NeMo-v2 on AMD ROCm

## 1. Overview and Motivation

As part of optimizing the `chigyusubs` pipeline for Japanese variety shows, we experimented with replacing `faster-whisper` (Kotoba) with **ReazonSpeech-NeMo-v2** (`reazon-research/reazonspeech-nemo-v2`).

### Why consider it?
1. **Zero Hallucinations:** Unlike Whisper (Encoder-Decoder), ReazonSpeech-NeMo-v2 uses an **RNN-T (Fast Conformer)** architecture. It only outputs tokens when speech is present, completely eliminating the "hallucination loops" Whisper suffers from during long intro BGM or crowd applause.
2. **Domain-Specific Training:** Trained on 35,000 hours of Japanese TV and broadcast data (ReazonSpeech v2.0 corpus), making it highly tuned for the specific cadence and slang of variety shows.
3. **Speed:** Fast Conformer models are typically much faster than Transformer-based models of similar size.

## 2. Environment Setup

We utilized the existing isolated NeMo environment (`.venv-nemo`) which already had ROCm PyTorch and `nemo_toolkit[asr]` installed.

To get the official Reazon wrapper:
```bash
uv pip install --python .venv-nemo "git+https://github.com/reazon-research/ReazonSpeech.git#egg=reazonspeech-nemo-asr&subdirectory=pkg/nemo-asr"
```

## 3. ROCm Instability and Workarounds

Running NeMo ASR models on AMD ROCm proved to be **highly unstable**, requiring a deep diagnostic separation between Python-level bugs in NeMo's decoding math and low-level AMD driver crashes.

### Issue 1: OOM on Long Audio
Feeding a full 45-minute `.wav` file into the model caused PyTorch CUDA OOM errors during the ConvSubsampling pre-encode phase.
* **Fix:** We wrote a wrapper script (`scripts/run_reazonspeech_nemo_chunked.py`) to split the audio into chunks using `librosa.effects.split(y, top_db=30)` and processed them sequentially.

### Issue 2: The `ALSD` IndexError (A Python Bug)
When attempting to use `alsd` (Align-Length-Sync Decoding) or standard `beam` search, the pipeline frequently crashed with:
`IndexError: list assignment index out of range` inside `align_length_sync_decoding`.
* **Diagnosis:** This is **not a ROCm problem**. It is a pure Python index bookkeeping bug within NeMo. If an audio chunk is very short, the internal `beam_state` list dynamically shrinks due to early termination/pruning, but the `sub_batch_ids` list still holds indices pointing to the original batch size, causing an out-of-bounds write.
* **Fix:** Force `batch_size=1` and strictly filter out chunks under 0.3 seconds (`librosa.effects.split` -> `min_samples = int(0.3 * sr)`).

### Issue 3: `HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION` (The AMD Driver Crash)
When we attempted to batch process chunks (e.g., `batch_size=8`), we hit catastrophic GPU core dumps and **severe garbling/visual artifacting** on the display.
* **Diagnosis:** ROCm fundamentally struggles with the highly dynamic, varying-length tensor padding required to batch RNN-T chunks. The driver throws a memory access fault when a vectorized element-wise kernel attempts to read out of bounds.
* **Fix:** Disable SDMA (`export HSA_ENABLE_SDMA=0`), tune the allocator (`export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:128`), and strictly use `batch_size=1`.

### Issue 4: The Final Nail - `gatherTopK` Hardware Exception
Thinking we had isolated the Python bugs (by using `batch_size=1`) and the memory faults (via `SDMA=0`), we attempted a "Golden Run" using `strategy: alsd` on the full episode. It immediately hard-crashed the GPU:
```text
Kernel Name: _ZN2at6native6sbtopk10gatherTopKIfjLi2ELb0EEEvNS...
Callback: Queue aborting with error : HSA_STATUS_ERROR_EXCEPTION: An HSAIL operation resulted in a hardware exception. code: 0x1016
```
* **Diagnosis:** The `gatherTopK` PyTorch kernel (which is the core operation of any Beam Search where the model must pick the top `N` highest probability tokens) is fundamentally broken or throws hardware-level exceptions on this specific ROCm/GPU architecture when executing NeMo's specific tensor shapes.

### The Only Working Configuration (Greedy)
To achieve a flawless, complete transcription run of a 45-minute episode without any crashes or visual artifacts, we had to abandon beam search entirely and run:
1.  **Decoding Strategy:** `greedy` (`max_symbols_per_step: 10`)
2.  **Batch Size:** `1`
3.  **ROCm Environment:** `export HSA_ENABLE_SDMA=0` and `export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:128`

## 4. Pipeline Limitations

Forcing the model into `greedy` decoding means the **beam size is effectively 1**. The model simply picks the single most likely token at every time step and never looks back. 

While stable on ROCm, this comes with a severe penalty to transcription quality for Japanese:
* **Kanji Selection (Homophones):** Without a beam to hold multiple sentence paths open, the model cannot use late-sentence context to correct early-sentence Kanji choices (e.g., guessing 期間 vs 機関).
* **Dropped Particles:** If the model's confidence drops for a single frame, it outputs a "blank" token, dropping crucial small particles (は, が, に) that a beam search would have smoothed over.

Additionally, the RNN-T architecture introduces two major workflow regressions for the `chigyusubs` pipeline:

### 1. No Initial Prompts / OCR Glossary Support
Whisper allows passing an `initial_prompt` to bias the decoder toward specific Kanji (crucial for OCR-derived names). **RNN-T models do not support textual prompts.** They strictly decode acoustic frames.
* To achieve hotword biasing in ReazonSpeech, you must use the ONNX version (`ReazonSpeech-k2-v2`) with `sherpa-onnx` and modified beam search, which completely abandons the NeMo ecosystem.

### 2. Timestamp Extraction Complexity
The native NeMo `model.transcribe()` returns raw hypothesis text strings for the chunk. To get segment/word-level timestamps, you either have to:
* Use the raw NeMo API and parse the `Hypothesis.timestamp` tensor (which requires mapping subword BPE tokens back to timeframes).
* Use the `reazonspeech.nemo.asr.transcribe` wrapper, which returns nice `Segment` objects, but does not support batch processing of multiple files at once.

## 5. Conclusion & Verdict

**Verdict: Do not use for transcription on AMD ROCm.**

While the theory of using a hallucination-proof, TV-trained RNN-T model is excellent, the reality of running NeMo's inference engine on ROCm is plagued by low-level kernel crashes, core dumps, and strict memory limits.

Furthermore, losing the ability to inject the global OCR glossary via `initial_prompt` degrades the spelling accuracy of proper nouns, which was the primary goal of the local OCR pipeline.

### Recommended Path Forward
* Keep **NeMo** strictly for **VAD and Speaker Diarization** (via `run_nemo_diarized.py`), as its MSDD (Multi-Scale Diarization Decoder) remains unparalleled for overlapping Japanese speech.
* Keep **faster-whisper (large-v3 or Kotoba)** for **Transcription**. It is stable on ROCm, supports batched/chunked processing flawlessly, and natively accepts the OCR glossary prompt.
* To solve Whisper's hallucination on BGM, pre-process the audio using **Voice Isolation (BS-RoFormer / Demucs)** or strictly adhere to the NeMo VAD timestamps to drop silent/BGM regions before feeding them to Whisper.