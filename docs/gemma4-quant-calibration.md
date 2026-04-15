# Gemma 4 Quantization + Calibration Sketch

Rough plan for quantizing `google/gemma-4-E4B-it` (and E2B) for vLLM serving
on the RX 9070 XT (16 GB) while keeping the audio/vision towers intact.

Status: **sketch**, not started. Circle back after the vLLM harness experiments
(see `project_gemma4_quant_calibration.md` memory) decide whether the vLLM path
is worth pursuing.

## Goal

Fit E4B + KV cache + multimodal input into 16 GB with headroom, so the 4B
audio-only variant (the one that actually matches Gemini coverage — see
`asr-comparison-2026-04-10.md`) can run under vLLM with prefix caching,
continuous batching, and structured outputs.

Current footprint (bf16, E2B, vLLM):
- Model weights: 10.56 GiB
- KV cache: 1.99 GiB
- Activations + multimodal input: rest of the 16 GB budget, tight

E4B in bf16 will not fit. Target: **INT8 W8A16** first (halves the weights,
leaves activations bf16), then **AWQ INT4** if we need more headroom for
bigger KV or video frames.

## What to quantize, what to skip

**Skip (ignore patterns for llmcompressor):**
- `re:.*audio_encoder.*` — the gemma4_audio encoder. Small, sensitive,
  quantization errors here hit transcription quality directly.
- `re:.*vision_tower.*` — same story for telop OCR.
- `re:.*multi_modal_projector.*` — the projectors between towers and the
  language model. Tiny, no point quantizing, and errors cascade.
- `lm_head` — standard llmcompressor default.

**Quantize:** everything else in the language model (attention + MLP
projections across all decoder layers).

## Formats to try, in order

1. **INT8 W8A16 (compressed-tensors)** — weights INT8, activations bf16.
   Safest, usually no measurable quality loss. Halves weight memory.
   Recipe: `GPTQModifier(targets=["Linear"], scheme="W8A16", ignore=[...])`.

2. **AWQ INT4 (compressed-tensors)** — weights INT4 group-quantized,
   activations bf16. More headroom, some quality cost. Well-supported path
   in vLLM. Needs calibration data; quality depends on it.

3. **FP8 (dynamic or static)** — ROCm 7.2 supports FP8 on gfx1201 via
   `torch.float8_*`. vLLM has an `fp8` quant method. Dynamic FP8 needs no
   calibration at all — worth trying as a baseline. Static FP8 needs
   calibration for activation scales.

Skip W4A4 and anything more aggressive for now — diminishing returns and
risks breaking tower interactions.

## Calibration data

**The trap:** generic calibration sets (wikitext, c4, Pile slices) don't
exercise the audio path. For audio-conditioned quantization, the calibration
batches must contain the same shape of multimodal prompts the model will see
at inference.

**The idea:** use the pipeline's own output as calibration data.

We already have everything we need:
- `samples/episodes/<slug>/chunks/` — VAD-chunked WAVs, same length
  distribution as inference time
- `samples/episodes/<slug>/transcription/*_gemini_raw.json` — high-quality
  Japanese transcripts aligned to those chunks
- Production system prompt from the audio-augment skill

**Assembly recipe:**
1. Pick 5–10 episodes spanning different shows (Great Escape, Killah Kuts,
   etc.) for calibration.
2. **Hold out Killah Kuts S01E01** entirely — that's the benchmark episode
   in `asr-comparison-2026-04-10.md`. Using it for calibration would leak.
3. Sample 256–512 chunks across the calibration set, stratified by chunk
   length (short/medium/long bins) so calibration sees the full range.
4. For each sampled chunk, build a `messages=[{system prompt}, {user:
   audio + instruction}, {assistant: gemini transcript}]` record.
5. Pass these through the processor's chat template to get tokenized
   prompts + audio feature tensors + labels.
6. Feed into llmcompressor's `oneshot()` call as the calibration dataset.

**Why this beats generic text calibration:**
- Activations inside the LM decoder during audio-conditioned generation are
  a different distribution from plain text. Calibrating on text alone and
  then running audio prompts means the INT4 scales are fit to the wrong
  distribution — the biggest quality drop hits exactly the path we care
  about.
- Matches genre (variety show Japanese) which dominates the vocab we need
  the model to predict well.

**Anti-leak rule:** never put the eval episode (Killah Kuts S01E01) in
calibration. Never reuse the same chunks for calibration and eval.

## Script sketches

Both live in `scripts/experiments/` for now (move to `scripts/` only if
we commit to the path).

### `build_calibration_dataset.py`
- Walks `samples/episodes/<slug>/` for a configurable list of episode slugs
- For each, loads chunks JSON + gemini_raw transcript
- Joins on `chunk_idx`, drops chunks with empty/failed transcripts
- Stratified sample N total chunks by duration bucket
- Writes a `.arrow` or `.jsonl` with `{wav_path, system, user_text, target}`
  per row
- Separate rendering step because llmcompressor wants a torch `Dataset`
  with already-prepared multimodal inputs

### `quantize_gemma4.py`
- Loads `google/gemma-4-E4B-it` with `Gemma4ForConditionalGeneration`
- Loads calibration dataset, wraps in a `Dataset` that yields processor
  outputs (input_ids, pixel_values=None, input_features from audio, labels)
- Calls `llmcompressor.transformers.oneshot(model=..., recipe=recipe,
  dataset=..., num_calibration_samples=256)`
- Recipe: either `GPTQModifier(scheme="W8A16", ignore=[tower patterns])`
  or AWQ equivalent
- Saves to `/mnt/models/huggingface/gemma-4-E4B-it-int8-audio-calib/`
- vLLM loads it with `--quantization compressed-tensors`

## Why E4B specifically (updated 2026-04-11 after harness experiments)

Initial findings from the vLLM harness experiments
(`docs/vllm-gemma4-harness-findings.md`) tighten the case for E4B rather
than weakening it:

- **E2B's vision tower saturates at `max_soft_tokens=280`**. Pushing
  past that regresses OCR accuracy instead of improving it. The
  capacity ceiling is the LM, not the pixel budget — which is exactly
  what quantizing up from E2B to a calibrated E4B should fix.
- **E2B can't use priming + vision simultaneously without conflict**
  on some segments. On the session-1 clip the vocabulary prime and
  the vision features fought for attention in a way the 2B model
  couldn't resolve. The hypothesis that E4B resolves this cleanly is
  testable only if we can actually load E4B.
- **Names at 27% recall across every E2B config.** `しんいち` is
  garbled at 0% even when vision reads the chyron clearly. The
  planned OCR pre-pass partially fixes this by injecting exact
  strings into the prime, but the ASR decoder also needs enough
  capacity to latch onto those strings — again pointing at E4B.

Concrete eval criterion for the quantized E4B: re-run
`scripts/experiments/vllm_gemma4_harness/run_bench.py` against
`eval_specs/killah_kuts_s01e01.json`. Success = name recall ≥ 60%
(vs 27% on E2B) AND katakana recall holds at 100%. Latency target:
≤ 2× the E2B per-chunk cost.

## Experiment order

1. **Harness path proof-of-worth is in progress.** Baseline results
   (`scripts/experiments/vllm_gemma4_harness/results/killah_kuts_s01e01__baseline_20260411.json`)
   show priming + vision on E2B reaches 100% kata recall and 27% name
   recall against the baseline 71/18. Worth pursuing further before
   quantization work, but the E2B ceiling is visible and the harness
   won't keep improving without a bigger model.
2. **FP8 dynamic E4B** — quickest win if it works. No calibration needed.
   Confirms we can fit E4B at all before spending time on calibration.
3. **INT8 W8A16 E4B with generic calibration (first)** — smoke test the
   quantization pipeline end to end with a trivial dataset to shake out
   tower-skip bugs.
4. **INT8 W8A16 E4B with pipeline calibration** — the real run. Evaluate
   on Killah Kuts S01E01 against the bf16 E4B baseline from the ASR
   comparison (25,407 chars).
5. **AWQ INT4 E4B with pipeline calibration** — only if INT8 leaves us
   short on KV cache or multimodal headroom.

## Open questions

- Does llmcompressor's GPTQ path handle `Gemma4ForConditionalGeneration`
  cleanly, or does it need a custom data collator for the audio features?
  Check their model support matrix before committing.
- Does vLLM's `compressed-tensors` loader tolerate the tower-skip pattern
  (quantized LM, bf16 towers in the same checkpoint)? Expected yes — this
  is the standard MM pattern — but verify on a 1-layer toy run first.
- FP8 on gfx1201 — ROCm 7.2 says supported, but vLLM's `fp8` quant method
  may assume Hopper/MI300 code paths. Try and see.

## References

- `docs/asr-comparison-2026-04-10.md` — baseline numbers; E4B bf16 = 25,407
  chars on Killah Kuts S01E01
- `project_gemma4_quant_calibration.md` memory — entry point
- llmcompressor: https://github.com/vllm-project/llm-compressor
- compressed-tensors: https://github.com/neuralmagic/compressed-tensors
