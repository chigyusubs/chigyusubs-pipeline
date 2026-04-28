# vLLM + Gemma 4 ASR harness findings

Running log of the experiment in `scripts/experiments/vllm_gemma4_harness/`.
Started 2026-04-11. Keep this doc in sync whenever a bench run changes
the headline numbers or reveals a new failure mode.

Status: **early / exploratory**. The harness can exploit visual context
and oracle-style name hints, but the 2026-04-28 no-show-prime baseline
shows that faster-whisper large-v3 still beats unprimed E4B on the same
hard clip windows across Killah Kuts S01E01-S01E04. Nothing is ready to
ship into the pipeline yet.

## Goal

Find out whether a vLLM-hosted Gemma 4 harness — domain-primed prompts,
optional vision grounding, prefix caching, structured outputs — can
produce cleaner Japanese transcription than the current Whisper/Cohere
pre-pass on variety-show audio. The benchmark episode is Killah Kuts
S01E01 (same one the ASR comparison in `asr-comparison-2026-04-10.md`
uses — but we **exclude it from any future quantization calibration set
to avoid leakage**).

## Environment

- vLLM nightly ROCm docker (`vllm/vllm-openai-rocm:nightly`) + layered
  image with `vllm[audio]` and transformers 5.5.3. See
  `scripts/experiments/vllm_gemma4_harness/server/README.md`.
- Gemma 4 E2B-it bf16, ~12.5 GiB VRAM, fits on the 9070 XT comfortably.
- E4B does not fit in bf16. Our local FP8 dynamic quantization
  (`quantize_e4b_fp8.py` via llm-compressor) fits at ~12.7 GiB + KV
  cache with `--max-model-len 4096 --gpu-memory-utilization 0.96`.
  Checkpoint at `/mnt/models/huggingface/gemma-4-E4B-it-FP8-Dynamic-local/`.
- `TRITON_ATTN` backend required (heterogeneous head dims).
- `ROCR_VISIBLE_DEVICES=0` to hide the Raphael iGPU.

## Eval methodology

1. Pick N hard segments from the episode's CTC-aligned gold transcript
   (`killah_kuts_s01e01_video_only_v1_ctc_words.json`). Hard =
   contains katakana jargon or named entities, 1–4 s duration,
   de-duplicated by 5 s proximity so we don't pick the same scene
   twice. Scoring: `2 * unique_katakana + 3 * name_hits`.
2. Extract an 8 s window centered on each segment + audio WAV + frames
   at 1 fps × 480 p.
3. Run each config against each segment's WAV + frames through the
   vLLM OpenAI chat completions endpoint.
4. Score each output on three metrics:
   - **Katakana recall**: substring presence of each expected katakana
     term (strict, no fuzzy).
   - **Name recall**: substring presence of each expected name
     (strict — `しんいち → シイチ` counts as a miss, which is what we
     want).
   - **LCS ratio**: `difflib.SequenceMatcher` against the full gold
     segment text.

This is deliberately crude — meant for trend tracking across config
changes, not to replace a real WER/CER eval. If you want finer grain,
swap in `chigyusubs.transcript_comparison` or `jiwer`.

## Current configs

Defined in `scripts/experiments/vllm_gemma4_harness/configs.py`:

- **A_audio_base** — audio-only, no system prime. Baseline of a naive
  "ASR via Gemma 4" call.
- **B_audio_primed** — audio-only, prompt includes a domain description
  and a vocabulary list (boxing/MMA terms + cast names) concatenated
  into the user turn. Prime is assembled from the kinds of inputs the
  existing pipeline produces (glossary, OCR, per-show domain kit).
- **C_vision_primed_1fps280** — same prime as B + 1 fps × 8 s frames
  at `max_soft_tokens=280` (the default image path, full-detail
  vision). Frames grounded against the dialogue prime.
- **D_audio_sysrole_primed** — same domain + vocab + cast as B but
  moved to a dedicated `system` role message, with "reference-only,
  do NOT output unless spoken" framing. User turn carries only the
  short transcription instruction.
- **E_vision_sysrole_primed_1fps280** — D plus the same 1 fps × 280
  vision frames as C.
- **F_audio_sysrole_guided_json** — D plus vLLM `guided_json` with
  schema `{dialogue: string}` and `additionalProperties: false`.
  Tests whether structurally locking the output shape kills the
  cast-list recitation.
- **G_vision_sysrole_guided_json_1fps280** — F plus vision frames.
- **H_audio_sysrole_oracle_names** — simulates a perfect OCR pre-pass:
  D's system prime, but with the specific cast names actually spoken
  in each segment (taken from CTC gold) injected as "Cast names
  spoken in THIS SPECIFIC clip (verbatim, do not transliterate):
  ...". This is a cheating upper bound for how well E2B's audio
  tower *can* do on names when told which ones to listen for. If H
  beats D, the OCR pre-pass plan is viable; if not, the audio
  tower itself is the ceiling.
- **I_vision_sysrole_oracle_names_1fps280** — H plus vision frames.
- **Q_vision_base_1fps280** — no system prime and no glossary/cast
  text; same base prompt as A, plus 1 fps frames at `mst=280`.
- **R_video_base** — no system prime and no glossary/cast text; same
  base prompt as A, plus `video_url` input.
- **S_vision_base_1fps560** — no system prime and no glossary/cast
  text; same base prompt as A, plus 1 fps frames at `mst=560`.

## Baseline results

Committed baseline: `results/killah_kuts_s01e01__baseline_20260411.json`,
produced against `eval_specs/killah_kuts_s01e01.json` (8 auto-picked
segments from CTC gold).

| Config | Kata recall | Name recall | Mean LCS | Mean wall/chunk |
|---|---|---|---|---|
| A_audio_base | 5/7 (71%) | 2/11 (18%) | 0.437 | ~0.45 s |
| B_audio_primed | **7/7 (100%)** | 3/11 (27%) | 0.426 | ~0.44 s |
| C_vision_primed_1fps280 | **7/7 (100%)** | 3/11 (27%) | 0.459 | ~0.85 s |
| D_audio_sysrole_primed (2026-04-11) | **7/7 (100%)** | 6/11 (55%) | 0.512 | ~0.43 s |
| E_vision_sysrole_primed_1fps280 (2026-04-11) | **7/7 (100%)** | 6/11 (55%) | **0.590** | ~0.87 s |
| F_audio_sysrole_guided_json (2026-04-11) | **7/7 (100%)** | **7/11 (64%)** | 0.548 | ~0.57 s |
| G_vision_sysrole_guided_json_1fps280 (2026-04-11) | **7/7 (100%)** | 6/11 (55%) | 0.565 | ~0.56 s |
| H_audio_sysrole_oracle_names (2026-04-11) | **7/7 (100%)** | **9/11 (82%)** | 0.576 | ~0.39 s |
| I_vision_sysrole_oracle_names_1fps280 (2026-04-11) | **7/7 (100%)** | 7/11 (64%) | 0.575 | ~0.94 s |
| **E4B FP8 local** A_audio_base (2026-04-12) | 5/7 (71%) | 2/11 (18%) | 0.449 | ~0.64 s |
| **E4B FP8 local** D_audio_sysrole_primed (2026-04-12) | 6/7 (86%) | 6/11 (55%) | **0.570** | ~0.59 s |
| **E4B FP8 local** H_audio_sysrole_oracle_names (2026-04-12) | 5/7 (71%) | **10/11 (91%)** | 0.522 | ~0.70 s |
| **E4B FP8 local** E_vision_sysrole_primed (2026-04-12) | 6/7 (86%) | 6/11 (55%) | 0.553 | ~1.28 s |
| **E4B FP8 local** I_vision_sysrole_oracle_names (2026-04-12) | 6/7 (86%) | **10/11 (91%)** | **0.610** | ~0.80 s |
| **E4B bf16** D_audio_sysrole_primed (2026-04-14) | **7/7 (100%)** | 6/11 (55%) | 0.585 | ~0.5 s |
| **E4B bf16** H_audio_sysrole_oracle_names (2026-04-14) | 5/7 (71%) | 9/11 (82%) | 0.504 | ~0.5 s |
| **E4B bf16** I_vision_sysrole_oracle_names_1fps280 (2026-04-14) | 6/7 (86%) | 8/11 (73%) | 0.538 | ~1.4 s |
| **E4B bf16** J_video_sysrole_primed (2026-04-14) | 6/7 (86%) | 5/11 (45%) | 0.512 | ~1.6 s |
| **E4B bf16** K_video_sysrole_oracle_names (2026-04-14) | **7/7 (100%)** | 8/11 (73%) | 0.560 | ~0.9 s |
| **E4B bf16** L_vision_sysrole_primed_1fps560 (2026-04-14) | 5/7 (71%) | 5/11 (45%) | 0.477 | ~2.5 s |
| **E4B bf16** M_vision_sysrole_primed_1fps1120 (2026-04-14) | **7/7 (100%)** | 5/11 (45%) | 0.502 | ~5.7 s |
| **E4B bf16** N_vision_sysrole_oracle_names_1fps560 (2026-04-14) | **7/7 (100%)** | 8/11 (73%) | 0.558 | ~2.4 s |
| **E4B bf16** O_vision_sysrole_oracle_names_1fps1120 (2026-04-14) | **7/7 (100%)** | 6/11 (55%) | 0.559 | ~5.6 s |

**E4B bf16 on 7900 XTX (2026-04-14):** new hardware slot unlocked
unquantized E4B with 16 GiB weights + 3.4 GiB KV (16k ctx, 9.8x
concurrency). Three findings versus E4B FP8 on the same spec:

1. **D improved** (kata 86→100%, LCS 0.570→0.585). The FP8 kata
   regression (`ノー ガード` spacing) was quant noise — bf16 recovers
   to E2B's clean 100%.
2. **I regressed sharply** (name 91→73%, LCS 0.610→0.538). FP8's
   "vision + oracle cooperate on E4B" headline did not reproduce on
   bf16 — seg03 `みなみかわ` and seg04 `しんいち` transliterate back
   to kanji under I-bf16. Both runs use identical weights pre-quant,
   same prompt, temp=0.0. Treating this as an unresolved quantization
   asymmetry rather than a new conclusion, n=8 is thin.
3. **K (video_url + oracle) is a new contender**: 100% kata, 73% name,
   0.560 LCS at **0.9 s/clip** — matches I-bf16 quality at roughly 60%
   of its latency, and contradicts the E2B finding that `video_url` is
   strictly worse than hand-picked frames. The hard-coded 70
   tok/frame × 32 frames is too sparse for E2B but E4B's bigger LM
   exploits the temporal coverage when oracle hints tell it what to
   listen for. On J (video_url without oracle) quality collapses
   (45% name), confirming that video_url is not a replacement for
   oracle text hints — it stacks with them.

**`max_soft_tokens` past 280 is a latency tax, not a quality dial even
on E4B.** L/M/N/O sweep: mst=1120 costs ~5.7 s/clip and never beats
mst=280 on headline metrics. The 280-tok default remains the right
per-frame budget; increasing it spends attention on vision tokens
without recovering anything that wasn't already there. The (70, 140,
280, 560, 1120) ladder is a mode switch on E4B too, not a resolution
dial. **Best vision config on bf16 by LCS is N at mst=560 (0.558)**,
but it's within noise of mst=280 (I: 0.538, K: 0.560) and 2–3x
slower. Default stays at mst=280.

Results files:
- D/H/I: `results/killah_kuts_s01e01__e4b_bf16_20260414.json`
- J/K/L/M/N/O: `results/killah_kuts_s01e01__e4b_bf16_knobs_20260414.json`

### n=24 re-run (de-noising K vs I) — 2026-04-14

Built a 3x denser spec (`eval_specs/killah_kuts_s01e01_n24.json`, also
from CTC gold, 24 hard segments) to check whether K's lead on n=8 was
noise and to re-read the I-FP8→I-bf16 regression puzzle. Results file
`results/killah_kuts_s01e01_n24__e4b_bf16_20260414.json`:

| Config | Kata | Name | LCS | Wall |
|---|---|---|---|---|
| D_audio_sysrole_primed | **24/34 (71%)** | 8/16 (50%) | 0.432 | ~0.5 s |
| H_audio_sysrole_oracle | 21/34 (62%) | **14/16 (88%)** | 0.418 | ~0.5 s |
| I_vision_sysrole_oracle_mst280 | 23/34 (68%) | 13/16 (81%) | 0.443 | ~1.5 s |
| J_video_sysrole_primed | **24/34 (71%)** | 7/16 (44%) | 0.427 | ~1.5 s |
| **K_video_sysrole_oracle** | **24/34 (71%)** | **14/16 (88%)** | **0.445** | **~0.9 s** |
| N_vision_sysrole_oracle_mst560 | 23/34 (68%) | 13/16 (81%) | 0.433 | ~2.5 s |

K is the strict winner — ties D/J on kata, ties H on name, highest
LCS of all six, and 1.7x faster than I. Both of K's name misses are
known failure modes: seg07 `しんいち` (documented audio-tower hard
ceiling) and seg03 `みなみかわ → 南川` (persistent kanji-transliteration
failure seen across many configs). Not things K can fix.

**Two structural findings from n=24:**

1. **The oracle-names system template introduces a tokenized-with-spaces
   output style** in audio-only (H) and image-path (I/N) configs, which
   reads as kata-recall regressions (`ノーガード` → `ノー ガード`
   fails strict substring match). Video_url (K) does NOT trigger this
   style — K's outputs are dense like D/J. So the FP8-era "kata is
   spacing artifacts, not quality" hypothesis was partially right: the
   artifact exists but it's prompt-shaped, not quant-shaped, and the
   video_url packing happens to suppress it.
2. **Video_url beats hand-picked frames only with oracle hints** — J
   (video, no oracle) is 44% name, the worst of this set. Video alone
   is not vision grounding; it's oracle hints + dense-ish style. The
   combination wins; each component alone loses.

**Episode-1 production rec for E4B bf16:
K_video_sysrole_oracle_names.** Faster than I, cleaner kata than H,
best LCS. The remaining gap to I_FP8's 91% is the audio-tower ceiling
plus seg03-style kanji preference. The episode-2 holdout below weakens
this from a global recommendation to an episode-1 result: K is still
efficient, but image-path oracle configs recover more katakana on the
medical-mystery episode.

On the I-FP8→I-bf16 regression puzzle: I-bf16 on n=24 is 81% name /
0.443 LCS. The n=8 dip to 73% was a single-segment noise spike
(seg04). The broader pattern is that I-FP8's 91% on n=8 is the
outlier, likely an FP8-quantization-specific tie-break that happened
to nail seg03/seg04. Not a reliable advantage.

### Second-episode n=24 holdout — 2026-04-28

Built a held-out spec from Killah Kuts S01E02
(`eval_specs/killah_kuts_s01e02_n24.json`) using the same CTC-gold
hard-segment picker, with episode-2 names from
`samples/episodes/killah_kuts_s01e02/glossary/episode_context.json`.
Results file:
`results/killah_kuts_s01e02_n24__e4b_bf16_20260428.json`.

| Config | Kata | Name | LCS | Wall |
|---|---|---|---|---|
| D_audio_sysrole_primed | 10/18 (56%) | 6/18 (33%) | 0.498 | ~0.4 s |
| H_audio_sysrole_oracle | 13/18 (72%) | 14/18 (78%) | 0.512 | ~0.4 s |
| I_vision_sysrole_oracle_mst280 | 14/18 (78%) | **15/18 (83%)** | 0.582 | ~1.3 s |
| J_video_sysrole_primed | 10/18 (56%) | 4/18 (22%) | 0.522 | ~1.5 s |
| K_video_sysrole_oracle | 13/18 (72%) | **15/18 (83%)** | 0.569 | ~0.8 s |
| **N_vision_sysrole_oracle_mst560** | **16/18 (89%)** | **15/18 (83%)** | **0.585** | ~2.4 s |
| P_audio_sysrole_minimal | 11/18 (61%) | 5/18 (28%) | 0.489 | ~0.4 s |

The episode-2 holdout changes the conclusion:

1. **Oracle hints generalize.** D/P without per-clip names are weak
   on episode-2 proper nouns; H/I/K/N all jump to 78-83% name recall.
2. **K is no longer the strict winner.** K remains the best
   quality/latency compromise, but N wins kata and LCS, and I nearly
   matches N at lower latency. The S01E01 "video_url suppresses the
   oracle spacing artifact" finding is not enough to make K globally
   best.
3. **The static Killah Kuts fight prime is mismatched for episode 2.**
   S01E02 is a medical mystery format with clues, suspects, and written
   dying messages. `P_audio_sysrole_minimal` is cheap but under-primes
   this episode: it misses `ともしげ`, `ニシダ`, `モグライダー`, and
   several suspect names that oracle configs recover.
4. **Image frames help clue terms.** The strongest N gains are on
   visual/text-heavy clue phrases like `アフロ` / `メガネ`, where
   video_url/audio paths often collapse to kanji or homophones.

Working recommendation after the holdout: keep K as the efficient
oracle-enabled candidate, but do not retire I/N. For non-fight formats
inside the same show, per-episode prime/OCR text is more important
than the video_url-vs-frame packing choice.

### S01E02 unprimed modality baseline — 2026-04-28

To separate raw model/media capability from show-specific prompting,
ran the same S01E02 n24 spec with no system message, no domain text,
no vocabulary list, no cast list, and no oracle names. The only user
instruction was the base transcription prompt. Results file:
`results/killah_kuts_s01e02_n24__e4b_bf16_unprimed_20260428.json`.

| Config | Media | Kata | Name | LCS | Wall |
|---|---|---|---|---|---|
| A_audio_base | audio | 9/18 (50%) | 4/18 (22%) | 0.484 | ~0.4 s |
| Q_vision_base_1fps280 | frames, mst280 | 12/18 (67%) | 6/18 (33%) | **0.583** | ~1.2 s |
| R_video_base | video_url | 9/18 (50%) | 4/18 (22%) | 0.498 | ~1.4 s |
| S_vision_base_1fps560 | frames, mst560 | **13/18 (72%)** | 6/18 (33%) | 0.574 | ~2.2 s |

Takeaways:

1. **Frames help even without priming.** Image-path configs lifted
   kata recall, name recall, and LCS over audio-only. On this episode,
   visible clue text is useful before any glossary/oracle injection.
2. **video_url is not a free visual baseline.** R barely moved LCS and
   did not improve kata/name recall over A. The S01E01 K win came from
   video_url + oracle packing, not from video_url alone.
3. **mst560 costs more than it earns unprimed.** S improved kata by one
   target over Q, but Q had the higher LCS at roughly half the latency.
4. **Unprimed name recall is the real floor.** 4-6/18 name hits means
   the model can hear some easy/common names, but the oracle jump to
   14-15/18 is still doing real work.

### Killah Kuts S01E01-S01E04 no-prime baseline vs Whisper — 2026-04-28

Ran the no-show-specific baseline on four Killah Kuts episodes using
the same n24 hard-window methodology, then compared against
faster-whisper `large-v3` on the exact same cached 8 s WAV windows.
E4B configs had no system message, no domain text, no vocabulary list,
no cast list, and no oracle names. Results files:

- `results/killah_kuts_s01e01_n24__e4b_bf16_unprimed_20260428.json`
- `results/killah_kuts_s01e02_n24__e4b_bf16_unprimed_20260428.json`
- `results/killah_kuts_s01e03_n24__e4b_bf16_unprimed_20260428.json`
- `results/killah_kuts_s01e04_n24__e4b_bf16_unprimed_20260428.json`
- `results/killah_kuts_s01e01_n24__whisper_large_v3_20260428.json`
- `results/killah_kuts_s01e02_n24__whisper_large_v3_20260428.json`
- `results/killah_kuts_s01e03_n24__whisper_large_v3_20260428.json`
- `results/killah_kuts_s01e04_n24__whisper_large_v3_20260428.json`

Best unprimed E4B by LCS per episode versus Whisper:

| Episode | Best unprimed E4B | E4B Kata | E4B Name | E4B LCS | Whisper Kata | Whisper Name | Whisper LCS |
|---|---|---:|---:|---:|---:|---:|---:|
| S01E01 | Q_vision_base_1fps280 | 21/34 (61.8%) | 2/16 (12.5%) | 0.417 | 24/34 (70.6%) | 1/16 (6.2%) | **0.497** |
| S01E02 | Q_vision_base_1fps280 | 12/18 (66.7%) | 6/18 (33.3%) | 0.583 | 16/18 (88.9%) | 11/18 (61.1%) | **0.591** |
| S01E03 | R_video_base | 17/25 (68.0%) | 7/16 (43.8%) | 0.395 | 19/25 (76.0%) | 9/16 (56.2%) | **0.507** |
| S01E04 | Q_vision_base_1fps280 | 11/17 (64.7%) | 5/22 (22.7%) | 0.408 | 13/17 (76.5%) | 8/22 (36.4%) | **0.469** |

Macro means across the four episode specs:

| Config | Kata | Name | Mean LCS |
|---|---:|---:|---:|
| A_audio_base | 58.6% | 24.6% | 0.390 |
| Q_vision_base_1fps280 | 63.3% | 26.5% | 0.430 |
| R_video_base | 57.4% | 26.4% | 0.419 |
| S_vision_base_1fps560 | 66.2% | 31.9% | 0.438 |
| whisper_large_v3 | **78.0%** | **40.0%** | **0.516** |

Character-count check, with whitespace stripped. The target count is
only the selected 1-4 s gold segment inside each 8 s eval window, so
output/target above 1.0x is expected. The useful signal is relative
verbosity:

| Config | Target chars | Output chars | Output/target |
|---|---:|---:|---:|
| A_audio_base | 1965 | 4074 | 2.07x |
| Q_vision_base_1fps280 | 1965 | 4530 | 2.31x |
| R_video_base | 1965 | 4146 | 2.11x |
| S_vision_base_1fps560 | 1965 | 4700 | 2.39x |
| whisper_large_v3 | 1965 | 3600 | 1.83x |

Takeaways:

1. **Whisper is the no-prime floor to beat.** It wins LCS on every
   episode and has the best macro katakana/name recall despite some
   visible proper-noun normalization misses.
2. **Whisper is also less verbose.** Because the windows contain
   surrounding dialogue, some over-target output is normal, but E4B
   produces more extra text than Whisper in every no-prime config.
3. **Raw E4B visual grounding helps, but not enough.** Frame inputs
   usually beat audio-only E4B, and `mst560` has the best E4B macro
   recall, but the gap to Whisper remains large.
4. **`video_url` is not consistently better without hints.** It is
   best E4B by LCS only on S01E03; on S01E02/S01E04 image frames win,
   and on S01E01 `video_url` trails Q/S.
5. **Do not spend time tuning show-specific primes until this baseline
   is tracked.** Prime/oracle experiments remain useful, but every new
   prompt path should report against this no-show-prime + Whisper
   comparison so we can separate real ASR improvement from prompt
   memorization or episode leakage.

### Local Gemma 26B OCR sweeps for S01E02-S01E04 — 2026-04-28

Ran local frame OCR with `bartowski/google_gemma-4-26B-A4B-it-GGUF:IQ4_XS`
via `llama-server` / llama.cpp, using the existing `ocr_sweep.py`
frame-batch path: `fps=0.5`, `height=720`, `batch_size=15`,
`backend=llama-cpp`, model alias `gemma-4-26B-A4B-it-IQ4_XS`. Existing
S01E01 comparison artifact:
`results/killah_kuts_s01e01_ocr_sweep_26bmoe_hip_fps05_20260426.json`.
New artifacts:

- `results/killah_kuts_s01e02_ocr_sweep_26bmoe_hip_fps05_20260428.json`
- `results/killah_kuts_s01e03_ocr_sweep_26bmoe_hip_fps05_20260428.json`
- `results/killah_kuts_s01e04_ocr_sweep_26bmoe_hip_fps05_20260428.json`

| Episode | Batches | Raw lines | Unique lines | Wall |
|---|---:|---:|---:|---:|
| S01E01 | 87 | 737 | 311 | 533.5 s |
| S01E02 | 97 | 886 | 625 | 568.8 s |
| S01E03 | 67 | 1355 | 715 | 446.9 s |
| S01E04 | 123 | 2399 | 486 | 769.8 s |

Operational findings:

1. **Local 26B OCR is strong enough to support transcript repair.**
   It recovered the large structured surfaces that matter for context:
   S01E02 suspect/name cards and clue boards, S01E03 rule/map/route
   overlays, and S01E04 contestant names, ages, experience counts, and
   round cards.
2. **The raw sweep is noisy and should not be injected wholesale.**
   It also captures subtitle-like captions, repeated persistent headers,
   map labels, credits, and occasional OCR hallucinations. The next
   repair pass needs a filtering/indexing layer before correction.
3. **`llama-server -np 1` was required for stability.** With the
   default auto parallelism, the 26B multimodal server crashed on E02
   batch 2 with `failed to find a memory slot for batch of size 264`.
   Restarting with one slot completed E02-E04.
4. **`ocr_sweep.py` now saves incrementally and resumes completed
   batches.** This is necessary for long local OCR jobs; a server crash
   should not lose hundreds of processed frames.

Design implication for transcript repair: an LLM correction pass makes
sense if it is constrained to line-preserving edits and is fed only
time-local, filtered OCR candidates. It should fix proper nouns,
visible rule terms, contestant metadata, and obvious homophones, not
rewrite timing, add missing dialogue freely, or turn OCR captions into
spoken transcript.

Priming nails katakana recall to 100% on this set. The A→B→C chain
adds ~9 pp on names and lifts LCS modestly. **Moving the prime into a
system-role message (B→D) doubled name recall from 27% → 55% on the
exact same eval spec**, with no change in latency and no katakana
regression, and also fixed the two B/C recitation failure modes
(seg02 vocab recitation and seg04 cast recitation). Vision on top of
the system-role prime (E) further lifts mean LCS to 0.590 — best of
any config tested so far — and uniquely produces the correct seg04
output `みなみかわ 対 お見送り芸人しんいち キックボクシング`
by grounding against the on-screen chyron.

Results files:
- D/E: `results/killah_kuts_s01e01__sysrole_20260411.json`
- F/G: `results/killah_kuts_s01e01__guided_20260411.json`
- H/I: `results/killah_kuts_s01e01__oracle_20260411.json`

**Oracle-names validates the OCR pre-pass plan.** Injecting the
exact cast strings actually spoken in each segment (from CTC gold)
into the system prime jumped audio-only name recall from 55% → 82%
and lifted LCS modestly. This is the **biggest single-knob name
recall win in the entire experiment** and proves that E2B's audio
tower *can* latch onto known-present name strings when told which
ones to expect — the earlier 0% on `しんいち` was an attention
priority problem, not an audio tower capacity problem.

Critically, **vision + oracle (I) underperforms audio + oracle (H)**
by 18 pp on name recall. On segments where the oracle prime already
tells the model the exact strings to emit, adding the visual token
budget introduces cross-modal noise — seg03 `みなみかわ` gets
transliterated to `南川` under I (vision) but stays hiragana under
H (audio). Vision is still best when oracle names are *absent*
(E's 0.590 LCS > H's 0.576 LCS), but oracle hints subsume most of
what vision was adding.

**What this means for the production design:**

1. OCR pre-pass → per-chunk oracle-style cast list → audio-only
   primed transcription is the winning topology for E2B.
2. Vision frames in the ASR pass are optional luxury, not
   essential, once the OCR pre-pass is in place.
3. The 2 remaining misses (seg07 `しんいち`, and I's seg03
   regressions) are the audio tower ceiling: seg07 fails because
   the model does not acoustically hear the name regardless of
   prime. That's the E2B hard wall.

`guided_json` surprise: it does **not** kill the seg04 cast
recitation — the model simply packs the entire cast list into the
`dialogue` string field as one sentence
(`みなみかわたい、お見送り芸人しんいち、大崎、設楽統、道尾、道雄、バナナマン、アンガールズ。`).
Structural output constraint ≠ semantic output constraint. The
+1-name improvement over D is almost entirely from seg03: F keeps
`みなみかわ` as hiragana whereas D transliterates to `南川`. That is
a useful side effect (the guided-JSON prior weakens the decoder's
preference for kanji on names) but it is not what we went looking
for. **Verdict on guided_json:** marginal quality gain, ~30% latency
penalty, and does not solve the recitation problem. Use only if we
specifically want to suppress kanji transliteration of name tokens.

### Reproducibility notes

- The same results file also contains a scratch "session 1" run against
  a different hand-picked segment set (`seg01_feint` through
  `seg08_referee`). That run produced:
  A_audio_base 50/25/0.400, B_audio_primed 70/25/0.457,
  C_vision_primed 80/38/0.491. It is not committed as a reproducible
  spec because the segments were hand-picked. The session-1 numbers
  live here as a reminder that **the gap between priming and
  vision-on-top depends on the segment set**; on katakana-dense but
  name-light segments, priming alone closes most of the gap.
- The auto-picker now de-duplicates repeated katakana within a segment
  so segments like `ストップ、ストップ、ストップ、ストップ` don't
  dominate the selection. Without that fix they crowded out more
  interesting cases.

## Key observations

### 1. Priming is the biggest single-knob win (+20+ pp katakana)

Adding a domain description + vocabulary list to the system prompt
fixes most katakana errors on E2B. The prime is assembled from
infrastructure the pipeline already has — glossary output, OCR
chyrons, per-show domain kit — so the cost is engineering, not data.

### 2. Vision stacks on top of priming but with caveats

On the earlier hand-picked set, vision+prime beat prime-alone by
+10 pp katakana and +13 pp names. On the auto-picked set the gain
collapsed to "different errors, same summary numbers." Vision's value
on E2B is probably highest on segments that contain on-screen text
directly related to what is being said (seg05 `みなみかわ` chyron
while a commentator says her name).

### 3. `しんいち` is recoverable with per-segment oracle hints, not global priming

Originally every B/C/D/E config garbled the audio rendering of
`しんいち` to `シイチ / C一 / し一 / しイチ / 新一 / 新一吉`, even
when the global prime included the exact string. The 2026-04-11
oracle-names run (H/I) changes that: telling the model per-segment
that `しんいち` appears in **this specific clip** (not merely that
it's a cast member of the show) lifts seg01 and seg02 from 0 → 1
and produces the exact string `しんいち`.

The wrinkle: **seg07 still fails even with the oracle hint**
(`ちゃんと読めばいいでしょうがチームやって…`). The model simply
does not acoustically hear the name in that clip. That's an
audio-tower ceiling, not a priming/grounding problem.

This is the actual argument for a dedicated OCR pre-pass:
extract chyron strings exactly, attach them per-chunk with
"speakers present in this clip" framing, and rely on the audio
decoder only to match them when acoustically present. Budget a
small residual of unrecoverable cases (~10–20% of name slots on
E2B) where the audio is too degraded for even a primed decoder.

### 4. Priming has two distinct failure modes — both fixed by system-role framing

**Miscategorization (seg08 `レフェリー` session 1):** when the prime
included `フェンシング` as a vocabulary term, the model out-competed
`レフェリー` with `フェンシング` because both sound plausible as
kata openings for "f-" sound. The vision config broke the tie by
grounding on the visible referee. **Mitigation:** only include
vocabulary terms with high confidence they appear verbatim. Do *not*
pre-load near-homophones hoping the model will pick the right one.

**Recitation (seg04 `みなみかわ対お見送り芸人しんいち`, seg02 vocab
list):** with the prime concatenated into the user turn, the model
output the **entire cast list** on seg04 and the **entire vocab
list** on seg02 as if they were the transcript. Classic
prompt-injection shape: the user-turn reference block bled into the
output distribution.

**Fix (2026-04-11):** moving the domain+vocab+cast block into a
dedicated `system` role message with "reference only, do NOT output
unless actually spoken" framing (configs D and E) **eliminated the
seg02 vocab-recitation failure entirely** and **sharply reduced the
seg04 cast recitation** — config E (vision + system prime)
produced the correct `みなみかわ 対 お見送り芸人しんいち` for
seg04, the only config so far to get it right. Config D still
partially recites the cast on seg04 (with `さん` suffixes), but
on all other segments it behaves cleanly. The system-role move also
doubled name recall from 27% → 55% and lifted mean LCS from 0.441 to
0.512 (D) / 0.590 (E). **Production rule: the prime goes in a
`system` message, not the user turn.**

### 5. `max_soft_tokens` is a mode switch, not a quality dial

Documented in detail in this session's scratch runs
(`/tmp/vllm_smoke/exp*`). Summary: the `(70, 140, 280, 560, 1120)`
`max_soft_tokens` ladder is the Gemini media-resolution ladder under
a different name, and E2B responds to it as a **mode switch**:
- 70 tok/frame → model can't OCR, falls back to audio
- 140 tok/frame → model enters OCR-aggregate mode (drops dialogue)
- 280 tok/frame → default image-path mode
- 560+ tok/frame → quality degrades (E2B vision tower saturates)

**E2B can't use HIGH-res frames.** Bumping past 280 regresses on OCR
accuracy (`スポーテストダンガン` → `スポーテストンガン`). This is a
capacity ceiling, not a resolution ceiling, and is a strong argument
for quantizing E4B rather than trying to squeeze more out of E2B.

### 6. `video_url` content part is strictly inferior to `image_url` + frame sampling

vLLM's `gemma4_mm.py` hard-codes `_VIDEO_MAX_SOFT_TOKENS = 70` for
video input, up to `_VIDEO_MAX_FRAMES = 32`. This is tuned to stay
under the E2B attention budget, but it gives us **no control** over
frame timing, count, or resolution. Hand-picking frames via
`image_url` reproduces everything video mode does, better.
**Rule: do not use `video_url` in the harness.**

### 7. Frame count, not per-frame resolution, is the vision grounding knob

On a 10 s clip, 20 frames × 70 tok reproduced video mode's "clean
dialogue with vision grounding" output; 5 frames × 280 (same visual
token budget) did not. Temporal spread beats per-frame detail for
grounding audio. Conversely, for OCR-extraction mode, fewer sharper
frames win (3 frames × 140 produced the cleanest single telop read
of anywhere tested). **Different tasks want different configs** —
which is why the design has split into "dialogue pass" and "OCR
pre-pass" with separate optimization targets.

## Current design sketch

Based on the above, the candidate production harness for an E2B-class
model is:

Revised 2026-04-11 after the oracle-names run. The vision-in-ASR-pass
branch is no longer the default — oracle text hints from the OCR
pre-pass subsume most of what vision was contributing, and stacking
vision on top actively regresses some names back to kanji.

```
episode OCR pre-pass (bigger model: Gemma 4 E4B quant / Qwen2.5-VL / manga-ocr)
  -> table of {time_range, text, kind=chyron_name|stats|caption|title}
  -> feeds glossary step
  -> indexed by time so per-chunk assembly can look up which
     cast names are on-screen *during each specific chunk*

per-chunk primed ASR pass (Gemma 4 E2B via vLLM, audio-only path)
  -> 6-10s chunks (smaller than current 20s) for decoder focus
  -> system role message containing:
       - per-episode domain description
       - high-confidence global vocab only (no near-homophones)
       - "Cast names spoken in THIS SPECIFIC clip (verbatim, do
          not transliterate): ..."  <- from the OCR time-index
  -> user turn = audio + short transcription instruction
  -> no vision frames (oracle text hints beat vision on E2B)
  -> vLLM prefix caching amortizes the system-constant portion
     across chunks; only the per-clip name line differs
```

Vision stays in scope for the OCR pre-pass itself — that's where
the pixel budget is spent, on a bigger model. In the ASR pass
vision is demoted to "optional fallback for chunks with empty
oracle name lists", not a default.

Open design questions listed below decide whether that sketch actually
wins over the current Whisper/Cohere pre-pass.

## Open questions (next sessions)

Not in priority order — each is roughly 0.5–2 hours of focused work.

1. ~~**Prime location and format.**~~ **Answered 2026-04-11.**
   Moving the prime into a `system` role message (configs D/E)
   eliminates the seg02 vocab recitation entirely and partially
   fixes seg04 cast recitation (E gets it right via vision
   grounding, D still recites with `さん` suffixes). Also doubles
   name recall 27% → 55% and lifts mean LCS to 0.512 (audio) /
   0.590 (vision). System-role is now the default for all primed
   configs going forward.
2. ~~**Guided JSON output.**~~ **Answered 2026-04-11.** No — forcing
   `{dialogue: string}` via `guided_json` does not suppress the
   cast-list recitation. The model packs the full list into the
   string field. Only unexpected win: guided decoding nudges names
   toward hiragana over kanji transliteration (+1 name on seg03).
   Latency cost ~30%. Not a production default.
3. **Chunk size sweep.** Re-run the bench at 4 s / 6 s / 8 s / 12 s
   windows. Hypothesis: smaller windows improve name recall because
   attention concentrates on fewer competing utterances, but hurt LCS
   because the model has less context to bridge fragments.
4. **Frame count sweep at fixed `mst=280`.** 2 / 4 / 6 / 8 / 12 frames
   over an 8 s window. The attention-budget saturation threshold
   depends on audio length; re-characterize for the full chunk range.
5. ~~**Name recall with OCR-injected strings.**~~ **Answered
   2026-04-11.** Yes — per-segment oracle names (config H) lifted
   audio-only name recall 55% → 82% (+27 pp) and fixed every
   previously-0% `しんいち` clip except seg07. **OCR pre-pass plan
   confirmed viable.** Secondary finding: vision + oracle (I)
   underperforms audio + oracle (H) by 18 pp — vision grounding is
   subsumed by oracle text hints and actively regresses some
   hiragana names back to kanji. The 2 remaining misses are the
   audio-tower ceiling: seg07 still fails even with the exact
   string in the prime, because the model does not acoustically
   hear `しんいち` there.
6. **Prefix-cache benefit at scale.** Current metrics show ~48% cache
   hit rate across sessions. Measure: run 30 chunks back-to-back with
   the same system prime, report p50/p95 wall time, compare against
   30 cold runs with randomized primes. Quantify the throughput
   upside of a stable prime.
7. **Cohere/Whisper comparison.** Partially answered 2026-04-28 for
   hard 8 s eval windows: faster-whisper `large-v3` beats unprimed E4B
   on S01E01-S01E04 by LCS and macro katakana/name recall. Still open
   for the full-episode shipping question: compare char counts,
   coverage, kata recall on full CTC gold, and name recall against the
   same 43.5-min episode.
8. ~~**E4B unlock.**~~ **Answered 2026-04-12.** Local FP8 dynamic
   quantization via llm-compressor (`quantize_e4b_fp8.py`) + clean
   tokenizer from google's upload (leon-se's had a poisoned
   `tokenizer.json` routing `max_length=None` into audio kwargs).
   E4B FP8 on config H reaches **91% name recall** (vs 82% E2B),
   recovering the seg07 `しんいち` that was E2B's hard ceiling.
   Katakana recall regressed on D/H (86%/71% vs 100%) due to
   spacing artifacts (`ノーガード` → `ノー ガード`) — likely a
   tokenization issue, not a quality drop. Mean LCS is mixed:
   D improved (0.512→0.570), H dropped (0.576→0.522). Vision
   configs E/I added 2026-04-12: **I (vision + oracle) hits
   91% name recall AND 0.610 mean LCS** — the best LCS of any
   config on either model. On E2B, vision + oracle actively
   regressed names (64% vs H's 82%); on E4B they cooperate,
   confirming the hypothesis that E4B's larger LM can resolve
   the cross-modal conflicts E2B couldn't. E4B is now a viable
   candidate for the ASR pass model, and vision is back in play
   as a default when paired with oracle names.
9. **OCR pre-pass model selection.** Benchmark Gemma 4 E4B (once
   quantized), Qwen2.5-VL 3B/7B, and manga-ocr on a few hundred chyron
   crops from the episode. Criteria: name-string exact match, latency
   per crop, integration cost.
10. **Structured output for OCR pre-pass.** Once the model is chosen,
    define the output schema: `{t_start, t_end, text, kind, bbox?}`.
    Use `guided_json` on the OCR model so we don't need to post-parse.

## References

- `scripts/experiments/vllm_gemma4_harness/` — experiment machinery,
  specs, results
- `scripts/experiments/vllm_gemma4_harness/server/` — vLLM docker
  recipe
- `docs/asr-comparison-2026-04-10.md` — broader ASR model comparison
  (where Killah Kuts S01E01 first showed up as the benchmark)
- `docs/gemma4-quant-calibration.md` — quant plan gated on harness
  outcomes (this doc). Update the quant doc's "experiment order"
  section whenever the harness numbers change its cost/benefit.
- `docs/transcription-research-roadmap.md` — broader research
  context, if any of this grows into a production path
