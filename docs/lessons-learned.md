# Lessons Learned

This document records what the repo has actually taught us so far, based on real runs against:

- `samples/episodes/great_escape1`
- `samples/episodes/dmm`

It is intentionally operational. The goal is to capture what worked, what did not, and what still needs to be fixed.

## Current Conclusions

### 1. Reusable artifacts were the right architectural move

Saving OCR, VAD, chunk boundaries, alignment outputs, and diagnostics as separate artifacts is materially better than recomputing them inside one monolithic script.

What this unlocked:

- rerunning OCR filtering without rerunning OCR
- rerunning translation without rerunning alignment
- comparing video-only vs OCR-assisted transcription on the same episode
- inspecting failures at the artifact boundary instead of guessing

### 2. CTC forced alignment replaced stable-ts and nearly eliminated stranded words

stable-ts uses Whisper's cross-attention for alignment, which is a byproduct of the generative model. This caused 13.4% of words to get zero-duration timestamps on `dmm`, even with chunked alignment.

CTC forced alignment (`scripts/align_ctc.py`) uses `NTQAI/wav2vec2-large-japanese` with `torchaudio.functional.forced_align`. This is a dedicated alignment mechanism — it finds the optimal path through frame-level character probabilities via Viterbi decoding.

Results on `dmm`:

| | stable-ts (chunked) | CTC wav2vec2-ja |
|---|---|---|
| Zero-duration segments | 42 (3.8%) | 3 (0.3%) |
| Zero-duration words | 947 (13.4%) | 3 (0.3%) |

The 3 remaining zero-duration segments are non-Japanese text (`Shit!`, `30?`, `50。`) with no characters in the model's Japanese vocabulary. These are trivially fixable.

Key properties of CTC alignment:

- errors are local — one bad region does not cascade and collapse a whole tail
- works at the character level, so even short utterances get placed
- deterministic given the audio features, no autoregressive drift
- runs on GPU via system python3.12 with ROCm

Model selection matters: `facebook/mms-1b-all` was tested first but had 10.8% character vocabulary gaps on real Japanese text (missing common kanji like `鬼`, `凄`, `喋`, plus all punctuation). `NTQAI/wav2vec2-large-japanese` has a proper Japanese vocabulary with 2174 tokens covering kanji, hiragana, and katakana.

CTC forced alignment is now the recommended default for alignment.

### 3. Chunked alignment is much more robust than whole-episode alignment

Whole-episode `stable-ts` alignment is too fragile. One bad transcript section can collapse a large tail of the episode to the final timestamp.

Chunked alignment fixed the major failure mode on `dmm`:

- whole-episode alignment collapsed a large late tail to `2442.51`
- chunked alignment removed the end-of-file collapse
- only local zero-duration alignment misses remained

Chunked alignment should remain the default. CTC forced alignment (lesson 2) also uses chunked alignment.

### 4. Zero-duration alignment misses should not be dropped blindly

With CTC alignment, zero-duration segments are now rare (0.3% on `dmm`) and were initially thought to be limited to non-Japanese text. But the principle still holds: do not drop them blindly.

The zero-duration segments from earlier stable-ts runs were not mostly garbage. Many were real short utterances or chunk-edge dialogue blocks.

Dropping them would make later English translation worse. The current reflow step preserves their text by merging them into nearby timed cues.

New failure mode seen on `great_escape_s01e01`:

- a short Japanese answer line (`蚊。`) failed alignment entirely inside a chunk
- `align_ctc.py` emitted it as `0.0 -> 0.0` inside the chunk
- the chunk offset then turned that into the chunk start timestamp, creating a large backward jump
- line-level reflow attached the zero-duration line to an earlier cue, and max-line trimming hid it

The fix is to repair all-unaligned line timings inside `align_ctc.py` before chunk offsetting:

- keep the text
- keep the original line order
- interpolate a small local slot between neighboring aligned lines instead of defaulting to chunk start
- run a final monotonic timing pass so tiny backward reversals do not survive into saved alignment JSON

This preserves short answers and other high-value lines without pretending the timing confidence is high.

This is the correct default for a translation-first pipeline, even if the repaired Japanese cue timing is imperfect.

### 5. Reflow must not use text-only word-timestamp lookup for repeated lines

Another real failure on `great_escape_s01e06` came from line-level reflow, not alignment:

- `_split_long_single_lines()` looked up character timestamps by line text only
- repeated strings like `うわ、すげえ。` appeared multiple times in one episode
- a later cue inherited the earlier occurrence's word timings
- that produced backward cue times and a negative-duration VTT cue

The fix is to resolve repeated lines by nearest `(text, start, end)` match, not by text alone.

### 6. Video-only Gemini is a serious path, not just an experiment

For both `great_escape1` and `dmm`, direct video transcription with Gemini was viable.

Why it matters:

- it removes the entire OCR preprocessing cost from the cloud path
- it naturally sees mission cards, question text, premise cards, counters, and subtitle-like telops
- it may be good enough often enough that OCR should become optional instead of mandatory for Gemini workflows

The main lesson:

- OCR is still useful
- but Gemini video-only is strong enough that it must be treated as a first-class baseline

### 7. Shorter video chunks were a big win

The first video-only trial on `great_escape1` used long fixed chunks and was too verbose and more failure-prone.

Switching to shorter chunks improved behavior substantially:

- less drift
- less visual over-description
- fewer repetition failures
- better local scene focus

Using VAD-derived chunk boundaries for video-only transcription worked better than naive long fixed chunks.

### 7. Bracketed visual context is the right output shape for video-only transcription

The best video-only prompt shape so far was:

- spoken text as `-- ...`
- visual-only context as `[画面: ...]`

This was better than forcing Gemini to mix spoken and visual information into one transcript.

Why:

- spoken text remains alignable
- visual scene/rule/question context is preserved
- `[画面: ...]` lines can be stripped before alignment

This format worked well on `great_escape1` and `dmm`.

### 8. OCR-first filtering became too conservative when it focused mainly on names

The initial OCR cleanup/classification work improved safety, but it over-optimized for name/entity anchors.

This was not enough for these shows.

What the examples showed:

- focused board questions are useful
- room/mission text is useful
- subtitle-like reaction telops are useful
- state readouts can matter when the scene depends on them

What is less useful on its own:

- only preserving performer names

The OCR filtering layer should keep more local semantic telop content, not just names.

### 9. OCR still varies too much across show formats to over-classify early

`great_escape1` and `dmm` share some properties, but the visual presentation differs a lot.

Examples:

- `great_escape1`: puzzle boards, focused question frames, counters, rules
- `dmm`: location cards, cast intros, premise cards, subtitle-like captions, constant service watermark

This argues against a heavy semantic classifier early in the OCR pipeline.

Better approach:

- keep OCR extraction relatively dumb
- keep filtering simple and local
- only strip obvious junk
- preserve a few useful local lines and keywords

### 10. Translation should be subtitle editing, not literal translation

The new `translate_vtt.py` direction is correct:

- translate in local cue batches
- preserve cue count/timing
- target readable English subtitle CPS
- allow local redistribution of meaning across adjacent target cues

The partial `dmm` English results are already much better than literal cue-by-cue translation would be.

Examples:

- `何これ?` -> `What's this?`
- `なんだよこのナレ。 ひでえな。` -> `What's with this narration? That's awful.`
- `「あの車に乗れ」言ってたぞ。` -> `He said, "Get in the car."`

The translation step is now behaving like subtitle editing, which is what this repo needs.

### 11. Cues under 0.5 seconds are too short to hand off, even if technically correct

Real episode review on `great_escape_s01e01` exposed a failure mode that the earlier policy still allowed:

- the restored answer `蚊。` survived in Japanese reflow
- but it lived in a `0.16s` cue
- the first English rerun also placed `Mosquito.` in that same micro-cue
- the answer became easy to miss in actual viewing even though it was "present"

Operational conclusion:

- cues under `0.5s` should be treated as structural blockers for the reflow -> translation handoff
- this is stricter than the earlier "short cues are advisory" rule, and intentionally so
- plausible short reactions can still exist above that threshold, but sub-`0.5s` cues are too easy to miss to trust in final subtitle output

Pipeline consequence:

- `repair_vtt_codex.py` / `chigyusubs.reflow_repair` should mark sub-`0.5s` cues as `red`
- `translate_vtt_codex.py prepare` should stop on sub-`0.5s` source cues instead of letting English translation try to compensate later

The purpose is not metric purity. The purpose is to prevent important content from being stranded in cues that barely register on screen.

### 12. Line-level micro-cue repair has to rewrap display lines, not just count raw transcript lines

Validated on `great_escape_s01e03`.

Failure mode:

- line-level reflow already had a residual micro-cue merge pass
- but it rejected merges whenever the combined raw line count would exceed the display line limit
- that left sub-`0.5s` cues behind even when the merged cue could still be shown cleanly in two subtitle lines

### 13. English draft reuse must be gated by exact cue timelines, not cue index

Validated on `great_escape_s01e04`.

Failure mode:

- an older Codex English draft existed for the episode
- the Japanese `*_ctc_words_reflow.vtt` was later regenerated with several local cue-boundary changes
- replaying the old English by cue number looked superficially plausible but drifted meaning into neighboring cues
- short answers like `ホッチキス。` and lines like `おしっこしたいんだけど。` then appeared at the wrong timestamps or vanished from the right cue

Operational rule:

- treat old English drafts as reusable only when cue count and cue timings match the current Japanese source exactly
- if even one cue boundary changed, do not seed by cue index
- any safe reuse path should validate exact cue-timeline equality before importing draft translations

Pipeline consequence:

- `translate_vtt_codex.py prepare --seed-from ...` now errors out unless the candidate draft matches the current Japanese cue timeline exactly
- manual cue-index replay should be treated as unsafe and avoided

Observed `e03` examples before the fix:

- `知らない。 / え?` followed by `つる、つる、 / つる本手。`
- `やったあ! / やったあ!` followed by `イエス! / よっしゃあ!`
- `おお。 / おお。` during the Tom Brown escape burst

Fix:

- `chigyusubs.reflow._merge_micro_cues()` now evaluates candidate merges through a display-text rewrap path
- merged cues can keep the original underlying raw-line list while rewriting visible cue text back down to the configured line limit

Result on `e03`:

- plain line-level reflow moved from `red` to `green`
- cue count dropped from `760` to `750`
- residual cues under `0.5s` dropped from `10` to `0`
- no negative durations and no overlaps

## Episode-Specific Findings

### `great_escape1`

What worked:

- video-only Gemini with `[画面: ...]` lines captured useful mission/question text
- shorter video chunks were substantially better than the first long-chunk run
- alignment was mostly good before the bad repeated-line failure

Main failure:

- one repeated `見、見、見...` line near the end poisoned tail alignment

Lesson:

- chunk-local loop detection/retry is necessary
- one localized failure can still ruin whole-episode alignment if not isolated

### `dmm`

What worked:

- OCR quality was generally usable on large telops
- VAD-based video-only transcription worked well
- chunked alignment fixed the major tail-collapse issue
- translation output is already reasonably natural in English

Main caveats:

- persistent overlay text like `DMM TV` / `月額550円` still needs explicit stripping in OCR-based paths
- some visual captions are editorial and not spoken
- transcript fragmentation still leaks into later translation and subtitle output

## Problems Still Open

### 1. Translation checkpointing exists, and it was necessary

The translator now checkpoints per batch to:

- `<output>.checkpoint.json`

That solved the resumability problem.

What is still open:

- `gemini-2.5-pro` still hits intermittent `429 RESOURCE_EXHAUSTED` depending on region
- the diagnostics need better cost/token accounting and less noisy short-cue CPS review logic
- Codex-interactive translation restarts should clear stale session/output/diagnostics history when intentionally restarted from a clean base

### 2. CPS validation is still too strict for very short cues

Some “hard CPS violations” are not real quality failures.

Examples:

- `What's this?`
- `A special.`

These only fail because the cue duration is tiny.

What should change:

- treat very short cue durations differently in translation diagnostics
- do not overcount these as meaningful readability failures

The same lesson applies to Japanese reflow review:

- short or tiny cues are not automatically broken
- one-word answers, reactions, and interruptions may be legitimate even below `0.8s`
- reflow review should use short/tiny counts as advisory metrics, not automatic repair triggers
- repair should focus on structural timing defects and artifact-like boundary errors

### 3. Line-level reflow eliminated split-word artifacts

The worst translation artifact was mid-word splits across adjacent cues (`地` / `獄` → `He-` / `-ll.`). This was caused by character-level reflow splitting at inter-character gaps within words.

Line-level reflow (`reflow_words.py --line-level`) solved this by treating each Gemini transcript line as an atomic unit. Lines are never split mid-word.

Additional fixes in the line-level reflow:
- Comma-based fallback splitting for long lines without sentence-ending punctuation
- Sparse-cue clamping: shrinks cues where CTC spreads few characters across disproportionately long audio windows (e.g., `なんなの!` spanning 10.8s → clamped to 7.0s)

Results after line-level reflow:
- ge1: 0 overlong cues (was 9 with character-level), max 7.0s, 0 >20 CPS
- dmm: 0 overlong cues (was 21), max 7.0s, 1 >20 CPS (real fast exchange)

The LLM-based `repair_vtt_local.py` is no longer needed in the default path. It solved symptoms (split words) that line-level reflow prevents at the source.

### 4. OCR context selection is still unresolved for the Gemini path

We learned that:

- names alone are not enough
- whole dense dumps are too much

What remains unclear is the best simple middle ground.

Most likely direction:

- local `line_hints`
- local `keyword_hints`
- narrow time locality
- minimal role classification

This still needs a concrete simplification pass before it should be treated as the stable OCR-assisted default.

### 5. Video-only transcription still needs automatic loop recovery baked into the maintained pipeline

The retry logic exists in the dedicated video script experiments, but the broader maintained path still needs a fully settled policy.

Current best behavior:

- detect loop-like output per chunk
- retry only the failed chunk
- slightly higher temperature on retry
- no rolling context on retry

This should become standard for video-only Gemini transcription.

### 6. Translation is now local-batch aware, but subtitle polishing is not finished

The English translation quality is already promising, but still incomplete:

- awkward source fragmentation still leaks through
- some lines are slightly too written or too literal
- diagnostics need one more pass to distinguish:
  - true bad subtitles
  - acceptable short-cue CPS spikes

### 7. Codex-interactive translation has a practical batch ceiling

When Codex itself did the subtitle translation interactively, quality degraded gradually rather than catastrophically:

- `60` cues: solid
- `84` cues: still held quality
- `120` cues: still usable, but noticeably flatter and weaker on puns/riddles

Current practical default for the Codex-interactive path:

- default batch tier: `84`
- automatic fallback tiers: `60`, then `48`
- one episode at a time
- stop on structural blockers or red-quality batches

This is now embodied in the maintained `translate_vtt_codex.py` workflow instead of relying on ad hoc chat-only checkpoint edits.

Follow-up correction:

- minimum-tier `yellow` batches should remain resumable when the only issue is hard CPS pressure
- short-cue CPS warnings are still too noisy to justify auto-stopping a full Codex-interactive episode run on their own
- in the Codex-interactive translation helper, `yellow` should be a continue-with-warning state; only structural errors or an explicit `red` should stop the run
- deterministic diagnostics are still useful, but they need a compact rollup so QA can distinguish:
  - warning-only batches
  - real structural blockers
  - obsolete history from a superseded session

### 8. Codex-interactive reflow repair should be the default skill fallback, not local LLM repair

After running the Codex repair path on weak reflowed VTTs:

- line-level deterministic reflow remains the right default producer
- Codex-interactive `repair_vtt_codex.py` is the right fallback for structurally valid but translation-hostile files
- local LLM cue-repair remains useful only as an alternative benchmark path, not the default workflow

What turned out to matter most in practice:

- exact flagged cue-id ranges
- before/after counts for short cues and tiny cues
- one recommended Japanese VTT handoff path for translation

So the useful next step is better deterministic review diagnostics, not putting a local server back into the default skill path.

### 9. CTC wav2vec2 alignment outperformed both stable-ts and Qwen forced alignment

Three alignment approaches were benchmarked:

- `stable-ts` (Whisper cross-attention): 13.4% zero-duration words on `dmm`
- `Qwen/Qwen3-ForcedAligner-0.6B`: high zero-duration word count in smoke test, not obviously better than stable-ts
- `NTQAI/wav2vec2-large-japanese` CTC: 0.3% zero-duration words on `dmm`

CTC forced alignment is the clear winner. The Qwen and stable-ts aligners remain as archived benchmarks.

## Current Best Practical Defaults

### For transcription experiments

Use video-only Gemini as a real baseline:

- Silero VAD
- VAD chunk boundaries
- Gemini video-only transcription
- spoken text as `-- ...`
- visual-only text as `[画面: ...]`
- strip `[画面: ...]` before alignment
- CTC forced alignment
- line-level reflow (`--line-level`)

### For OCR-assisted experiments

Keep OCR reusable and local:

- Qwen OCR JSONL
- OCR spans
- light local filtering/classification
- do not overfit the classifier to one show

### For alignment

- CTC forced alignment (`align_ctc.py`) is the new default
- uses `NTQAI/wav2vec2-large-japanese` + `torchaudio.functional.forced_align`
- runs on system python3.12 with ROCm GPU
- chunked: aligns per-chunk from Gemini raw transcription JSON
- keep per-chunk diagnostics
- preserve zero-duration segment text for translation support (rare with CTC, but still applies to non-Japanese text)

### For translation

- batch-based subtitle editing
- preserve cue count/timing
- enforce CPS-aware diagnostics
- checkpoint after every batch

## Next High-Value Work

1. Finish one full `dmm` translation run with checkpointing.
2. Make translation diagnostics less noisy on very short cues.
3. Decide whether Gemini video-only becomes the default cloud path.
4. Simplify OCR context selection into a stable local `line_hints + keyword_hints` model if OCR remains part of the Gemini path.
