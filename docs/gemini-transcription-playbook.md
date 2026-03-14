# Gemini Transcription Playbook

This document is the current operating playbook for Gemini-based Japanese transcription in this repo.

Use it for:

- model-tier selection
- chunking defaults
- media-resolution defaults
- cost-aware experimentation
- escalation / fallback decisions

For raw evidence and experiment history, see `docs/lessons-learned.md`.

## Current Default

For serious Gemini transcription experiments:

- keep local encoding at `1 FPS`
- keep source resolution by default; only downscale intentionally
- use video-only first
- keep `temperature=0.0`
- keep `rolling_context_chunks=0` on fragile models
- prefer VAD-derived chunks
- compare suspicious regions against faster-whisper

Current free-tier default for real episode work:

- `gemini-2.5-flash`
- `video`
- `spoken_only`
- `media_resolution=high`
- `temperature=0.0`

Current free-tier production rollover:

- run `gemini-2.5-flash` first on the canonical chunk plan
- keep `rolling_context_chunks=1` for real production runs
- if `2.5-flash` RPD is exhausted, continue from the next chunk with `gemini-3-flash-preview`
- do not restart the episode on `3-flash` unless the user explicitly wants a comparison run

Named presets worth remembering:

- `flash25_free_default` — maintained `2.5-flash` production baseline with no thinking override
- `flash_free_default` — maintained free-tier real-run baseline
- `flashlite_debug_transcript` — cheap debug transcript preset with no rolling context and bounded retries
- `pro_quality_video` — higher-quality paid reference baseline

Current paid/reference quality anchor:

- `gemini-2.5-pro`
- `video`
- `spoken_plus_visual`
- `media_resolution=high`
- `temperature=0.0`

## Model Tier Policy

Current practical policy:

- use Flash-tier models for most transcription experiments
- reserve Pro-tier runs for:
  - quality benchmarking
  - episodes where Flash-tier lexical fidelity is clearly insufficient
  - sections where visual/rule-text precision materially affects downstream translation

Current qualitative ordering for useful Japanese transcription on variety content:

- `gemini-2.5-pro`
- `gemini-3-flash-preview` / `gemini-2.5-flash`
- `gemini-3.1-flash-lite-preview`

`flash-lite` is still useful for cheap probing, but it is not the default “trust this for a full episode” choice.

## Media Resolution

This is now a first-class decision, not a hidden default.

Working rule:

- for Gemini 3.x evaluations, test `media_resolution=high` before concluding the model is weak
- default / unspecified Gemini 3 media resolution appears materially lower than 2.5-era multimodal accounting
- for text-heavy or cast-card-heavy variety shows, `high` is the right comparison setting

What we learned:

- `flash-lite + high` improves some visual/name details
- but `flash-lite + high` still times out on dense rule-card chunks and still makes bad lexical mistakes
- `2.5-flash` handled the same short dense rule-card chunks quickly and captured useful rule text

Practical policy:

- use `high` when evaluating a serious Gemini path on visual-heavy content
- do not assume `high` makes weak models good enough

## FPS

Keep `1 FPS` as the default.

Why:

- this pipeline uses video for more than OCR
- brief cast cards and rule cards matter
- lowering FPS is more likely to miss short visual information than to fix model quality

Do not prioritize FPS experiments before media-resolution experiments.

If FPS is tested later, use explicit Gemini-side video sampling controls rather than relying only on ffmpeg downsampling.

## Local Encoding Resolution

Keep source resolution by default.

Why:

- Gemini `media_resolution=high` is an internal processing tier, not a published fixed pixel size
- pre-downscaling throws away detail that Gemini cannot recover
- variety content often depends on small cast cards, counters, and prompt text

Practical policy:

- preserve source width for maintained Gemini video chunking by default
- use `--width` only for intentional small-probe experiments or strict payload constraints
- do not assume Gemini `high` compensates for a low-resolution upload

## Chunking

Default chunk policy:

- prefer reviewed semantic chunks around `90s` for real episode work
- treat `target + 30s` as the default hard ceiling
- that means `90s -> 120s` is the current maintained reviewed-chunk default
- use `120s` to probe fragile models
- if Flash Lite is specifically the model under test and a bad span is already identified, `60s` is a reasonable surgical probe size
- go shorter only when there is clear evidence chunk length is the blocker

Operational naming rule:

- keep the primary plan in `vad_chunks.json` or `vad_chunks_semantic_<target>.json`
- use `*_repair*.json` only after a real chunk failure
- use `probes/*exact_chunks_60s*.json` for Flash Lite debug work, not as the canonical episode chunk plan

Current lesson:

- on `great_escape_s02e04`, a `90s` semantic plan was materially easier to transcribe, review, and translate than the earlier `180s` path
- shorter chunks help ordinary dialogue
- shorter chunks did not rescue `flash-lite + high` on dense rule-card sections
- on a stable non-looping dialogue section, `flash-lite + high` completed at `60s`, `120s`, and `150s`, but a `90s` probe still hit `504 DEADLINE_EXCEEDED`
- on a problematic `great_escape_s02e03` dialogue span, `flash-lite + high` at `120s` produced a `10053`-char loop-like chunk before retry, while a `60s` split completed cleanly at nearly the same token cost
- on the full `great_escape_s02e03` episode, a VAD-guided `~60s` plan still failed early because one chunk stretched to `64.576s`; a strict exact-`60s` plan got much further before the next repeated timeout
- stubborn timeout chunks should be split surgically and resumed, not hammered with long retry loops

Interpretation:

- Flash Lite instability is not monotonic with chunk duration
- shorter chunks are still a useful safety lever
- but one successful short chunk length does not prove the neighboring lengths will be reliable

So chunk size is a second-order lever after:

- model tier
- media resolution

## Temperature And Context

Recommended defaults:

- `temperature=0.0`
- retry at `0.3`
- `rolling_context_chunks=0` on fragile Gemini video-only paths
- do not assume higher thinking helps transcription; test it explicitly before paying the latency/cost

Reason:

- low temperature keeps the transcript more literal
- zero rolling context reduces loop/repetition carryover on weaker models
- transcription failures so far have looked more like visual-density / lexical-ASR failures than under-reasoning failures
- on `great_escape_s02e01` Flash Lite probes, `temperature=0.0`, `0.1`, and `0.2` all completed on the same stable `120s` section, but `0.3` hit `504 DEADLINE_EXCEEDED`
- `0.0` was the cleanest of the cheap Flash Lite probes structurally; `0.1` compressed more aggressively; `0.2` added slightly more drift

Current model-specific thinking guidance:

- `gemini-3-flash-preview`:
  - `video + spoken_only + low` is currently the best compromise setting we have tested
  - `high` is usable, but more cleanup-oriented
  - `medium` looped on the beach chunk and should be avoided on that scene type
- `gemini-2.5-flash`:
  - keep default thinking
  - turning thinking fully off regressed badly on the beach scene

## Cost Strategy

Current cost policy:

- use `count_tokens` preflight whenever exact input cost matters
- treat Pro runs as scarce / intentional
- budget with retries, not just ideal-case single-pass estimates
- after changing transcription code or switching model tier, validate with
  `--stop-after-chunks 1` before committing a full-episode run

Operational heuristic:

- Flash-tier models are cheap enough for chunk-shape and prompt-shape experiments
- Pro-tier experiments should be run only after the experiment design is already narrowed down

## Escalation / Fallback Ladder

When a model struggles, escalate in this order:

1. keep `1 FPS`
2. set `media_resolution=high`
3. shorten chunks
4. keep `temperature` low and `rolling_context_chunks=0`
5. move up to a better Flash-tier model
6. move up to Pro only if the quality gap still matters

Do not assume shorter chunks alone will rescue a weak model on dense visual sections.

Operational stop rules:

- if one chunk hits repeated timeout failures, split that chunk and resume
  rather than increasing retries indefinitely
- if the API starts returning repeated quota/rate-limit errors, stop and resume
  after reset instead of letting the helper sit in long backoff loops
- if one chunk returns suspiciously large output, treat that as a chunk-local
  failure and repair/split it before trusting the raw transcript

## Recommended Evaluation Pattern

Do not spend full-episode runs first.

Use 2-3 diagnostic chunks:

- cold open / cast-card chunk
- dense rule-card or mission-text chunk
- one lexical-fidelity chunk with known difficulty

If those chunks are not better, do not spend a full-episode run.

For manual AI Studio experiments:

- audio runs should use `spoken_only`
- video runs can compare `spoken_only` vs `spoken_plus_visual`
- record any UI safety-setting changes as part of the result, especially on distress-heavy scenes

## Current Recommendation

If choosing today:

- stick to Flash-tier models for most transcription work
- use `media_resolution=high` for meaningful visual-heavy evaluation
- keep temperature at `0.0`
- fall back to shorter chunks before wasting a full run
- escalate to a stronger model tier when looping, timeout, or lexical fidelity remains unacceptable

For the current free tier specifically:

- treat `gemini-3-flash-preview` as the primary “one episode per day” model
- use `gemini-2.5-flash` as the secondary comparison path when quota allows
- keep `gemini-3.1-flash-lite-preview` for auxiliary OCR/audio-draft experiments, not the main transcript
