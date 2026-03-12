# Transcription Research Roadmap

This document tracks promising transcription ideas that are not yet the maintained default.

Use it for:

- scene-dependent model-setting hypotheses
- hybrid pipeline ideas
- Codex-skill candidates
- focused experiment queues

For validated production guidance, see:

- `docs/gemini-transcription-playbook.md`
- `docs/lessons-learned.md`

## Current Working Hypotheses

These are not stable rules yet. They are the best current read from the `great_escape_s02e01` AI Studio and API probes.

### 1. Thinking level is modality-sensitive on Gemini 3.x

Current evidence suggests:

- for `video`, lower thinking is often safer than higher thinking
- for `audio`, higher thinking can sometimes recover more structure from audio alone

But this is not universal.

Observed examples:

- `gemini-3-flash-preview` beach scene:
  - `video + spoken_plus_visual + low` was the best 3-flash setting
  - `video + spoken_plus_visual + medium` looped catastrophically
  - `video + spoken_plus_visual + minimal` regressed on names (`しんいち` -> `森口`)
- `gemini-3-flash-preview` rule block:
  - `audio + spoken_only + high` was clearly better than `audio + spoken_only + minimal`
- `gemini-3.1-pro-preview` rule block:
  - `video + spoken_plus_visual + low` beat the `high` variant
  - `audio + spoken_only + low` also beat the `high` variant

So the right conclusion is:

- do not use one global thinking policy for all modalities and scene types
- treat thinking as a scene-dependent control, not a monotonic quality knob

### 2. Video is still the best single default, but audio-only is better than expected

Video remains the safest one-size-fits-most mode because it preserves:

- rule cards
- cast cards
- premise text
- prompt text
- scene grounding

But strong models recovered much more from audio-only than expected, especially on:

- banter-heavy dialogue
- some prompt-reading scenes
- cases where video prompting encouraged visual overreach

This suggests a possible future hybrid architecture:

- audio-first for cheap spoken coverage
- targeted video reruns only on visually important or suspicious chunks

### 3. Model ranking is scene-dependent

On current evidence:

- beach / comedy-banter scene:
  - `gemini-2.5-pro` looked strongest
  - `gemini-2.5-flash` was close behind
  - `gemini-3-flash-preview` became competitive with better settings
- rule / prompt-reading scene:
  - `gemini-3.1-pro-preview` with `video + spoken_plus_visual + low thinking` was highly competitive
  - `gemini-3-flash-preview` also performed well

So “best model” should probably be chosen by chunk type, not only by episode.

## Hybrid Pipeline Ideas

### Idea A. Faster-Whisper first, then smarter chunking

Proposal:

1. run `faster-whisper large-v3` early on the episode
2. combine:
   - faster-whisper text
   - faster-whisper word timings
   - Silero VAD
3. derive smarter chunks that are:
   - speech-aware
   - semantically coherent
   - less likely to split setup/punchline pairs
   - less likely to blend dense visual prompts into surrounding chatter

Potential value:

- better chunk boundaries before Gemini
- lower loop risk on banter-heavy chunks
- more targeted video reruns on suspicious regions

Main risk:

- turning “semantic chunking” into a brittle heuristic mess

Recommendation:

- prototype as an experiment, not as the default chunker
- preserve all intermediate artifacts

### Idea B. Codex skill for chunk planning / arbitration

Proposal:

Create a new Codex skill that reads:

- Silero VAD
- faster-whisper transcript
- OCR spans/context
- optionally prior Gemini raw output

and emits:

- revised chunk boundaries
- chunk labels like:
  - `banter`
  - `visual_prompt`
  - `name_card`
  - `suspicious_audio_only_candidate`
  - `needs_video_rerun`

Potential value:

- explicit reasoning about chunk type
- easier hybrid per-chunk modality selection
- reusable planning artifact for experiments

Main risk:

- over-automating what should remain a reviewable advisory step

Recommendation:

- make this a planning skill first
- its output should be a saved JSON artifact, not hidden decisions

### Idea C. Revive OCR context, but as a separate evidence layer again

Proposal:

Use the Qwen OCR pipeline again for:

- glossary candidates
- visual cues
- scene-local prompt text
- cast/name extraction

Then use a Codex skill to turn OCR artifacts into one or both of:

- `glossary/glossary.json`
- chunk-local visual cue hints

Possible modes:

- episode-wide glossary build
- chunk-wise visual-cue extraction
- chunk-wise “line_hints + keyword_hints” generation

Why this may be worth revisiting:

- current video-only Gemini path is strong, but still misses exact visible text sometimes
- OCR can be reused cheaply once extracted
- this keeps visual support separate from spoken transcript generation

Main risk:

- poisoning the transcript if OCR hints are injected too aggressively

Recommendation:

- keep OCR outputs explicit and inspectable
- do not force OCR-derived text directly into spoken transcript format

### Idea D. Flash-Lite as a cheap auxiliary model, not the main model

Proposal:

Use `gemini-3.1-flash-lite-preview` in narrowly scoped auxiliary roles:

- OCR-like visual extraction
- cheap audio-only transcript draft
- cheap chunk probes
- cheap rerun candidate on high-volume tasks

Specific idea to test:

- `flash-lite`
- audio-only
- high thinking

Reason:

- stronger Flash models showed that audio-only can recover more than expected
- Flash Lite might still be useful as a cheap spoken draft if the role is limited and downstream review is explicit

Current evidence on the `beach_panel_banter` clip:

- `flash-lite + audio + high thinking` was the best Flash Lite beach result so far
- but `faster-whisper large-v3` still beat it on the key lexical trap (`オーシャンビューよ` vs Flash Lite `大さん橋よ`)
- that keeps Flash Lite in the auxiliary-draft bucket, not the benchmark bucket

Important constraint:

- do not treat Flash Lite as the main source of truth
- it remains an auxiliary/budget model unless evidence improves materially

## Candidate New Skills

### 1. `chunk-planner`

Inputs:

- VAD segments
- faster-whisper transcript
- OCR spans/context
- optional Gemini raw transcript

Outputs:

- revised chunk boundary plan
- chunk taxonomy labels
- modality recommendation per chunk

### 2. `visual-context-builder`

Inputs:

- OCR JSONL
- OCR spans
- episode transcript if available

Outputs:

- glossary candidates
- visual cue candidates
- chunk-local prompt hints

### 3. `transcription-arbiter`

Inputs:

- primary Gemini output
- faster-whisper output
- OCR-derived visual text
- chunk metadata

Outputs:

- ranked suspicious regions
- recommendation:
  - keep primary
  - patch from Whisper
  - rerun as audio-only
  - rerun as video

## Near-Term Experiment Queue

High priority:

1. Compare `faster-whisper + Silero` driven semantic chunk proposals against current VAD chunks on one episode.
2. Prototype chunk labels:
   - `banter`
   - `visual_prompt`
   - `mixed`
3. Test whether OCR-derived chunk hints improve only the visually dense chunks without harming banter chunks.
4. Test `flash-lite` as a cheap audio-only auxiliary draft with higher thinking.

Medium priority:

1. Build a saved JSON artifact for chunk-type recommendations.
2. Try per-chunk modality routing:
   - banter -> audio
   - prompt/rule -> video
3. Evaluate whether Codex should make the routing decision or only review it.

Low priority:

1. Revisit FPS once modality/routing is clearer.
2. Revisit prompt wording only after chunking and routing stabilize.

## Current Provisional Recommendation

If work had to continue today:

- keep the maintained pipeline simple
- keep `video` as the global default
- use stronger models for primary transcript quality
- use faster-whisper as the standing second-opinion artifact

But the most promising research direction is:

- chunk-aware hybrid routing, not one universal model/mode for the whole episode
