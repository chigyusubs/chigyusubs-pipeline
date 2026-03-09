---
name: subtitle-translation
description: Translate subtitle files such as VTT and SRT into natural English using Codex itself as the subtitle editor, while preserving cue count, cue order, and timings. Use when the user wants Codex to perform batch-by-batch subtitle translation interactively instead of calling an external translation API, especially for Japanese variety-show workflows with glossary use, CPS constraints, checkpointing, resumable progress, and transcript-quality review before translation.
---

# Subtitle Translation

Use this skill when Codex should perform the translation work directly in the session. Do not default to `scripts/translate_vtt.py` or another external model call unless the user explicitly asks for an API-backed run.

Use `scripts/translate_vtt_codex.py` as the maintained helper workflow:

- `prepare`
- `next-batch`
- `apply-batch`
- `status`
- `finalize`

If you need to restart a translation from a clean base, use `prepare --force`.
That should be treated as a fresh session reset, not a resume.

## Core Rules

- Preserve cue count, cue order, and cue timings exactly.
- Translate as subtitle editing, not literal translation.
- Keep every cue non-empty.
- Prefer natural English that reads well on screen.
- Preserve speaker and scene continuity across nearby cues.
- Treat reflow problems as blockers when they are severe enough to damage translation.

## Configuration

Ask once or infer from repo context:

- source subtitle file
- output subtitle path
- glossary path, if any
- show summary/context, if any
- preferred model (`gpt-5.4` by default)
- preferred thinking level
- preferred temperature

Important:

- In Codex-interactive mode, `model`, `thinking level`, and `temperature` are session preferences, not hard runtime controls. Record them in the working notes or checkpoint, but do not claim they were enforced unless the platform explicitly exposes those controls.
- If the user wants hard model/temperature control, that is the API-backed path, not this skill’s default path.

See [references/workflow.md](./references/workflow.md) for repo-specific defaults and wording.

## Workflow

### 1. Preflight

- Inspect the input VTT/SRT before translating.
- Check for:
  - negative cue durations
  - cue overlaps
  - obviously broken reflow
  - pathological repetition
  - extreme CPS or unreadable line breaks
- If those problems are severe, stop and repair or ask the user whether to repair first.

### 2. Set Translation Shape

Use these defaults unless the user says otherwise:

- target language: English
- subtitle style: natural, concise, readable
- target CPS: about 17
- hard CPS: about 20 as a warning/review threshold, not a stop condition by itself
- preferred line length: about 42 chars
- max lines: 2

### 3. Batch Loop

**This is the core of the workflow. You MUST loop until all cues are translated or a `red` review stops the session.**

Repeat this cycle:

1. Run `next-batch` to get the current batch payload.
2. Translate every cue in the batch (see translation rules below).
3. Write the translations JSON and run `apply-batch`.
4. Check the apply output: if status is `completed` or `stopped`, exit the loop. Otherwise, go back to step 1 immediately.

Do NOT stop after a single batch. Do NOT wait for user confirmation between batches unless the review is `red`. A `yellow` review means continue with the next batch (the tier auto-downgrades). A `green` review means continue normally.

Batch settings:

- Use nearby context on both sides.
- Keep the current batch small enough to reason about coherence.
- Carry forward terminology, names, and recurring jokes consistently.
- Default to the repo workflow's `84 -> 60 -> 48` batch-tier policy.

For long files, use the helper script's session/checkpoint in `translation/`, not ad hoc JSON edits.

### 4. Translation Rules

For each cue in a batch:

- keep cue timings unchanged
- keep cue count unchanged
- rewrite Japanese into natural English subtitle lines
- compress where needed for readability
- do not leave empty cues
- do not invent content that is unsupported by the source

If one cue is semantically weak in isolation, use adjacent cues to distribute meaning more naturally, but still keep one non-empty output per cue.

### Visual Cues

If `visual_cues` is present in the batch payload, these are on-screen text elements
from the video (name plates, captions, game instructions, location text). They have
chunk-level timestamps, not cue-level precision.

- Use visual cues to resolve ambiguous names, quiz content, and game rules
- They are context only — do not emit them as subtitle lines
- On-screen instructions often clarify what speakers are referring to

### 5. Write Incrementally

- Save progress after every batch through `apply-batch`.
- Let the helper regenerate the partial VTT and diagnostics.
- Keep reruns resumable from the last completed batch.
- If you intentionally restart with `prepare --force`, expect the helper to clear the old session, partial output, final output, and diagnostics so stale batch history does not leak into the new run.
- After each `apply-batch`, immediately loop back to `next-batch` — do not pause or summarize between batches.

### 6. Review

After translation:

- verify cue count/order/timings still match
- review the helper's deterministic batch summary and hard CPS outliers
- review very short cues that became awkward in English
- review named entities and recurring terms
- summarize any risky regions for the user

Important:

- hard CPS diagnostics are useful, but in this repo they are still noisy on short cues
- do not stop a whole Codex-interactive run on CPS or ordinary warning-level issues when the actual subtitle quality is still acceptable

## When To Use The Script Instead

Use `scripts/translate_vtt.py` only when the user explicitly wants an API-backed run or hard runtime controls for backend/model/location/temperature.

If the user wants Codex itself to do the translator role, stay in the interactive workflow and do not offload the actual translation to Gemini/OpenAI automatically.
