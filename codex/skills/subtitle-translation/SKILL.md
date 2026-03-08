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
- hard CPS: about 20
- preferred line length: about 42 chars
- max lines: 2

### 3. Batch the Work

Translate in local cue batches, not the whole file at once.

- Use nearby context on both sides.
- Keep the current batch small enough to reason about coherence.
- Carry forward terminology, names, and recurring jokes consistently.
- Default to the repo workflow's `84 -> 60 -> 48` batch-tier policy.

For long files, use the helper script's session/checkpoint in `translation/`, not ad hoc JSON edits.

### 4. Translate

For each batch:

- keep cue timings unchanged
- keep cue count unchanged
- rewrite Japanese into natural English subtitle lines
- compress where needed for readability
- do not leave empty cues
- do not invent content that is unsupported by the source

If one cue is semantically weak in isolation, use adjacent cues to distribute meaning more naturally, but still keep one non-empty output per cue.

### 5. Write Incrementally

- Save progress after every batch through `apply-batch`.
- Let the helper regenerate the partial VTT and diagnostics.
- Keep reruns resumable from the last completed batch.

### 6. Review

After translation:

- verify cue count/order/timings still match
- review hard CPS outliers
- review very short cues that became awkward in English
- review named entities and recurring terms
- summarize any risky regions for the user

## When To Use The Script Instead

Use `scripts/translate_vtt.py` only when the user explicitly wants an API-backed run or hard runtime controls for backend/model/location/temperature.

If the user wants Codex itself to do the translator role, stay in the interactive workflow and do not offload the actual translation to Gemini/OpenAI automatically.
