# Subtitle Translation Workflow

## Purpose

This reference captures the repo-specific expectations for Codex-interactive subtitle translation.

## Repo Defaults

- preserve cue count
- preserve cue order
- preserve cue timings
- optimize for natural English subtitle editing
- target about 17 CPS
- treat about 20 CPS as a hard warning
- prefer max 2 lines
- prefer about 42 chars per line

## Input Quality Gates

Before translating, inspect for:

- negative cue durations
- cue overlaps
- split-word or split-clause artifacts
- pathological transcript repetition
- unusable OCR or visual-text junk carried into dialogue cues

Do not translate clearly broken subtitle timing artifacts without flagging them.

## Session Preferences

The user may specify:

- preferred model
- preferred thinking level
- preferred temperature

In this Codex-interactive path, treat those as working preferences unless the current Codex product/runtime explicitly exposes those controls. Record them in notes or checkpoint metadata when useful, but do not claim they were enforced automatically.

## Recommended Checkpoint Shape

If no existing checkpoint is present, a small JSON file next to the output is enough:

```json
{
  "input": "source.vtt",
  "output": "source_en.vtt",
  "target_lang": "English",
  "preferred_model": "gpt-5.4",
  "preferred_thinking": "medium",
  "preferred_temperature": 0.2,
  "completed_batches": [0, 1, 2]
}
```

The exact format can vary; resumability matters more than schema purity.

## Trigger Phrases

This skill should trigger on requests like:

- translate this VTT with Codex
- use Codex as the subtitle translator
- do the subtitle translation here, not via Gemini
- batch-translate this subtitle file interactively
