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
- cues under `0.5s`
- split-word or split-clause artifacts
- pathological transcript repetition
- unusable OCR or visual-text junk carried into dialogue cues

Do not translate clearly broken subtitle timing artifacts without flagging them.
Treat any cue under `0.5s` as a structural blocker for the handoff, even if the text itself is technically correct.
If `translate_vtt_codex.py prepare` loads alignment diagnostics, treat those interpolation warnings as advisory context for the affected cues, not as a structural blocker.
When using `translate_vtt_codex.py next-batch`, preserve each target cue's `source_text_hash` into the corresponding translation item you pass back to `apply-batch`; cue IDs alone are no longer treated as sufficient validation.

If you reuse an older English draft as seed material, only do it through a path
that validates exact cue-count and cue-timeline equality against the current
Japanese source. Never replay old English by cue index when reflow has changed.

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

When alignment diagnostics are available, the session and per-batch payloads should also retain:

- episode-level counts of interpolated all-unaligned source lines
- affected cue IDs
- sampled repaired source lines for the current batch

## Trigger Phrases

This skill should trigger on requests like:

- translate this VTT with Codex
- use Codex as the subtitle translator
- do the subtitle translation here, not via Gemini
- batch-translate this subtitle file interactively
