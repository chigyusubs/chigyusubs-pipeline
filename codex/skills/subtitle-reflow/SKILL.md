---
name: subtitle-reflow
description: Reflow aligned Japanese subtitle artifacts into translation-ready VTTs using the maintained line-level path, review the output for translation-risk issues, and optionally run the local cue-repair workflow when the reflowed VTT is weak but structurally valid. Use when the user wants Codex to prepare or review a Japanese subtitle file between CTC alignment and translation.
---

# Subtitle Reflow

Use this skill when Codex should prepare or evaluate the Japanese subtitle artifact before translation.

Default maintained tools:

- `scripts/reflow_words.py`
- `scripts/repair_vtt_local.py`

Do not default to experimental reflow scripts when the maintained line-level path is available.

## Core Rules

- Prefer `reflow_words.py --line-level` for CTC alignment output.
- Treat line-level reflow as the default path and local cue repair as conditional fallback.
- Preserve the original reflow artifact if repair is run.
- Stop on structural timing blockers instead of trying to LLM-repair them.
- Optimize for translation readiness, not perfect Japanese subtitle polish.

## Input Shape

Accept any of:

- `*_ctc_words.json`
- an existing reflowed `*.vtt`
- both

If only `*_ctc_words.json` is available, derive the default reflow output path in the sibling `transcription/` directory as `<stem>_reflow.vtt`.

If both are available, treat the VTT as the current artifact under review.

See [references/workflow.md](./references/workflow.md) for repo-specific commands, review gates, and output naming.

## Workflow

### 1. Reflow

Use the maintained default:

- `PYTHONPATH=. python3 scripts/reflow_words.py --line-level --stats`

Do not switch to word-level reflow unless the user explicitly asks for it or the input is not a CTC-aligned artifact.

### 2. Review

Inspect the resulting VTT for:

- negative cue durations
- cue overlaps
- obvious split-word or split-clause artifacts
- pathological short fragment clusters
- clearly broken or translation-hostile line breaks

Review should combine:

- objective file-level checks
- spot-checks from early, middle, and late regions

### 3. Decision

Use this policy:

- `green`: structurally valid and translation-ready
- `yellow`: structurally valid but weak enough that cue repair is justified
- `red`: structurally broken or obviously corrupt; stop and report

### 4. Optional Repair

Only run local repair on `yellow`.

Repair requirements:

- input VTT exists
- aligned `*_ctc_words.json` exists
- local cue-repair server is reachable

Default repair stack:

- server: `scripts/start_gemma_cue_repair_server.sh`
- client: `scripts/repair_vtt_local.py`

Default repair output naming:

- VTT: `<stem>_reflow_repaired.vtt`
- decisions: `<stem>_reflow_repaired.decisions.json`

Do not overwrite the original reflow VTT.

### 5. Handoff

End by naming the single recommended Japanese VTT path for translation and summarizing:

- structural status
- whether repair was skipped, run, or blocked
- the main remaining translation risks, if any

## What Not To Do

- Do not run repair on every episode by default.
- Do not use cue repair to hide negative durations or overlaps.
- Do not silently proceed to translation when review found a real blocker.
- Do not discard the original reflow artifact when a repaired artifact is created.
