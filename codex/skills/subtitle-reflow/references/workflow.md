# Subtitle Reflow Workflow

## Purpose

This reference captures the repo-specific expectations for Codex-interactive Japanese reflow preparation before translation.

## Repo Defaults

- use `scripts/reflow_words.py --line-level` for CTC output
- prefer `transcription/<stem>_reflow.vtt` as the default reflow artifact
- keep original and repaired artifacts separate
- treat repair as fallback, not the default path
- optimize for translation readiness

## Default Commands

### Reflow from aligned words

```bash
PYTHONPATH=. python3 scripts/reflow_words.py \
  --input samples/episodes/<slug>/transcription/<stem>_ctc_words.json \
  --output samples/episodes/<slug>/transcription/<stem>_reflow.vtt \
  --line-level --stats
```

### Optional local repair

Expected server:

```bash
scripts/start_gemma_cue_repair_server.sh
```

Repair command:

```bash
python3 scripts/repair_vtt_local.py \
  --input-vtt samples/episodes/<slug>/transcription/<stem>_reflow.vtt \
  --input-words samples/episodes/<slug>/transcription/<stem>_ctc_words.json \
  --output-vtt samples/episodes/<slug>/transcription/<stem>_reflow_repaired.vtt \
  --output-decisions samples/episodes/<slug>/transcription/<stem>_reflow_repaired.decisions.json
```

## Review Gates

### Immediate Red

- any negative-duration cue
- any cue overlap
- obviously corrupt or nonsensical VTT output

On `red`, stop. Do not run local cue repair.

### Yellow

Use `yellow` when the file is structurally valid but clearly weak for translation, for example:

- obvious split-word or split-clause artifacts in sampled regions
- a visible pattern of very short fragment cues
- localized pathological regions that would clearly damage translation

On `yellow`, run repair if the aligned words and local server are available. If they are not available, report that repair is indicated but unavailable.

### Green

Use `green` when:

- no structural blocker exists
- sampled regions read cleanly
- no obvious split-word or fragment-cluster problem remains

## Review Method

Check all of:

- structural timing sanity
- cue density and fragment patterns
- opening sample
- middle sample
- ending sample

The purpose is not exhaustive QA. The purpose is to decide whether the VTT is safe to hand to translation.

## Output Policy

Always end with one recommended artifact path:

- original reflow VTT if `green`
- repaired VTT if repair materially improved the file
- no recommendation if the file is `red`

Keep the summary short and explicit:

- structural status
- repair status
- translation readiness

## Trigger Phrases

This skill should trigger on requests like:

- reflow this aligned episode
- prep this VTT for translation
- review the reflow
- run reflow and fix weak cue boundaries
- get this subtitle file translation-ready
