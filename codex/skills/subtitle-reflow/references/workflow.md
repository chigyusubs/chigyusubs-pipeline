# Subtitle Reflow Workflow

## Purpose

This reference captures the repo-specific expectations for Codex-interactive Japanese reflow preparation before translation.

## Repo Defaults

- use `scripts/reflow_words.py --line-level` for CTC output
- prefer `transcription/<stem>_reflow.vtt` as the default reflow artifact
- keep original and repaired artifacts separate
- treat Codex-interactive repair as fallback, not the default path
- optimize for translation readiness

## Default Commands

### Reflow from aligned words

```bash
python3 scripts/check_raw_chunk_sanity.py \
  --input samples/episodes/<slug>/transcription/<stem>_gemini_raw.json

# Stop here and repair the raw transcript first if the report contains any
# red chunk. Yellow chunks are review targets, not automatic blockers.

# Optional, not automatic:
# Run only when alignment diagnostics flag possible visual narration
# substitution, or when a reusable whole-episode secondary transcript already
# exists and can be compared without rerunning faster-whisper.
python3.12 scripts/pre_reflow_second_opinion.py \
  --words samples/episodes/<slug>/transcription/<stem>_ctc_words.json

PYTHONPATH=. python3 scripts/reflow_words.py \
  --input samples/episodes/<slug>/transcription/<stem>_ctc_words.json \
  --output samples/episodes/<slug>/transcription/<stem>_reflow.vtt \
  --line-level --stats
```

Important:

- do not rerun faster-whisper by default during reflow prep just because a
  sibling `*_gemini_raw.json` exists
- if there is no reusable whole-episode secondary artifact, proceed to reflow
  and keep the missing second-opinion note explicit in review

### Optional Codex-interactive repair

```bash
python3 scripts/repair_vtt_codex.py prepare \
  --input samples/episodes/<slug>/transcription/<stem>_reflow.vtt \
  --words samples/episodes/<slug>/transcription/<stem>_ctc_words.json \
  --output samples/episodes/<slug>/transcription/<stem>_reflow_repaired.vtt

# If <stem>_ctc_words.json.diagnostics.json exists, prepare auto-loads it and
# carries interpolated alignment lines into session diagnostics and region payloads.
# If <stem>_ctc_words.json preserves Gemini turn metadata, prepare also carries
# cue-level turn-boundary context for merged multi-turn cues.
```

Then iterate:

```bash
python3 scripts/repair_vtt_codex.py next-region \
  --session samples/episodes/<slug>/transcription/<stem>_reflow_repaired.vtt.checkpoint.json

python3 scripts/repair_vtt_codex.py apply-region \
  --session samples/episodes/<slug>/transcription/<stem>_reflow_repaired.vtt.checkpoint.json \
  --repair-json /tmp/<stem>_repair_region.json

python3 scripts/repair_vtt_codex.py finalize \
  --session samples/episodes/<slug>/transcription/<stem>_reflow_repaired.vtt.checkpoint.json
```

## Review Gates

### Immediate Red

- any negative-duration cue
- any cue overlap
- any cue under `0.5s`
- obviously corrupt or nonsensical VTT output

On `red`, stop. Do not run repair in this workflow.

### Yellow

Use `yellow` when the file is structurally valid but clearly weak for translation, for example:

- cue timing visibly starts too early before the spoken line in sampled regions
- obvious split-word or split-clause artifacts in sampled regions
- artifact-like short clusters or non-terminal fragments
- localized pathological regions that would clearly damage translation

On `yellow`, use `repair_vtt_codex.py` and keep the repair local to the flagged regions. Preserve the full source text of each region and let the helper rebuild timings deterministically inside the region span.

### Green

Use `green` when:

- no structural blocker exists
- sampled regions read cleanly
- no obvious split-word or fragment-cluster problem remains

## Review Method

Check all of:

- structural timing sanity
- cue density and fragment patterns
- whether cue starts are anchored near the first spoken line instead of popping in conspicuously early
- opening sample
- middle sample
- ending sample
- deterministic helper metrics:
- negative durations / overlaps
- hard-stop micro cues under `0.5s`
- advisory counts for short cues under `0.8s` and `1.0s`
- advisory counts for tiny cues (`<=4` chars)
- flagged region ranges and sampled previews
- advisory alignment warnings for cues that overlap interpolated all-unaligned source lines

Important policy:

- cues under `0.5s` should stop the handoff
- short/tiny cues above that threshold alone should not force repair
- interpolated alignment warnings alone should not force repair
- plausible fast reactions and one-word answers are acceptable
- a small anticipatory lead is acceptable, but cues that sit in silence long before the line are not
- repair should target artifact-like boundaries, not optimize the metrics mechanically

The purpose is not exhaustive QA. The purpose is to decide whether the VTT is safe to hand to translation.

## Output Policy

Always end with one recommended artifact path:

- original reflow VTT if `green`
- repaired VTT if repair materially improved the file
- no recommendation if the file is `red`

Keep the summary short and explicit:

- structural status
- repair status
- before/after deterministic metrics if repair ran
- translation readiness

## Trigger Phrases

This skill should trigger on requests like:

- reflow this aligned episode
- prep this VTT for translation
- review the reflow
- run reflow and fix weak cue boundaries
- get this subtitle file translation-ready
