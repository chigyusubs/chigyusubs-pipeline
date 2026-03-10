# Codex Skills

This repo now has two Codex-interactive skills for subtitle work:

- `subtitle-reflow`
- `subtitle-translation`

These skills are not replacements for the maintained scripts. They are Codex
workflows that make the interactive agent use the existing repo tools and review
logic consistently.

The canonical tracked skill source now lives in:

```text
codex/skills/
```

The live install target is still usually:

```text
~/.codex/skills/
```

Use the repo helper to install or refresh them:

```bash
python3 scripts/install_codex_skills.py
python3 scripts/install_codex_skills.py --skill subtitle-reflow
```

## When To Use Them

Use the skills when the user wants Codex itself to do the work interactively in
this session rather than sending the task to an external translation or repair
API.

Use the maintained scripts directly when:

- you want a fully API-backed run
- you need hard backend/model/location controls
- you are benchmarking non-Codex paths

## `subtitle-reflow`

Purpose:

- prepare a Japanese subtitle artifact for translation
- keep the default path deterministic
- only escalate to Codex-interactive cue repair when the reflowed VTT is weak enough to hurt
  translation

Default behavior:

1. Run maintained line-level reflow:

```bash
PYTHONPATH=. python3 scripts/reflow_words.py \
  --input samples/episodes/<slug>/transcription/<stem>_ctc_words.json \
  --output samples/episodes/<slug>/transcription/<stem>_reflow.vtt \
  --line-level --stats
```

2. Review the VTT for:

- negative cue durations
- cue overlaps
- split-word or split-clause artifacts
- pathological short-fragment clusters
- obviously translation-hostile cue boundaries

3. Decide:

- `green`: recommend the reflowed VTT for translation
- `yellow`: run Codex-interactive cue repair
- `red`: stop and report the blocker

Optional repair path:

```bash
python3 scripts/repair_vtt_codex.py prepare \
  --input samples/episodes/<slug>/transcription/<stem>_reflow.vtt \
  --words samples/episodes/<slug>/transcription/<stem>_ctc_words.json \
  --output samples/episodes/<slug>/transcription/<stem>_reflow_repaired.vtt

python3 scripts/repair_vtt_codex.py next-region \
  --session samples/episodes/<slug>/transcription/<stem>_reflow_repaired.vtt.checkpoint.json

python3 scripts/repair_vtt_codex.py apply-region \
  --session samples/episodes/<slug>/transcription/<stem>_reflow_repaired.vtt.checkpoint.json \
  --repair-json /tmp/<stem>_repair_region.json

python3 scripts/repair_vtt_codex.py finalize \
  --session samples/episodes/<slug>/transcription/<stem>_reflow_repaired.vtt.checkpoint.json
```

Important:

- line-level reflow is still the default
- cue repair is conditional fallback, not the default path on every episode
- the default repair path is Codex-interactive and does not require a local model server
- the repair helper now writes deterministic before/after metrics, sampled flagged regions, and one recommended Japanese VTT path for translation
- the original reflow VTT should be preserved if repair runs

Example invocation in Codex:

```text
Use $subtitle-reflow on samples/episodes/<slug>/transcription/<stem>_ctc_words.json
```

## `subtitle-translation`

Purpose:

- translate a Japanese VTT or SRT into natural English with Codex itself acting
  as the subtitle editor
- keep the process checkpointed and resumable

Default behavior:

1. Prepare a session:

```bash
python3 scripts/translate_vtt_codex.py prepare \
  --input samples/episodes/<slug>/transcription/<stem>_reflow.vtt \
  --output samples/episodes/<slug>/translation/<stem>_reflow_en_codex.vtt
```

If a sibling CTC alignment diagnostics sidecar exists, `prepare` auto-loads it and later `next-batch` payloads include advisory `alignment_warnings` for affected cues.

2. Work batch-by-batch with:

- `next-batch`
- Codex translation of the emitted cues
- `apply-batch`
- `finalize`

The maintained helper automatically:

- preserves cue count, order, and timings
- writes a checkpoint/session JSON
- writes a partial VTT in `translation/`
- writes a deterministic batch summary in diagnostics
- carries advisory alignment warnings from CTC diagnostics into batch payloads and diagnostics when present
- uses the `84 -> 60 -> 48` batch-tier fallback
- keeps minimum-tier CPS overruns as diagnostics/warnings instead of auto-stopping the whole run
- continues through `yellow` batches by default; only structural errors or explicit `red` stop the session
- clears old session/output/partial/diagnostics artifacts when restarted with `prepare --force`

Example invocation in Codex:

```text
Use $subtitle-translation on samples/episodes/<slug>/transcription/<stem>_reflow.vtt
```

## Recommended Handoff

The intended order is:

```text
aligned words
  -> subtitle-reflow
  -> recommended Japanese VTT
  -> subtitle-translation
  -> English VTT in translation/
```

Use `subtitle-reflow` first when the Japanese subtitle artifact still needs
boundary review or repair.

Use `subtitle-translation` once there is a single recommended Japanese VTT that
is safe to hand to English subtitle editing.

## Limits

The skills are interactive Codex workflows, not callable model backends.

That means:

- they help Codex perform the work here in-session
- they do not replace `translate_vtt_api.py` or `repair_vtt_codex.py` as local
  libraries
- model/thinking/temperature settings in the Codex-interactive path are working
  preferences unless the product runtime exposes hard controls

## Source Of Truth

Rules:

- edit tracked skills in `codex/skills/`
- install them into `~/.codex/skills/` with `scripts/install_codex_skills.py`
- do not treat the live home copy as the canonical version
- do not copy `.system` skills into the repo
