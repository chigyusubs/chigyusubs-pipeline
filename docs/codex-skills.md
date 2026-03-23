# Codex Skills

This repo now has five Codex-interactive skills for subtitle work:

- `chunk-review`
- `subtitle-reflow`
- `glossary-context`
- `speaker-diarization`
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
python3 scripts/install_codex_skills.py --skill chunk-review
python3 scripts/install_codex_skills.py --skill subtitle-reflow
python3 scripts/install_codex_skills.py --skill glossary-context
python3 scripts/install_codex_skills.py --skill speaker-diarization
python3 scripts/install_codex_skills.py --skill subtitle-translation
```

## When To Use Them

Use the skills when the user wants Codex itself to do the work interactively in
this session rather than sending the task to an external translation or repair
API.

Use the maintained scripts directly when:

- you want a fully API-backed run
- you need hard backend/model/location controls
- you are benchmarking non-Codex paths

## `chunk-review`

Purpose:

- replace purely acoustic VAD-based chunking with semantically informed boundaries
- use a faster-whisper pre-pass transcript so Codex can judge whether each
  silence gap is a natural sentence boundary

Default behavior:

1. Run the prepare step (Silero VAD + faster-whisper pre-pass):

```bash
python scripts/build_semantic_chunks.py prepare \
  --video samples/episodes/<slug>/source/video.mp4
```

2. Loop through candidates: `next-candidate` -> Codex decision -> `apply-candidate`

3. Finalize:

```bash
python scripts/build_semantic_chunks.py finalize \
  --session samples/episodes/<slug>/transcription/vad_chunks.json.checkpoint.json
```

Important:

- the only decision per candidate is: sentence boundary (split) or mid-sentence (skip)
- do not evaluate topic/scene changes, only sentence boundaries
- use both duration and transcript density as chunk budgets
- when approaching max chunk duration or transcript character budget, bias toward splitting
- output is standard `vad_chunks.json` consumed by downstream transcription scripts

Example invocation in Codex:

```text
Use $chunk-review on samples/episodes/<slug>/source/video.mp4
```

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

## `glossary-context`

Purpose:

- build episode translation context before English subtitle editing
- extract stable show terms and episode-local names/rules from the Gemini raw transcript

Default behavior:

1. Read `transcription/<slug>_gemini_raw.json`
2. Use both spoken `-- ...` lines and visual `[画面: ...]` lines
3. If present, also use `ocr/*_flash_lite_chunk_ocr.json` as supporting evidence for visible spellings, names, title cards, and rule text
4. Write:

```text
glossary/glossary.json
glossary/episode_context.json
```

Important:

- `glossary.json` should hold stable cross-episode terms
- `episode_context.json` should hold episode-local cast, guests, rule text, and one-off terms
- the maintained interactive flow expects this step between Japanese reflow review and English translation

Example invocation in Codex:

```text
Use $glossary-context on samples/episodes/<slug>/transcription/<slug>_gemini_raw.json
```

## `speaker-diarization`

Purpose:

- identify anonymous voice clusters from `speaker_map.json` by cross-referencing
  sample turns with the glossary
- produce `named_speaker_map.json` with per-cluster identifications, merge decisions,
  and a pre-resolved effective speakers rollup

Default behavior:

1. Read `transcription/*_speaker_map.json` — extract per-cluster stats and ~10 longest
   sample turns per cluster
2. Read `glossary/glossary.json` + `glossary/episode_context.json` — extract cast/person
   entries
3. Cross-reference turn content with glossary names, identify patterns (self-introductions,
   quiz-reading, singing, speech register, dialect)
4. Write `transcription/<stem>_named_speaker_map.json`

Important:

- every cluster in the source speaker map must get an identification entry
- merges are single-hop only (no chains), never merge speakers who interact
- the named speaker map is auto-discovered by `translate_vtt_codex.py prepare`
- speaker context appears in `next-batch` payloads as advisory per-cue labels

Example invocation in Codex:

```text
Use $speaker-diarization on samples/episodes/<slug>/transcription/<stem>_speaker_map.json
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
- auto-discovers a sibling chunkwise OCR sidecar in `ocr/*_flash_lite_chunk_ocr.json` and includes filtered visual cues in batch payloads when present
- carries advisory alignment warnings from CTC diagnostics into batch payloads and diagnostics when present
- auto-discovers a sibling `*_named_speaker_map.json` and includes per-cue speaker context in batch payloads when present
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
video
  -> chunk-review
  -> vad_chunks.json
  -> transcription (Gemini or local)
  -> aligned words
  -> subtitle-reflow
  -> recommended Japanese VTT
  -> glossary-context
  -> glossary/episode_context.json
  -> cluster_speakers.py (automated)
  -> speaker-diarization (interactive)
  -> named_speaker_map.json
  -> subtitle-translation (auto-discovers named speaker map)
  -> English VTT in translation/
  -> publish_vtt.py
  -> source/<video_stem>.vtt
```

Use `chunk-review` before transcription when you want semantically informed
chunk boundaries instead of pure VAD-based splitting.

Use `subtitle-reflow` first when the Japanese subtitle artifact still needs
boundary review or repair.

Use `glossary-context` once the raw transcript is available and before English
translation starts.

Use `speaker-diarization` after voice clustering to name anonymous clusters
using glossary context. The named speaker map is auto-discovered by translation.

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
