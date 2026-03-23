# AGENTS.md

## Purpose

This repo is a Japanese variety-show subtitle pipeline.

The real goal is not "perfect Japanese subtitles." The goal is high-quality English subtitles with low human input.

Optimize for:

- reusable artifacts
- resumability
- inspectability
- translation quality
- robustness across very different show formats

Do not optimize for:

- elegant one-shot pipelines
- brittle show-specific heuristics
- dropping ambiguous text too early

## Working Rules

### Artifact-first workflow

Keep these as reusable saved artifacts whenever possible:

- OCR JSONL
- OCR spans/context
- Silero VAD segments
- VAD chunk boundaries
- Gemini/local raw transcript
- chunked alignment outputs
- translation diagnostics

Do not bury VAD, OCR filtering, or chunking inside one transcription call if they can be saved and reused.

### Documentation is part of the work

If you materially change pipeline behavior, update the docs in the same task.

At minimum, check whether these need changes:

- `README.md`
- `docs/current-architecture.md`
- `docs/lessons-learned.md`
- `docs/scripts-reference.md`

Update `docs/lessons-learned.md` when:

- a new path is validated on a real episode
- a failure mode is clearly identified
- a previous assumption turns out to be wrong

Do not let important operational findings live only in chat history.

## Current Best Practices

### Alignment

Use CTC forced alignment (`scripts/align_ctc.py`) by default.

This uses `NTQAI/wav2vec2-large-japanese` with `torchaudio.functional.forced_align` for character-level CTC alignment. It reduced zero-duration words from 13.4% (stable-ts) to 0.3% on `dmm`.

Runs on system python3.12 with ROCm GPU. Chunked: aligns per-chunk from Gemini raw transcription JSON.

`-- ` speaker-turn markers and `[画面: ...]` lines must be stripped before alignment.

Zero-duration aligned text should not be dropped blindly:

- keep it in raw artifacts
- preserve it into nearby cues for translation support
- only treat its timing as unreliable
- with CTC alignment, zero-duration is now rare and limited to non-Japanese text

### Transcription

For Gemini, video-only is a serious baseline.

Current promising cloud path:

- Silero VAD
- VAD chunks
- Gemini video-only transcription
- spoken lines as `-- ...`
- visual-only lines as `[画面: ...]`
- chunked alignment
- reflow
- CPS-aware English translation

Use OCR-assisted Gemini only when OCR is clearly earning its cost.

### OCR

Do not overcomplicate the OCR step.

Keep OCR extraction relatively dumb:

- high recall
- deterministic
- minimal semantic interpretation

Do not force OCR filtering into a heavy taxonomy unless there is clear evidence it generalizes.

Current lesson:

- names alone are not enough
- whole dense dumps are too much
- if OCR is used for Gemini, it likely needs simple local `line_hints + keyword_hints`

### Translation

Translation is subtitle editing, not literal translation.

`scripts/translate_vtt_api.py` should preserve:

- cue count
- cue order
- cue timings

But it should optimize for:

- natural English subtitles
- local cue-set coherence
- CPS

Checkpoint after every translation batch. Long `gemini-2.5-pro` runs are not reliable enough without resume.

For Codex-interactive translation with `scripts/translate_vtt_codex.py`:

- prefer `next-batch --output-json /tmp/<episode>_batch<N>.json` over reading giant JSON payloads from stdout
- keep using the helper session/checkpoint; do not hand-edit partial VTTs
- repeated short acknowledgements (`ありがとうございます`, applause-like runs, etc.) should be translated minimally and consistently rather than literally expanding them
- `yellow` batches caused only by short-cue CPS pressure are not a stop condition by themselves

After a final English VTT is ready, publish it back to the episode `source/` folder with:

- `python3 scripts/publish_vtt.py <final_translation_vtt>`

Do not do ad hoc manual copies when the helper can infer the source video stem.

## Known Failure Modes

### Repetition loops

Gemini can loop:

- spoken text
- visual `[画面: ...]` text

Detect loop-like chunk output and retry only that chunk:

- slightly higher temperature
- no rolling context on retry

### Split-word cue artifacts

Reflow/alignment can split compounds across adjacent cues, for example:

- `地` / `獄`
- `オ` / `ムツ`

This damages English translation later.

If you touch reflow, prioritize repairing these before translation.

### CPS diagnostics

Hard CPS warnings are currently too noisy for very short cues.

Do not assume every `>20 CPS` warning is a real subtitle-quality failure.

Very short repeated cues can produce impossible-looking CPS even with good subtitle choices.

Treat those as a signal to keep the English minimal, not as evidence that the whole translation batch is bad.

## Before Big Changes

Before changing architecture, check:

- `docs/current-architecture.md`
- `docs/lessons-learned.md`

If a new idea sounds clever but ignores those findings, it is probably a regression.

## Preferred Direction

If you need to choose where to invest next, prefer:

1. chunked robustness
2. checkpointing/resume
3. reflow repair of broken cue boundaries
4. better English subtitle editing
5. simple OCR context selection

Prefer evidence from real episode artifacts over theoretical prompt ideas.

## End-to-End Episode Pipeline

When the user asks to process an episode end-to-end, follow this sequence.
The only required input is the video path. Infer the episode directory from it.

All paths below use `<ep>` for the episode directory and `<stem>` for the
run-ID-prefixed artifact stem that each step produces.

### Phase 1 — Semantic chunking (interactive)

```bash
PYTHONPATH=. python3 scripts/build_semantic_chunks.py prepare \
  --video <ep>/source/<video>.mp4 \
  --target-chunk-s 90
```

This runs Silero VAD + faster-whisper pre-pass internally.
Then use the `$chunk-review` skill to loop through all candidates.
Finalize when done.

### Phase 2 — Transcription + OCR sidecar (automated)

Run transcription with Flash 3 (concurrent, auto-falls back to Flash 2.5
on quota exhaustion):

```bash
PYTHONPATH=. python3 scripts/transcribe_gemini_video.py \
  --video <ep>/source/<video>.mp4 \
  --output <ep>/transcription/<stem>_gemini_raw.json \
  --chunk-json <ep>/transcription/vad_chunks_semantic_90.json \
  --preset flash_free_default
```

The preset runs 5 concurrent workers RPM-limited to 5 req/min (free-tier
ceiling), saves per-chunk results to individual files, and automatically
falls back to `gemini-2.5-flash` if the primary model's quota is exhausted.

Then run the OCR sidecar on the same chunk plan:

```bash
PYTHONPATH=. python3 scripts/extract_gemini_chunk_ocr.py \
  --video <ep>/source/<video>.mp4 \
  --chunk-json <ep>/transcription/vad_chunks_semantic_90.json \
  --preset flashlite_ocr_sidecar
```

Then run the raw chunk sanity gate:

```bash
PYTHONPATH=. python3 scripts/check_raw_chunk_sanity.py \
  --input <ep>/transcription/<stem>_gemini_raw.json
```

Stop on any chunk-level `red`. Retry red chunks before proceeding.

### Phase 3 — Glossary (interactive)

Use the `$glossary-context` skill on the raw transcript from Phase 2.
The OCR sidecar from Phase 2 is auto-discovered as supporting evidence.

This runs before alignment so that glossary work fills the time between
transcription and the automated alignment/reflow steps.

### Phase 4 — Alignment + reflow (automated)

```bash
PYTHONPATH=. python3 scripts/align_ctc.py \
  --video <ep>/source/<video>.mp4 \
  --chunks <ep>/transcription/<stem>_gemini_raw.json \
  --output-words <ep>/transcription/<stem>_ctc_words.json

PYTHONPATH=. python3 scripts/reflow_words.py \
  --input <ep>/transcription/<stem>_ctc_words.json \
  --output <ep>/transcription/<stem>_reflow.vtt \
  --line-level --stats
```

### Phase 5 — Reflow review (interactive)

Use the `$subtitle-reflow` skill to review the reflowed VTT.
Follow its green/yellow/red decision policy. Only repair on yellow.

### Phase 6 — Translation (interactive)

Use the `$subtitle-translation` skill on the recommended Japanese VTT
from Phase 5. Glossary and OCR sidecar are auto-discovered.

### Phase 7 — Publish (automated)

```bash
python3 scripts/publish_vtt.py <ep>/translation/<final_en>.vtt
```

### Defaults summary

| Setting | Default |
|---|---|
| Semantic chunk target | 90 s |
| Transcription preset | `flash_free_default` (Gemini 3-Flash) |
| Transcription fallback | `flash25_free_default` (Gemini 2.5-Flash) |
| OCR sidecar preset | `flashlite_ocr_sidecar` (Gemini 3.1-Flash-Lite) |
| Alignment | CTC (`align_ctc.py`) on system python3.12 |
| Reflow | line-level (`reflow_words.py --line-level`) |
| Translation | Codex-interactive (`translate_vtt_codex.py`) |

### Resumability

Every phase produces saved artifacts. If a run is interrupted, re-entering
the pipeline should skip completed phases by checking for existing artifacts:

- Phase 1 done: `vad_chunks_semantic_90.json` exists
- Phase 2 done: `*_gemini_raw.json` + `*_flash_lite_chunk_ocr.json` exist
- Phase 3 done: `glossary/glossary.json` + `glossary/episode_context.json` exist
- Phase 4 done: `*_ctc_words.json` + `*_reflow.vtt` exist
- Phase 5 done: reflow review completed (check `preferred.json`)
- Phase 6 done: `*_en_codex.vtt` exists in `translation/`

## Codex Skills

This repo has tracked Codex-interactive skills under:

- `codex/skills/chunk-review`
- `codex/skills/subtitle-reflow`
- `codex/skills/glossary-context`
- `codex/skills/speaker-diarization`
- `codex/skills/subtitle-translation`

Treat the repo copy as canonical. Do not treat `~/.codex/skills/` as the source of truth.

If the live Codex install needs refreshing, use:

- `python3 scripts/install_codex_skills.py`
- `python3 scripts/install_codex_skills.py --skill chunk-review`
- `python3 scripts/install_codex_skills.py --skill subtitle-reflow`
- `python3 scripts/install_codex_skills.py --skill subtitle-translation`

### When to use `chunk-review`

Use the chunk-review skill when building chunk boundaries for a new episode.
Semantic chunking with interactive review is the default path.

Default behavior should be:

- `scripts/build_semantic_chunks.py prepare --target-chunk-s 90`
- `next-candidate -> Codex split/skip decision -> apply-candidate` loop
- `scripts/build_semantic_chunks.py finalize`

Do not skip chunk review and fall back to automatic `build_vad_chunks.py`
unless the user explicitly asks for unreviewed chunking.

### When to use `subtitle-reflow`

Use the reflow skill when the task is:

- reviewing a Japanese reflowed VTT for translation readiness
- running line-level reflow from `*_ctc_words.json`
- repairing weak cue boundaries before translation

Default behavior should be:

- deterministic `scripts/reflow_words.py --line-level`
- deterministic review of the VTT
- `scripts/repair_vtt_codex.py` only if the VTT is `yellow`

Do not use a local model server as the default reflow-skill fallback.
`repair_vtt_local.py` is an alternative path, not the normal one.

### When to use `glossary-context`

Use the glossary-context skill when starting a new episode or show and the user wants to build a glossary and episode context before translation.

Default behavior should be:

- read Gemini raw JSON for the episode
- extract cast names, show terms, game terms, guest names
- classify as global (cross-episode) or local (episode-specific)
- output `glossary/glossary.json` and `glossary/episode_context.json`
- merge with existing global glossary if present

### When to use `speaker-diarization`

Use the speaker-diarization skill after `cluster_speakers.py` has produced a `speaker_map.json` and before translation begins.

Default behavior should be:

- read `transcription/*_speaker_map.json` — extract per-cluster stats and sample turns
- read `glossary/glossary.json` + `glossary/episode_context.json` — extract cast/person entries
- cross-reference turn content with glossary names, identify patterns
- produce `transcription/*_named_speaker_map.json` with identifications, merges, and effective speakers

The named speaker map is auto-discovered by `translate_vtt_codex.py prepare` and injected as per-cue `speaker_context` in `next-batch` payloads.

### When to use `subtitle-translation`

Use the translation skill when the user wants Codex itself to do subtitle translation interactively instead of an API-backed translation run.

Default behavior should be:

- `scripts/translate_vtt_codex.py prepare`
- `next-batch -> translate -> apply-batch` loop
- `finalize` at the end

Important translation-skill rules:

- preserve cue count, order, and timings
- keep `yellow` batches resumable and continue by default
- only structural errors or explicit `red` should stop the session
- restart with `prepare --force` when a truly clean session reset is needed

### Handoff order

Preferred end-to-end handoff is:

1. `chunk-review` (semantic chunking, 90 s target)
2. Gemini transcription + OCR sidecar + sanity gate (automated)
3. `glossary-context` (needs raw transcript)
4. CTC alignment + reflow (automated)
5. `subtitle-reflow` (review, optional repair)
6. Speaker clustering (automated) + `speaker-diarization` (interactive)
7. `subtitle-translation` (batch loop, auto-discovers named speaker map)
8. `publish_vtt.py` (automated)

See "End-to-End Episode Pipeline" above for full commands and defaults.
