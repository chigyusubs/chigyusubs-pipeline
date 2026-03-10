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

## Codex Skills

This repo has tracked Codex-interactive skills under:

- `codex/skills/subtitle-reflow`
- `codex/skills/glossary-context`
- `codex/skills/subtitle-translation`

Treat the repo copy as canonical. Do not treat `~/.codex/skills/` as the source of truth.

If the live Codex install needs refreshing, use:

- `python3 scripts/install_codex_skills.py`
- `python3 scripts/install_codex_skills.py --skill subtitle-reflow`
- `python3 scripts/install_codex_skills.py --skill subtitle-translation`

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

Preferred Codex-interactive handoff is:

1. aligned words
2. `subtitle-reflow`
3. `glossary-context`
4. `subtitle-translation`
5. English VTT in `translation/`
