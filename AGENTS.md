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

`scripts/translate_vtt.py` should preserve:

- cue count
- cue order
- cue timings

But it should optimize for:

- natural English subtitles
- local cue-set coherence
- CPS

Checkpoint after every translation batch. Long `gemini-2.5-pro` runs are not reliable enough without resume.

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
