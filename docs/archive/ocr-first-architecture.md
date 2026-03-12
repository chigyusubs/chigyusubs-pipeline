# OCR-First Architecture

This document preserves the older OCR-heavy architecture framing that was used when the repo treated OCR context as a central prerequisite for Gemini transcription.

It is kept for historical reference only.

## Old Default Shape

```text
source video
  -> fixed-rate frames
  -> Qwen OCR JSONL
  -> OCR spans
  -> OCR context
  -> VAD
  -> chunk boundaries
  -> Gemini transcription with OCR hints
  -> alignment
  -> reflow
  -> translation
```

## Why It Moved Out Of The Main Architecture Doc

This shape is no longer the maintained default because:

- Gemini video transcription became good enough that OCR is often optional
- OCR context selection remained unresolved and sometimes poisoned transcription
- the simpler VAD -> Gemini -> CTC path became the practical mainline
- Flash Lite OCR-only side artifacts now look more promising than the older OCR-first transcription coupling

## What Still Remains Useful From This Architecture

- lossless OCR JSONL
- OCR spans as reusable visual evidence
- OCR-derived glossary/context work
- OCR-only experiments on dense visual chunks

## Scripts Still Relevant To This Older Path

- `scripts/run_qwen_ocr_episode.py`
- `scripts/classify_ocr_spans_local.py`
- `scripts/transcribe_gemini_raw.py`

Use this archived framing when:

- revisiting older OCR-assisted experiments
- comparing current behavior to the older OCR-heavy design
- deciding whether OCR should re-enter the maintained path in a narrower form
