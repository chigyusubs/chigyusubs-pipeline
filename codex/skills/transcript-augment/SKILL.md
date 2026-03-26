---
name: transcript-augment
description: Merge Flash Lite audio re-transcription additions into a Gemini transcript, recovering missing dialogue (reactions, interjections, back-channel) before CTC alignment. Use after Gemini transcription when you want a fuller transcript for alignment.
---

# Transcript Augment

Use this skill after Gemini transcription and Flash Lite audio correction to
merge recovered dialogue into the original transcript before CTC alignment.

Use `scripts/augment_transcript_codex.py` as the maintained helper workflow:

- `prepare`
- `next-chunk`
- `apply-chunk`
- `status`
- `finalize`

## Prerequisites

Before using this skill, run the Flash Lite audio pass:

```bash
GOOGLE_GENAI_USE_VERTEXAI=false python3.12 scripts/correct_transcript_flash_lite.py \
  --gemini-raw <path>_gemini_raw.json \
  --video <video.mp4>
```

This produces `<path>_corrected.json` alongside the original.

## Core Rules

- All original lines are kept verbatim. Never modify, rephrase, or remove an
  original line.
- All `[画面: ...]` visual annotation lines are preserved in their original
  positions. These come from Gemini's video context and cannot be recovered
  from audio.
- Additions from Flash Lite are opt-in. Review each one and decide whether to
  accept or reject.
- Modified lines (where Flash Lite rewrote an original) are always rejected in
  favor of the original. They appear in the `modifications` field for context
  only.

## Decision Heuristics

**Accept** additions that are:
- Natural reactions and interjections (`うん`, `えー`, `おお`, `マジで？`)
- Back-channel responses during someone else's turn
- Short confirmations or laughter that fill gaps in conversational flow
- Lines that clearly correspond to dialogue you can infer from surrounding context

**Reject** additions that are:
- Duplicates of existing content in slightly different words
- Hallucinations inconsistent with the surrounding dialogue
- Fragments that don't form a complete utterance

**When in doubt, accept.** CTC alignment validates every line against the
actual audio waveform -- lines that don't match will surface in alignment
diagnostics.

## Workflow

### 1. Prepare

```bash
python scripts/augment_transcript_codex.py prepare \
  --gemini-raw <path>_gemini_raw.json \
  --corrected <path>_corrected.json
```

Diffs each chunk and writes a session checkpoint. Reports how many chunks have
candidate additions.

### 2. Chunk Loop

**Loop until all chunks are reviewed.**

1. Run `next-chunk` to get the current chunk's diff payload.
2. Read `original_lines` to understand the chunk content.
3. Review each entry in `additions`:
   - Each has `addition_idx`, `insert_after_orig` (position), and `text`.
   - Decide: accept (include in merged output) or reject.
4. Check `dropped_originals` for any visual annotations Flash Lite missed --
   these are always preserved automatically, no action needed.
5. Write a merge decision JSON and run `apply-chunk`.
6. Loop back to step 1.

Do NOT wait for user confirmation between chunks. Process all chunks in a
continuous loop.

### 3. Finalize

```bash
python scripts/augment_transcript_codex.py finalize --session <session.json>
```

Produces `*_augmented.json` in gemini_raw format and updates `preferred.json`
so CTC alignment picks it up automatically.

### 4. Verify Saved Output

After `finalize`, read the saved augmented JSON from disk and confirm the chunk
count and total line count look reasonable.

## Decision JSON Format

```json
{
  "chunk_index": 5,
  "accepted_additions": [0, 2, 3],
  "rejected_additions": [1],
  "notes": "Accepted reactions, rejected duplicate of line 4"
}
```

Notes are optional but helpful for review.

## What Not To Do

- Do not modify original lines. The original Gemini transcript is the baseline.
- Do not accept additions that duplicate existing content.
- Do not stop the loop early. Process every pending chunk.
- Do not modify the session JSON directly. Use the helper commands.
- Do not skip reading the original lines context before deciding on additions.
