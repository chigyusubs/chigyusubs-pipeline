# Chunk Review Workflow

## Purpose

Replace purely acoustic VAD-based chunking with semantically informed chunk
boundaries. A faster-whisper pre-pass gives Codex enough transcript context to
judge whether each silence gap is a natural sentence boundary.

## Why Only Sentence Boundaries

Chunk boundaries affect transcription quality because:

- A chunk that starts mid-sentence forces the ASR to guess without prior context
- A chunk that ends mid-sentence may truncate or hallucinate the remainder

Scene changes, topic shifts, and speaker changes do NOT hurt transcription.
The ASR handles those fine within a single chunk.

## Defaults

- Target chunk duration: 240s (4 min)
- Maximum chunk duration: 360s (6 min) -- force a split before this
- Target transcript budget: derived from the faster-whisper pre-pass density
  at the target duration
- Maximum transcript budget: 1.2x target transcript budget
- Minimum silence gap: 1.5s
- Transcript context window: 30s on each side of the gap
- faster-whisper model: large-v3

## Decision Heuristics

Good split signals:
- Sentence-final punctuation before the gap: `。` `？` `！`
- Sentence-final particles before the gap: `よ` `ね` `か` `な` `さ`
- Complete thought/clause ending before the gap

Bad split signals:
- Text before the gap ends with a particle like `は` `が` `を` `に` `で` `と`
- Text before the gap ends mid-word or with a conjunction like `けど` `から` `て`
- Text after the gap clearly continues the previous sentence

When in doubt and `approaching_max` is true, split.

`approaching_max` can be triggered by:

- chunk duration nearing the configured maximum
- transcript character count nearing the configured maximum

This matters because two chunks with the same duration can be very different
transcription workloads if one is much denser.

## Output

The finalized `vad_chunks.json` is in the same format as `build_vad_chunks.py`
output and is consumed by the same downstream scripts:

- `transcribe_local.py`
- `transcribe_gemini_raw.py`
- `align_ctc.py`

After finalize, verify the saved file itself instead of relying on prior
terminal output. If `finalize` was rerun, or if a turn was interrupted, older
printed chunk listings can be stale even though the saved JSON is correct.

## Trigger Phrases

- review chunk boundaries for this episode
- use semantic chunking on this episode
- run chunk review before transcription
