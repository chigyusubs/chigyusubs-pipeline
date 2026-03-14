---
name: chunk-review
description: Review candidate chunk boundaries semantically using a faster-whisper pre-pass transcript, deciding at each VAD silence gap whether it falls between sentences (split) or mid-sentence (skip). Use when the user wants Codex to interactively review and approve chunk boundaries before transcription instead of relying on purely acoustic VAD-based chunking.
---

# Chunk Review

Use this skill when Codex should review chunk boundary candidates interactively
rather than using the automatic `build_vad_chunks.py` path.

Use `scripts/build_semantic_chunks.py` as the maintained helper workflow:

- `prepare`
- `next-candidate`
- `apply-candidate`
- `status`
- `finalize`

## Core Rules

- The only question at each candidate is: does this silence gap fall between
  sentences or in the middle of one?
- Do not try to detect topic changes, scene boundaries, or speaker changes.
  Those do not affect transcription quality.
- Treat transcript density as a real chunk budget, not just wall-clock time.
  Dense chunks are harder to transcribe than quiet ones.
- When `approaching_max` is true in the payload, prefer splitting even at an
  imperfect boundary. `approaching_max` may be triggered by duration or by
  transcript character budget. An oversized or overly dense chunk is worse than
  a slightly awkward split.
- Trust the transcript context as rough signal. The pre-pass is not the final
  transcription -- minor errors are fine as long as sentence boundaries are
  readable.

## Workflow

### 1. Prepare

```bash
python scripts/build_semantic_chunks.py prepare \
  --video samples/episodes/<slug>/source/video.mp4
```

This runs Silero VAD + faster-whisper pre-pass and writes a session JSON with
all candidate gaps and their surrounding transcript context.

### 2. Candidate Loop

**Loop until all candidates are reviewed.**

1. Run `next-candidate` to get the current candidate with transcript context.
2. Read the `transcript_context.before` and `transcript_context.after` text.
3. Decide:
   - `split` if the gap falls between sentences (look for sentence-final
     punctuation like `。？！` or sentence-final particles like `よ ね か な`)
   - `skip` if the gap falls mid-sentence or mid-clause
4. Write a decision JSON and run `apply-candidate`.
5. Loop back to step 1.

Do NOT wait for user confirmation between candidates. Process all candidates
in a continuous loop.

When `chunk_state.approaching_max` is true, bias toward `split` unless the
boundary is clearly destructive (e.g. splitting a word). Check both:

- duration state (`time_since_last_split_s`)
- transcript density state (`chars_since_last_split`)

### 3. Finalize

```bash
python scripts/build_semantic_chunks.py finalize --session <session.json>
```

Produces `vad_chunks.json` in the standard format used by downstream scripts.

### 4. Verify Saved Output

After `finalize`, always read the saved chunk JSON from disk and inspect the
actual chunk boundaries or duration summary that will be used downstream.

Do not rely on previously printed chunk listings or earlier finalize output if:

- the turn was interrupted
- `finalize` was rerun after code changes
- a stale JSON summary may still be in context

The file on disk is the source of truth.

## Decision JSON Format

```json
{
  "candidate_id": 0,
  "decision": "split",
  "notes": "sentence ends with 。before the gap"
}
```

Notes are optional but helpful for review.

## What Not To Do

- Do not evaluate topic or scene coherence. Only sentence boundaries matter.
- Do not skip candidates without reading the transcript context.
- Do not stop the loop early. Process every candidate.
- Do not modify the session JSON directly. Use the helper commands.
- Do not assume earlier printed finalize output is still current. Re-read the
  saved chunk JSON after the last finalize run.
