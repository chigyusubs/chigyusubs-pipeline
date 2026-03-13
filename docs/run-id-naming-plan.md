# Run ID Naming Plan

Short run IDs for lineage artifacts. Stable names for published outputs.
Settings move into file headers and existing `.meta.json` sidecars.

## Problem

Current filenames encode the entire configuration lineage:

```
great_escape_s02e02_video_spoken_only_semantic_density_repaired_ctc_words_reflow_repaired_english_codex_glossary.vtt
```

This is 120+ characters, fragile to rename, and doesn't even capture chunk-level
variation (e.g. mixed models after repair).

## Key Principle: Two Naming Regimes

Not all artifacts should be named the same way.

**Published final outputs** (the VTT that ships with the video) get a stable
name matching the source video. Provenance goes in a VTT NOTE header and
`.meta.json` sidecar, not the filename.

**Lineage artifacts** (intermediate pipeline outputs where multiple competing
runs may coexist) get run-ID-based names. These live in `transcription/` and
draft `translation/` and include:

- Gemini raw JSON
- CTC aligned words
- JP reflow VTT
- Draft EN VTT

This keeps the existing separation between stable canonical paths and the
run ledger under `logs/runs/`.

## Design

### Run ID

- 8-char hex hash derived from `(step, timestamp)`
- Generated in `start_run()`, propagated to all artifacts from that run
- Derived from the existing ledger run ID in `chigyusubs/metadata.py` (a
  human-readable timestamp+step string like
  `20260312_223453_000000_plus_0000__transcribe_gemini`)
- The short ID is just the first 8 chars of a hash of the ledger ID
- The ledger directory keeps its current long name for audit trails
- The mapping is deterministic and reconstructable — no need to store it
  explicitly
- Example: `r3a7f01b`

### Published Final Output

Stable name matching the source video basename:

```
source/第2話.mp4
source/第2話.vtt          <- published subtitle
source/第2話.vtt.meta.json <- points back to run
```

The VTT carries provenance in a NOTE header but the filename never changes
between runs. `publish_vtt.py` copies the best draft to this location.

### Lineage Artifacts

Run-ID-based names in `transcription/` and `translation/`:

```
transcription/r3a7f01b_gemini_raw.json
transcription/r3a7f01b_ctc_words.json
transcription/r3a7f01b_reflow.vtt
translation/r3a7f01b_en.vtt
```

Episode slug is implicit from the directory path — no need to repeat it in
every filename.

### Current-Best Pointers

Instead of "resolve by latest" (fragile, implicit), use explicit preferred
manifests per area:

```
transcription/preferred.json
translation/preferred.json
```

Each manifest points to artifacts within its own directory:

```json
// transcription/preferred.json
{
  "gemini_raw": "r3a7f01b_gemini_raw.json",
  "ctc_words": "r3a7f01b_ctc_words.json",
  "reflow": "r3a7f01b_reflow.vtt"
}
```

```json
// translation/preferred.json
{
  "en_draft": "r3a7f01b_en.vtt"
}
```

Scripts that need to auto-resolve an input read the relevant `preferred.json`
first. Manual overrides still work by passing explicit paths. No cross-directory
relative paths needed.

### Two Levels of Metadata

**Run level** — what was intended:
- Pipeline invocation, default model, default settings, chunk plan, episode
- Stored in `.meta.json` sidecar and file headers
- The run ID is the lookup key

**Chunk level** — what actually happened:
- Which model, which attempt, temperature, retry/repair/subsplit status, token counts
- Already exists in Gemini raw JSON (`model`, `prompt_token_count`, `repair_provenance`)
- No change needed — this is per-chunk by nature

This split exists because chunks can use different models and settings after
repair cascades (e.g. S02E02: 3-Flash primary, 2.5 Flash fallback on 4 chunks).

### VTT NOTE Header

VTT supports `NOTE` blocks after the `WEBVTT` line. Used for both published
and lineage VTTs:

```
WEBVTT

NOTE
run: r3a7f01b
step: reflow
episode: great_escape_s02e02
source: r3a7f01b_ctc_words.json
created: 2026-03-12T22:34:53Z
models: gemini-3-flash-preview (11/15), gemini-2.5-flash (4/15)
chunks: 15 (semantic_density, 2 repaired, 2 subsplit)
```

The `models` line is a summary. Full per-chunk detail lives in the Gemini raw
JSON and `.meta.json` sidecar.

### JSON `_run` Header

Top-level key in Gemini raw and CTC words files:

```json
{
  "_run": {
    "id": "r3a7f01b",
    "step": "transcribe_gemini",
    "episode": "great_escape_s02e02",
    "created": "2026-03-12T22:34:53Z",
    "default_model": "gemini-3-flash-preview",
    "chunk_plan": "semantic_density",
    "chunks_total": 15,
    "chunks_repaired": 2,
    "chunks_subsplit": 2
  },
  "chunks": [...]
}
```

Parsers already skip unknown keys, so this is backwards-compatible.

## What Changes

- `chigyusubs/metadata.py`: `start_run()` generates a short run ID alias
  derived from the existing ledger run ID (no ledger changes needed)
- `chigyusubs/vtt.py` + `chigyusubs/translation.py`: VTT serializers accept an
  optional NOTE header
- Scripts that write lineage artifacts: use `{run_id}_{type}` pattern
- Scripts that read artifacts: check `preferred.json` first, fall back to
  explicit path
- `publish_vtt.py`: copies preferred draft to stable source-video-basename path

## What Doesn't Change

- Published output naming (stable, matches source video)
- Chunk-level metadata in Gemini raw JSON (already per-chunk)
- Run ledger in `logs/runs/`
- Session/checkpoint files for Codex skills
- `.meta.json` sidecar system (just gains a `run_id` field)

## Migration

- New episodes use run ID naming for lineage artifacts automatically
- Existing episodes keep their current filenames
- No bulk rename needed — old files still work, just verbose
- `preferred.json` can be backfilled for existing episodes if needed
