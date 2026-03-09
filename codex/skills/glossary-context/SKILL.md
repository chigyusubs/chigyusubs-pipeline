---
name: glossary-context
description: "Extract show glossary and episode context from a Gemini raw transcript. Reads both dialogue lines (-- ...) and visual cues ([画面: ...]) to identify cast names, show segments, game terms, and guest names. Produces a global glossary and episode-specific context file in the standard glossary format. Use when starting a new episode or show to build translation context before subtitle-translation."
---

# Glossary & Context Builder

Use this skill to build structured glossary and episode context from a Gemini raw transcript before translation.

## Inputs

1. **Gemini raw JSON** — path to `transcription/<slug>_gemini_raw.json`
2. **Existing global glossary** (optional) — path to `glossary/glossary.json` if one already exists for the show

## Extraction Process

### 1. Read the transcript

Read all chunks from the Gemini raw JSON. Process both:

- `-- ` dialogue lines (spoken text)
- `[画面: ...]` visual cue lines (on-screen text)

### 2. Identify entities

Extract:

- **Cast names** — regular cast members with Japanese names and romanizations
- **Guest names** — guests introduced in the episode
- **Show/segment names** — recurring show titles, segment names, game names
- **Game terms** — rules, scoring terms, location names specific to challenges
- **Catchphrases** — recurring phrases that need consistent translation

For each entity, note:

- `source` — Japanese text as it appears
- `target` — English translation or romanization
- `context` — brief note on usage (e.g., "cast member", "game name", "location")
- `is_hotword` — `true` if this term should bias ASR/alignment (proper nouns, show names)

### 3. Classify as global or local

- **Global** — stable across episodes: cast names, show title, recurring segment names, standard phrases
- **Local** — episode-specific: guest names, episode-specific game rules, location names, one-off terms

### 4. Output

Use this format for both files:

```json
{"entries": [{"source": "str", "target": "str", "context": "str", "is_hotword": true}]}
```

Output paths (relative to the episode directory):

- Global: `glossary/glossary.json`
- Local: `glossary/episode_context.json`

### 5. Merge rules

When an existing global glossary is provided:

- Preserve all existing entries
- Add new entries that are clearly global
- Do not duplicate entries with the same `source` text
- Flag uncertain romanizations with `"context": "romanization uncertain"` so the user can review

## Quality Checks

- Every proper noun should have a romanization
- Common variety-show terms (e.g., MC, VTR, CM) should use standard English equivalents
- If a name appears in both dialogue and visual cues, cross-reference for accuracy
- Visual cues with English text are high-confidence sources for romanizations

## Trigger Phrases

This skill should trigger on requests like:

- build a glossary for this episode
- extract names and terms from the transcript
- create episode context for translation
- set up glossary before translating
