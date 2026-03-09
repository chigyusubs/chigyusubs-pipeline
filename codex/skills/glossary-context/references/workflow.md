# Glossary Context Workflow

## Purpose

This reference covers the step-by-step workflow for extracting glossary and episode context from Gemini raw transcripts.

## Inputs

- Gemini raw JSON: `samples/episodes/<slug>/transcription/<slug>_*_gemini_raw.json`
- Existing global glossary (optional): `samples/episodes/<slug>/glossary/glossary.json`

## Step-by-Step Extraction

### Step 1: Load transcript chunks

```python
import json
chunks = json.load(open(gemini_raw_path))
```

Each chunk has: `chunk`, `chunk_start_s`, `chunk_end_s`, `text`, `chunk_size_mb`.

### Step 2: Parse lines

For each chunk's `text`, split by newlines and classify:

- Lines starting with `-- ` are spoken dialogue
- Lines matching `[画面: ...]` are visual cues (on-screen text)
- Other lines are narration or description

### Step 3: Entity extraction

From dialogue and visual cues, identify:

1. **Names** — look for katakana sequences (guest/cast names), kanji names, and English names in visual cues
2. **Show terms** — recurring capitalized phrases in visual cues, segment titles
3. **Game rules** — instructions, scoring terms, challenge names
4. **Locations** — place names relevant to the episode

### Step 4: Cross-reference

- If a name appears in both `[画面: ...]` (often with English/romaji) and dialogue (in Japanese), use the visual cue as the authoritative romanization
- If a visual cue shows English text that matches a Japanese dialogue term, use that as the `target`

### Step 5: Classify global vs local

**Global** (stable across episodes):
- Main cast members
- Show title and standard segments
- Recurring catchphrases
- Standard variety-show terminology

**Local** (episode-specific):
- Guest names
- Episode-specific locations
- One-off game names and rules
- Episode-specific references

### Step 6: Write output

Global glossary: `glossary/glossary.json`
Episode context: `glossary/episode_context.json`

Both use the same format:

```json
{
  "entries": [
    {
      "source": "伊藤",
      "target": "Ito",
      "context": "cast member",
      "is_hotword": true
    }
  ]
}
```

## Global-vs-Local Classification Rules

| Signal | Classification |
|--------|---------------|
| Appears in multiple chunks with same usage | Likely global |
| Named in show opening/title sequence | Global |
| Guest introduction in a single segment | Local |
| Game name used only in this episode | Local |
| Standard MC/host reference | Global |
| Location specific to episode challenge | Local |

## Merge with Existing Glossary

When merging into an existing `glossary.json`:

1. Load existing entries
2. Index by `source` text
3. For new entries: add if `source` not already present
4. For existing entries: do not overwrite — the existing version is user-reviewed
5. If a new entry conflicts with an existing one, skip it and note the conflict

## Trigger Phrases

- build glossary for `<episode>`
- extract names from transcript
- create episode context
- glossary from Gemini raw
- set up translation context
