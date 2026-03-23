---
name: speaker-diarization
description: Identify anonymous voice clusters from speaker_map.json by cross-referencing sample turns with the glossary, then produce a named_speaker_map.json for downstream translation. Use after cluster_speakers.py has run and before translation begins.
---

# Speaker Diarization — Identify & Name Voice Clusters

Use this skill when an episode has a `speaker_map.json` (produced by `cluster_speakers.py`) and needs human-readable speaker identifications before translation.

## Inputs

1. **Speaker map** — `transcription/*_speaker_map.json`
   - Contains `speakers` (per-cluster stats) and `turns` (per-turn entries with `speaker`, `start`, `end`, `text`)
2. **Glossary** — `glossary/glossary.json` + `glossary/episode_context.json`
   - Contains cast names (Japanese + romanized), roles, and episode-specific context

Both are auto-discoverable from the episode directory.

## Procedure

### 1. Load & Survey

- Read `speaker_map.json` — note how many clusters exist, their turn counts and durations
- For each cluster, pick the ~10 longest turns (by duration) as representative samples
- Read glossary entries, focusing on person/cast entries

### 2. Analyze Each Cluster

Look for these identification signals, in rough priority order:

1. **Self-introduction** — speaker says their own name (「○○です」, 「○○の」)
2. **Glossary name match** — turn text contains a glossary person's name used in first-person context
3. **Name-card correlation** — if visual cues/OCR data is available, name plates appearing during a cluster's turns
4. **Role inference** — quiz-reading patterns (presenter), singing (contestant in karaoke), phone-call register
5. **Speech register/dialect** — Kansai-ben, formal MC speech, characteristic catchphrases
6. **Temporal grouping** — clusters that only appear in certain time ranges (room-based segments)
7. **Process of elimination** — after high-confidence IDs, remaining clusters in a room group

### 3. Assign Identifications

For each `spk_N`, determine:

- **name** — romanized display name (from glossary if possible)
- **name_ja** — Japanese name
- **role** — one of the controlled vocabulary: `presenter`, `contestant`, `phone-caller`, `narrator`, `staff`, `unknown`
- **group** — optional scene/room grouping (e.g., `phone_quiz_room`, `karaoke_room`, `studio`)
- **confidence** — `high` (glossary match + clear evidence), `medium` (process of elimination), `low` (guess)
- **evidence** — brief explanation of why this identification was made

### 4. Merge Decisions

Only merge clusters when there is clear evidence they are the same person:

- Singing voice vs speaking voice for the same person
- Emotional register split (shouting/whispering variant)
- Very short fragment clusters that clearly belong to an identified speaker

**Never merge two speakers who interact with each other** (appear in overlapping or adjacent turns in conversation).

Merges are single-hop and non-circular: `spk_8 → spk_3` is valid, but `spk_8 → spk_3 → spk_1` is not.

### 5. Write Output

Write `<stem>_named_speaker_map.json` to the `transcription/` directory, where `<stem>` matches the speaker_map stem (minus `_speaker_map`).

## Output Format

```json
{
  "version": 1,
  "source_speaker_map": "<filename of source speaker_map.json>",
  "identifications": {
    "spk_0": {
      "name": "Tetsuya Morita",
      "name_ja": "森田哲矢",
      "role": "contestant",
      "group": "phone_quiz_room",
      "confidence": "high",
      "evidence": "Reads quiz questions, glossary match for Saraba member"
    },
    "spk_8": {
      "name": "Iguchi",
      "name_ja": "井口",
      "role": "contestant",
      "group": "karaoke_room",
      "confidence": "medium",
      "evidence": "Singing voice variant of spk_3",
      "merge_into": "spk_3"
    }
  },
  "merges": [
    { "source": "spk_8", "target": "spk_3", "reason": "singing voice variant" }
  ],
  "effective_speakers": {
    "Tetsuya Morita": {
      "spk_ids": ["spk_0"],
      "role": "contestant",
      "group": "phone_quiz_room"
    },
    "Iguchi": {
      "spk_ids": ["spk_3", "spk_8"],
      "role": "contestant",
      "group": "karaoke_room"
    }
  }
}
```

### Field Descriptions

- **identifications** — every `spk_N` from the source map, mapped to name/role/evidence
- **merges** — flat list, single-hop only, no circular chains
- **effective_speakers** — pre-resolved rollup keyed by display name; this is what downstream consumers use
- **group** — optional scene/room grouping for spatial context

## Guidelines

- Every cluster in the source `speakers` dict must appear in `identifications` — do not skip any
- If a cluster cannot be identified, use `"name": "Unknown Speaker N"`, `"role": "unknown"`, `"confidence": "low"`
- Prefer glossary names over ad-hoc romanizations
- The `effective_speakers` section should reflect the final merged state — merged clusters appear under the target speaker's entry
- Keep evidence strings concise but specific enough to audit later
