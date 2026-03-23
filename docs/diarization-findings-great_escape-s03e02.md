# Diarization Findings: `great_escape_s03e02_youtube`

This note summarizes what the current speaker clustering was and was not able to
recover on `great_escape_s03e02_youtube`, plus the practical downstream value.

## Scope

Artifacts inspected:

- `samples/episodes/great_escape_s03e02_youtube/transcription/ra76357b0_speaker_map.json`
- `samples/episodes/great_escape_s03e02_youtube/transcription/ra76357b0_named_speaker_map.json`
- `samples/episodes/great_escape_s03e02_youtube/transcription/ra76357b0_reflow.vtt`
- `samples/episodes/great_escape_s03e02_youtube/translation/ra76357b0_en.vtt`

This was an interactive identification pass using the `speaker-diarization`
skill plus manual episode knowledge.

## High-level result

The diarization was useful, but only as a partial speaker anchor layer.

It was not good enough for:

- full per-line named-speaker recovery
- trusting every `spk_N` as a real person
- aggressive automatic speaker-aware reflow

It was useful for:

- anchoring several major speakers with high or medium confidence
- finding perspective errors in translation (`we` vs `they`)
- identifying impure clusters that should not be trusted downstream
- giving soft constraints for future reflow review

## Source map quality

From `ra76357b0_speaker_map.json`:

- total turns: `1415`
- assigned turns: `818`
- unassigned turns: `597`
- speaker clusters: `19`

This is the core limitation. Almost 42% of turns are unassigned, and at least
one cluster is a clear catch-all bucket spanning multiple rooms and people.

## What was successfully identified

These identifications are the main wins from the pass:

- `spk_0` -> `Higashibukuro`
  Evidence: `俺や。袋や。`
- `spk_5` -> `Tetsuya Morita`
  Evidence: room pairing with Higashibukuro plus matching cigarette comments
- `spk_6` -> `Michio`
  Evidence: user-confirmed `00:10:34` `えんとつ町のプペル。`
- `spk_12` -> `Yoichi Okano`
  Evidence: by elimination inside the clean two-speaker letter-room stretch at `10:10-10:30`
- `spk_10` -> `Omiokuri Geinin Shinichi`
  Evidence: user-confirmed karaoke speaking cluster
- `spk_8` -> merge into `spk_10`
  Evidence: validated Shinichi singing-voice split
- `spk_3` -> `Hiroyuki Iguchi`
  Evidence: explicit self-ID `南川さん、井口です`
- `spk_2` -> `Minamikawa`
  Evidence: user-confirmed far-side vent voice
- `spk_4` -> `Masanari Takano`
  Evidence: user-confirmed second dark-room participant with Minamikawa
- `spk_1` -> `Bakarhythm`
  Evidence: user-confirmed studio commentary at `36:40`
- `spk_14` -> `Eiji Kotoge`
  Evidence: user-confirmed studio commentary at `37:07`

This means the main cast is mostly recoverable despite the weak map.

## What failed cleanly

### `spk_13` is not a person

The most important failure mode was `spk_13`.

`spk_13` contains lines from clearly different scenes and speakers, including:

- early room chatter
- phone/itch room reactions
- letter-room comments like `00:10:38.697` `使えなそうだけどな、この辺も。`
- later `OK Google / Alexa` room lines

User-confirmed evidence directly contradicted any single-person mapping:

- one `spk_13` line in the letter room belonged to `Okano`
- later `spk_13` lines in the trapped-room device sequence belonged to `Kuro-chan`

Conclusion:

- `spk_13` is a mixed cluster
- it must not be mapped to a single named speaker
- downstream consumers should treat it as untrusted

This was represented in the named map as:

- `name: Unknown Speaker 13`
- `role: unknown`
- `group: mixed_cluster`

### `Kuro-chan` is present but not recoverable as a clean cluster

Episode knowledge confirms `Kuro-chan` speaks in the episode, but the current
cluster map does not expose him as a clean stable `spk_N`.

Practical implication:

- do not force a fake `Kuro-chan -> spk_X` mapping
- preserve the fact that he is present in episode reasoning
- treat his lines as episode knowledge, not cluster knowledge

## Why the map still helped

Even with weak clustering, the pass still improved translation in small but real
ways.

### 1. Pronoun perspective fixes

Two clear translation fixes came from speaker context:

- `00:26:38-00:26:43`
  `We're breaking down...` -> `They're breaking down...`
- `00:37:28`
  `We're peeking down at it from above.` -> `They're peeking down from above.`

The key point is not “named speaker labels solved translation.”
The key point is “speaker perspective made a few ambiguous lines safer.”

### 2. Stronger confidence about when not to trust a cue

The map was most useful as negative information:

- some clusters are stable enough to use
- some clusters are too impure to drive any wording choice

This is especially important for short reaction lines and room-crossing edits.

## Practical downstream guidance

### Translation

Use diarization only as a soft editorial cue:

- safe to use for explicit self-identification and room-confirmed anchors
- safe to use for perspective cleanup (`we` vs `they`)
- unsafe to use for aggressive name insertion everywhere
- unsafe to use when the line belongs to a mixed or low-confidence cluster

### Reflow

Diarization may still be useful for reflow, but only as a soft constraint.

Good uses:

- avoid merging across confident speaker changes
- preserve separate reply lines in rapid back-and-forth sections
- allow overlap treatment when confident speakers differ

Bad uses:

- using speaker clusters as the primary timing source
- forcing hard splits from low-confidence or mixed clusters
- trusting catch-all clusters like `spk_13`

In short:

- good for `don't merge these`
- weak for `split exactly here`

## Recommended policy

For the current diarization path, the safest policy is:

1. treat named speaker clusters as optional metadata, not truth
2. only trust `high` or strong `medium` identifications
3. explicitly preserve `mixed_cluster` / impure-cluster states
4. let diarization influence translation and reflow review only when it reduces ambiguity
5. never force full named-cast recovery when the source map quality is weak

## Bottom line

This episode validated a modest but useful diarization role:

- not strong enough for full speaker attribution
- strong enough for anchor identities, perspective correction, and reflow review hints

The biggest lesson is that diarization can still be operationally valuable even
when it fails at the “every line belongs to a named person” goal.
