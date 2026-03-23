# Great Escape S03E04 Transcript Repair Notes

This note records the concrete transcript and translation repairs made during the
Codex review session on `samples/episodes/great_escape_s03e04_tvcut`.

The point is not to preserve chat history. The point is to preserve:

- exactly what failed
- how it was validated
- which artifact layer was wrong
- what kind of automation should catch the same issue next time

## Summary

The episode was broadly usable on the maintained Gemini -> CTC -> reflow path,
but it still needed a cluster of manual repairs in three categories:

1. real raw-transcript omissions
2. short high-value ASR mishears
3. translation-only speaker/pronoun nuance fixes

The most important architectural takeaway is that omission-only second-opinion
checks are not enough. This episode also needed disagreement checks on short
high-value lines such as names and brief question/response turns.

## Repairs

### 00:08-00:18 opening dialogue missing entirely

Observed failure:

- the preferred Japanese VTT started at `00:00:12.142`
- the English VTT and published `source/第4話.vtt` therefore also missed the
  opening lines
- the omission was already present in `rab192d6e_gemini_raw.json`, not created
  later by alignment or reflow

Validation:

- `transcription/whisper_prepass_transcript.json` contained the missing opening
  lines from about `00:08.43`
- a local faster-whisper clip pass on the first ~25 seconds confirmed:
  - `ここから12時間後`
  - `あ、ここでエイム`
  - `なるほど、なるほど`
  - `そこが最後だ`

Repair:

- patched the missing lines into `rab192d6e_gemini_raw.json` and
  `rab192d6e_gemini_raw_chunks/chunk_000.json`
- reran CTC alignment with network access
- patched the first weak CTC anchors with the validated local-whisper timings
- regenerated `rab192d6e_ctc_words.json` and `rab192d6e_reflow.vtt`
- synced `rab192d6e_reflow_repaired.vtt`, `r9c223bed_en.vtt`, and
  `source/第4話.vtt`

Desired automation:

- omission checks must flag missing early spoken coverage even when the rest of
  the chunk is dense and looks plausible
- opening regions deserve special scrutiny because a full first chunk can still
  look healthy while silently dropping the first spoken exchange

### 05:59 `潰してどうすんだよ`

Observed failure:

- the line had been translated too literally as `Why would you crush it?`

Context:

- Minamikawa tricks Takano into drinking a bitter mystery liquid by pretending
  it is water
- the following line is `潰す意味がわからないですよ。何の意味もない。仲間を。`

Repair:

- translated the line as `Why would you screw him over like that?`
- changed the follow-up to `I don't get why you'd screw over your own teammate.`

Desired automation:

- none at the ASR layer; this is a translation/editing nuance

### 06:04 speaker perspective was singular, not plural

Observed failure:

- `He lied to us!`

Repair:

- changed to `You lied to me!`

Rationale:

- the speaker is Takano reacting directly to Minamikawa

Desired automation:

- none at the ASR layer; this is translation review

### 16:15-16:19 pronouns were wrong in the back-and-forth

Observed failure:

- `He is improving, but...`
- `He's improving?`
- `He is improving, but still...`

Japanese:

- `成長はしてるけど`
- `成長してる？`
- `成長はしてるけどそのもう`

Repair:

- `You're improving, but...`
- `I'm improving?`
- `You're improving, but still...`

Desired automation:

- none at the ASR layer; this is translation speaker-tracking

### 27:36 person reference corrected to Michio-san

Observed failure:

- English line said `The way Nishimura laughs means something is definitely up.`

Japanese artifact at the time:

- `西村さんの笑い方絶対なんかある。`

Repair:

- updated the English to `The way Michio-san laughs means something is definitely up.`

Important note:

- this was a scene-identity correction from manual review, not a fix validated
  by the current Whisper artifacts
- keep this separate from the fully validated `お腹` / `高野` mishear cases below

### 31:06 `お腹減ってる？` misheard as `小田さん減ってる？`

Observed failure:

- Japanese artifacts had `小田さん減ってる？`
- English became `Has Oda-san shrunk?`

Validation:

- `whisper_prepass_transcript.json` clearly had:
  - `お腹減ってる?`
  - `減ってますよ`
  - `水あるある`
- the following water/food talk strongly supports the hunger reading

Repair:

- patched:
  - `rab192d6e_gemini_raw.json`
  - `rab192d6e_gemini_raw_chunks/chunk_020.json`
  - `rab192d6e_ctc_words.json`
  - `rab192d6e_reflow.vtt`
  - `rab192d6e_reflow_repaired.vtt`
- updated the English to:
  - `Are you hungry?`
  - `I am.`

Desired automation:

- second-opinion review must catch short high-value disagreement lines, not only
  missing speech regions
- this is the clearest example from the session of a plausible-but-wrong short
  line that survives all the way through alignment and translation if nobody
  intervenes

### 33:00 `高野` misheard as `パタノ`

Observed failure:

- Japanese artifacts had `パタノか` and `パタノのケツにしては...`
- English became `Maybe Patano's?` / `Patano's butt`

Repair:

- patched the raw Gemini chunk, `ctc_words`, both Japanese VTTs, the English
  draft, and the published VTT to use `高野` / `Takano`

Validation:

- manual review of the scene and cast context

Desired automation:

- short name-like disagreement review should flag this class of failure for
  human inspection before translation finalization

## Root Cause Pattern

This episode does not argue for replacing Gemini wholesale with Whisper.

It argues for treating the maintained path as:

- Gemini for the base transcript
- CTC for timing
- faster-whisper as an explicit second opinion
- deterministic reports for both:
  - omission-like coverage gaps
  - short high-value disagreement cues

The omission report alone is not enough. `s03e04` had both:

- a real missing opening block
- plausible short-line mishears that did not look like omissions at all

## Follow-up

Implemented during this session:

- `scripts/report_short_line_disagreements.py`
- wired into `scripts/pre_reflow_second_opinion.py`

Intended use:

- run alongside the existing coverage-gap and raw-omission reports
- treat it as an advisory report for short high-value raw lines where Whisper
  strongly disagrees with the Gemini-based transcript
- use it to prioritize manual checks around names, short questions, and compact
  scene-critical lines

Validation outcome on this episode:

- first full-pass prototype against a real `large-v3` second-opinion artifact
  was too noisy: `271` short-line candidates
- the useful signal was present, but the selector was overfitting to nearest
  timing instead of best nearby text match
- after tightening candidate selection to prefer nearby text agreement, widening
  the local search window, and narrowing the "high-value short" heuristic, the
  same episode dropped to `27` candidates
- the tuned report still surfaced the high-value name disagreement class we
  care about, including `高野か` vs `タカノか`
- omission coverage remained important in parallel: the separate raw-omission
  report still captures the missing-opening / compressed-coverage family better
  than the short-line disagreement report does
