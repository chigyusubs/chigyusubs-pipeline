# CTC Weak-Anchor Failure Modes

This note documents a recurring class of subtitle failures seen in the current
`Gemini raw -> CTC align -> line-level reflow -> translation` path.

The goal is to capture the issue precisely enough for a second-opinion review.
This is not a design proposal by itself. It is a problem statement with
validated examples, observed artifact behavior, and concrete questions.

## Short Version

CTC alignment is generally strong, but some lines behave as weak anchors:

- very short answers
- numeric or unit-bearing answers
- mixed-script wake words / commands
- repeated short command phrases

These lines may:

- fail as zero-duration orphan segments, or
- receive technically non-zero timings that are still too weak/brittle to trust

The downstream problem is not just "CTC missed a line." The bigger issue is that
reflow currently lets these weak lines propagate as if they were ordinary
aligned dialogue. That can cause:

- a prompt cue followed by the wrong answer from a later question
- a blank answer window where a short answer should appear
- short wake-word commands merging with unrelated adjacent dialogue or device replies

## Pattern Name

`weak-anchor lines`

Meaning:

- the text is high-value and semantically important
- the line is too short or too mixed-script to anchor robustly
- downstream code should not trust its timing the same way it trusts an
  ordinary full Japanese sentence

## Example 1: Numeric Orphan Answer Leak

Episode:

- `samples/episodes/great_escape_s03e02_youtube`

Primary affected region:

- around `00:23:45` to `00:24:00`

### Expected content

The later prompt is:

- `大鶴肥満の本日の体重を答えよ。`

The answer should be:

- `190kg。`

The later `161cm。` answer belongs to the next Kitajima Saburo statue question,
not to the Himan weight prompt.

### Evidence

Raw Gemini chunk:

- [great_escape_s03e02_youtube_video_only_v2_gemini_raw.json](/home/npk/code/chigyusubs/pipeline/samples/episodes/great_escape_s03e02_youtube/transcription/great_escape_s03e02_youtube_video_only_v2_gemini_raw.json)

Confirmed local excerpt in the raw chunk text:

- `肥満の体重。`
- `大鶴肥満の、`
- `本日の体重を答えよ。`
- `190kg。`

Original aligned words:

- [great_escape_s03e02_youtube_video_only_v2_ctc_words.json](/home/npk/code/chigyusubs/pipeline/samples/episodes/great_escape_s03e02_youtube/transcription/great_escape_s03e02_youtube_video_only_v2_ctc_words.json)

Observed aligned-word behavior:

- `本日の体重を答えよ。` is at `1429.090-1430.430`
- `190kg。` exists only as a zero-duration orphan at `1185.826-1185.826`
- `161cm。` exists only as a zero-duration orphan at `1436.546-1436.546`
- `70。` exists only as a zero-duration orphan at `1436.546-1436.546`

Local Whisper check on a narrow clip:

- clip: `/tmp/great_escape_s03e02_2344_2402.mp4`
- transcript: [/tmp/whisper_2344/great_escape_s03e02_2344_2402.vtt](/tmp/whisper_2344/great_escape_s03e02_2344_2402.vtt)

Whisper output:

```text
00:00.000 --> 00:08.340
大鶴肥満の本日の体重を答えよう 190キロ
```

This confirms that the intended content in that local window is the Himan
weight prompt followed by `190kg`, not `161cm`.

### Broken downstream result

Original reflow artifact:

- [great_escape_s03e02_youtube_video_only_v2_ctc_words_reflow.vtt](/home/npk/code/chigyusubs/pipeline/samples/episodes/great_escape_s03e02_youtube/transcription/great_escape_s03e02_youtube_video_only_v2_ctc_words_reflow.vtt)

Before repair, that region had:

- the Himan weight prompt
- then a blank answer gap
- then `161cm。` / `70。` in the next cue

Original English draft showed the same error:

- [r101456d8_en.vtt](/home/npk/code/chigyusubs/pipeline/samples/episodes/great_escape_s03e02_youtube/translation/r101456d8_en.vtt)

Before repair, it read:

- `What's his weight today?`
- then later: `161 centimeters. / 70.`

### What this proves

This is not just a translation mistake.

It is a source-artifact failure where:

1. the correct short answer survived only as zero-duration orphan text
2. a later unrelated short answer also survived as orphan text
3. reflow preserved the prompt but attached the wrong orphan answer later

## Example 2: Repeated Mixed-Script Wake-Word Phrases

Episode:

- `samples/episodes/great_escape_s00e03`

Representative regions:

- around `16:05-16:21`
- around `33:08-34:00`

### Evidence

Aligned words:

- [r69545317_ctc_words.json](/home/npk/code/chigyusubs/pipeline/samples/episodes/great_escape_s00e03/transcription/r69545317_ctc_words.json)

Observed lines include:

- `Google、Google。`
- `ヘイ、Google。`
- `オッケー、Google。`
- `オーケー 、 グーグル 。`

Important detail:

- these lines are not all zero-duration
- they do have timestamps
- but they are still brittle because they are short, repeated, and mixed-script

Examples from aligned words:

- `965.513-965.593 | Google、Google。`
- `969.954-970.114 | ヘイ、Google。`
- `970.294-970.674 | オッケー、Google。`
- `971.234-971.714 | オッケー、Google。`

Reflow artifacts:

- [r69545317_reflow.vtt](/home/npk/code/chigyusubs/pipeline/samples/episodes/great_escape_s00e03/transcription/r69545317_reflow.vtt)
- [r69545317_reflow_repaired.vtt](/home/npk/code/chigyusubs/pipeline/samples/episodes/great_escape_s00e03/transcription/r69545317_reflow_repaired.vtt)

English draft:

- [re16ff3a7_en.vtt](/home/npk/code/chigyusubs/pipeline/samples/episodes/great_escape_s00e03/translation/re16ff3a7_en.vtt)

### Failure shape

These `OK Google` lines are often close enough to survive timing-wise, but not
stable enough to behave like ordinary anchors.

Observed brittle behaviors:

- wake phrase shares a cue with unrelated adjacent text
- assistant/device response follows immediately and the local boundaries become fragile
- repeated short wake phrases cluster together and become easy to merge or misread

This is a different surface expression of the same underlying family:

- short, high-value, weakly anchored lines are allowed downstream without being
  marked as special-risk lines

## Common Properties Across Both Examples

These failure cases share most of the following:

- very short text
- semantically high-value
- often repeated nearby
- often mixed-script or unit-bearing
- often easy for humans to verify
- timing confidence is much weaker than a normal full Japanese sentence

Common categories:

- `190kg。`
- `161cm。`
- `44回。`
- `35発。`
- `OK Google`
- `オッケー、Google。`
- clipped English/Japanese interjections or command phrases

## Current Pipeline Boundary Where The Bug Escapes

The failure is crossing the alignment/reflow boundary.

### Alignment stage

File:

- [align_ctc.py](/home/npk/code/chigyusubs/pipeline/scripts/align_ctc.py)

Current good behavior:

- full all-unaligned lines are repaired locally before chunk offsetting
- monotonic timing pass runs afterward
- per-chunk diagnostics are saved

Current gap:

- some weak-anchor lines still survive as zero-duration or degenerate segments
- some mixed-script short commands receive non-zero timings but remain brittle
- these lines are not clearly promoted to a stronger downstream review state

### Reflow stage

File:

- [reflow.py](/home/npk/code/chigyusubs/pipeline/chigyusubs/reflow.py)

Current good behavior:

- line-level reflow avoids split-word artifacts
- zero-duration text is preserved instead of silently dropped

Current gap:

- preserving the text is not enough if the attachment rule is weak
- nearest-cue attachment can leak a short orphan answer into the wrong prompt
- mixed-script repeated command lines can be treated like ordinary lines even
  when their local timing confidence is poor

## What This Issue Is Not

It is not best described as:

- "CTC cannot align numbers"
- "translation hallucinated"
- "reflow is generally broken"

More accurate description:

- CTC is usually strong
- a small but important class of weak-anchor lines is still fragile
- reflow currently handles "preserve the text" better than "preserve the right
  local semantic attachment"

## Hypotheses Worth Reviewing

These are hypotheses, not validated fixes.

### Hypothesis A

Arabic numerals with attached units (`kg`, `cm`, `回`, `発`, `万`) are harder to
anchor than ordinary Japanese lines because they are short and mixed-format.

### Hypothesis B

Repeated mixed-script command phrases such as `OK Google` are not necessarily
zero-duration failures, but they are weak enough that downstream code should
treat them as special-risk lines.

### Hypothesis C

The main bug is not alignment alone. The bigger bug is that downstream reflow
does not distinguish:

- robust aligned line
- repaired all-unaligned line
- weak-anchor short line with brittle timing

### Hypothesis D

A local rescue path is likely better than global normalization:

- flag weak-anchor lines
- inspect only the local span
- if needed, use local Whisper or another bounded second-opinion aligner

## Candidate Fix Directions To Evaluate

These are the main directions worth second-opinion review.

### 1. Stronger post-CTC repair for weak-anchor lines

Instead of repairing only `_unaligned` lines, also catch:

- zero-duration segments
- near-zero segments
- short unit-bearing answers
- repeated mixed-script command phrases

### 2. Reflow should not blindly nearest-attach weak answers

Especially for:

- numeric orphan answers
- short repeated wake words

These should either:

- attach only to the immediately preceding prompt/setup in local transcript order, or
- be surfaced as repair-needed rather than silently attached

### 3. Local second-opinion fallback

When a line is flagged as weak-anchor:

- compare against raw Gemini chunk text
- optionally run a local Whisper clip for only that span
- possibly compare against stable-ts only as a bounded local fallback

### 4. Alignment-only normalization layer

Possible but risky:

- convert digits/units to spoken-form aliases for matching only
- keep saved source text unchanged

This needs caution because number readings are ambiguous in Japanese and unit
handling can get messy.

## Questions For Second Opinion

If reviewing this issue externally, the useful questions are:

1. Is the right abstraction "zero-duration orphan lines", or the broader
   concept of "weak-anchor lines"?
2. Should the primary fix live in `align_ctc.py`, `reflow.py`, or both?
3. Is there a robust way to classify weak-anchor lines deterministically?
4. Is an alignment-only spoken-form normalization for short numeric/unit lines
   worth the complexity?
5. For repeated mixed-script wake phrases like `OK Google`, should they be
   treated as a dedicated special case or as one subtype of weak-anchor lines?
6. Is a local Whisper rescue pass the right maintained fallback for flagged
   spans?

## Suggested Files To Inspect

- [align_ctc.py](/home/npk/code/chigyusubs/pipeline/scripts/align_ctc.py)
- [reflow.py](/home/npk/code/chigyusubs/pipeline/chigyusubs/reflow.py)
- [great_escape_s03e02_youtube_video_only_v2_ctc_words.json](/home/npk/code/chigyusubs/pipeline/samples/episodes/great_escape_s03e02_youtube/transcription/great_escape_s03e02_youtube_video_only_v2_ctc_words.json)
- [great_escape_s03e02_youtube_video_only_v2_ctc_words_reflow.vtt](/home/npk/code/chigyusubs/pipeline/samples/episodes/great_escape_s03e02_youtube/transcription/great_escape_s03e02_youtube_video_only_v2_ctc_words_reflow.vtt)
- [great_escape_s03e02_youtube_video_only_v2_gemini_raw.json](/home/npk/code/chigyusubs/pipeline/samples/episodes/great_escape_s03e02_youtube/transcription/great_escape_s03e02_youtube_video_only_v2_gemini_raw.json)
- [r69545317_ctc_words.json](/home/npk/code/chigyusubs/pipeline/samples/episodes/great_escape_s00e03/transcription/r69545317_ctc_words.json)
- [r69545317_reflow.vtt](/home/npk/code/chigyusubs/pipeline/samples/episodes/great_escape_s00e03/transcription/r69545317_reflow.vtt)
- [/tmp/whisper_2344/great_escape_s03e02_2344_2402.vtt](/tmp/whisper_2344/great_escape_s03e02_2344_2402.vtt)

