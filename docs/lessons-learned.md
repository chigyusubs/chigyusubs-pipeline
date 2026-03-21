# Lessons Learned

This document records what the repo has actually taught us so far, based on real runs against:

- `samples/episodes/great_escape1`
- `samples/episodes/oni_no_dokkiri_de_namida_ep2`

It is intentionally operational. The goal is to capture what worked, what did not, and what still needs to be fixed.

## Current Conclusions

### 1. Reusable artifacts were the right architectural move

Saving OCR, VAD, chunk boundaries, alignment outputs, and diagnostics as separate artifacts is materially better than recomputing them inside one monolithic script.

What this unlocked:

- rerunning OCR filtering without rerunning OCR
- rerunning translation without rerunning alignment
- comparing video-only vs OCR-assisted transcription on the same episode

Named Gemini presets are worth keeping for the maintained path.

- there are too many coupled Gemini settings to rely on ad hoc flag combinations
- maintained presets make real-run comparisons easier and reduce accidental drift
- a named Flash Lite debug preset is worth keeping separate from real-run presets so cheap smoke tests do not silently inherit production settings
- raw flags should still exist for experiments, but the common paths should have names
- inspecting failures at the artifact boundary instead of guessing
- maintained Gemini/API-backed CLI entrypoints should auto-load the repo `.env` so Codex and direct terminal runs do not fail just because keys were not shell-exported

Chunk-plan intent also needs to be surfaced explicitly.

- once semantic plans, repair plans, and debug probe plans accumulate, filenames alone stop being self-explanatory
- maintained Gemini helpers should log a human-readable chunk-plan label and duration summary when `--chunk-json` is supplied
- docs should treat `vad_chunks.json`, `vad_chunks_semantic_<target>.json`, `*_repair*.json`, and `probes/*exact_chunks_<target>s*.json` as distinct operator-facing categories

ROCm faster-whisper env defaults matter operationally.

- faster-whisper on ROCm should always run with `CT2_CUDA_ALLOCATOR=cub_caching`
- forgetting that can turn a normal pre-pass into a GPU hang / allocator failure
- helper scripts that invoke faster-whisper as part of a maintained workflow should set that env themselves instead of expecting the operator to remember it
- Codex sandbox runs also need escalation before ROCm is actually visible; setting the env inside an already-started Python process is not enough to expose the device
- for `ctranslate2` and `faster-whisper`, `LD_LIBRARY_PATH=/opt/rocm-7.2.0/lib` needs to be present when Python starts, not added later after import failures

Semantic chunk review should also reuse pre-pass artifacts when possible.

- `build_semantic_chunks.py prepare` does not need to rerun faster-whisper on an episode that already has a usable `transcription/whisper_prepass_transcript.json`
- reusing that artifact is faster, cheaper, and avoids introducing extra GPU instability into a chunk-review pass whose goal is boundary review, not ASR benchmarking
- add an explicit `--rerun-whisper` switch for the cases where a fresh pre-pass is actually desired

Hard-max chunking also needs a short-gap fallback before true forced splits.

- a dense dialogue span on `great_escape_s02e03` had no `>=1.5s` silence gap late enough to stay under the hard max
- both the plain VAD path and semantic finalize would otherwise cut through active speech at the hard cap
- the better fallback is to accept a shorter real silence gap, down to about `0.75s`, before ever inserting a mid-speech split

Spoken-only Gemini chunks can occasionally volunteer visual prompt text even when the base prompt did not ask for `[画面: ...]` lines.

- on `great_escape_s02e05`, a spoken-only chunk leaked `[画面: クロちゃんの現在の貯金額を答えよ]`
- this was rare, and not enough evidence by itself to justify a broader spoken-only prompt rewrite
- but it is important to remember during transcript review, because one leaked visual line can become a bad English subtitle later if nobody catches it
- treat this as a manual review watchpoint, especially around obvious on-screen prompt cards and quiz text

### 2. CTC forced alignment replaced stable-ts and nearly eliminated stranded words

stable-ts uses Whisper's cross-attention for alignment, which is a byproduct of the generative model. This caused 13.4% of words to get zero-duration timestamps on `oni_no_dokkiri_de_namida_ep2`, even with chunked alignment.

CTC forced alignment (`scripts/align_ctc.py`) uses `NTQAI/wav2vec2-large-japanese` with `torchaudio.functional.forced_align`. This is a dedicated alignment mechanism — it finds the optimal path through frame-level character probabilities via Viterbi decoding.

Results on `oni_no_dokkiri_de_namida_ep2`:

| | stable-ts (chunked) | CTC wav2vec2-ja |
|---|---|---|
| Zero-duration segments | 42 (3.8%) | 3 (0.3%) |
| Zero-duration words | 947 (13.4%) | 3 (0.3%) |

On the original `oni_no_dokkiri_de_namida_ep2` benchmark, the 3 remaining
zero-duration segments were non-Japanese text (`Shit!`, `30?`, `50。`) with no
characters in the model's Japanese vocabulary. That result was encouraging, but
later real episodes showed the residual failure surface is broader than "only
trivial non-Japanese leftovers."

Key properties of CTC alignment:

- errors are local — one bad region does not cascade and collapse a whole tail
- works at the character level, so even short utterances get placed
- deterministic given the audio features, no autoregressive drift
- runs on GPU via system python3.12 with ROCm

Model selection matters: `facebook/mms-1b-all` was tested first but had 10.8% character vocabulary gaps on real Japanese text (missing common kanji like `鬼`, `凄`, `喋`, plus all punctuation). `NTQAI/wav2vec2-large-japanese` has a proper Japanese vocabulary with 2174 tokens covering kanji, hiragana, and katakana.

CTC forced alignment is now the recommended default for alignment.

### 3. Chunked alignment is much more robust than whole-episode alignment

Whole-episode `stable-ts` alignment is too fragile. One bad transcript section can collapse a large tail of the episode to the final timestamp.

Chunked alignment fixed the major failure mode on `oni_no_dokkiri_de_namida_ep2`:

- whole-episode alignment collapsed a large late tail to `2442.51`
- chunked alignment removed the end-of-file collapse
- only local zero-duration alignment misses remained

Chunked alignment should remain the default. CTC forced alignment (lesson 2) also uses chunked alignment.

### 4. Zero-duration alignment misses should not be dropped blindly

With CTC alignment, zero-duration segments are now rare (0.3% on `oni_no_dokkiri_de_namida_ep2`) and were initially thought to be limited to non-Japanese text. But the principle still holds: do not drop them blindly.

The zero-duration segments from earlier stable-ts runs were not mostly garbage. Many were real short utterances or chunk-edge dialogue blocks.

Dropping them would make later English translation worse. The current reflow step preserves their text by merging them into nearby timed cues.

New failure mode seen on `great_escape_s01e01`:

- a short Japanese answer line (`蚊。`) failed alignment entirely inside a chunk
- `align_ctc.py` emitted it as `0.0 -> 0.0` inside the chunk
- the chunk offset then turned that into the chunk start timestamp, creating a large backward jump
- line-level reflow attached the zero-duration line to an earlier cue, and max-line trimming hid it

The fix is to repair all-unaligned line timings inside `align_ctc.py` before chunk offsetting:

- keep the text
- keep the original line order
- interpolate a small local slot between neighboring aligned lines instead of defaulting to chunk start
- run a final monotonic timing pass so tiny backward reversals do not survive into saved alignment JSON
- keep diagnostics for those repaired lines so QA can still count them and inspect which lines were interpolated
- carry those diagnostics forward into reflow review and Codex translation as advisory context, not as a hard stop
- when the repaired line falls into a tiny reflow gap, attach the warning to the nearest cue instead of dropping it from downstream review

This preserves short answers and other high-value lines without pretending the timing confidence is high.

This is the correct default for a translation-first pipeline, even if the repaired Japanese cue timing is imperfect.

Another real failure on `great_escape_s03e02_youtube` showed a later variant of the same problem:

- one quiz-answer line (`190kg。`) survived only as a zero-duration orphan in the aligned words JSON
- a later unrelated answer (`161cm。` / `70。`) also survived as zero-duration orphan text
- line-level reflow preserved the later question prompt (`大鶴肥満の本日の体重を答えよ。`) but then filled the next cue with the wrong later answer
- the published English VTT therefore showed a blank answer window around `23:51` and then the wrong `161 centimeters / 70` text at `23:56`

Operational rule:

- when a prompt-card cue is followed by a long silence gap and the next cue contains a short numeric answer with weak alignment provenance, compare that region against the raw Gemini chunk or a local Whisper clip before accepting the reflow
- treat this as a local `yellow`/repair case even if the rest of the episode is structurally healthy

Another recurring CTC-adjacent pattern shows up on short repeated wake-word or
command phrases such as `OK Google` / `オッケー、Google`:

- these lines are short, mixed-script, and often repeated several times within a small span
- they may not fail as full zero-duration orphans, but they still behave like weak alignment anchors
- in `great_escape_s00e03`, several `Google` wake phrases aligned, but the local cue boundaries stayed brittle enough that adjacent assistant/chatter text could get merged into the same cue
- this is the same family of issue as the numeric-answer leak: a very short high-value line with weak timing confidence is allowed to propagate downstream as if it were ordinary aligned dialogue

Operational rule:

- treat short repeated mixed-script command phrases (`OK Google`, brand wake words, device commands, clipped English interjections) as weak-alignment lines even when they have non-zero timings
- surface them in review alongside zero-duration orphan answers, especially when they appear in repeated bursts or next to assistant/device responses
- if such a line lands inside a cue that also contains unrelated adjacent dialogue, prefer a local raw/Whisper check over trusting the CTC boundary blindly

Another real failure on `great_escape_s03e02_youtube` showed that rescue itself
can create a second-order timing bug if the saved JSON is not re-clamped:

- stable-ts rescue moved `8、9。` earlier than the preceding `うん、うん、うん。`
- stable-ts rescue also left `35発。` ending after the following `あ、来た来た来た!`
- line-level reflow preserved transcript order, so those backward jumps became negative-duration cues in the Japanese VTT

The fix is to run the monotonic timing pass again after weak-anchor rescue and
before writing `*_ctc_words.json`, not only before rescue.

### 5. Reflow must not use text-only word-timestamp lookup for repeated lines

Another real failure on `great_escape_s01e06` came from line-level reflow, not alignment:

- `_split_long_single_lines()` looked up character timestamps by line text only
- repeated strings like `うわ、すげえ。` appeared multiple times in one episode
- a later cue inherited the earlier occurrence's word timings
- that produced backward cue times and a negative-duration VTT cue

The fix is to resolve repeated lines by nearest `(text, start, end)` match, not by text alone.

### 6. Video-only Gemini is a serious path, not just an experiment

For both `great_escape1` and `oni_no_dokkiri_de_namida_ep2`, direct video transcription with Gemini was viable.

Why it matters:

- it removes the entire OCR preprocessing cost from the cloud path
- it naturally sees mission cards, question text, premise cards, counters, and subtitle-like telops
- it may be good enough often enough that OCR should become optional instead of mandatory for Gemini workflows

The main lesson:

- OCR is still useful
- but Gemini video-only is strong enough that it must be treated as a first-class baseline

### 7. Shorter video chunks were a big win

The first video-only trial on `great_escape1` used long fixed chunks and was too verbose and more failure-prone.

Switching to shorter chunks improved behavior substantially:

- less drift
- less visual over-description
- fewer repetition failures
- better local scene focus

Using VAD-guided chunk boundaries for video-only transcription worked better than naive long fixed chunks.

Important refinement:

- VAD should place boundaries, not filter content
- for Gemini video transcription, chunk coverage should stay continuous from `0` to episode end
- splitting at the midpoint of a silence gap is safer than cutting the silence out entirely
- speech-bounded chunk spans are more fragile and can hide real missing-dialogue gaps until manual review

### 7. Bracketed visual context is the right output shape for video-only transcription

The best video-only prompt shape so far was:

- spoken text as `-- ...`
- visual-only context as `[画面: ...]`

This was better than forcing Gemini to mix spoken and visual information into one transcript.

Why:

- spoken text remains alignable
- visual scene/rule/question context is preserved
- `[画面: ...]` lines can be stripped before alignment

This format worked well on `great_escape1` and `oni_no_dokkiri_de_namida_ep2`.

### 7a. Visual-text extraction can substitute for narration instead of complementing it

Validated on `killah_kuts_s01e03`.

Failure mode:

- Gemini did capture map-summary/status text such as `それぞれの現在地はご覧の通り` and `未だウエストランドがやや優勢といった状況`
- but it captured them as `[画面: ...]` lines instead of spoken `-- ...` narration
- `align_ctc.py` correctly stripped those visual lines before speech alignment
- the aligned words JSON and reflowed VTT therefore looked structurally healthy while the narrator VO was missing from the spoken transcript

Operational conclusion:

- asking for visual cues was not a mistake
- but visual cues cannot be treated as proof that spoken coverage is complete
- dense `[画面: ...]` runs in the middle of active chunk dialogue are a real warning sign, especially when the text looks like narration, rules, or status summaries

Pipeline consequence:

- `align_ctc.py` diagnostics should flag `possible_visual_narration_substitution`
- `scripts/pre_reflow_second_opinion.py` should be the maintained gate that consumes that flag
- when that flag appears, the preferred repair path is a local `faster-whisper large-v3` second-opinion pass plus a transcript coverage diff before reflow
- do not assume CTC itself can detect missing speech when the text never reached the spoken transcript

This is a manual-review aid, not a reason to remove visual cues from the Gemini format entirely.

### 7b. Gemini 3 default media resolution is a real confounder in cost and quality comparisons

Validated on `great_escape_s02e01` chunk 1 with `gemini-3.1-flash-lite-preview`.

On the same inline video payload:

- default / unspecified:
  - `prompt_token_count=15982`
  - video tokens `15738`
  - output tokens `386`
- `HIGH`:
  - `prompt_token_count=50236`
  - video tokens `49992`
  - output tokens `410`

That is roughly a `3.1x` prompt-token increase, driven almost entirely by video tokens.

Quality also changed in meaningful ways:

- `袋` became `ブクロ`
- `解いてっててた` became `解いてって言ってた`
- but `HIGH` did not make Flash Lite fully trustworthy; the same chunk still produced questionable lines like `偽物や` and `勝ち目せんわ`

Operational implication:

- comparisons between `gemini-2.5-pro` and Gemini 3.x are not apples-to-apples if Gemini 3 is left at default media resolution
- for text-heavy or cast-identification-heavy video-only transcription, test `HIGH` before concluding Gemini 3 quality is inherently worse
- Gemini API `count_tokens` currently does not support `media_resolution` overrides, so exact preflight counts for `low`/`medium`/`high` need Vertex or a real generation request with `usage_metadata`

### 7b1. Local ffmpeg downscaling is a separate confounder from Gemini `media_resolution`

Validated during later AI Studio and API probe review.

Important distinction:

- Gemini `media_resolution=high` is an internal model-side processing tier
- it is not a published fixed output pixel size
- if the local ffmpeg chunker downscales first, Gemini never sees the lost detail

Operational implication:

- preserve source resolution by default for maintained Gemini video chunking
- keep `1 FPS`, but do not downscale width unless payload size is the explicit experiment variable
- do not interpret Gemini `high` as compensation for a low-resolution upload

### 7c. Flash Lite chunk length and temperature tuning help at the margins, but they do not remove model instability

Validated on a stable non-looping `great_escape_s02e01` section around the beach / panel-banter opening using `gemini-3.1-flash-lite-preview` with `media_resolution=high`.

Chunk-length sweep from the same start time:

- `60s`: completed in `4.8s`
- `90s`: failed with `504 DEADLINE_EXCEEDED`
- `120s`: completed in `6.8s`
- `150s`: completed in `7.5s`

This is important because it means Flash Lite instability is not monotonic with chunk duration. A failing chunk length does not imply that every longer chunk will also fail, and a successful shorter chunk does not prove the neighboring lengths are safe.

Follow-up validation on a problematic `great_escape_s02e03` span that had already gone bad in a production run strengthened the practical short-chunk case for Flash Lite. The same `232.1s` scene was clipped and probed two ways with `spoken_only`, `media_resolution=high`, `temperature=0.0`, and no rolling context:

- `120s + 112s`: chunk 2 emitted `10053` chars / `1257` lines, tripped the loop detector, and only became usable after the automatic retry
- `60s + 60s + 60s + 52s`: all 4 chunks completed cleanly on the first attempt with sane output sizes
- total token usage and estimated cost were almost unchanged:
  - `120s` probe: `67,422` prompt / `1,149` output tokens, about `$0.01858`
  - `60s` probe: `67,663` prompt / `1,163` output tokens, about `$0.01866`

So while chunk length is still not a monotonic knob on Flash Lite, `60s` can be a materially safer probe setting than `120s` on fragile dialogue spans, and the cost penalty can be negligible.

There is also an important distinction between "about 60s" and exact `60s`. On `great_escape_s02e03`, a VAD-guided "`60s`" plan still produced a `64.576s` early chunk and Flash Lite died there after two `504 DEADLINE_EXCEEDED` errors. A strict full-coverage exact-`60s` plan for the same episode got much further: it completed chunks `0-16`, including one loop-retry recovery and one timeout-retry recovery, before stalling on another chunk that hit two timeout-type errors. So exact `60s` can materially improve survivability over VAD-guided `~60s`, but it still does not make Flash Lite a reliable full-episode transcript model.

Temperature sweep on the same `120s` section:

- `0.0`: completed in `4.9s`
- `0.1`: completed in `4.6s`
- `0.2`: completed in `3.5s`
- `0.3`: failed with `504 DEADLINE_EXCEEDED`

Quality differences were modest but still directional:

- `0.0` preserved the dialogue structure best among the cheap Flash Lite probes
- `0.1` compressed adjacent short reactions more aggressively
- `0.2` stayed fast but introduced slightly more drift
- none of the successful temperatures fixed the core lexical miss around `オーシャンビュー`

Operational implication:

- for cheap Flash Lite probing, keep temperature at `0.0` or `0.1`
- avoid treating chunk length as a clean monotonic tuning knob on Flash Lite
- if a specific Flash Lite span is looping or timing out at `~120s`, testing `~60s` is justified before giving up on the scene
- if Flash Lite is still failing on a VAD-guided `~60s` plan, try strict exact `60s` before concluding the whole episode is unsalvageable
- use these probes to compare settings cheaply, but do not interpret a locally good Flash Lite result as evidence that the model is ready for full-episode production

### 7d. On Flash Lite, video can improve exact rule-text capture while making surrounding speech less literal

Validated on a short `30s` rule-card slice from `great_escape_s02e01` using `gemini-3.1-flash-lite-preview` at `temperature=0.0`.

Compared paths:

- video input with `media_resolution=high`
- audio-only input
- same spoken-only prompt

What happened:

- video captured the rule line cleanly as `クイズに正解してこの部屋から脱出してください`
- audio also captured the same line, but with tokenized spacing artifacts
- however, video made the surrounding spoken banter worse in several places:
  - `せやろな` drifted into `捨てたろな` / `捨てやろな`
  - early complaint lines were less faithful than the audio-only version
- audio-only preserved the spoken Kansai lines more plausibly, even though formatting was rougher

Operational implication:

- on Flash Lite, video is not automatically better for faithful spoken transcription
- video may be worth paying for when exact rule cards or title text matter
- audio-only can still be the better literal speech-recognition path on short dialogue-heavy slices
- if the workflow needs both, treat Whisper or audio-only as the spoken-truth check and treat video as context / rule-text support

Follow-up multiscene probe on `great_escape_s02e01` with `gemini-3.1-flash-lite-preview`, `temperature=0.0`, and a spoken-only prompt showed the pattern is scene-dependent:

- opening cast-card / room-reveal section:
  - audio-only was cheaper and generally more literal
  - video introduced invented or more distorted lines like `偽物や` and `勝ち目ないわ`
- beach / panel-banter section:
  - both video and audio missed `オーシャンビュー`
  - audio preserved names a bit better (`シンイチ` vs video's `真一`)
  - video remained slightly more normalized and visually grounded, but not clearly more faithful
- later quiz-prompt block:
  - video clearly helped with exact rule / prompt text
  - audio degraded proper nouns and dense prompt reading much more badly

So for Flash Lite, the practical rule is not “video good” or “audio good.” It is:

- dialogue-heavy banter sections: audio-only may be the cleaner spoken-truth probe
- prompt / rule / on-screen-information sections: video is worth it if those details matter

### 7e. `gemini-2.5-flash` handles the same video-vs-audio rule-card slice much better than Flash Lite

Validated on the same short `30s` `great_escape_s02e01` rule-card slice at `temperature=0.0` with the same spoken-only prompt.

Compared with Flash Lite:

- `2.5-flash` video captured the rule line cleanly and also preserved the surrounding spoken banter much better
- `2.5-flash` audio also preserved the spoken banter well, with only minor tokenization artifacts like `いち` for `1`
- the damaging Flash Lite video failure (`せやろな` -> `捨てたろな` / `捨てやろな`) did not recur on `2.5-flash`

Residual differences still matter:

- `2.5-flash` video wrote `なんてな` where audio had the more plausible `なんでな`
- `2.5-flash` audio wrote `なんぼ言ってもいいってこと？`, which is more plausible for the spoken line than video's `なんも言ってもいいってこと。`
- `2.5-flash` video incurred a much larger prompt-token cost than audio, and exposed `thoughts_token_count=45` on this probe

Operational implication:

- the “video helps visual text but hurts spoken banter” result should currently be treated as a Flash Lite weakness, not a general Gemini rule
- on `2.5-flash`, video is a much more defensible default for visual-heavy comedy/game content
- audio-only remains a useful comparison path when a spoken regional phrase looks suspicious

Follow-up multiscene probe on `great_escape_s02e01` with the opening cast-card section and the beach / panel-banter section strengthened that conclusion:

- opening section:
  - neither modality was clean, but `2.5-flash` still avoided the worst Flash Lite beach-level lexical collapse
  - audio was cheaper and somewhat more grounded on obvious visible content like `1000万`
  - video incurred large extra cost and even exposed `thoughts_token_count=1055` on this probe
- beach section:
  - both video and audio got `オーシャンビューよ` right
  - this is a major quality difference from Flash Lite, where both modalities still missed it
  - video remained more compressive than audio, but not catastrophically so

Operational implication:

- on `2.5-flash`, the main value of audio-only is cost and occasional spoken-phrase sanity checking
- on `2.5-flash`, video is generally a defensible default even outside dense rule cards
- the strongest anti-video caution from current evidence is specific to Flash Lite, not to Flash-tier Gemini as a whole

### 7f. `gemini-3-flash-preview` currently looks operationally friendlier on video than on audio in this workflow

Validated on the same `great_escape_s02e01` multiscene probe setup used for Flash Lite and `2.5-flash`:

- spoken-only prompt
- `temperature=0.0`
- three scenes: opening, beach, later prompt block

Observed behavior on this run:

- all three video probes completed
- all three audio probes failed before returning usable text
  - two ended with read timeouts
  - one ended with `503 UNAVAILABLE`

This is not enough to conclude that `gemini-3-flash-preview` audio transcription is inherently worse. It is enough to conclude that, under the current API conditions and prompt path, the model was operationally less reliable on audio-only than on video for this experiment.

The completed video outputs were useful but not perfect:

- opening section recovered visible-value details like `1000万`
- beach section still reduced `オーシャンビュー` to `オーサンビュー`
- rule/prompt section preserved the important prompt text well

Operational implication:

- do not assume `gemini-3-flash-preview` audio-only is the safer or cheaper default path until it proves stable on repeated probes
- for now, `gemini-3-flash-preview` video-only remains the more validated path in this repo

### 7g. AI Studio safety filters can block spoken-only beach/distress scenes even when the transcription task is legitimate

Validated on the `great_escape_s02e01` beach / panel-banter scene in manual AI Studio runs.

Observed behavior:

- spoken-only prompting on the beach scene triggered safety blocks related to sexual / harassment categories
- the run became possible only after lowering safety settings in the AI Studio UI

Most plausible explanation:

- the scene contains distressed shouted speech (`助けて`, `埋められてんじゃん`) plus beach/burial context
- without visual-cue framing, the moderation layer appears to over-read the content as abusive or sexualized

Operational implication:

- treat UI-level safety refusals as a separate experimental variable from model transcription quality
- for manual AI Studio comparisons, record when safety had to be lowered to get a result
- do not compare a safety-blocked spoken-only run against an unblocked spoken-plus-visual run as if they were equivalent conditions

### 7h. On `gemini-3.1-pro-preview`, lower thinking improved the beach-scene result shape but did not fully fix lexical drift

Validated on the manual AI Studio `great_escape_s02e01` beach / panel-banter scene.

Compared outcomes:

- `spoken_plus_visual` with higher/default thinking recovered visual cues well but missed `オーシャンビュー`
- `spoken_only` with lower thinking produced cleaner plain-text formatting once, but another run regressed badly into `王様ビーチ` / `砂浜`
- `spoken_plus_visual` with low thinking recovered `オーシャンビュー` and kept output shape stable

Residual issues still remained in the low-thinking visual run:

- `サラバ` instead of `さらば`
- `すごい番組` instead of the more plausible `すごいメンバー`
- `俺サイコロなんすか` / `あれなんかあれ怖い` remained degraded

Operational implication:

- on `gemini-3.1-pro-preview`, lowering thinking looks more promising than leaving it high for strict transcription
- but even with low thinking, `3.1-pro-preview` still does not clearly beat the best `2.5-flash` result on this chunk

### 7i. `faster-whisper large-v3` remains a stronger speech-truth benchmark than Flash Lite audio-only on the beach scene

Validated on the `great_escape_s02e01` `beach_panel_banter` experiment clip, comparing:

- `faster-whisper large-v3` on the extracted audio clip
- `gemini-3.1-flash-lite-preview` with `audio + spoken_only + high thinking`

Key result:

- Whisper recovered the core lexical trap correctly as `オーシャンビューよ`
- Flash Lite still misheard it as `大さん橋よ`

Whisper was not clean enough to be treated as a final transcript on its own. The same run still had visible roughness:

- `あの街で` for the buried-beach distress line
- `地面いたつき`
- `黒ちゃん`
- `サラバ`
- a split-cue artifact around `オ / ーシャンビュー`

But the comparison still matters operationally:

- Flash Lite audio-only did improve over Flash Lite video on some name-shape and speech-only details
- even so, Flash Lite did not catch up to Whisper on the key contested spoken line
- Whisper remains the better benchmark and second-opinion artifact when we want to know what was actually said

Operational implication:

- treat `faster-whisper large-v3` as the speech-truth benchmark for cheap-model experiments
- treat Flash Lite audio-only as an auxiliary draft path only
- do not promote Flash Lite to the primary transcript source just because audio-only plus higher thinking looks better than its video runs

### 7j. `gemini-3-flash-preview` audio-only can get much closer to Whisper on banter-heavy speech than Flash Lite does

Validated on the same `great_escape_s02e01` `beach_panel_banter` clip, comparing:

- `gemini-3-flash-preview` with `audio + spoken_only + high thinking`
- `faster-whisper large-v3`

Compared with Flash Lite, `3-flash` closed most of the important gap:

- it recovered `オーシャンビューよ`
- it preserved `さらば`
- it kept `信一` instead of collapsing the name to a wrong place-like reading

Residual differences versus Whisper still remained:

- `地面板付き` / `地面いたつき` were still both wrong in different ways
- `リピーター。 / いらっしゃいますけどね。` was more normalized than the rougher Whisper read
- `どこ、ここ？` and some short reaction lines were cleaned up rather than kept maximally literal

Operational implication:

- on this scene, `gemini-3-flash-preview` audio-only is much closer to Whisper than Flash Lite audio-only is
- Whisper still remains the better benchmark when the question is “what was actually said?”
- but `3-flash` audio-only is strong enough to be treated as a real comparison path, not just a failed experiment

### 7k. On current free-tier models, `gemini-3-flash-preview` is the strongest non-Pro compromise path so far

Validated on `great_escape_s02e01` across the beach / panel-banter chunk and the rule / prompt block.

Most important current finding:

- `gemini-3-flash-preview` with `video + spoken_only + low thinking + media_resolution=high + temperature=0.0` is the best single compromise mode we have tested on the free tier

Why it currently leads:

- better spoken cleanliness than `spoken_plus_visual` on banter-heavy chunks
- better visual disambiguation than audio-only
- fewer meaning-changing regressions than `gemini-2.5-flash` on the beach scene
- much stronger overall quality than `gemini-3.1-flash-lite-preview`

Counterpoints:

- `gemini-2.5-flash` is still competitive and remains useful as a secondary comparison path
- `gemini-2.5-flash` should stay at default thinking; turning thinking fully off regressed badly
- `gemini-3-flash-preview` still benefits from chunk-type awareness, and `spoken_plus_visual` may still be preferable on dense rule-card sections when the visual artifact itself matters

Operational implication:

- for free-tier daily episode work, prefer `gemini-3-flash-preview` first
- keep `media_resolution=high`
- keep `temperature=0.0`
- prefer `spoken_only`
- prefer `thinking=low` on video

### 8. OCR-first filtering became too conservative when it focused mainly on names

The initial OCR cleanup/classification work improved safety, but it over-optimized for name/entity anchors.

This was not enough for these shows.

What the examples showed:

- focused board questions are useful
- room/mission text is useful
- subtitle-like reaction telops are useful
- state readouts can matter when the scene depends on them

What is less useful on its own:

- only preserving performer names

The OCR filtering layer should keep more local semantic telop content, not just names.

### 9. OCR still varies too much across show formats to over-classify early

`great_escape1` and `oni_no_dokkiri_de_namida_ep2` share some properties, but the visual presentation differs a lot.

Examples:

- `great_escape1`: puzzle boards, focused question frames, counters, rules
- `oni_no_dokkiri_de_namida_ep2`: location cards, cast intros, premise cards, subtitle-like captions, constant service watermark

This argues against a heavy semantic classifier early in the OCR pipeline.

### 9a. Chunk-wise OCR-only prompting may be useful, but beach/banter scenes can collapse to low-recall name-card extraction

Validated on the `great_escape_s02e01` `beach_panel_banter` clip with a manual AI Studio `ocr_only` prompt on `gemini-2.5-flash`.

Observed output:

- `クロちゃん`
- `安田大サーカス`
- `バカリズム`
- `バイきんぐ`
- `小峠英二`

This is not useless, but it is narrow:

- it mostly captured visible name cards
- it did not surface richer contextual telops
- it did not add much beyond what the stronger video-transcription runs were already implicitly using

Operational implication:

- chunk-wise OCR-only prompting is probably not worth the extra step on banter-heavy scenes
- it is still worth testing on dense rule/prompt scenes, where the visible text is the point
- if OCR-only is kept, treat it as a chunk-type-specific tool, not a global preprocessing default

### 9b. The same OCR-only prompt becomes much more valuable on dense rule/prompt chunks

Validated on the `great_escape_s02e01` `rule_prompt_block_2` clip with manual AI Studio `ocr_only` prompting on `gemini-3.1-flash-lite-preview`.

Observed output included:

- the full `Take2深沢邦之...一番高い山...` prompt
- the full `TSUTAYA西五反田店18禁ナンパコーナー...` prompt
- the full `本日東京地方裁判所...` prompt
- the full `日比谷公園の噴水...` prompt
- smaller local items like `テレホンカード` and `料金 100.000円`

This is materially more useful than the beach-scene OCR-only result:

- the visible text is the main information-bearing content of the chunk
- OCR-only output captures reusable prompt text that can support transcription, glossary building, and translation review
- even Flash Lite is good enough here as a cheap visual-text extractor

Operational implication:

- OCR-only prompting is worth keeping as a chunk-type-specific tool for dense visual prompt sections
- it should not be run blindly on every chunk
- the best current role for Flash Lite may be exactly this: cheap OCR-like visual extraction on text-heavy chunks

Follow-up comparison on the same chunk with `thinking=minimal`:

- retained the major prompt cards cleanly
- improved a numeric format detail (`100,000円` instead of `100.000円`)
- added a few extra low-value arithmetic fragments (`12+12=`, `10+12=`, `3+7=?`)

Current practical read:

- Flash Lite OCR-only is promising because it is fast, free-tier friendly, and preserves reusable visual text on the chunks where visual text actually matters
- `minimal` and `high` are both usable here
- `minimal` may recover slightly more literal surface text, but it can also admit more low-value visual clutter

Better approach:

- keep OCR extraction relatively dumb
- keep filtering simple and local
- only strip obvious junk
- preserve a few useful local lines and keywords

### 10. Translation should be subtitle editing, not literal translation

The `translate_vtt_api.py` direction is correct:

- translate in local cue batches
- preserve cue count/timing
- target readable English subtitle CPS
- allow local redistribution of meaning across adjacent target cues

The partial `oni_no_dokkiri_de_namida_ep2` English results are already much better than literal cue-by-cue translation would be.

Examples:

- `何これ?` -> `What's this?`
- `なんだよこのナレ。 ひでえな。` -> `What's with this narration? That's awful.`
- `「あの車に乗れ」言ってたぞ。` -> `He said, "Get in the car."`

The translation step is now behaving like subtitle editing, which is what this repo needs.

### 11. Cues under 0.5 seconds are too short to hand off, even if technically correct

Real episode review on `great_escape_s01e01` exposed a failure mode that the earlier policy still allowed:

- the restored answer `蚊。` survived in Japanese reflow
- but it lived in a `0.16s` cue
- the first English rerun also placed `Mosquito.` in that same micro-cue
- the answer became easy to miss in actual viewing even though it was "present"

Operational conclusion:

- cues under `0.5s` should be treated as structural blockers for the reflow -> translation handoff
- this is stricter than the earlier "short cues are advisory" rule, and intentionally so
- plausible short reactions can still exist above that threshold, but sub-`0.5s` cues are too easy to miss to trust in final subtitle output

Pipeline consequence:

- `repair_vtt_codex.py` / `chigyusubs.reflow_repair` should mark sub-`0.5s` cues as `red`
- `translate_vtt_codex.py prepare` should stop on sub-`0.5s` source cues instead of letting English translation try to compensate later

The purpose is not metric purity. The purpose is to prevent important content from being stranded in cues that barely register on screen.

### 12. Line-level micro-cue repair has to rewrap display lines, not just count raw transcript lines

Validated on `great_escape_s01e03`.

Failure mode:

- line-level reflow already had a residual micro-cue merge pass
- but it rejected merges whenever the combined raw line count would exceed the display line limit
- that left sub-`0.5s` cues behind even when the merged cue could still be shown cleanly in two subtitle lines

Fix:

- `chigyusubs.reflow._merge_micro_cues()` now evaluates candidate merges through a display-text rewrap path
- merged cues can keep the original underlying raw-line list while rewriting visible cue text back down to the configured line limit

Result on `e03`:

- plain line-level reflow moved from `red` to `green`
- cue count dropped from `760` to `750`
- residual cues under `0.5s` dropped from `10` to `0`
- no negative durations and no overlaps

### 13. English draft reuse must be gated by exact cue timelines, not cue index

Validated on `great_escape_s01e04`.

Failure mode:

- an older Codex English draft existed for the episode
- the Japanese `*_ctc_words_reflow.vtt` was later regenerated with several local cue-boundary changes
- replaying the old English by cue number looked superficially plausible but drifted meaning into neighboring cues
- short answers like `ホッチキス。` and lines like `おしっこしたいんだけど。` then appeared at the wrong timestamps or vanished from the right cue

Operational rule:

- treat old English drafts as reusable only when cue count and cue timings match the current Japanese source exactly
- if even one cue boundary changed, do not seed by cue index

### 14. Full-episode faster-whisper is the standard always-on second opinion

Validated on `killah_kuts_s01e03` and `killah_kuts_s01e02`.

What worked:

- a full-episode `faster-whisper large-v3` pass recovered several real spoken lines that Gemini+CTC had missed or compressed
- the highest-signal misses were found by comparing the two transcripts as time-local coverage windows rather than by cue index
- glossary-assisted Whisper (via initial_prompt from episode glossary) improves name recognition in the coverage pass
- VAD cross-reference in the coverage comparison distinguishes real coverage gaps from Whisper hallucination in silence

What not to do:

- do not switch the whole maintained pipeline to faster-whisper by default
- do not auto-replace Gemini output wholesale from a second model

Current rule:

- keep Gemini + CTC as the main path
- always run faster-whisper as a second-opinion artifact — it is cheap (~2-3 min/episode) and catches both visible (visual-substitution) and invisible (silent omission) Gemini drops
- when a faster-whisper second-opinion artifact already exists, reuse it by default instead of rerunning Whisper; use `--rerun-whisper` only when a clean baseline is actually needed
- do not treat OCR-prompted or otherwise prompt-biased Whisper probe artifacts as the canonical second-opinion baseline; on `great_escape_s02e03`, replacing an OCR-prompted Whisper comparison with a fresh plain `large-v3` baseline reduced flagged coverage regions from `163` to `93`
- compare `*_ctc_words.json` against `*_faster_*_words.json` with a deterministic coverage-diff pass before reflow or source patching
- when `*_gemini_raw.json` exists, classify the remaining gaps against raw Gemini spoken lines and visual `[画面: ...]` lines so review can separate:
  - `visual_substituted_narration`
  - `missing_narration_high_confidence`
  - `compressed_vs_missing_unclear`
- when VAD segments are available, the coverage diff marks each flagged region as `vad_confirmed` or `possible_hallucination`

This preserves the main quality path while giving review a concrete way to find missing speech without rerunning Gemini.

The second opinion was originally gated on `possible_visual_narration_substitution` from alignment diagnostics. This was changed to always-on after lesson 15 showed Gemini can silently omit speech with no signal at all.
- any safe reuse path should validate exact cue-timeline equality before importing draft translations

### 17. Raw chunk QA has to happen before alignment/reflow, not after subtitles already look wrong

Validated on `great_escape_s02e03` and `great_escape_s02e04`.

Failure mode:

- on `s02e03`, a bad classroom raw chunk made it all the way into CTC and reflow
- CTC then anchored the wrong text plausibly enough that the failure first showed up as "subtitles appear too early"
- by that point the real problem was upstream transcript suitability, not reflow itself

Operational rule:

- run a deterministic raw chunk sanity gate on saved `*_gemini_raw.json` before alignment/reflow
- treat chunk-level `red` issues as upstream repair blockers, not translation-time cleanup
- keep `yellow` as review targets, not automatic stop conditions

Current `red` conditions:

- thought / meta transcription leakage
- spoken chunks that lost all `-- ` turn markers
- visual-only substitution
- pathological repetition loops

Pipeline consequence:

- `scripts/check_raw_chunk_sanity.py` is now the maintained pre-alignment gate for raw transcript artifacts
- CTC remains the default aligner, but it should not be asked to rescue obviously unsuitable source chunks

### 19. Raw chunk QA also has to live inside the transcription loop when rolling context or model rollover is in play

Validated while tightening the maintained Gemini video path after `great_escape`
`s02e03` and `s02e04`.

Failure mode:

- a bad chunk discovered only before reflow is already too late for
  `rolling_context_chunks=1`
- the next chunk may inherit poisoned context, and later `2.5-flash -> 3-flash`
  rollover can continue on the same damaged raw lineage

Operational rule:

- run the same deterministic red-chunk checks immediately after each chunk
  response on the maintained video path
- give a red chunk one no-context retry at the retry temperature
- if it is still red, stop resumably and repair/split it before continuing
- on resume, refuse to continue from a saved raw lineage that already contains
  red chunks

Pipeline consequence:

- `scripts/transcribe_gemini_video.py` now reuses the shared raw chunk sanity
  rules in-run, not just before alignment/reflow
- rollover only continues on a lineage that has already cleared the red-chunk
  gate

### 18. `90s` semantic chunks are the better reviewed default, and `2.5-flash -> 3-flash` rollover is operationally clean

Validated on `great_escape_s02e04`.

What worked:

- the reviewed `90s` semantic plan was materially easier to transcribe, inspect, and translate than the older `180s` path
- `gemini-2.5-flash` was good enough to be the main free production pass
- when `2.5-flash` RPD ran out, continuing the same lineage with `gemini-3-flash-preview` worked cleanly
- full chunkwise Flash-Lite OCR sidecar remained useful as translation support without contaminating the spoken transcript path

Operational rule:

- when semantic review is worth the operator time, prefer `90s` target chunks with the default `120s` hard max
- for free-tier production, run `2.5-flash` first and use `3-flash` only as overflow/backfill on the same chunk plan
- do not ladder the whole episode through multiple Flash tiers unless the user explicitly wants a comparison run

Pipeline consequence:

- `translate_vtt_codex.py prepare --seed-from ...` now errors out unless the candidate draft matches the current Japanese cue timeline exactly
- manual cue-index replay should be treated as unsafe and avoided

Observed `e03` examples before the fix:

- `知らない。 / え?` followed by `つる、つる、 / つる本手。`
- `やったあ! / やったあ!` followed by `イエス! / よっしゃあ!`
- `おお。 / おお。` during the Tom Brown escape burst

### 14. Codex batch apply must validate source cue identity, not just cue IDs

Validated on `killah_kuts_s01e02`.

Failure mode:

- a Codex translation batch was written with the correct `cue_id`s
- but the English lines drifted onto the wrong source cues partway through the batch
- `apply-batch` accepted it because it only checked cue ID coverage/order
- the final English VTT then looked like a timing problem even though the cue timestamps themselves were unchanged

Operational rule:

- cue IDs alone are not enough to validate a translation batch
- each translated item should also prove which source cue text it belongs to
- a lightweight per-cue source-text hash is enough for this

Pipeline consequence:

- `translate_vtt_codex.py next-batch` now emits `source_text_hash` for each cue payload
- `translate_vtt_codex.py apply-batch` now rejects items that omit `source_text` / `source_text_hash`, or whose source signature does not match the current batch

This catches batch-local semantic drift immediately instead of letting it masquerade as subtitle-timing drift later.

### 15. Video-only Gemini can compress or skip narrated premise VO even when nearby cards are correct

Validated on `killah_kuts_s01e02`.

Failure mode:

- an early narrated premise block around `01:50-02:18` was clearly present in audio
- Gemini preserved part of the surrounding explanation (`グッドアイデア`, `麻酔ってなった`)
- but it did not preserve the full spoken setup as spoken lines
- instead, some of that setup survived only as nearby `[画面: ...]` cards like `被害者 刑事` and `設定 診察で病院を訪れていた被害者`
- CTC then aligned the shortened Gemini text correctly, so the downstream artifact looked coherent even though the spoken transcript was incomplete

Operational lesson:

- this is not an alignment failure; it is an upstream transcript-completeness failure
- CTC diagnostics cannot detect missing speech that never appears in the source transcript
- manual review is still required for premise VO, rules narration, and other off-camera explanation blocks

Recommended workaround:

- keep Gemini + CTC as the default path
- when review finds a suspicious local miss, extract only that short clip
- run `faster-whisper large-v3` locally on the clip as a second-opinion artifact
- patch the affected Japanese cues from that local result instead of rerunning Gemini for the whole episode
- then regenerate only downstream artifacts that depend on that region

Why this is the right compromise:

- it keeps the main artifact chain stable
- it is much cheaper than rerunning Gemini
- it gives a saved local comparison artifact for the exact repaired window
- it avoids overreacting to a local Gemini miss by replacing the whole transcription path

### 16. Gemini speaker turns are useful as downstream metadata, but not as visible subtitle text

Validated on the maintained Gemini -> CTC -> reflow -> Codex workflow.

Operational rule:

- preserve Gemini turn boundaries in the aligned words JSON as metadata
- do not restore literal `-- ` markers into Japanese or English VTT text
- surface turn-boundary awareness to reflow review and translation as advisory context only

Why:

- visible turn markers make subtitle text read like a transcript
- but losing turn metadata entirely makes it harder to notice cues that merge multiple rapid speaker turns
- the right compromise is cue-level awareness such as "this cue spans 2 source turns", not visible markup

## Episode-Specific Findings

### `great_escape1`

What worked:

- video-only Gemini with `[画面: ...]` lines captured useful mission/question text
- shorter video chunks were substantially better than the first long-chunk run
- alignment was mostly good before the bad repeated-line failure

Main failure:

- one repeated `見、見、見...` line near the end poisoned tail alignment

Lesson:

- chunk-local loop detection/retry is necessary
- one localized failure can still ruin whole-episode alignment if not isolated

### `oni_no_dokkiri_de_namida_ep2`

What worked:

- OCR quality was generally usable on large telops
- VAD-based video-only transcription worked well
- chunked alignment fixed the major tail-collapse issue
- translation output is already reasonably natural in English

Main caveats:

- persistent overlay text like `DMM TV` / `月額550円` still needs explicit stripping in OCR-based paths
- some visual captions are editorial and not spoken
- transcript fragmentation still leaks into later translation and subtitle output

## Problems Still Open

### 1. Translation checkpointing exists, and it was necessary

The translator now checkpoints per batch to:

- `<output>.checkpoint.json`

That solved the resumability problem.

What is still open:

- `gemini-2.5-pro` still hits intermittent `429 RESOURCE_EXHAUSTED` depending on region
- the diagnostics still need less noisy short-cue CPS review logic
- Gemini API cost accounting is now partially solved for the video-only path:
  - `count_tokens` can preflight exact multimodal input tokens for the real inline video request payload
  - streamed responses expose `usage_metadata`, so prompt/output/total token counts can be recorded after a real run
  - output tokens still cannot be known exactly before generation, only estimated
- Codex-interactive translation restarts should clear stale session/output/diagnostics history when intentionally restarted from a clean base

### 2. CPS validation is still too strict for very short cues

Some “hard CPS violations” are not real quality failures.

Examples:

- `What's this?`
- `A special.`

These only fail because the cue duration is tiny.

What should change:

- treat very short cue durations differently in translation diagnostics
- do not overcount these as meaningful readability failures

The same lesson applies to Japanese reflow review:

- short or tiny cues are not automatically broken
- one-word answers, reactions, and interruptions may be legitimate even below `0.8s`
- reflow review should use short/tiny counts as advisory metrics, not automatic repair triggers
- repair should focus on structural timing defects and artifact-like boundary errors

### 3. Line-level reflow eliminated split-word artifacts

The worst translation artifact was mid-word splits across adjacent cues (`地` / `獄` → `He-` / `-ll.`). This was caused by character-level reflow splitting at inter-character gaps within words.

Line-level reflow (`reflow_words.py --line-level`) solved this by treating each Gemini transcript line as an atomic unit. Lines are never split mid-word.

Additional fixes in the line-level reflow:
- Comma-based fallback splitting for long lines without sentence-ending punctuation
- Sparse-cue clamping: shrinks cues where CTC spreads few characters across disproportionately long audio windows (e.g., `なんなの!` spanning 10.8s → clamped to 7.0s)
- Boundary expansion now caps pre-speech lead to `0.2s` so short reactions do not appear conspicuously early and blunt comedic timing

Results after line-level reflow:
- ge1: 0 overlong cues (was 9 with character-level), max 7.0s, 0 >20 CPS
- oni_no_dokkiri_de_namida_ep2: 0 overlong cues (was 21), max 7.0s, 1 >20 CPS (real fast exchange)

The cap matters because the previous readability-driven expansion could pull some cue starts well into silence before the first spoken line. On real episodes this reached `1.28s`, which looked sloppy even when the CTC alignment itself was accurate. The maintained default now preserves a small anticipatory lead for readability while preventing obviously early subtitle pop-in.

The LLM-based `repair_vtt_local.py` is no longer needed in the default path. It solved symptoms (split words) that line-level reflow prevents at the source.

### 4. OCR context selection is still unresolved for the Gemini path

We learned that:

- names alone are not enough
- whole dense dumps are too much

What remains unclear is the best simple middle ground.

Most likely direction:

- local `line_hints`
- local `keyword_hints`
- narrow time locality
- minimal role classification

This still needs a concrete simplification pass before it should be treated as the stable OCR-assisted default.

### 5. Sequential retry loops waste time — concurrent chunk-first is better

The original sequential transcription loop spent too much time retrying one bad chunk before knowing the episode-wide failure pattern. On episodes with fragile models or quota limits, this meant:

- minutes spent in backoff loops on a single chunk while other chunks sat idle
- quota used up on retries instead of getting a first pass across all chunks
- no visibility into whether a failure was chunk-local or episode-wide until very late

The concurrent redesign (March 2026) fixes this:

- 5 concurrent workers by default, RPM-limited to 5 req/min (free-tier ceiling)
- per-chunk results saved to individual files for concurrent-safe writes and trivial resume
- tight retry policy: one attempt + one retry per chunk per model (not 8 internal retries)
- transient errors (network, 500) get one same-settings retry
- quality errors (loop, red QA, timeout) get one retry with temperature bump to `0.1`
- quota errors trigger immediate model fallback, not retry
- the full first pass completes across all chunks before spending more effort
- rolling context is disabled in concurrent mode (chunks run out of order)

Tradeoff: losing rolling context means slightly less continuity between adjacent chunks. In practice, rolling context was already set to 0 on fragile models and the quality impact was minimal. Robust chunk completion across the full episode matters more than inter-chunk continuity hints.

The old retry policy still exists in `transcribe_gemini.py` for `max_retries` in the low-level API call, but the orchestrator now passes `max_retries=1` so the outer policy controls everything.

Previous retry-focused findings (still relevant for understanding the behavior):

New failure mode from `great_escape_s02e01` on `gemini-3.1-flash-lite-preview`:

- some video-only chunks returned thousands of short reaction lines like `-- おお。` or `-- うわあ。`
- the current loop detector missed these because the lines were short, not abnormally long, and each line was individually plausible
- shrinking VAD-derived chunks from ~240s to ~180s helped, but did not eliminate the problem
- setting `rolling_context_chunks=0` materially improved stability and stopped repeated deadline failures on the next chunk
- two chunks still required targeted single-chunk repair with a stronger anti-repeat retry prompt and a higher temperature

Operational consequence:

- loop detection must count dominant repeated short lines, not just repeated clauses inside one line or overlong line lengths
- for `gemini-3.1-flash-lite-preview` video-only runs, defaulting to no rolling context is currently safer than carrying prior chunk text

This should become standard for video-only Gemini transcription.

### 6. `gemini-3-flash-preview` looks usable; `flash-lite` still looks like a budget fallback

Validated on `great_escape_s02e01`.

Comparison against `gemini-3.1-flash-lite-preview`:

- `gemini-3-flash-preview` finished the episode without catastrophic looped chunks
- most chunks completed on the first attempt; one chunk had a fast `500` retry and one chunk needed three attempts due to read timeouts
- saved chunk outputs stayed in a sane line-count range and did not require the manual loop repairs that `flash-lite` needed
- transcript fidelity was still imperfect: likely `オーシャンビューよ` was misheard as `大三廟`
- alignment on the `gemini-3-flash-preview` run also surfaced one visual-substitution-risk chunk, so it is still not a literal-truth model

Operational conclusion:

- `gemini-3-flash-preview` is currently good enough as the primary cloud transcription draft when paired with:
  - ~180s VAD-derived chunk boundaries
  - `rolling_context_chunks=0`
  - CTC alignment
  - faster-whisper second-opinion coverage check
- `gemini-3.1-flash-lite-preview` remains useful only as a cheaper fallback path, not the preferred model
- based on this run alone, stepping up to a paid Pro model is not yet justified; first validate `gemini-3-flash-preview` on a few more real episodes and measure whether the remaining errors are mostly local lexical misses or true spoken-coverage failures

### 6a. `flashlite_debug_transcript` is good for structural smoke tests, not transcript-quality evaluation

Validated on `great_escape_s02e03` with a one-chunk smoke test against the repaired semantic chunk plan.

Observed behavior:

- the cheap debug preset completed a `166s` chunk cleanly and produced a structurally sane raw artifact
- the output was good enough to confirm that chunk encoding, request plumbing, checkpoint writing, and retry behavior were working
- lexical quality was still clearly weaker than the stronger Pro-path probes on the same episode, with compressed or drifted lines that would be misleading if treated as a quality benchmark

Operational conclusion:

- keep `flashlite_debug_transcript` for one-chunk smoke tests, cheap chunk-shape probes, and regression checks after code changes
- do not use it to judge the best transcript quality path on hard episodes
- when a chunk passes structurally under the debug preset, escalate to a stronger model before drawing quality conclusions

### 7. Translation is now local-batch aware, but subtitle polishing is not finished

The English translation quality is already promising, but still incomplete:

- awkward source fragmentation still leaks through
- some lines are slightly too written or too literal
- diagnostics need one more pass to distinguish:
  - true bad subtitles
  - acceptable short-cue CPS spikes

### 8. Codex-interactive translation has a practical batch ceiling

When Codex itself did the subtitle translation interactively, quality degraded gradually rather than catastrophically:

- `60` cues: solid
- `84` cues: still held quality
- `120` cues: still usable, but noticeably flatter and weaker on puns/riddles

Current practical default for the Codex-interactive path:

- default batch tier: `84`
- automatic fallback tiers: `60`, then `48`
- one episode at a time
- stop on structural blockers or red-quality batches

This is now embodied in the maintained `translate_vtt_codex.py` workflow instead of relying on ad hoc chat-only checkpoint edits.

Follow-up correction:

- minimum-tier `yellow` batches should remain resumable when the only issue is hard CPS pressure
- short-cue CPS warnings are still too noisy to justify auto-stopping a full Codex-interactive episode run on their own
- in the Codex-interactive translation helper, `yellow` should be a continue-with-warning state; only structural errors or an explicit `red` should stop the run
- deterministic diagnostics are still useful, but they need a compact rollup so QA can distinguish:
  - warning-only batches
  - real structural blockers
  - obsolete history from a superseded session

### 9. Codex-interactive reflow repair should be the default skill fallback, not local LLM repair

After running the Codex repair path on weak reflowed VTTs:

- line-level deterministic reflow remains the right default producer
- Codex-interactive `repair_vtt_codex.py` is the right fallback for structurally valid but translation-hostile files
- local LLM cue-repair remains useful only as an alternative benchmark path, not the default workflow

What turned out to matter most in practice:

- exact flagged cue-id ranges
- before/after counts for short cues and tiny cues
- one recommended Japanese VTT handoff path for translation

So the useful next step is better deterministic review diagnostics, not putting a local server back into the default skill path.

### 10. CTC wav2vec2 alignment outperformed both stable-ts and Qwen forced alignment

Three alignment approaches were benchmarked:

- `stable-ts` (Whisper cross-attention): 13.4% zero-duration words on `oni_no_dokkiri_de_namida_ep2`
- `Qwen/Qwen3-ForcedAligner-0.6B`: high zero-duration word count in smoke test, not obviously better than stable-ts
- `NTQAI/wav2vec2-large-japanese` CTC: 0.3% zero-duration words on `oni_no_dokkiri_de_namida_ep2`

CTC forced alignment is the clear winner. The Qwen and stable-ts aligners remain as archived benchmarks.

### 11. Semantic chunk review is workable on a real episode, but the candidate surface is still too noisy

The new `chunk-review` skill and `scripts/build_semantic_chunks.py` were tried on
`great_escape_s02e02` using:

- cached Silero VAD
- faster-whisper `large-v3` pre-pass
- semantic review of silence gaps

Observed behavior:

- the helper produced `139` candidate gaps for a `48.0 min` episode
- the pre-pass transcript was usable enough to judge many boundaries
- a first semantic pass accepted `11` split gaps and produced `12` chunks
- the resulting chunk durations were reasonable: `181.4s` min, `237.5s` avg, `276.4s` max

Comparison with the existing acoustic chunking:

- existing `vad_chunks.json`: `12` chunks, `191.1s` min, `238.2s` avg, `282.8s` max
- semantic first pass: also `12` chunks, but with a few different boundary choices

Main lesson:

- the idea is viable, but `139` review candidates is too many for a literal
  one-by-one manual pass on a normal episode
- the helper needs either:
  - tighter candidate generation
  - ranking/prioritization of likely good boundaries
  - or a guided first-pass heuristic before Codex reviews edge cases

So semantic chunking is promising, but not yet a drop-in default replacement for
`build_vad_chunks.py`.

### 12. Whisper transcript density is a better secondary chunk budget than duration alone

The semantic chunking helper was then extended to track a faster-whisper
character budget in addition to wall-clock duration:

- target chars: derived from pre-pass transcript density at the target chunk duration
- max chars: `1.2x` the target transcript budget
- `approaching_max` now becomes true when either duration or transcript density
  is near its configured ceiling

On `great_escape_s02e02`, a density-aware first pass produced:

- `12` chunks
- durations: min `187.1s`, avg `237.9s`, max `268.4s`
- transcript chars: min `971`, avg `1325.8`, max `1603`

Compared with the existing acoustic chunks:

- old durations: min `191.1s`, avg `238.2s`, max `282.8s`
- old transcript chars: min `846`, avg `1356.9`, max `1857`

Main lesson:

- duration alone hides very dense chunks that are harder to transcribe
- the whisper pre-pass character budget made chunk density more even on a real episode
- the late dense chunks in `great_escape_s02e02` were pulled earlier and the
  worst transcript-heavy chunk dropped from `1857` chars to `1603`

So the best direction for semantic chunk review is:

- keep duration as the hard ceiling
- use faster-whisper transcript density as the secondary split budget
- avoid treating all `240s` chunks as equivalent workloads

### 12b. The semantic chunk-review helper needs real-session validation, not just single-step testing

On `great_escape_s02e03`, a full `chunk-review` pass with `--target-chunk-s 180`
hit a real helper bug:

- `next-candidate` crashed after the second accepted split because it tried to
  `sorted()` accepted candidate dicts without a sort key

This is now fixed by sorting accepted candidates by `candidate_id`.

The same reviewed pass produced:

- `16` chunks
- durations: min `81.5s`, avg `152.6s`, max `232.1s`
- pre-pass transcript chars per chunk: min `416`, avg `640.8`, max `883`

Main lesson:

- long interactive helpers need resume-path testing after multiple applied
  decisions, not just `prepare` and the first couple of candidates
- a `180s` target plus transcript-density balancing can legitimately produce
  some `80-120s` chunks on dense comedy/problem-solving sections, and that is
  preferable to carrying `>900`-char transcript-heavy chunks forward into
  Gemini transcription
  rather than trying to keep every chunk near the wall-clock target

Another `great_escape_s02e03` failure exposed a separate persistence bug in
`scripts/transcribe_gemini_video.py`:

- successful chunks were only appended and written when `retry_info` existed
- a normal all-success run could therefore write `.meta.json` and
  `preferred.json` but leave the raw transcript JSON missing

This is now fixed so every completed chunk is checkpointed to disk, regardless
of whether loop-retry metadata exists.

Follow-up hardening from the same episode:

- `scripts/transcribe_gemini_video.py` now supports `--stop-after-chunks 1`
  for cheap smoke tests after code/model changes
- repeated timeout-heavy or rate-limit-heavy chunks now stop earlier instead of
  burning many retries on the same failing request
- suspiciously oversized chunk outputs now trigger retry/stop behavior rather
  than being silently accepted into the raw artifact

### 13. A repaired semantic-density raw transcript aligned cleanly on a real episode

On `great_escape_s02e02`, a mixed-model repaired raw transcript was assembled from:

- mostly `gemini-3-flash-preview`
- targeted repair chunks from `gemini-2.5-flash`
- density-aware semantic chunking

CTC alignment result on the merged raw JSON:

- `15` chunks
- `1567` segments
- `11987` words
- `0` zero-duration segments
- `0` zero-duration words
- `6` interpolated all-unaligned segments across `3` chunks
- `0` visual-only narration risk chunks

Main lesson:

- the mixed-model repair strategy is alignment-safe
- replacing only the unstable chunks is a viable recovery path
- the repaired semantic-density raw artifact stayed structurally clean enough to
  continue into reflow/translation without special alignment handling

### 14. Flash Lite chunkwise OCR is useful as a separate structured sidecar, not as inline transcript context by default

The best current role for `gemini-3.1-flash-lite-preview` is no longer primary
transcription. It is:

- chunkwise video OCR
- reusable side artifact
- free-tier abundant
- especially useful on dense rule/prompt chunks

Implementation direction:

- keep OCR separate from the main spoken transcript call
- store chunk-scoped JSON items rather than only raw `[画面: ...]` lines
- use a tiny classifier only:
  - `title_card`
  - `name_card`
  - `info_card`
  - `label`
  - `other`
- keep timings chunk-scoped (`timing_basis=chunk_span`) instead of inventing
  fake precise timings

So the current maintained OCR-like path is:

- `scripts/extract_gemini_chunk_ocr.py`
- usually `gemini-3.1-flash-lite-preview`
- `video`
- `media_resolution=high`
- separate artifact for review/glossary/translation support
- not automatically injected into transcription prompts by default

Validated on `great_escape_s02e03` using the repaired semantic `180s` chunk
plan. The sidecar completed cleanly with:

- `20` chunks
- `107` extracted items
- `0` chunks with parse warnings
- kind mix:
  - `2` `title_card`
  - `16` `info_card`
  - `81` `label`
  - `3` `name_card`
  - `5` `other`

Two wiring bugs in `scripts/extract_gemini_chunk_ocr.py` surfaced during that
run:

- `run_chunk_ocr()` did not accept the `preset_name` that `main()` passed
- `run_chunk_ocr()` then failed to pass `preset_name` down into
  `extract_ocr_chunk_result()`

Both are now fixed. This matters because the maintained OCR path is supposed to
be driven by presets such as `flashlite_ocr_sidecar`, and without that
plumbing the preset-backed CLI path failed before sending any useful requests.

Downstream use should stay conservative:

- glossary/context building should consume the sidecar when present
- translation should consume filtered local OCR context when present
- OCR should remain supporting evidence, not a replacement for spoken transcript review

### 15. Deterministic line-level reflow must preserve internal turn markers when rescuing micro-cues

On `great_escape_s02e01`, the first line-level reflow from the `gemini-3-flash`
CTC words artifact produced `65` cues under `0.5s`, which made the file a hard
`red` blocker for the maintained repair workflow.

The important fix was not raising `--min-cue-s`. That did not help. The actual
problem was:

- residual micro-cue rescue refused to merge across turn boundaries
- merged cues only rendered a visible turn marker for the first source turn

That left unreadable one-word or reaction-only cues like `- ん。` stranded at
`0.04-0.24s`, even though merging them into an adjacent cue would have been safe.

The maintained deterministic fix is:

- preserve visible turn markers for internal turn starts inside merged cues
- allow the hard micro-cue rescue pass to merge across turn boundaries

This improved the same `e01` reflow from:

- `1112` cues
- `65` cues under `0.5s`
- `33` cues over `20 CPS`

to:

- `1048` cues
- `0` cues under `0.5s`
- `2` cues over `20 CPS`

and downgraded the file from `red` to `yellow` for Codex repair.

## Current Best Practical Defaults

### For transcription experiments

Use video-only Gemini as a real baseline:

- Silero VAD
- VAD chunk boundaries
- Gemini video-only transcription
- prefer `gemini-3-flash-preview` over `gemini-3.1-flash-lite-preview` when quota allows
- use ~180s VAD-derived chunks for the current video-only path
- treat `target_chunk_s + 30s` as the default hard max for maintained chunk plans
- default `rolling_context_chunks=0` for the current Gemini video-only path
- spoken text as `-- ...`
- visual-only text as `[画面: ...]`
- strip `[画面: ...]` before alignment
- CTC forced alignment
- line-level reflow (`--line-level`)

### For OCR-assisted experiments

Keep OCR reusable and local:

- Qwen OCR JSONL
- OCR spans
- light local filtering/classification
- do not overfit the classifier to one show

### For alignment

- CTC forced alignment (`align_ctc.py`) is the new default
- uses `NTQAI/wav2vec2-large-japanese` + `torchaudio.functional.forced_align`
- runs on system python3.12 with ROCm GPU
- chunked: aligns per-chunk from Gemini raw transcription JSON
- keep per-chunk diagnostics
- preserve zero-duration or weakly aligned segment text for translation support
  (rare with CTC, but still relevant for mixed-script short answers, wake-word
  commands, and other weak-anchor lines)

### For translation

- batch-based subtitle editing
- preserve cue count/timing
- enforce CPS-aware diagnostics
- checkpoint after every batch

### Gap audit after chunking/transcription

- `great_escape_s02e01` exposed a real missing-dialogue failure from a chunk coverage gap in the main Gemini path
- the active raw transcript chunks covered `0.000-172.766s`, then resumed at `205.794s`, leaving `172.766-205.794s` uncovered
- this caused missing dialogue around `03:10-03:25` in both the Japanese reflow and published English VTT
- faster-whisper second-opinion artifacts made the gap obvious and provided enough local text to patch the affected cues
- after chunking/transcription, audit for long uncovered spans before alignment/reflow/publish
- if a long uncovered span contains Whisper speech, treat it as a real transcript gap, not just silence

Maintained fix direction:

- VAD should place chunk boundaries, not define coverage
- chunk plans for Gemini video work should cover the full episode continuously
- split at the midpoint of usable silence gaps instead of dropping the silence span
- legacy speech-bounded chunk JSONs should be rebuilt or rejected

### Sandbox no-progress hangs should fail fast

- on `great_escape_s02e07`, sandboxed Gemini video runs could sit for minutes with no first token, no chunk file writes, and no useful exception
- the same commands worked immediately once network access was escalated, which means the bad behavior was environmental, not chunk-specific
- maintained fix: each request attempt now has a first-token watchdog (`60s` default)
- if the initial worker wave all hits that `no_progress` watchdog before any chunk gets a real response, abort the whole run immediately with a clear blocked-network/sandbox hint
- probe runs (`--stop-after-chunks`) should not replace the main episode raw lineage in `preferred.json`
- resume should key off the requested output anchor, not only `preferred.json`, so debug probes do not strand the main lineage

### Dense aftershow talk can break a nominally sane semantic chunk plan without requiring a full episode restart

Validated on `great_escape_s00e01`.

What happened:

- a reviewed `90s` semantic plan finalized to `15` chunks with a superficially acceptable average duration, but it still needed `5` forced hard-cap splits
- the dense cast roundtable section produced several `120s` maxed-out chunks and one tiny orphan tail chunk
- Gemini still completed `14/15` chunks, so the real failure surface was one bad span, not the whole episode

What worked:

- do not throw away the whole raw lineage just because the chunk plan was ugly
- salvage the episode if the failures are localized and the finished chunks are structurally usable
- for `great_escape_s00e01`, a single missing `~60s` span was repaired cleanly with an AI Studio pack, then merged back into the saved raw lineage
- after that, the standard path still held: raw chunk sanity -> glossary -> CTC alignment -> line-level reflow -> Codex reflow repair -> translation

Operational takeaway:

- dense aftershow / panel-talk episodes should bias chunk review more aggressively toward transcript density, not just wall-clock duration
- a weak semantic chunk plan is still recoverable if only one or two chunks fail; prefer targeted AI Studio or repair-split recovery over restarting the episode
- line-level reflow from the salvaged raw transcript can still be translation-ready after a small Codex repair pass, so bad chunk aesthetics alone are not a reason to abandon the lineage
- if a chunk plan shows multiple forced hard-cap splits plus tiny orphan tails on a talk-heavy special, tighten the next plan before spending more quota

### If only one or two original production chunks fail, AI Studio should try the exact failed spans first before fallback subsplits

Validated on `great_escape_s00e02`.

What happened:

- the reviewed `90s` plan finalized to `6` chunks and needed `2` forced hard-cap splits
- the main `gemini-3-flash-preview` production run still saved `4/6` chunks successfully
- only the two middle production chunks failed, at `142.800-214.704` and `214.704-334.704`, both by timeout

What worked:

- keep the saved raw lineage and OCR sidecar; do not restart the episode
- build an AI Studio pack that includes the exact failed production spans as first-pass clips
- use smaller fallback sub-scenes only if the full failed span still compresses or fails in AI Studio
- for `great_escape_s00e02`, `gemini-3.1-pro-preview` handled both original failed spans directly, which let the repaired Japanese flow back into the main raw lineage without an extra repair transcription run
- once those two spans were promoted, the normal path held cleanly: raw sanity `green` -> glossary -> CTC alignment -> line-level reflow -> Codex translation -> publish

Operational takeaway:

- when the failure surface is already localized to one or two original chunks, prefer "exact failed span first" over immediately micro-splitting for AI Studio
- keep fallback subsplits ready, but treat them as fallback, not default
- preserving the original chunk boundaries in the repair transcript makes the final raw lineage easier to inspect against the failed production run
- this is especially useful when Flash timed out for runtime reasons rather than producing obviously broken text

## Next High-Value Work

1. Finish one full `oni_no_dokkiri_de_namida_ep2` translation run with checkpointing.
2. Make translation diagnostics less noisy on very short cues.
3. Decide whether Gemini video-only becomes the default cloud path.
4. Simplify OCR context selection into a stable local `line_hints + keyword_hints` model if OCR remains part of the Gemini path.
- lineage artifact naming is easier to manage when:
  - published outputs in `source/` keep stable source-video-basename names
  - intermediate artifacts in `transcription/` and draft `translation/` use short run-ID filenames
  - `preferred.json` manifests point to the current preferred lineage artifacts
- the readable timestamp-based ledger ID in `logs/runs/` is still worth keeping for audit/debugging
  - short run IDs should augment it, not replace it
