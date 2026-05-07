---
name: title-cards
description: Insert title-card cues (segment intros, round markers, episode titles, location stamps) into a translated VTT. Cross-references TransNetV2 shot brackets with Flash Lite chunk OCR; you decide per candidate whether the shot is a card worth captioning. Less is more — when in doubt, skip.
---

# Title Cards

Augment a translated EN VTT with cues for full-screen title cards that the
audio-driven pipeline can't see (because they're typically unnarrated). Each
candidate is a shot bracket (precise timing) paired with the chunk's OCR menu
(plausible JA texts).

Use `scripts/title_cards_codex.py` as the maintained helper workflow:

- `prepare`
- `next-card`
- `apply-card`
- `status`
- `finalize`

## Prerequisites

Before using this skill the episode must have:

- A translated EN VTT (`samples/episodes/<slug>/translation/<run>_en.vtt`)
- The JA reflow VTT it was translated from
- TransNetV2 shot detection cached:
  ```bash
  python3.12 scripts/detect_shots.py --video samples/episodes/<slug>/source/<file>.mp4
  ```
- Flash Lite chunk OCR (default OCR backend; `<stem>_flash_lite_chunk_ocr.json`)

## Core Rule: Less Is More

The base subtitle is already complete. We only add a card when **a viewer would
genuinely lose information without it.** Most candidates should be skipped.

Target inclusion rate: roughly **20–40 % of candidates** on a card-heavy
variety show, **single-digit** on a talk show. If your inclusion rate is
higher, you are probably accepting decorative or redundant cards.

## Decision Heuristics

**Include** when the shot is clearly:

- A **segment intro** that names what's happening: 「1回戦 第1試合」, 「第2ラウンド」, 「最終決戦」
- An **episode title / segment title**: 「スポーツスタンガン」, 「命名:〇〇」, 「ドッキリ大成功」
- A **location / time stamp** that orients the viewer: 「5分後」, 「楽屋にて」
- An **on-screen rule** that pays off later in the segment: 「ルール: 1vs1でスタンガンを食らった方が負け」
- An introduction **name plate for a guest** who isn't named verbally in the
  surrounding dialogue.
- A **persistent name card** that stays on screen across multiple shot cuts
  while a newly-appearing person is shown — even if the OCR menu calls it a
  lower-third / `name_card`. The signal is "the same JA name text spans the
  shot bracket cleanly and the surrounding dialogue never says that name."
  One person at a time only — if 3+ name cards are simultaneously on screen
  (panel reveal), skip the whole batch; the host will name them verbally.
- A **silent opening legal / content card** at the very start of the episode
  (network disclaimer, content warning, sponsorship notice) that runs over
  no dialogue. These give context the viewer would otherwise stare at
  untranslated.

**Skip** when:

- The card is **branding** that recurs: show logos like `KILLAH KUTS` between
  every match, network bumpers.
- The card is **redundant** with adjacent EN dialogue (the host just announced
  the round number — no card needed).
- The card is a **lower-third banner over active dialogue** (`dialog_overlap`
  is non-zero, host is mid-sentence). Persistent name cards over silent
  reaction shots are the include case above — the distinguishing signal is
  whether speech is covering the same window.
- The card is a **WINNER announcement** that's already obvious from the
  preceding action (`WINNER みなみかわ` after we just watched him win — skip).
- The shot duration is borderline (<1 s) — likely a flash, not a card.
- **Anything you're unsure about.** Skip is the safe default.

## Wording

- **Bracketed, terse, no trailing punctuation.**
  - `[Round 1, Match 1]`
  - `[5 minutes later]`
  - `[Spot the Difference]`
  - `[Rule: First to be tased loses]`
- **Sentence case** for proper nouns; otherwise lowercase except the first
  word.
- **Match Netflix-ish brevity** — under 42 chars / line, one or two lines max.
- Don't translate decorative text literally. If the JA card is `スポーツスタンガン`
  shown as a title, `[Sport Tasing]` or `[Sports Stun-Gun]` is fine; pick the
  EN that fits the show's voice as established in `en_context_before`.

## Workflow

### 1. Prepare

```bash
python3.12 scripts/title_cards_codex.py prepare \\
  --en-vtt samples/episodes/<slug>/translation/<run>_en.vtt \\
  --reflow samples/episodes/<slug>/transcription/<run>_reflow.vtt \\
  --shots samples/episodes/<slug>/shots/<stem>.shots.json \\
  --chunk-ocr samples/episodes/<slug>/ocr/<stem>_flash_lite_chunk_ocr.json
```

Builds the candidate list and writes a session checkpoint. Reports total
candidate count and the menu kind distribution.

### 2. Candidate Loop

**Loop until all candidates are reviewed.** Don't wait for user confirmation
between candidates.

1. Run `next-card` to get the current candidate's payload.
2. Read `shot_start`/`shot_end`/`shot_duration`, `dialog_overlap`, the
   `ocr_menu`, and the surrounding JA + EN context.
3. Apply the decision rules above.
4. Write a decision JSON and run `apply-card`.
5. Loop back to step 1.

### 3. Finalize

```bash
python3.12 scripts/title_cards_codex.py finalize --session <session.json>
```

Splices approved cards into the EN VTT, writes `*_en_with_cards.vtt`,
updates `preferred.json`, and runs sanity checks (no overlap, no
microcues, sane added-card ceiling).

### 4. Verify

After `finalize`, read the saved VTT from disk and sanity-check that the
added cards are correctly timed and worded.

## Decision JSON Format

Include:
```json
{
  "candidate_id": 12,
  "action": "include",
  "picked_ja": "1回戦 第1試合",
  "text": "[Round 1, Match 1]",
  "reason": "Segment intro card, no narration covers it"
}
```

Skip:
```json
{
  "candidate_id": 13,
  "action": "skip",
  "reason": "Recurring KILLAH KUTS branding, no information gained"
}
```

## What Not To Do

- Do not include redundant or decorative cards just because OCR detected them.
- Do not write more than 2 lines or exceed ~42 chars / line.
- Do not translate verbatim if the JA is stylized — pick something that reads
  as a subtitle.
- Do not skip the loop early. Process every pending candidate.
- Do not edit the session JSON directly — use the helper commands.
