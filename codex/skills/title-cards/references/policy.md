# Title Cards Policy Reference

## Pipeline Position

```
gemini transcription → ... → reflow → [repair] → translation → [title-cards] → final
```

The title-cards stage runs *after* translation. It augments the translated EN
VTT with cues for full-screen, typically unnarrated title cards. CTC alignment
and translation can't see these because there's no speech to align — but the
viewer needs them to follow the show.

## Why Shot Boundaries

Variety-show audio is continuous (narration over reaction shots, speakers
talking through cuts), so cut boundaries are not a useful signal for *most*
subtitles. The exception is **standalone full-screen cards**: those are
typically introduced and ended by hard cuts, and they have **no speech**
covering them, so the shot bracket is the only timing signal available.

## Why Coarse OCR Is Enough

Flash Lite chunk OCR has `timing_basis: "chunk_span"` (~240 s windows). It
tells us *what cards exist somewhere in this chunk*, not exactly when. We
treat it as a **menu**: for each candidate shot, the menu contains the
plausible JA texts that might be on screen during that shot. You match the
shot to a menu entry by judgment (shot duration, position in chunk,
surrounding cues), not by frame-precise OCR.

## Inclusion Rate Target

| Show type        | Inclusion rate |
| ---------------- | -------------- |
| Talk show        | <10 %          |
| Game / battle    | 20–35 %        |
| Sketch / variety | 15–30 %        |

If you are including more than this, the policy is too permissive — re-read
the *Skip* heuristics in `SKILL.md`.

## Common Mistakes

- **Translating WINNER announcements** when the viewer just watched the win.
- **Adding location stamps** that are already obvious from the visuals.
- **Including every name plate** even when the host just introduced the
  guest verbally.
- **Translating decorative show logos** that recur every minute.
- **Picking the wrong menu entry** — the menu lists everything in a 240 s
  window, including overlay banners. Pick something whose duration and
  context match the shot.
- **Skipping the silent waiting beat** — opening legal / content cards
  and persistent name cards over silent reaction shots are *includable*.
  The skip rule is "lower-third over active speech," not "any lower-third."

## Edge Cases

- **Phone-call / waiting beats**: a few seconds with no dialogue while a
  caller is dialed; a name plate often identifies who's being called. This
  is an include — the dialogue gap *is* the signal that the card carries
  the only information.
- **Persistent name card across cuts**: the card stays on through 2-3
  consecutive shots while one new person is on screen. Treat the *first*
  card-shaped shot as the include and skip the duplicates.
- **Panel reveal (3+ names at once)**: skip. The list will be too long for
  a clean cue and the host will name them verbally within seconds.

## Wording Conventions

- Use square brackets `[ ... ]` so the viewer reads the cue as on-screen text,
  not dialogue.
- Sentence case unless the original is a proper noun.
- No trailing punctuation.
- Keep under one line where possible; never exceed two lines or ~42 chars
  per line.
- Match the show's translated voice as established in surrounding EN cues.

## Examples

| JA card                                  | Good EN                          | Bad EN                              |
| ---------------------------------------- | -------------------------------- | ----------------------------------- |
| `1回戦 第1試合`                          | `[Round 1, Match 1]`             | `[FIRST ROUND - FIRST MATCH!!]`     |
| `5分後`                                  | `[5 minutes later]`              | `[Five minutes later.]`             |
| `命名: スポーツスタンガン`               | `[Naming: Sports Stun-Gun]`     | `[The activity is named...]`        |
| `ルール: 1vs1でスタンガンを食らった方が負け` | `[Rule: First to be tased loses]` | `[1vs1, getting hit by a stun-gun = loss]` |
