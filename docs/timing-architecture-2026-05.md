# Subtitle Timing Architecture — 2026-05

Design doc covering the CTC-drift problem, research findings, and the staged
plan for fixing it. Captures decisions made during the san-nomi investigation
so future work can pick up without re-deriving the tradeoffs.

Companion to `docs/ctc-weak-anchor-failure-modes.md` (a related but distinct
class of CTC failures).

## The Problem

`san-nomi_ep1_NNLE1SciaQE` cue 0 ("おお、") shows at `00:09.424` but the actual
speech starts ~22s in. 12 seconds of drift before any audio. The CTC
forced-aligner placed individual characters across the silent intro:

```
お    @  9.424
お、  @ 13.306   ← 3.84s gap
よ    @ 14.246
し    @ 18.088   ← 3.84s gap
よ    @ 18.528
```

This isn't a one-off. It's the predictable failure mode whenever a segment
spans pre-roll silence, BGM-masked passages, or other low-confidence regions:
CTC must place every transcript token *somewhere*, so when the model's
posterior over blank collapses it sprinkles tokens evenly across the silence.

The same pattern shows up at lower magnitudes elsewhere in the episode (cue
17: 6.5s drift; cue 768: 1.1s drift). VAD cross-references confirm: drifted
cues land in regions Silero marks as silence.

## Research findings

Single research agent, three topic areas, ~30k tokens. Citations are working
URLs verified by the agent.

### Industry timing rules (Netflix as primary anchor)

- Pre-speech lead-in tolerance: **1–2 frames @24fps (≈ 40–80ms)**. Anything
  earlier is a violation.
  ([Netflix Subtitle Timing Guidelines](https://partnerhelp.netflixstudios.com/hc/en-us/articles/360051554394-Timed-Text-Style-Guide-Subtitle-Timing-Guidelines))
- Post-speech lead-out: ≥0.5s when no following cue.
- Min cue duration: **500ms (12 frames @24fps)**. Already matches reflow's
  `_HARD_MIN_CUE_S`.
- Inter-cue gap: exactly 2 frames OR ≥500ms — no in-between values. Closer
  than 2 frames must be closed; gaps in 3–11 frames must be reduced to 2.
- CPS — English: **20 CPS adult, 17 CPS children**.
  ([Netflix EN-USA TTSG](https://partnerhelp.netflixstudios.com/hc/en-us/articles/217350977-English-USA-Timed-Text-Style-Guide))
- CPS — Japanese: **4 CPS regular, 7 CPS SDH** (full-width = 1, half-width = 0.5).
  ([Netflix JP TTSG](https://partnerhelp.netflixstudios.com/hc/en-us/articles/215767517-Japanese-Timed-Text-Style-Guide))
- Line constraints: EN 42 chars × 2 lines max; JP 13 full-width × 2 lines (16 SDH).

The 12s drift in san-nomi is **150× the Netflix lead-in spec**.

### CTC forced-alignment drift in silence

- The "tokens distributed across silence" pathology is **not inherent to CTC**.
  Per the [torchaudio CTC tutorial](https://docs.pytorch.org/audio/stable/tutorials/ctc_forced_alignment_api_tutorial.html),
  the blank token is treated as "silence between words" — long silences should
  be absorbed by blank emissions. Drift only happens when the **model's
  posterior over blank collapses** (low confidence / domain mismatch). For
  variety shows: pre-roll music + ambient = exactly that domain-mismatch pocket.
- MMS uses an explicit `<star>` slot for missing/uncertain transcript regions
  ([Pratap 2023](https://arxiv.org/abs/2305.13516)) — non-target audio gets
  siphoned into junk class rather than into real word tokens.
- **WhisperX is the canonical repair pattern**: it gates alignment on
  pyannote VAD chunks — alignment is run **per VAD chunk, never across
  silences**. ([m-bain/whisperX](https://github.com/m-bain/whisperx))
- stable-ts `suppress_silence` (default on): post-hoc snap of token
  timestamps to non-speech boundaries via loudness diff or Silero VAD.
- Detection signals in the wild: **>0.5s inter-word gap is the consensus
  suspicion threshold**, plus VAD overlay, plus frame-level emission probability.

### Translate-then-segment vs segment-then-translate

- MuST-Cinema (Karakanta 2020,
  [LREC](https://aclanthology.org/2020.lrec-1.460/)) preserves subtitle break
  symbols inline so MT models can emit segmentation jointly with translation.
- "Is 42 the Answer…" (Karakanta IWSLT 2020,
  [aclanthology](https://aclanthology.org/2020.iwslt-1.26.pdf)) found joint
  translate+segment outperforms translate-then-segment cascade on chars-per-line
  conformity.
- IWSLT 2023 subtitling task ([iwslt.org](https://iwslt.org/2023/subtitling))
  defined the problem as joint generation of translation + segmentation under
  CPS / line / char constraints.
- FBK direct speech-to-subtitle ([2023.iwslt-1.11](https://aclanthology.org/2023.iwslt-1.11/))
  beats cascades on SubER.
- AppTek length-controlled MT ([2023.iwslt-1.22](https://aclanthology.org/2023.iwslt-1.22/))
  prepends a target-length token to fit translations into source-timed cues.
- Cross-cue context (pronouns, register) needs sentence-level scope —
  segmentation evaluated in isolation hides errors that surface only when
  segments are read sequentially.
  ([arXiv 2205.09360](https://arxiv.org/abs/2205.09360))

## Architectural decision

The literature says: don't decide segmentation in isolation, use sentence-level
context, length-control the MT to fit timing.

**Our setup is different from the lit's assumed setup:**

The Karakanta/AppTek/FBK work assumes a small MT model with limited context
window. The win from joint segmentation comes from coupling decisions those
small models would otherwise make independently. **We use Codex** — a frontier
LLM that already holds whole-document context, already takes anonymous turn
context, already understands target-CPS prompts. Most of the joint-decision
benefit is available *without* changing the architecture.

What actually hurts us isn't the translate-vs-segment ordering. It's the **1:1
cue contract** enforced by `_validate_seed_timeline` in
`scripts/translate_vtt_codex.py`: Japanese reflow decides cue boundaries,
English text is forced to fit them.

### Key insight (user, 2026-05-04)

> If we nail some sort of cue cluster with japanese timestamp artifact we can
> safely retime, reflow and actually use specific word timing for translation,
> doesn't need to be one shot. If codex can identify the punchline for example
> that would be another great benefit. The key is giving the model the timing
> info so it can be reconstructed in english, probably more important than
> perfect one-shot timing.

Reframe: the unit of timing isn't the cue, it's the **word cluster** (group
of words separated by short pauses, < ~0.5s gaps). Clusters are derived from
CTC word timestamps. If we package each translation batch with its cluster
artifact (per-word timings, cluster boundaries, total speech-active duration),
Codex has the raw signal to reconstruct timing in English — even if the
English cue count and boundaries differ from Japanese.

This makes the existing strengths of the LLM (sentence-level coherence, comedic
timing, punchline recognition) bear on segmentation, instead of fighting
against the Japanese-cue mold.

## Phased plan

### Phase 1 — Drift corrector (SHIPPED 2026-05-05)

Pre-pass on `ctc_words.json`. Clusters words by intra-word gap, validates
against Silero VAD, retimes the cue when it shows the gross-drift signature.
Implemented in `chigyusubs/ctc_drift.py`, CLI at `scripts/correct_ctc_drift.py`.

#### What it does
- Cluster words on `cluster_gap_s` (default 0.5s), VAD-bridge-aware so
  natural pauses inside speech don't split clusters.
- A segment is **suspect** iff its largest intra-word gap ≥ `min_intra_gap_s`
  (default 3.0s — see "Threshold story" below).
- Suspect + has any real cluster → trim cue span to the real-cluster region.
- Suspect + all-ghost → relocate near the nearest VAD onset, or pack against
  next segment if no anchor.
- **Never drops words.** Ghost-cluster word timestamps are redistributed across
  the new span by character weight.

| Param | Default | Source |
|---|---|---|
| `--min-intra-gap-s` | **3.0s** | empirical, see below |
| `--cluster-gap-s` | 0.5s | WhisperX/stable-ts consensus |
| `--max-pre-speech-lead-s` | 0.08s | Netflix 1–2 frames @24fps |
| `--min-cue-s` | 0.5s | Netflix min |
| `--target-jp-cps` | 4.0 | Netflix JP regular |
| `--ghost-vad-overlap-s` | 0.1s | min VAD overlap for "real" cluster |
| `--vad-bridge-s` | 0.05s | min VAD speech inside a gap to treat as natural pause |

#### Threshold story (why min_intra_gap_s = 3.0s)

Initial design fired on multiple-cluster-with-ghost OR single-cluster-no-VAD-onset.
On san-nomi that triggered 5 segments — but Whisper sanity-check + user listen
test revealed only 2 were real gross drift; the other 3 were noise:

| Seg | max gap | First-pass action | Listen test verdict |
|---|---|---|---|
| 0  | 3.842s | relocated -12s → -2s residual | ✅ real gross drift, fix correct |
| 17 | 5.724s | trimmed -6.3s | ✅ real gross drift, fix correct |
| 128 | 0.0s | relocated 80ms cue 360ms forward | ❌ pointless churn |
| 220 | 0.82s | trimmed 860ms (`何kg` retimed) | ❌ Gemini-correct text, sub-second drift fine |
| 753 | 0.4s | relocated +1s → +3s late | ❌ regression — pushed past actual speech |

The pattern: a 3s+ intra-word gap is the blank-collapse signature (CTC sprinkling
tokens across silence). Gaps under that scale are CTC's normal jitter and both
Silero and Whisper have their own ~1s timing errors there — fighting noise with
noise produces regressions. Tightening to gap ≥ 3.0s collapses 5 modifications
to 2, exactly the egregious cases.

Practical viewing argument (user, 2026-05-05):

> Subtitles lingering or being 1s early is really not a dealbreaker, considering
> our translation will have some freedom for cps fit anyway.

Sub-second timing residuals don't justify the regression risk. Gross drift does.

#### Why no Whisper integration (deferred)

Considered: cross-checking clusters against Whisper segments to recover from
Silero misses. Empirical Silero-vs-Whisper-vs-ground-truth on san-nomi:

- **Seg 753**: Silero had a 2.18s silence gap; Whisper placed `ズレ感があってことですよね`
  inside that gap (start 2059.76 ✓, end 2062.76 inflated by 1.7s).
  Resolution: this turned out to be **seg 752** correctly aligned at 2059.695-2060.876.
  Seg 753 is a separate Gemini-only cue (possibly a rephrase or a double-transcription)
  that the gross-drift threshold now leaves alone.
- **Seg 0**: Silero VAD onset at 22.146 was correct. The corrector's window
  (`search_hi = next_seg_start - 0.05`) excluded it because seg 1 starts before
  22.146. Result: relocation packed against next-seg start (~21.4) instead of
  using the real VAD onset (~22.1). Residual ~2s.

Both Silero and Whisper have ~1s timing errors. Adding Whisper-overlap as a
second realness signal would complicate the corrector without clearly improving
gross-drift behavior, and Phase 2 (cluster artifact for translation) is the more
valuable next investment.

#### Side fixes flagged but not landed
- `chigyusubs/reflow.py:_MAX_PRE_SPEECH_LEAD_S = 0.2` → tighten toward
  Netflix's 0.08s. Verify this doesn't regress short reaction cues.
- Audit `_TARGET_CPS = 14.0` in reflow.py — that's English-territory but
  reflow operates on Japanese text. Check whether it's driving Japanese cue
  expansion incorrectly.

#### Outputs
- `<input>_drift_corrected.json` — corrected ctc_words (standalone tool)
- `<output>.diagnostics.json` — per-seg actions, params, summary
- Run `scripts/reflow_words.py --line-level` against the corrected json to
  produce a parallel reflow VTT for A/B comparison.

#### Pipeline integration (SHIPPED 2026-05-06)

Drift correction is now folded into `scripts/align_ctc.py` as a post-pass and
runs on every alignment by default. Rationale:

- **Drift is a CTC failure mode.** wav2vec2 must place every transcript token
  somewhere on the audio; where the blank-token posterior collapses (pre-roll
  music, BGM-masked passages) tokens smear across silence. The fix conceptually
  belongs to the same step that produced the broken timestamps — splitting it
  into a separate pipeline phase advertises a problem the user shouldn't need
  to know about.
- **Gross-drift only, false-positive rate ~0.** With `min_intra_gap_s = 3.0s`,
  san-nomi triggers on exactly the 2 segments confirmed by listen-test (seg 0
  pre-roll, seg 17 BGM gap). Other 827/829 segments untouched. Always-on is
  safe at this threshold.
- **Clean phase count.** Pipeline stays at 6 phases. No new `preferred` keys,
  no branching consumer logic in reflow / second opinion.
- **Audit trail preserved.** When any modification fires, the pre-correction
  segments are saved as `*_ctc_words_raw.json` alongside the corrected output.
  Drift diagnostics fold into the existing `*.diagnostics.json` under a
  `"drift"` key (status, params, per-segment actions, summary). Metadata stats
  carry `drift_status`, `drift_segments_modified`, `drift_max_abs_shift_s`.
- **Opt-out path.** `--no-drift-correction` disables; `--vad-segments PATH`
  overrides VAD location; `--min-intra-gap-s` tunes the threshold. Graceful
  skip with a logged reason if VAD JSON is absent.
- **Standalone tool retained.** `scripts/correct_ctc_drift.py` still works for
  re-running correction with different params on cached CTC output without a
  full re-alignment.

#### Anomalies surfaced (separate from drift, Gemini-text concerns)
- san-nomi: `おお、よしよし` appears twice in the transcript (seg 0 at 19.4s
  after correction, plus an unrelated cue at 33.2s). Likely a Gemini
  double-transcription.
- san-nomi: seg 752 (`ズレ感があってことですね`) and seg 753
  (`ズレ感がってことですね` — missing は) at 2059.7s and 2062.2s. Either a
  real rephrase from a second speaker, or a Gemini hallucinated repeat.
- These are out of scope for the drift corrector; flag for a separate
  Gemini-transcript-dedup pass if they recur on other episodes.

### Phase 2 — Cluster artifact + post-translation retiming (NEXT)

Once the corrector is producing clean cluster timings, the cluster JSON
becomes a first-class artifact that travels with each translation batch:

```
cue_cluster_artifact = {
  "cue_id": 17,
  "japanese_text": "...",
  "clusters": [
    {"words": [...], "start": 64.13, "end": 64.67, "duration": 0.54},
    {"words": [...], "start": 65.63, "end": 68.25, "duration": 2.62},
  ],
  "total_speech_s": 3.16,
  "total_silence_s": 0.96,
  "japanese_cue_span": [64.13, 68.25],
}
```

`translate_vtt_codex.py` includes the cluster artifact in each batch's
context. Codex translates with timing awareness — it can:
- recognize a punchline cluster and weight English brevity accordingly
- propose splitting a Japanese cue into two English cues if a long pause
  separates two thoughts
- propose merging two short Japanese cues into one English cue when the
  English collapses naturally (e.g., short greeting exchanges)

A new post-translation retiming script consumes the (English text + cluster
artifact) and produces an English VTT with timing derived from clusters, not
from Japanese cue boundaries. Phase-2 retains text-level work in Codex but
relaxes the timing tie to Japanese cues.

### Phase 3 — Translate-then-segment branch (DEFER)

Only if Phase 2 underperforms: experimental branch where Codex translates the
whole turn without per-cue boundaries, then a segmenter re-flows English
against clusters end-to-end. A/B against Phase 2.

## Tradeoffs

- **Phase 2 breaks the 1:1 cue index assumption** held by some downstream
  tools — repair workflow, second-opinion diagnostics, alignment review. Need
  to either preserve mapping in metadata (`source_cue_ids: [17, 18]`) or
  migrate downstream tools to lookup-by-time-range.
- **Cluster identification depends on the corrector being correct.** If the
  corrector mis-clusters (e.g., merges two real utterances into one cluster
  because their gap was just under 0.5s), Codex inherits that error. Phase 1
  needs strong diagnostics so misclusters are detectable post-hoc.
- **Pre-speech lead = 80ms is aggressive.** Subtitlers sometimes give 100–200ms
  for "anticipation." Netflix is the strict spec; YouTube/casual viewing tolerates
  more. Worth validating with the user after Phase 1 lands.
- **Frontier-LLM dependency.** The cluster-artifact approach assumes Codex
  reasons well about timing budgets. If the LLM ignores the artifact (treats
  it as noise), Phase 2 degrades to current behavior. Mitigation: prompt-side
  unit tests on a few known cases.

## Open questions

- Does forced-alignment confidence (per-frame emission probability) give a
  cleaner drift signal than gap heuristics? torchaudio's `forced_align`
  returns scores per token — we don't currently use them.
- san-nomi intro `おお、よしよし` is real speech at 21.5-22.5s (user listen
  test 2026-05-05). The corrector lands it at 19.4-21.4 — ~2s early because
  of `next_seg_start` clamp. Could be improved by allowing the relocation
  window to extend past `next_seg_start` when adjacent segments share a VAD
  onset region; not urgent given practical-viewing tolerance.
- Inter-cue gap quantization (Netflix 2-frame OR ≥500ms rule) — should reflow
  enforce this, or only the publish step (`format_vtt_netflix.py`)? Currently
  neither does.
- Regression harness: tiny `drift_cases.json` pinning known cases (san-nomi
  seg 0/17 fire, seg 128/220/753 don't) so threshold tweaks have a fast
  feedback loop. Worth building once we have ≥3 episodes' worth of verified
  cases; until then, re-running on san-nomi + diffing diagnostics is
  sufficient.
