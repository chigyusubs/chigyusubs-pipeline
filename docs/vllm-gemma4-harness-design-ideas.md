# vLLM + Gemma 4 ASR harness — design ideas

Companion to `vllm-gemma4-harness-findings.md` (which is the running
results log). This doc is a brainstorm of where the harness could go
next. Each idea is roughly independent — pick by ROI, not order.

Tags:
- **S/M/L** = rough effort (S ≤ 1 h wire-up, M = an afternoon,
  L = multi-session).
- **value: high / mid / low / speculative** = expected payoff if it
  works, not probability of working.
- **status: open / done / partial** = whether some version of this is
  already in the harness.

Anchoring facts (from `vllm-gemma4-harness-findings.md` and the
`feedback_e4b_prime_budget.md` memory):

- **K_video_sysrole_oracle_names** is current bench winner on n=24
  (E4B bf16, 100% kata, 88% name, 0.445 LCS, ~0.9 s/clip).
- **P_audio_sysrole_minimal** (4-name cast) is the production default
  for full-episode runs via `transcribe_episode_e4b.py`.
- Two known hard ceilings on KK S01E01: `しんいち` audio-tower miss in
  some clips, `山根 → 山田` substitution.
- Eval gold = CTC-aligned transcript of one episode → leakage risk as
  configs get tuned to it.

---

## Quick triage

**Top picks (high value, ≤ M effort):**
- §1.1 OCR-derived vocab (not just names)
- §1.4 Ablate the domain block — does it earn its tokens?
- §2.1 Logit bias on cast-name tokens
- §2.4 Repetition-penalty safety net for catastrophic loops
- §4.1 Second-episode spec to detect overfit
- §4.4 Reaction-recall metric (E4B's claimed strength is currently unscored)
- §4.5 Precision alongside recall (hallucination rate)
- §6.1 Full-episode bench at K (currently we extrapolate from n=24)
- §7.1 Result diff CLI + canonical regression set
- §8.1 Whisper-augment fusion (already drafted in
  `project_whisper_gemma_augment.md`)

**Skip-for-now (likely low value or already converged):**
- More guided-JSON variants (§9.1)
- Higher `max_soft_tokens` (§9.2)
- More vision-only configs without oracle (§9.3)

---

## 1. Priming content & framing

### 1.1 OCR-derived vocab, not just names — `M`, value: high
Currently `_DEFAULT_VOCAB` in `configs.py` is a hand-curated list of
boxing/MMA terms. The OCR sweep already extracts on-screen text per
batch — the rule terms / section markers / show terms also live there.
Extend `build_per_clip_oracle.py` to emit `oracle_vocab` alongside
`oracle_names`, and add `system_template` slots for both. Hypothesis:
per-clip vocab grounding fixes near-homophone failures (e.g., the
`フェンシング → レフェリー` regression from session 1) without bloating
the static prime.

### 1.2 Negative prime / anti-list — `S`, value: mid speculative
For known confusion pairs (`山根` vs `山田`), tell the model:
"Names that sound similar but are NOT in this clip: 山田". Cheap to
test on the n=24 spec — add a fourth oracle source. May backfire if
the model treats absence-instructions as presence cues; worth a single
config to find out.

### 1.3 Few-shot (audio, transcript) anchor pairs — `M`, value: mid
Pick 2 short reference segments from the same episode (or show) where
the gold is known and the audio is clean. Inject them as a system-role
preamble: "Reference (audio + transcript): ... Now transcribe the
following clip:". Hypothesis: gives the audio tower a concrete prior
for show-specific reaction tokens. Risk: doubles audio tokens per
request and could leak phrases into outputs.

### 1.4 Ablate the domain block — `S`, value: high
The 3-sentence "Killah Kuts is a stun-gun cage show..." description has
never been ablated. Run D and K with that paragraph removed, prime
left as just vocab + cast. If LCS / kata / name don't move, drop it
and reclaim the tokens. If they do move, we know domain framing is
load-bearing — useful when porting to a new show.

### 1.5 Tag prime sections explicitly — `S`, value: low
Today the system prime is one block of prose. Try
`# Domain\n...\n# Vocabulary\n...\n# Cast\n...` so the model can
pattern-match section boundaries during attention. Cheap probe; if
it doesn't move numbers, drop.

### 1.6 Show-level static + episode-level dynamic — `M`, value: mid
Split the prime into two messages: a show-level system message
(domain, MC, recurring vocab) that's identical across the whole
season — large prefix-cache win — and an episode-level system message
(per-episode cast + episode-specific OCR oracle) that varies. Test
prefix-cache hit rate at scale (§6.2).

### 1.7 Prime-length sweep — `S`, value: mid
The current minimal prime is 4 names. We have intuition that ~17 was
too many and 0 was too few, but no curve. Run 0/2/4/8/12/17 cast sizes
on the n=24 spec; chart kata/name/LCS vs prime size. Locks in the
"how minimal is too minimal" boundary so future episode primes have a
defensible budget.

---

## 2. Decoder controls

### 2.1 Logit bias on cast-name tokens — `S`, value: high
vLLM supports `logit_bias` on the OpenAI completions endpoint. Bias
the kana-fighter token IDs (`ショーゴ`, `みちお`, `しんいち`, `ダイアン`)
positively. Hypothesis: a small bias (+1 to +3) lifts marginal-acoustic
hits without forcing recitation. Risk: the model spams them; mitigate
with very small magnitudes and a per-segment trigger (only bias names
in oracle_names). Add a config `Q_audio_sysrole_logit_bias_oracle`.

### 2.2 Negative logit bias on substitution traps — `S`, value: mid
Same mechanism as 2.1, but on the wrong-but-similar tokens we've
observed: `山田`, `南川`. Risk of overshooting (kanji `南川` is
sometimes the right output). Pair with §1.2.

### 2.3 Self-consistency: temp>0, N samples, majority — `M`, value: mid speculative
Instead of one greedy run per chunk, run N=5 at temp=0.5 and pick the
majority output by character n-gram overlap. Cost: 5x latency. Worth
testing on the 10-20% of chunks that flag with low confidence (low
cps, low cjk_ratio) rather than as default. Could double as a
confidence signal.

### 2.4 Repetition-penalty safety net — `S`, value: high
We've seen catastrophic repetition in whisper-large-v3 (chunk 78:
1916 chars of `ありがとうございます`). E4B hasn't shown the same yet,
but the failure mode is silent and sample-dependent. Add a
`repetition_penalty=1.05` and `no_repeat_ngram_size=12` to the runner
and verify the fix doesn't degrade kata/name on n=24. Cheap insurance.

### 2.5 Two-pass: draft → critique — `M`, value: mid speculative
Pass 1: standard transcribe call. Pass 2: feed the draft back to the
same model with the audio + oracle and ask "Are any of these names
wrong? Reply with corrected transcript or 'OK'." Latency cost ~2x.
Test only on chunks where draft includes a name (or any kanji that's
in the substitution-trap list). If P2 changes anything on >5% of
chunks, productionize.

### 2.6 Cascade: oracle-empty → vision fallback — `S`, value: mid
Today every chunk uses the same config. Empty `oracle_names` chunks
have no name-disambiguation help. For those specifically, fall back
to E_vision_sysrole_primed (which has the highest LCS with no-oracle
hints). Implement as a router in `run_bench.py` / the production
runner: `cfg = K if oracle_names else E`.

---

## 3. Multi-modal stacking

### 3.1 Dynamic frame inclusion — `S`, value: mid
Same lever as §2.6 but at the modality level. If `oracle_names` is
non-empty, audio-only path; if empty, audio + 1 chyron-detected frame.
Doesn't change the model; just changes the request shape per-chunk.
Cheap, no new config.

### 3.2 Pre-OCR'd telops as text parts — `M`, value: high
Today the OCR sweep produces strings, but the harness consumes only
the *names* derived from them. Inject the full OCR'd text blob for the
overlapping batch into the system prime as "Text visible on-screen
during this clip:". This is the cheapest possible vision substitute —
no pixels, just strings. Hypothesis: subsumes most of vision-grounding
value at zero pixel-token cost. Useful especially for OCR-heavy
sections (rule introductions, fighter cards).

### 3.3 Mixed text-OCR + pixel hybrid — `L`, value: speculative
For chunks with rich OCR + visible action (a fighter card on top of a
scoreboard), include both pre-OCR'd telop strings AND a single still
frame. Tests whether redundant text+pixel grounding helps. Risk: the
model double-cites the OCR text. Out of scope until §3.2 is done.

### 3.4 Frame-by-chyron-change selector — `M`, value: mid
Today vision configs sample at fixed fps. Instead, detect frames where
on-screen text *changed* (diff a binarized chyron crop frame-to-frame)
and pick those. Hypothesis: 1-2 chyron-change frames at mst=280 beat
8 fixed-cadence frames because the model isn't averaging across stale
text. Requires a small chyron-detector (manga-ocr binarized region
diff is enough).

---

## 4. Eval methodology gaps

### 4.1 Second-episode spec to detect overfit — `M`, value: high
All bench numbers are on KK S01E01. Configs are tuned to it; primes
are tuned to it. Build a second eval spec from a different episode
(KK S01E02 if CTC-aligned, otherwise a different show entirely).
Acceptance: best config on E01 should be within 5pp on E02; if it
isn't, we've been overfitting. Critically, mark the new spec
"calibration" and never tune to it.

### 4.2 TTS-from-gold ceiling — `M`, value: mid
Take the gold transcript, TTS it (any Japanese voice), run K against
it. Should hit ~100% on every metric. If it doesn't, we know the
model has a hard ceiling unrelated to acoustic conditions. Useful for
distinguishing "audio is hard" from "the model can't be primed
correctly". One-off run, ~30 min.

### 4.3 Disagreement eval (E4B vs whisper) — `S`, value: high
We have whisper-large-v3 outputs from `transcribe_episode_whisper.py`
and E4B outputs from `transcribe_episode_e4b.py` on the same packed
chunks. For every chunk where they disagree non-trivially (LCS < 0.7),
hand-rate which is right and which kind of error each made.
Categorize: substitution / drop / addition / repetition / OCR-grounded
correction. This is the actual error taxonomy we need to design
fusion configs (§8.1) against.

### 4.4 Reaction-recall metric — `M`, value: high
E4B's claimed advantage over whisper is "picks up reactions/
interjections". The current scorer measures only kata + names + LCS,
which doesn't reward reactions. Add a metric: count `そうそう / うん /
えー / マジで / うわー / おお` etc. tokens in gold vs output, report
recall. Without this, every config change is biased toward the
metrics we have.

### 4.5 Precision alongside recall — `M`, value: high
Today scoring is recall-only. The seg04 cast-recitation failure mode
is invisible to recall — it would *increase* name-recall while being
catastrophic. Add precision-of-cast-tokens ("of the cast names emitted
in output, what fraction were actually spoken according to gold?").
Probably needs gold name-set per segment (already in `seg["names"]`).

### 4.6 Adversarial regression set — `S`, value: high
Curate a set of ~10 segments that are known failure modes:
- seg07 `しんいち` (audio-tower ceiling)
- seg with `山根 → 山田` substitution
- seg08 from session-1 (`フェンシング` confusion)
- a chunk that triggered whisper repetition
- a vision-grounding-required chunk (chyron name only on screen)
Lock these down as `eval_specs/regressions.json`. Every new config
must pass all of them or be rejected.

### 4.7 Realism eval: drop name metadata — `M`, value: mid
Today specs include `seg["names"]` for scoring. The harness also has
access to it via `oracle_names` (when patched). Build a "realism"
spec where names are intentionally *wrong* (random shuffled) and see
how badly the harness degrades. If it doesn't degrade much, the
oracle isn't doing as much as we think; if it tanks, oracle quality
is critical and we need to bench the OCR pre-pass against gold.

### 4.8 Variance / p95 reporting — `S`, value: mid
Currently we report per-config aggregate kata/name/LCS. Add p50/p95
LCS and per-segment LCS variance. A config with 0.45 mean LCS and 0.05
variance is more shippable than 0.50 mean / 0.20 variance.

---

## 5. Failure-mode probes

### 5.1 Clip-duration sweep — `M`, value: mid
The full-episode runner uses VAD-packed 18s windows; the n=24 bench
uses 8s windows. We don't know whether duration helps or hurts E4B.
Run K at 4 / 6 / 8 / 12 / 18 s windows on the same content. Hypothesis:
shorter helps name recall (less competing utterance), longer helps
LCS (reactions context). If true, route by content type.

### 5.2 Cross-chunk context: previous transcript injection — `M`, value: mid speculative
Inject the previous chunk's transcript as a "context" line in the
system prime: "Previous chunk ended with: ...". Hypothesis: helps
mid-sentence chunks where E4B currently restarts. Risk: error
propagation — a chunk-N hallucination poisons N+1.

### 5.3 Phoneticize-first prompt — `S`, value: low speculative
Before transcribing, ask the model to emit raw kana/romaji
("Phonetics first, then commit:"). Hypothesis: forces audio-tower
output before LLM substitution kicks in, recovering kana fighter
names that get over-written by the LLM's kanji preference. Likely
brittle but cheap to test.

### 5.4 Audio-quality probe — `M`, value: low
Down-sample to 8 kHz, add white noise at -20 dB SNR, etc. Run K on
each variant. Tells us how much headroom there is in audio
preprocessing (denoising, vocal isolation) — feeds into a downstream
"should we add audio cleanup" decision.

### 5.5 Acoustic emphasis — `M`, value: speculative
Take the 5-10 chunks where `しんいち` was missed. Slow them down 10%
and isolate vocals (Demucs / UVR). Re-run K and see if ANY of them
recover. If yes, audio preprocessing is a real lever. If none recover,
the audio-tower ceiling is below the acoustic content — no amount of
prompt tuning will help.

---

## 6. Production gates

### 6.1 Full-episode K bench — `S`, value: high
We've shown K wins on n=24. We haven't actually run K on all 131
chunks of an episode. Wire it into `transcribe_episode_e4b.py` (or a
new runner) and produce the same artifacts as `_e4b_raw.json`.
Compare full-episode kata/name/LCS against P_audio_sysrole_minimal.
This is the actual "should we ship K" number.

### 6.2 Prefix-cache hit rate at n=131 — `S`, value: mid
With a stable system prime + per-clip oracle tail, what's the actual
cache-hit rate vLLM reports? Pull the metric from `/metrics`. Quantify
the throughput upside before committing. If prefix-cache wins are
small, the oracle-tail design is engineering overhead for no perf gain.

### 6.3 Latency at concurrency — `S`, value: mid
Today we measure single-stream wall time. Run 4 chunks in parallel
through the vLLM endpoint (it's a queue under the hood) and report
total throughput. If concurrency speedup is sub-linear, batching
gives diminishing returns and we should pipeline serially.

### 6.4 GPU-memory headroom check — `S`, value: low
At full episode + KV cache + concurrent requests, what's peak VRAM?
Currently 16 GiB weights + 3.4 GiB KV at 16k ctx leaves ~4 GiB on the
24 GB 7900 XTX. That's fine for n=131 sequential but might pinch
under §6.3 batching. Monitor with `rocm-smi` during a full run.

### 6.5 Drift gate / config-PR template — `S`, value: mid
Make `make bench` (or a `bench.sh`) the canonical "did this config
regress" check: runs the n=24 spec on K + the new config, prints
deltas, exits non-zero if any metric drops > N pp. Wire into the
adversarial regression set (§4.6).

---

## 7. Harness tooling

### 7.1 Result diff CLI — `S`, value: high
`diff_results.py a.json b.json` walks each segment and prints
side-by-side `kata / name / lcs` deltas, color-coded for regressions.
Useful for every config change. Currently we hand-eyeball JSON files;
diff cuts that to seconds.

### 7.2 Per-segment HTML report with embedded audio — `M`, value: mid
`report.py result.json --out report.html` renders a table of
`segment | gold | each-config-output | wav (embedded)`. Lets a human
listen to the audio while comparing. Massive accelerator for the
"what's going on with seg07" debugging cycle. ~100 lines of Python +
a static HTML template.

### 7.3 Result replay — `S`, value: mid
`rescore.py result.json --metrics reaction_recall,precision`
re-scores an existing result file with new metrics — no model rerun.
Means we can add §4.4 / §4.5 retroactively to all past results
without re-running the bench. Important for trend continuity.

### 7.4 Result-name embedding (config + spec + model hash) — `S`, value: low
Today result filenames have a date stamp. When a config changes
without a date bump, results silently mean different things. Embed
short hashes: `killah_kuts_s01e01_n24__K-3a2_M-bf16_S-9c4.json`.
Less human-readable, but unambiguous for the result-diff tooling
in §7.1.

### 7.5 CONFIGS as YAML — `S`, value: low
`configs.py` has 16 configs and counting. Move to per-file YAML so
PR diffs are clean. Doesn't change behavior. Worth doing only if
we're going to add ≥ 5 more configs (this doc adds at least 6).

### 7.6 vLLM/transformers version pinning — `S`, value: mid
Findings doc warns: "container reboot with a different transformers/
vllm version" can drift numbers. Add a `versions.lock` next to each
result file, populated from `pip freeze | grep -E 'transformers|vllm'`.
Diff before declaring regression.

---

## 8. Genuinely new axes

### 8.1 Whisper-augment fusion — `M`, value: high
Already drafted in `project_whisper_gemma_augment.md`. New config
`R_audio_sysrole_oracle_whisperaug`: includes the whisper-large-v3
draft for this chunk in the system prime ("The following draft
transcript may be missing reactions; add them where you hear them in
the audio, otherwise keep it unchanged"). Tests a single-pass fusion
vs the parallel-merge approach. Stack with §4.3 disagreement eval to
target it at chunks where E4B and whisper actually disagree.

### 8.2 Speaker-count / role hint — `S`, value: low speculative
Inject "This clip has 2 speakers" or "Commentator + fighter" from a
diarization pre-pass. Hypothesis: helps the model split turns
correctly when it currently runs them together. Cheap if diarization
is already a pipeline step (it is — see `chigyusubs/turn_context.py`).

### 8.3 Self-RAG: similar past chunks — `L`, value: speculative
Embed each chunk's audio (Whisper encoder or wav2vec2) → nearest
neighbors from past episodes' gold → inject their transcripts as
few-shot anchors. Heavy infra (audio embedding store) but could
generalize show-knowledge across episodes without hand-curating
primes.

### 8.4 OCR-pre-pass model bake-off — `M`, value: high
The OCR pre-pass is the upstream of every oracle config we run.
Today it's E4B at mst=140; we have probe results for 26b-MoE and
31b. Score them all against `score_ocr_sweep.py` with the same
glossary, pick the best by recall × throughput, and freeze the
choice. The wrong OCR model silently caps every downstream config.

### 8.5 Per-show prime template — `M`, value: mid
Right now `configs.py` hard-codes a Killah Kuts domain string. To
ship to a second show, we'd copy/paste. Define a
`prime_templates/<show>.py` (or YAML) that the runner loads by
episode → show mapping. No new model behavior; pure refactor that
lets us bench across shows without code edits.

### 8.6 Streaming output for long chunks — `L`, value: low
vLLM supports streaming. For 18s chunks, stream tokens and stop early
if the output exceeds expected length × 2 (catches repetition loops
before they consume 2k tokens). Engineering complexity for an edge
case; only worth it if §2.4 fails to catch loops.

---

## 9. Out of scope (and why)

### 9.1 More guided_json variants — won't pay off
Findings doc settled this on 2026-04-11: guided_json ≠ semantic
constraint. The model packs the cast list into the string field
verbatim. Adding more schema variants doesn't change the underlying
attention dynamics.

### 9.2 `max_soft_tokens` past 280 — confirmed unhelpful
The L/M/N/O sweep on 2026-04-14 showed mst > 280 is a latency tax
without quality gain. Don't burn more bench cycles here.

### 9.3 More vision-only configs without oracle — diminishing returns
J (video, no oracle) was 44% name on n=24 — strictly the worst.
Vision wins only when paired with oracle text (K's headline). Skip
configs that vary vision without oracle.

### 9.4 E2B → bigger model — wrong direction
E2B → E4B already happened. The gap to a hypothetical E12B isn't
funded; finite local hardware (24 GiB VRAM peak) constrains us.
Better local routes are quantization (NVFP4, AWQ, GPTQ) and
distillation (§8.3 cousins), not bigger weights.

### 9.5 Replacing CTC alignment with vLLM — out of harness scope
Aligning words back to audio is a separate problem from transcribing
them. CTC alignment (`align_ctc.py`) already works. The harness
shouldn't grow into that surface; CrispASR is the alternative
runtime if we want consolidation (see `reference_crispasr.md`).

---

## How to pick what to do next

If you have **one afternoon** to spend on the harness:
- §6.1 (full-episode K bench) → answers "should we ship K" today.
- §7.1 (result diff CLI) → makes every future change cheaper.

If you have **one week** to spend:
- §4.1 + §4.6 (second-episode spec + adversarial regression set)
  before any more config tweaks. Without these, we're tuning blind.
- §4.4 + §4.5 (reaction recall, precision) so the metrics reflect
  what we actually care about.
- §1.1 (OCR-derived vocab) and §2.4 (rep-penalty safety) — both
  cheap, both plausibly material.

If you're optimizing for **long-term harness health**:
- §7.2 (HTML report) and §7.3 (result replay) compound across all
  future bench cycles. They're not flashy but they multiply
  iteration speed.

What to *avoid* doing without a hypothesis:
- Adding configs by analogy with past wins (`O_..._mst1120` style).
  Configs are cheap to write but expensive to maintain; every config
  is a tax on every future result-diff.
- Bumping prime length without a §1.7 sweep first — our finite
  attention budget intuition isn't quantified.
