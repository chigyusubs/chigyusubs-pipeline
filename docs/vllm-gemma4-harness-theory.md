# vLLM + Gemma 4 ASR harness — theoretical scaffolding

Companion to `vllm-gemma4-harness-findings.md` (empirical results) and
`vllm-gemma4-harness-design-ideas.md` (open design space). This doc is
the theoretical lens — the standard concepts from speech ML and LM
literature that already exist for problems like this, mapped onto our
specific levers.

The point is not to introduce new theory. It's to stop re-deriving
intuitions that have decades of grounding, and to make the harness's
design choices legible: each one corresponds to a known move in a
known framework.

## What's already in this repo

Search the docs and the picture is empirical, not theoretical:

- **Failure-mode catalogs** (`ctc-weak-anchor-failure-modes.md`,
  `lessons-learned.md`): named patterns ("weak-anchor lines",
  "recitation", "audio-tower ceiling"), but no formal model of why
  they happen.
- **Comparison tables** (`asr-comparison-2026-04-10.md`,
  `transcription-model-comparison.md`): chars/sec, RTF, anecdotal
  per-model strengths. Useful but doesn't generalize across new
  candidates without rerunning.
- **Findings doc** (`vllm-gemma4-harness-findings.md`): ~30 KB of
  config tables, ablations, and decisions. It treats `mst` as a
  "mode switch" and the prime as an "attention budget" — both
  approximations of formal decoder-state and information-bottleneck
  ideas, but without the formalism.
- **Quant calibration** (`gemma4-quant-calibration.md`): the closest
  thing to an explicit theoretical framing in the repo (calibration
  data distributions matter, leakage matters), but scoped to one
  decision.
- **Codebase**: reads as a clean implementation; no theoretical
  comments beyond local-correctness reasoning.

So: pragmatic codebase, sparse internal theory. The rest of this doc
fills in the lens we've been using implicitly.

---

## 1. The Bayesian decomposition (the master frame)

For an audio-conditioned LM transcribing audio `a` into text `t` with
context `c`:

```
P(t | a, c) ∝ P(a | t)  ·  P(t | c)
              ─acoustic   ─language-and-context
              likelihood   prior
```

Every lever in the harness moves one of these two factors.

| Lever | Moves | Notes |
|---|---|---|
| Audio preprocessing (denoise, vocal isolation, slow-down) | `P(a\|t)` | `§5.4`, `§5.5` in design-ideas. Cleans the acoustic likelihood. |
| Larger / better audio encoder | `P(a\|t)` | E2B → E4B unlock. Audio-tower capacity. |
| System prime (domain, vocab, cast) | `P(t\|c)` | Shifts the prior toward show-relevant strings. |
| Per-clip oracle names | `P(t\|c)` | Per-segment prior shift. |
| Logit bias on cast tokens | `P(t\|c)` | A direct, post-encoder prior nudge. |
| Vision frames | `P(t\|c)` | Cross-modal conditioning of the prior. |
| Repetition penalty | sampling distribution from `P(t\|·)` | Modifies the decoded distribution, not the model. |
| Self-consistency vote | sampling distribution | Averages over samples from `P(t\|a, c)`. |
| Two-pass critique | `P(t\|a, c, draft)` | Conditions on its own output as additional context. |

This decomposition predicts our central empirical finding:
**oracle-names is the biggest single-knob name-recall win** because
names are exactly the case where the LLM's default `P(t | c=∅)` prior
is wrong for our domain (it prefers common kanji), but the audio
likelihood `P(a | しんいち)` is fine. Moving the prior fixes it; no
amount of audio-side work would.

It also predicts what *won't* work: when `P(a | t)` is too flat (chunk
where `しんいち` is acoustically too degraded — seg07), no prior
strong enough to discriminate among hypotheses can recover the right
answer. That's the audio-tower ceiling. Bayes says: zero likelihood
beats any prior.

---

## 2. Audio LLMs as speech-conditioned language models

The classical ASR factorization is the same Bayesian one, but the
language prior `P(t)` was a small N-gram or Transformer LM on top of
an acoustic model. The audio path dominated.

In Gemma 4 (and Whisper, AudioPaLM, Qwen-Audio, SeamlessM4T):

- Audio encoder produces a fixed-size latent representation.
- That representation is mixed into the **decoder LM's** input as
  soft tokens.
- Decoding is one-shot generation by the LM, not a beam search over
  acoustic frames.

Implications:

1. **The LM prior dominates.** At inference, the LM is computing
   `P(t | a) ∝ P(a | t) · P(t)` but `P(t)` is *enormous* — billions of
   parameters of language preferences. This is why our model writes
   `南川` when the audio said `みなみかわ`: the kanji prior is much
   stronger than the acoustic discriminator at distinguishing kana
   homophones. Classical Whisper has the same issue but smaller LM
   = weaker prior = less script substitution.

2. **Prompt-conditioning is shallow fusion.** Putting the cast list in
   the system prime is mathematically equivalent to multiplying every
   cast-name token's prior probability up. The strength of the shift
   depends on how the model was instruction-tuned to attend to the
   system prompt. Logit bias (`§2.1` in design-ideas) is the same
   operation made *explicit and uniform* at the token level rather
   than mediated by the model's learned attention.

3. **The "audio-tower ceiling" is information loss in the encoder.**
   If `P(a | しんいち)` ≈ `P(a | シンイチ)` ≈ `P(a | 新一)` because the
   audio encoder collapsed the relevant acoustic distinctions, then
   no prior can recover the correct script — the encoder threw away
   the information needed to discriminate. Larger encoder, slower
   playback, isolated vocals all attack this from the `P(a | t)` side.

This framing makes "fix names with bigger model" and "fix names with
better prior" two different things, not interchangeable.

---

## 3. Contextual biasing / shallow fusion (where most of our levers live)

The classical-ASR literature has spent ~15 years on the problem of
"how do you make a fixed model attend to a small list of important
words at inference time, without retraining?" The standard answers:

- **Hot-word lists** (Whisper's `initial_prompt`, NVIDIA Riva's
  context lists, Google STT's speech adaptation): a soft prior that
  tells the model "expect these tokens".
- **Shallow fusion** with an external LM: linear combination of
  scores at decode time. `score(t) = score_ASR(t) + λ · log P_LM(t)`.
- **Class-based LMs**: replace `<NAME>` slots in the LM with a
  cast-specific list. Used in classical ASR for proper nouns.
- **TCPGen** (tree-constrained pointer generator) and
  **biasing transducers**: trainable variants where the contextual
  list is a structural input, not just text.

Our system-role prime, per-clip oracle, and logit-bias proposals are
all variations on hot-word prompting. They differ in:

- **Strength**: logit bias is the strongest lever (uniform), prime
  text is moderate (mediated by attention), oracle text in user turn
  is weakest (most easily ignored).
- **Selectivity**: oracle names per-clip is targeted (only when the
  name is plausible); static cast prime is global.
- **Failure mode**: too-strong biasing causes recitation
  (the seg04 cast-list dump). Too-weak biasing fails to override the
  kanji prior.

The empirical lesson from this literature, which our findings
independently confirm: **bias strength must match the noise**. A
kana-only fighter name that the audio tower can't pin down needs
strong biasing; a name the model already gets right needs none.
"Minimal prime is correct" (`feedback_e4b_prime_budget.md`) is the
classical-ASR result that biasing irrelevant items hurts because it
shifts mass away from where it's needed.

This also predicts §1.2 (anti-prime / negative biasing) is risky:
suppressing `山田` to recover `山根` works only if the model
distributions on those tokens are well-separated. If they collapse to
the same internal representation under the audio encoder, negative
biasing on one suppresses both.

---

## 4. In-context learning as conditioning

When we put text in the prompt and the model behaves as if it had
been fine-tuned on that text, that's in-context learning. The
mechanism is debated (induction heads, implicit Bayesian inference,
gradient-descent-in-attention) but the practical implications are
well-established:

- **More demonstrations help** — to a point, then plateau or regress.
  Our `_DEFAULT_CAST` size sweep (8 → 17 → 4) traces this curve
  empirically. The §1.7 sweep formalizes it.
- **Demonstration similarity matters**. Few-shot anchors (§1.3) work
  best when the example chunks are acoustically and semantically
  close to the target. A reference clip from the same episode is
  better than from a different show.
- **Position effects.** System role > pre-user-turn instruction >
  trailing instruction > buried in user content. Our D vs B finding
  is exactly this.
- **Recency bias.** The model attends most strongly to nearby tokens.
  This argues for putting the per-clip oracle *closer to* the audio
  input than the static prime.

Caveat: in-context learning is not a free lunch. The model can
"learn" wrong patterns if the demonstrations are misleading. The
seg04 recitation failure was the model learning "this prompt
expects a cast list as output" from the structure of the user-turn
prime. Moving to system role broke that pattern.

---

## 5. Constrained decoding

Three flavors of decode-time constraint, in order of strength:

1. **Soft prior shift** (logit bias, hot words): tilts the
   distribution but allows any output.
2. **Grammar / schema constraint** (vLLM `guided_json`, Outlines,
   LMQL, lm-format-enforcer): enforces output structure, but not
   semantics. The seg04 finding — guided JSON didn't kill recitation,
   it just packed it into the string field — is the canonical example
   that *structural constraint ≠ semantic constraint*.
3. **Hard token forcing** (forced decoder prefix, constrained beam
   search per Hokamp & Liu's grid beam search): forces specific
   tokens to appear in the output. Strong but rigid; mismatches with
   what the audio actually says cause forced-error outputs.

Our harness uses (1) heavily, has tested (2), and has not used (3).
For (3) the test would be: force-decode the oracle names as required
substrings via constrained beam search. Likely too rigid for
conversational ASR (some clips don't actually contain the oracle
name even when an OCR'd telop overlaps), but worth knowing about.

The `repetition_penalty` and `no_repeat_ngram_size` levers (§2.4) are
distinct from the above — they're *negative* constraints on the
sampling distribution, born from Holtzman et al.'s work on degenerate
decoding ("The Curious Case of Neural Text Degeneration"). Greedy or
low-temperature sampling on long contexts collapses into loops; the
penalty breaks the loop without changing the model. The whisper
chunk-78 failure is a textbook degeneracy pattern, and the same
mitigation applies to E4B if it ever exhibits it.

---

## 6. Multimodal grounding

The literature on visually-grounded ASR (Look-Listen-and-Read style
work, AV-HuBERT, video-conditioned Whisper variants) has a
consistent finding: **vision helps most when the audio likelihood is
ambiguous and the visible text or scene resolves the ambiguity.**

This is exactly the conditional information story. Vision modality
adds information `I(t; v | a, c)`. If the prior `c` already gives the
model the answer (oracle names), vision's marginal contribution is
near zero — and our findings confirm this directly: K (audio + oracle)
≈ I (audio + oracle + vision) on our spec. The "vision regresses
some hiragana names back to kanji" finding is the same cross-modal
noise effect documented in the multimodal literature: when modalities
disagree, the dominant prior wins, and the LM's kanji prior reasserts
itself when vision contributes ambiguous evidence.

The pre-OCR'd-telops idea (§3.2) is the practical takeaway: most of
vision's contribution is the on-screen *text*, and we can deliver
that as text directly without paying the pixel-token cost. Vision
matters mainly when the on-screen content isn't text (referee
gesturing, fighter posture).

---

## 7. Decoding pathologies and their canonical names

Knowing the names cuts debugging time. The patterns we've seen, with
literature names:

| Symptom in our runs | Canonical name | Standard mitigations |
|---|---|---|
| Whisper chunk 78: 1916 chars of `ありがとう` | Degenerate repetition (Holtzman) | repetition penalty, no_repeat_ngram, top-k/p sampling |
| seg04 cast list as transcript | Prompt recitation / instruction echoing | system-role separation, schema-locking partial fix, RLHF normally addresses |
| `みなみかわ → 南川` | Script substitution / Bayesian-prior dominance | per-clip oracle, logit bias, kanji penalty |
| `山根 → 山田` | Plausible-substitute hallucination (faithful to LM prior, not audio) | logit bias, more audio context, retraining |
| `しんいち → シイチ / 新一` | Acoustic ambiguity collapse at the encoder | larger encoder, audio cleanup, oracle hint as last resort |
| anime-whisper outputs `チンポ` for `フェイント` | Domain-prior poisoning (training data dominates inference) | model selection, rerank with cleaner LM |
| Gemma 2B occasional "invented MC speeches" | LM hallucination unconstrained by acoustics | shallow-fusion biasing, rejection sampling, smaller models avoid this |

Most of these are well-mapped in the LM literature. The one we have
the *least* established theory for is the "vision regresses names
back to kanji" interaction effect, which seems specific to Gemma's
fusion architecture. That'd be a useful narrow probe (§5 in
design-ideas).

---

## 8. Self-consistency and ensemble decoding

Wang et al.'s self-consistency idea (multiple samples → majority vote)
exploits a property well-known in classification ensembles: errors
are uncorrelated across samples in a way that correct answers aren't.
At low temperature it's a no-op (samples collapse to the same output);
at moderate temperature (0.5–0.8) it can recover.

Two practical modes:

- **Self-consistency on the same model** (§2.3): cheap if latency
  budget allows, but gains are bounded by the model's mode coverage.
  If the wrong answer is always the modal output, voting reinforces
  it.
- **Cross-model ensemble** (the parallel-merge idea, §8.1's whisper-
  augment alternative): different architectures make different
  errors. Whisper and E4B failure modes are demonstrably distinct
  (clean kanji + repetition vs reactions + script substitution).
  Ensembling them is a bet that the disagreement is correctable.

The disagreement-eval idea (§4.3) is the prerequisite for both: you
can't design a good ensemble without knowing the conditional error
distributions.

---

## 9. Confidence calibration

Greedy decoding gives no confidence. Several established proxies:

- **Token logprobs** (vLLM exposes them): rough but cheap.
- **Length-normalized sequence logprob**: standard for ranking
  hypotheses.
- **Self-consistency vote margin**: stronger when calibrated.
- **Prompt-perturbation stability** (paraphrase the prime, see if
  output changes): a robustness signal.
- **Holdout-likelihood under a separate LM**: how plausible is the
  output to *another* model.

Our harness has none of these wired up. The `flags` system in
`transcribe_episode_e4b.py` (low cps, low cjk_ratio) is a poor-man's
calibrator — useful but not principled. A cascade strategy (route
low-confidence chunks to a stronger config, §2.6) needs a real
confidence signal first; a flag based on cps catches some failures
but misses substitution errors entirely (the `山根 → 山田` chunk has
fine cps).

---

## 10. Cascade vs end-to-end factoring

The OCR-pre-pass + ASR-pass design is a cascade. Cascades are
known-suboptimal in the limit (joint optimization beats stage-wise),
but practically dominant when:

- One stage is much more expensive than the other (we can afford a
  big OCR model once per episode but not per chunk).
- Failure modes of the stages are independent (OCR misses are
  recoverable from a glossary; ASR misses are recoverable from
  oracle hints).
- Each stage produces a clean, inspectable artifact (OCR sweep JSON,
  per-clip oracle, glossary).

The conditions hold for us. The relevant theoretical concern is
**error compounding**: if OCR pre-pass misses a name 20% of the time
and the ASR pass needs the oracle for that name to score, the joint
miss rate is ≥ 20%. The §4.7 realism eval (drop oracle metadata) is
the cascade-reliability probe.

End-to-end alternatives exist (train a single model on
audio + frames → transcript with frozen LM) but require fine-tuning
infrastructure we don't have. They're listed for context, not on
the table.

---

## 11. Eval theory

Two bodies of literature matter here:

**Speech-recognition metrics:**
- WER / CER are the standard measures. Substring recall is *not*
  WER. We use it because it's cheap, interpretable, and stable
  across long outputs where WER alignment can fail. But it's biased
  toward rare-word recall and hides whole-sequence quality. The
  `mean_lcs` LCS ratio is an approximation of the structural part
  WER captures.
- **Hallucination-aware metrics**: precision of named entities, not
  just recall (`§4.5`). Standard in summarization-eval literature
  (factuality scores), almost never reported in ASR comparisons —
  but it should be for LLM-based ASR, which can hallucinate in ways
  classical ASR can't.

**ML eval methodology:**
- **Train/dev/test discipline.** We're tuning configs on KK S01E01
  and reporting on the same episode. That's dev-set-as-test. The
  §4.1 "second-episode spec" is the standard fix.
- **Generalization gap.** Expect a drop when moving to held-out data;
  if there's none, suspect leakage. The OCR-derived oracle is
  partly leakage-resistant (it doesn't see CTC gold) but our
  vocab/cast lists were curated by inspecting CTC gold — that *is*
  leakage.
- **Ceiling analysis** (Andrew Ng's term): "if I had perfect X, what
  would the metric be?" The §4.2 TTS-from-gold and §4.4 oracle-
  names-as-CTC-gold are ceiling probes. They tell us how much room
  is left and where the loss is.

---

## 12. Japanese-specific phenomena worth naming

These aren't universal, but they're well-documented enough in
Japanese-NLP that we should treat them as known patterns rather than
re-deriving them every time.

- **Script ambiguity** (kanji vs hiragana vs katakana for the same
  reading): means that for many tokens, the audio likelihood is
  identical across multiple text outputs. The LLM picks one based
  on prior. This is *the* core driver of our name-substitution
  failures. The traditional NLP fix is dictionary-based reading
  resolution; the LLM fix is biasing toward the desired script.
- **Reactions / aizuchi (相槌)**: short interjections (`うん`,
  `そうそう`, `えー`, `はあ`) that classical ASR drops because they
  have low acoustic energy and weak language-model support. LLM-
  based ASR catches them because the LM expects conversational
  patterns. This is E4B's main edge over Whisper, and it should be
  measured directly (§4.4).
- **Telop integration**: variety-show on-screen text is *part of*
  the meaning, not metadata. OCR'd telops as ASR context (§3.2)
  isn't just grounding — it's recovering content the audio doesn't
  carry.
- **Speaker overlap and crosstalk**: more frequent in variety than
  in news/interview audio. Diarization quality bounds reaction
  recall (you can't transcribe a reaction you can't separate from
  the dominant speaker).
- **Mora-timing and katakana-borrowed terms**: katakana fighter
  vocabulary follows English mora-stress rather than Japanese pitch
  accent, which audio encoders trained on standard Japanese
  occasionally collapse onto similar-sounding native words. Hence
  hot-word biasing on katakana terms is disproportionately
  effective.

---

## 13. Where we lack theoretical grounding (open questions)

Honest gaps. Areas where the harness is making decisions that the
literature *would* inform but we haven't connected to:

1. **Optimal prime length as a function of prior strength.** The
   `_DEFAULT_CAST` sweep (§1.7) is the empirical version. The
   theoretical version is "how does in-context-learning slope
   change with relevant vs irrelevant context?" — there are
   scaling-curve papers that could give us a closed-form expectation
   instead of empirical search.
2. **Cross-modal interference in audio + vision LLMs.** Why does
   vision regress some names? Standard multimodal-fusion theory
   (modality dropout, attention-weight analysis) might explain it.
   We haven't looked.
3. **Quantitative confidence-calibration baseline.** What logprob
   threshold corresponds to which error rate? Standard
   calibration-curve methods (reliability diagrams, ECE) would tell
   us. Not wired up.
4. **Oracle-hint sensitivity to OCR error rate.** If the OCR
   pre-pass injects a *wrong* name into the oracle list, does the
   audio model emit the wrong name verbatim? This is the cascade
   error-compounding question and it's measurable (inject perturbed
   oracles, measure output drift). §4.7 design.
5. **Domain-prior interaction at quantization.** FP8 vs bf16
   produced asymmetric results on I (vision + oracle). Standard
   quantization-error analysis (Hessian-based sensitivity scoring,
   activation distribution analysis) would tell us which layer
   carries the script-prior signal. Maps to
   `gemma4-quant-calibration.md` planning.

These are research questions, not engineering ones. They'd each be
real work to address, and most aren't on the critical path. Worth
naming so we know what we're not doing.

---

## 14. Design-idea ↔ theory cross-reference

Forward index from `vllm-gemma4-harness-design-ideas.md` into the
sections above. Use this when you're picking something to implement
and want to know what theory predicts about it (the "what theory
predicts" column is the value-add — design-ideas already states what
the idea *is*). A back-index from theory section to grounded ideas
follows.

### Design ideas → theory

#### Design-ideas §1 — Priming content & framing

| Idea | Theory | Prediction |
|---|---|---|
| 1.1 OCR-derived vocab | §1, §3 | Same `P(t\|c)` shift as cast names; per-clip selectivity dominates static for the same biasing-must-match-noise reason. Should compound with oracle_names, not overlap. |
| 1.2 Negative prime / anti-list | §3 | Effective only if `山田` vs `山根` are well-separated in the encoder. If the audio collapses them, suppressing one suppresses both. Quick falsification: ablate on a 山根-heavy chunk set. |
| 1.3 Few-shot audio anchors | §4 | Demonstration similarity matters; same-show clips should outperform same-domain clips. Watch for verbatim copying of anchor phrases — classic ICL failure mode. |
| 1.4 Ablate domain block | §4 | Tests whether the prose paragraph pays its irrelevant-context cost. Likely small effect on this episode (KK-specific phrasing rarely appears in spoken dialogue) but load-bearing for cross-show transfer. |
| 1.5 Tag prime sections | §4 | Position effects say structure helps; magnitude is small. Cheap to verify, cheap to drop. |
| 1.6 Show-level + episode-level split | §4, infra | Two distinct wins: stable prefix for vLLM cache (infra), recency bias toward the per-episode tail (§4). |
| 1.7 Prime-length sweep | §4, §13.1 | Expect inverted-U: too few = no signal, too many = irrelevant-context noise. The empirical curve is what §13.1 wants in closed form. |

#### Design-ideas §2 — Decoder controls

| Idea | Theory | Prediction |
|---|---|---|
| 2.1 Logit bias on cast tokens | §3, §5 (soft) | Strongest `P(t\|c)` shift available, applied uniformly. Small magnitudes (≤ +3) should suffice; larger magnitudes will recite (the same dynamic that broke configs B/C). |
| 2.2 Negative logit bias on traps | §3, §5 (soft) | Same separability caveat as 1.2. Adds risk of suppressing legitimate uses (`南川` is sometimes correct). |
| 2.3 Self-consistency vote | §8, §9 prereq | Bounded by mode coverage — if the wrong answer is the modal output, voting reinforces it. Best deployed only on chunks already flagged low-confidence. |
| 2.4 Repetition-penalty safety net | §5 (Holtzman), §7 | Insurance against the documented degeneracy pattern. Should not move the headline metrics on already-stable runs; will catch the next chunk-78 silently. |
| 2.5 Two-pass critique | §1 (extension to `P(t\|a, c, draft)`) | Adds conditioning but also propagates draft errors. Best on chunks where draft includes a token in the substitution-trap list. |
| 2.6 Cascade: oracle-empty → vision | §6, §9, §10 | Cross-family lever (modality + content). Quality bound by the upstream confidence/coverage signal — see §9 gap. |

#### Design-ideas §3 — Multi-modal stacking

| Idea | Theory | Prediction |
|---|---|---|
| 3.1 Dynamic frame inclusion | §6, §10 | Same routing principle as 2.6, applied at modality level. Doesn't change the model — just changes the request shape per chunk. |
| 3.2 Pre-OCR'd telops as text | §6 | Most of vision's information *is* on-screen text. Substituting text for pixels should retain most of vision's gain at zero pixel-token cost; the residual is non-text content (gestures, posture). |
| 3.3 Mixed text-OCR + pixel | §6 | Diminishing returns once 3.2 is in. Worth testing only on chunks where vision *and* OCR-text both add information independently. |
| 3.4 Frame-by-chyron-change | §6 | Information selection: 1 informative frame > 8 redundant ones at fixed token budget. The standard "salience-aware sampling" move. |

#### Design-ideas §4 — Eval methodology gaps

| Idea | Theory | Prediction |
|---|---|---|
| 4.1 Second-episode spec | §11 | Expect a regression on first try; that *is* the generalization gap, not a problem with the new spec. If there's no regression, suspect leakage. |
| 4.2 TTS-from-gold ceiling | §11 | If the ceiling is well below 100%, the bottleneck is non-acoustic (eval metric harshness, decoding constraints, prompt format) — useful to know which. |
| 4.3 Disagreement eval | §7, §8 | Maps the conditional error distributions: prereq for any cross-model ensemble or fusion (§8.1). |
| 4.4 Reaction-recall metric | §12 | Aizuchi-specific. Without it, our metrics structurally under-reward E4B's main edge. |
| 4.5 Precision alongside recall | §11 | Hallucination-aware eval. Necessary because LLM-based ASR can hallucinate plausibly-named outputs that classical-ASR metrics never anticipated. |
| 4.6 Adversarial regression set | §11 | Locks past lessons into future guards. Standard regression-test discipline. |
| 4.7 Realism eval (perturbed oracle) | §10, §13.4 | Cascade error-compounding probe. Tells us the slope of the OCR-quality → ASR-quality relationship — a number we currently have zero of. |
| 4.8 Variance / p95 reporting | §11 | Distribution-aware eval. Two configs at the same mean LCS aren't equivalent if one has a long left tail. |

#### Design-ideas §5 — Failure-mode probes

| Idea | Theory | Prediction |
|---|---|---|
| 5.1 Clip-duration sweep | §1, §4 | More audio = more `P(a\|t)` evidence but stretches LM coherence. Expect non-monotone with a sweet spot in the 8–18 s range we already work in. |
| 5.2 Cross-chunk context | §1, §4 (recency) | Helps mid-sentence chunks, propagates errors elsewhere. Needs a confidence gate — otherwise net-zero. |
| 5.3 Phoneticize-first prompt | §2, §5 | Re-orders decoding so audio-tower output precedes LM substitution. Recovers cases where the LM kanji-prior overruled a correct kana hearing. Brittle but cheap. |
| 5.4 Audio-quality probe | §1, §2 | Bounds the headroom on `P(a\|t)`. If degraded audio produces the same outputs, the encoder isn't using the missing detail anyway. |
| 5.5 Acoustic emphasis on しんいち | §2 | Direct test of the encoder ceiling. If any of the failed chunks recover under slowdown / vocal isolation, the ceiling is acoustic, not architectural. Strong signal either way. |

#### Design-ideas §6 — Production gates

| Idea | Theory | Prediction |
|---|---|---|
| 6.1 Full-episode K bench | §11 | n=131 is enough for stable means; n=24 is not for tail metrics. Expect ~5–10 pp shift on tail metrics vs the n=24 numbers. |
| 6.2 Prefix-cache hit rate | infra | Quantifies the throughput half of the §1.6 design split. |
| 6.3 Latency at concurrency | infra | Tells us if vLLM batching is sub-linear at our request shape. |
| 6.4 GPU-mem headroom | infra | Constraint check, not theory. |
| 6.5 Drift gate / config-PR | §11 | Operationalizes regression discipline. |

#### Design-ideas §7 — Harness tooling

Mostly meta — these change the cost of *running* experiments, not
their content. Theory-light by design. §7.6 (version pinning) is an
instance of the reproducibility-discipline subset of §11.

#### Design-ideas §8 — Genuinely new axes

| Idea | Theory | Prediction |
|---|---|---|
| 8.1 Whisper-augment fusion | §1 (extension), §7, §8 | Cross-model ensemble: bets on uncorrelated errors between Whisper's dominant-speaker-clean and E4B's reaction-rich modes. The §4.3 disagreement eval is the prereq that tells us if the bet is sound. |
| 8.2 Speaker-count hint | §3, §12 | Different `P(t\|c)` axis. Modest gain expected on overlap-heavy clips, near-zero on clean single-speaker. |
| 8.3 Self-RAG | §4 | In-context demos retrieved by audio similarity. Acoustic similarity > semantic similarity for ASR few-shot — embedding choice matters more than retrieval architecture. |
| 8.4 OCR pre-pass model bake-off | §10 | Caps every downstream cascade config. Single biggest lever you'd ever want to tune in a cascade — the upstream stage's quality bounds everything below. |
| 8.5 Per-show prime template | infra | Refactor; enables §4.1 cleanly. |
| 8.6 Streaming output / early stop | §5 | Catches degeneracy mid-generation rather than after. Runtime-layer counterpart to 2.4. |

### Theory → grounded design ideas (back-index)

| Theory section | Grounded design ideas |
|---|---|
| §1 Bayesian decomposition | Touched by every idea; primary anchor for 1.x, 2.x, 5.4–5.5, 8.1. |
| §2 Audio LLMs as conditioned LMs | 5.3 (phoneticize), 5.4–5.5 (encoder ceiling). |
| §3 Contextual biasing / shallow fusion | 1.1, 1.2, 2.1, 2.2, 8.2. |
| §4 In-context learning | 1.3–1.7, 5.2, 8.3. |
| §5 Constrained decoding | 2.1, 2.2, 2.4 (Holtzman), 8.6. |
| §6 Multimodal grounding | 2.6 (cascade lever), 3.1–3.4. |
| §7 Decoding pathology table | 2.4 (rep penalty), 4.3 (disagreement), 8.1 (fusion). |
| §8 Self-consistency / ensembles | 2.3 (same-model), 4.3 (prereq), 8.1 (cross-model). |
| §9 Confidence calibration | 2.3, 2.5, 2.6 prereq, 5.2 gating. |
| §10 Cascade vs end-to-end | 2.6, 3.1, 4.7, 8.4. |
| §11 Eval theory | 4.1–4.8, 6.1, 6.5, 7.6. |
| §12 Japanese-specific phenomena | 1.1 (vocab), 3.2 (telop), 4.4 (aizuchi), 8.2 (overlap). |
| §13 Open questions | 1.7 ↔ 13.1; 3.x ↔ 13.2; 2.3/2.6 ↔ 13.3; 4.7 ↔ 13.4; quant work ↔ 13.5. |

Three observations from looking at this index together:

- **§1, §3, §11 ground the most ideas.** That's a fingerprint of where
  the headroom actually is: prior shifting, contextual biasing, and
  eval methodology. Match for the empirical finding that oracle hints
  are the biggest knob and our eval has known gaps.
- **§9 (confidence calibration) is grounding nothing concrete yet but
  is a prereq for many.** That's the "build this before everything
  else" tell — 2.3, 2.5, 2.6, 5.2 all need a real confidence signal
  to work well, and we don't have one.
- **§13's open questions cluster on the cross-modal and quantization
  axes**, which is honestly where we'd need to do real research,
  not just engineering. Knowing this means we don't have to retry it
  hopefully every session.

---

## How to use this doc

- When proposing a new harness config: identify which factor in §1
  it moves. If it doesn't move any, it's probably a wash.
- When debugging a failure: cross-reference §7's pathology table.
  The named-pattern usually has a known mitigation.
- When designing an eval: cross-reference §11. If we're not measuring
  a relevant axis (precision, generalization, ceiling), say so
  explicitly rather than implying coverage we don't have.
- When picking levers from `vllm-gemma4-harness-design-ideas.md`:
  ideas that target `P(t|c)` (prime, oracle, logit bias) compound
  with each other; ideas that target `P(a|t)` (audio cleanup, model
  size) compound with each other; cross-family combinations have the
  most independent gains.

The TL;DR: **prime/oracle/bias are all the same thing
(`P(t|c)` shifts) at different strengths and selectivities, and
they're the bulk of our headroom on names. Audio-side work attacks a
different and smaller bucket. End-to-end ensembling and
cascade-error analysis are where the genuinely new gains would
come from once those buckets are exhausted.**
