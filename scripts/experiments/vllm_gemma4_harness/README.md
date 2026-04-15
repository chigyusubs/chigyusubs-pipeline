# vLLM + Gemma 4 ASR harness experiment

Exploring whether a vLLM-hosted Gemma 4 ASR harness (prime-biased
prompts + optional vision grounding + prefix caching + structured
outputs) can beat the current Cohere/Whisper pre-pass on Japanese
variety-show transcription quality.

See `docs/vllm-gemma4-harness-findings.md` for running results and the
current design. This folder is the reproducible machinery that produced
those results; keep the findings doc in sync when you re-run.

## Layout

```
vllm_gemma4_harness/
├── server/              # dockerfile + run recipe for the vLLM ROCm image
├── configs.py           # harness configurations being compared (A, B, C, ...)
├── scoring.py           # kata/name recall + LCS-to-gold utilities
├── build_eval_spec.py   # extracts tight windows + frames from CTC-gold segments
├── run_bench.py         # runs all CONFIGS across a spec, writes results JSON
├── eval_specs/          # generated spec files (checked in for repro)
├── results/             # timestamped benchmark outputs (checked in, small JSON)
└── README.md            # this file
```

## Reproducing the current numbers

1. **Start the vLLM server** — see `server/README.md`.
2. **Build an eval spec** for Killah Kuts S01E01 (the baseline episode):
   ```bash
   python3 build_eval_spec.py \
     --episode killah_kuts_s01e01 \
     --video samples/episodes/killah_kuts_s01e01/source/KILLAH.KUTS.S01E01.1080p.AMZN.WEB-DL.DDP2.0.H.264-Emmid.mkv \
     --out eval_specs/killah_kuts_s01e01.json \
     --count 10
   ```
3. **Run the bench**:
   ```bash
   python3 run_bench.py \
     --spec eval_specs/killah_kuts_s01e01.json \
     --out  results/killah_kuts_s01e01__$(date +%Y%m%d_%H%M%S).json
   ```
4. Compare against the snapshot result in `results/` with the earliest
   timestamp. If numbers have regressed, the most likely causes are:
   container reboot with a different transformers/vllm version, model
   file drift in `/mnt/models/huggingface`, or a prompt edit in
   `configs.py`.

## Extending the experiment

### Adding configs

Edit `configs.py` and append to `CONFIGS`. Keep the name prefix letters
stable (`A_`, `B_`, ...) so old results stay comparable — assign a new
letter to each new config rather than mutating existing ones.

### Adding episodes

Run `build_eval_spec.py` with a different `--episode` / `--video`. If
the episode has different named entities, pass `--names` to override
the default list. Save the spec in `eval_specs/`.

### Adding harder / easier segments

Edit `build_eval_spec.py`'s `pick_segments` scoring or tune
`--min-dur`/`--max-dur`/`--count`. The current heuristic is:
`score = 2 * katakana_count + 3 * name_hits`, segments 1–4 s long,
de-duped by 5 s proximity.

### Scoring upgrades

`scoring.py` currently does substring kata/name recall and LCS ratio.
This is intentionally crude — meant for trend tracking, not absolute
WER. If you want finer grain, swap in `chigyusubs.transcript_comparison`
utilities or wire up `jiwer`. Keep the crude metrics too so old results
stay comparable.

## Current findings (2026-04-11 snapshot)

Killah Kuts S01E01, 8 hard segments, E2B-it via vLLM ROCm nightly:

| Config | Kata recall | Name recall | Mean LCS |
|---|---|---|---|
| A_audio_base | 5/10 (50%) | 2/8 (25%) | 0.400 |
| B_audio_primed | 7/10 (70%) | 2/8 (25%) | 0.457 |
| C_vision_primed_1fps280 | 8/10 (80%) | 3/8 (38%) | 0.491 |

Headline conclusions (full analysis in `docs/vllm-gemma4-harness-findings.md`):

- **Priming wins ~+20pp on katakana**, free, uses infrastructure the
  pipeline already has (glossary + OCR).
- **Vision on top of priming wins another ~+10pp katakana and +13pp
  names**, at the cost of ~0.6 s latency per 8 s chunk.
- **`しんいち` breaks at 0% across all configs** — this is the strongest
  argument for a dedicated OCR pre-pass that extracts chyron strings
  exactly and injects them into the prime, rather than asking Gemma to
  OCR while transcribing.
- **Priming can mislead**: seg08 `レフェリー` regressed to `フェンシング`
  when `フェンシング` was included in the prime vocab. Only include
  terms with high confidence they appear verbatim.

## Open questions

See the "Open questions" section of
`docs/vllm-gemma4-harness-findings.md` — kept there rather than here
because it evolves every session.
