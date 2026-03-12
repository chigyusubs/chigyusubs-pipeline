# Gemini Pricing Notes

This document tracks the practical pricing and value conclusions for the Gemini models used in this repo.

It is not a full product matrix. It exists to answer:

- which model is the best paid/default compromise
- which model is the best free-tier compromise
- when Pro is worth paying for
- whether auxiliary OCR on Flash Lite is worth keeping

For operating defaults, see `docs/gemini-transcription-playbook.md`.
For raw experiment evidence, see `docs/lessons-learned.md`.

## Current Practical Conclusion

For the current free tier, the best working default is:

- `gemini-3-flash-preview`
- `video`
- `spoken_only`
- `thinking=low`
- `media_resolution=high`
- `temperature=0.0`

For cheap visual-text side artifacts, the best current auxiliary path is:

- `gemini-3.1-flash-lite-preview`
- `video`
- `ocr_only`

This combination is currently the best value path we have found:

- `3-flash` handles the main spoken transcript well enough to be the primary free-tier transcription model
- `flash-lite` is weak as a main transcript model
- but `flash-lite` is good enough as a cheap OCR-like extractor on dense visual prompt chunks

## Does `3-flash` Transcript + `flash-lite` OCR Get Close Enough To Pro?

Yes, for the default workflow.

No, if the question is “does it fully replace the best Pro outputs?”

Current practical read:

- it gets close enough that Pro is no longer the obvious default choice
- it is probably the right default for day-to-day work
- it does not fully replace Pro when you want the cleanest Japanese transcript on hard banter-heavy chunks

More concretely:

- `gemini-2.5-pro` still produced the strongest beach / comedy-banter transcript
- `gemini-3-flash-preview` with `video + spoken_only + low` is the best non-Pro compromise we tested
- `gemini-3.1-flash-lite-preview` OCR-only on dense rule/prompt chunks is good enough to recover useful visual text cheaply

So the working recommendation is:

- use `3-flash` as the primary transcript model
- use `flash-lite` OCR-only as a side artifact
- keep Pro for benchmarking, difficult episodes, or when transcript quality clearly matters more than cost / quota

## Paid Pricing Snapshot

The following prices are the current practical reference values for the models we tested.

### `gemini-3.1-pro-preview`

- input:
  - `$2.00 / 1M` tokens for prompts `<= 200k`
  - `$4.00 / 1M` tokens for prompts `> 200k`
- output, including thinking:
  - `$12.00 / 1M` tokens for prompts `<= 200k`
  - `$18.00 / 1M` tokens for prompts `> 200k`

### `gemini-3.1-flash-lite-preview`

- input:
  - `$0.25 / 1M` for text / image / video
  - `$0.50 / 1M` for audio
- output, including thinking:
  - `$1.50 / 1M`

### `gemini-3-flash-preview`

- input:
  - `$0.50 / 1M` for text / image / video
  - `$1.00 / 1M` for audio
- output, including thinking:
  - `$3.00 / 1M`

### `gemini-2.5-pro`

- input:
  - `$1.25 / 1M` for prompts `<= 200k`
  - `$2.50 / 1M` for prompts `> 200k`
- output, including thinking:
  - `$10.00 / 1M` for prompts `<= 200k`
  - `$15.00 / 1M` for prompts `> 200k`

## Practical Cost Implications

Two important consequences from current pricing plus our measured token counts:

1. For Gemini 3.x Flash-family models, video is not just higher quality for this repo. It is also cheaper than audio on the paid tier.

2. Pro still costs enough more than Flash that it should be treated as an escalation tier, not the default first choice.

Approximate episode-scale costs from our current chunked-video experiments on `great_escape_s02e01`:

- `gemini-3.1-flash-lite-preview`: about `$0.07` per episode
- `gemini-3-flash-preview`: about `$0.15` per episode
- `gemini-3.1-pro-preview`: about `$0.60-$0.70` per episode
- `gemini-2.5-pro`: about `$0.95-$1.05` per episode

These are operational estimates, not formal billing guarantees.

## Free-Tier Policy

Current free-tier usage policy:

- spend `gemini-3-flash-preview` on the main transcript
- use `gemini-2.5-flash` as the secondary comparison path when needed
- use `gemini-3.1-flash-lite-preview` for cheap OCR-like extraction and auxiliary probes

Roughly:

- `3-flash`: scarce, high-value main transcription calls
- `2.5-flash`: secondary benchmark / comparison calls
- `flash-lite`: abundant OCR / probe budget

## Current Recommendation

If choosing today:

- default to `3-flash` for transcript generation
- default to video, not audio
- keep `spoken_only` for the main transcript
- keep `flash-lite` OCR-only as a reusable side artifact
- escalate to Pro only when the quality delta is materially worth paying for
