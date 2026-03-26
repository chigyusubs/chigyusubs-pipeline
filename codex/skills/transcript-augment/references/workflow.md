# Transcript Augment Workflow

## Purpose

Recover missing dialogue from Gemini transcripts by merging in Flash Lite
audio re-transcription additions. Gemini's video-only pass produces clean but
compressed turns; real spoken dialogue has many more reactions, interjections,
and back-channel responses that matter for subtitle timing and naturalness.

## Pipeline Position

```
gemini transcription → [Flash Lite audio pass] → [transcript-augment] → CTC alignment
```

The augmented transcript replaces `gemini_raw` in `preferred.json`, so CTC
alignment consumes it automatically.

## Flash Lite Configuration

- **Model**: `gemini-3.1-flash-lite-preview`
- **API**: AI Studio (`GOOGLE_GENAI_USE_VERTEXAI=false`)
- **Rate limit**: 500 free requests per day, 10 RPM default
- **Temperature**: 0.1

## Output Naming

- Flash Lite output: `<run_id>_gemini_raw_corrected.json`
- Augmented output: `<run_id>_augmented.json`
- Session checkpoint: `<run_id>_augmented.json.session.json`

All artifacts live in the episode's `transcription/` directory.

## What Flash Lite Captures

Flash Lite re-transcribes from audio with the existing transcript as context.
It is particularly good at recovering:

- Short reaction lines (`うん`, `おお`, `えー`)
- Back-channel responses during phone calls or group conversations
- Overlapping dialogue that Gemini compresses into clean turns
- Filler and interjections that Gemini omits for readability

It is NOT reliable for:
- Preserving original line boundaries (it re-transcribes fully)
- Visual annotations (`[画面: ...]`) which come from video context
- Lines derived from on-screen text rather than speech

This is why the merge is done externally by Codex rather than trusting Flash
Lite's output directly.

## Trigger Phrases

- augment transcript with audio
- merge flash lite additions
- run audio gap-fill before alignment
- recover missing dialogue from audio
