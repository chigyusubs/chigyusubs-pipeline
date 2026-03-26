# Chigyusubs Pipeline

Japanese variety show subtitle pipeline. Scripts in `scripts/`, episode data in `samples/episodes/<slug>/`.

## Pipeline order

```
whisper pre-pass → semantic chunking → gemini transcription → [audio augment] → OCR → glossary
→ CTC alignment → second opinion → reflow → [repair] → translation
```

Key artifacts per episode:
- `*_gemini_raw.json` → (`*_augmented.json`) → `*_ctc_words.json` → `*_reflow.vtt` → `*_en.vtt`
- Second opinion reports live in `transcription/diagnostics/`
- Pre-pass transcript doubles as second-opinion source (no redundant whisper run)

## Environment

- **Python**: Use system `python3.12` for all GPU work. `.venv-nemo` lacks ROCm access.
- **GPU**: AMD RX 9070 XT (gfx1201, RDNA 4). Officially supported in ROCm 7.2 — no `HSA_OVERRIDE_GFX_VERSION` needed.
- **ROCm env vars must be set before Python starts** (ctranslate2 loads shared libs at import time):
  ```
  LD_LIBRARY_PATH=/opt/rocm/lib CT2_CUDA_ALLOCATOR=cub_caching python3.12 ...
  ```
  Setting these via `os.environ` in Python is too late for ctranslate2 (used by CTC alignment).

## Architecture notes

- CTC alignment (`align_ctc.py`) is the default. Uses `NTQAI/wav2vec2-large-japanese` + `torchaudio.functional.forced_align`.
- When filtering CTC segments per chunk, assign by **start time** (`seg.start >= chunk_start and seg.start < chunk_end`), not by overlap. Overlap-based filtering causes segment leak across chunk boundaries.
- Whisper pre-pass uses openai-whisper (not faster-whisper — faster-whisper silently stops producing segments partway through long files). Uses `condition_on_previous_text=False` with a consecutive-dupe strip as safety net.
- Shared transcript comparison utilities live in `chigyusubs/transcript_comparison.py`.
- Audio augmentation (`correct_transcript_flash_lite.py` + `augment_transcript_codex.py`) is an optional step between Gemini transcription and CTC alignment. Flash Lite re-transcribes from audio to recover missing reactions/interjections, then Codex merges additions into the original transcript. Updates `preferred.json` so alignment picks up the augmented file.
- Translation uses Codex (OpenAI) interactively via `translate_vtt_codex.py`. Turn context from alignment provides anonymous speaker boundaries for pronoun tracking.
