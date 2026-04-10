# ASR Model Comparison — 2026-04-10

Benchmark comparing local ASR models against Gemini on Japanese variety show audio.

**Test episode**: Killah Kuts S01E01 (43.5 min, multi-speaker, overlapping dialogue, telops)
**Chunking**: Silero VAD → semantic boundaries, 20s target / 30s max (123 chunks)
**GPU**: AMD RX 9070 XT (16GB VRAM, ROCm 7.2)

## Results

| Model | Params | Chars | Chars/sec | Wall time | RTF | Local | VRAM |
|---|---|---|---|---|---|---|---|
| Gemini (video+audio) | ? | 24,750 | 9.5 | API | API | No | — |
| Gemma4 E4B-it (audio) | 4B | 25,407 | 9.7 | 11.2 min | 0.26x | Yes | ~8 GB |
| Gemma4 E2B-it (video+audio) | 2B | 18,143 | 7.0 | 10.0 min | 0.23x | Yes | ~14.7 GB peak |
| Whisper large-v3 | 1.5B | 12,526 | 4.8 | 5.3 min | 0.12x | Yes | ~4 GB |
| Cohere Transcribe | 2B | 10,966 | 4.2 | 1.1 min | 0.03x | Yes | ~4 GB |
| Granite 4.0-1B | 1B | 7,252 | 2.8 | 3.4 min | 0.08x | Yes | ~2 GB |

## Coverage tiers

Two distinct groups emerge on variety show audio:

**High coverage (~10 chars/sec)**: Gemini, Gemma4 4B audio. These capture reactions, crosstalk, fillers, and overlapping speech that traditional ASR drops.

**Low coverage (~4-5 chars/sec)**: Whisper, Cohere, Granite. These transcribe the dominant speaker cleanly but miss roughly half the spoken content.

Gemma4 2B video+audio sits in between (7.0 c/s) — the smaller model has less capacity, though video input helps.

## Per-model notes

### Gemini (baseline)
- Best proper nouns (設楽統, みなみかwa, ダイアン津田)
- Captures on-screen text `[画面: ...]` for free via video input
- No hallucination, clean formatting
- API-only, highest cost per episode

### Gemma4 4B audio-only
- **Highest local coverage** — matches Gemini's character density
- Garbles proper nouns (下田尾サム for 設楽統, 道尾 for 道雄)
- Inconsistent word-level spacing in some chunks (tokenizer artifacts)
- Occasional looping/repetition (chunk 90)
- Good at reactions and fillers that other models miss
- **Best candidate for second-opinion / gap-fill source**

### Gemma4 2B video+audio
- Reads telops: 8 `[画面:]` captures across the episode
- 720p at 0.5fps, capped at 10 frames per chunk fits in 16GB VRAM (14.7GB peak)
- Correct "スポーツスタンガン" where audio-only got "ポーズスタンガン"
- Hallucinates more than 4B — invented entire MC speeches, last chunk produced 1233 chars for 11s
- Video on the 4B model would be ideal but OOMs on 16GB (vision tower + audio + 4B weights)
- Also hit MIOpen kernel error on vision tower with 4B (gfx1201/RDNA4 compatibility)

### Whisper large-v3
- Drops ~50% of speech — misses entire opening, skips reactions/crosstalk
- Most accurate per-word on what it captures
- Clean proper nouns (梅木, ダイアン津田)
- Minimal hallucination
- `condition_on_previous_text=False` required (otherwise loops on long files)
- Needs numba shim due to numpy 2.4 incompatibility

### Cohere Transcribe
- **Fastest by far** — 67s for 43 min (33x real-time)
- Similar coverage to Whisper but in continuous flow (less segmented)
- Clean output, good proper nouns (設楽治, 道雄, 大崎)
- No speaker turn markers (pure ASR model, ignores prompt instructions)
- No punctuation (wall-of-text output)
- **Best whisper pre-pass replacement candidate** — faster, comparable quality
- Worth exploring: tuning to reduce occasional output truncation

### Granite 4.0-1B
- Accurate on clean single-speaker audio
- Loops badly on noisy multi-speaker sections ("これやっぱり道を短期決戦..." repeats)
- Lowest coverage (2.8 chars/sec) — too small for variety show audio
- Good punctuation when it works
- GGUF/CrispASR binary has UTF-8 encoding bug for Japanese — used HF transformers instead

## Technical notes

### Environment setup
- All local models need `ROCR_VISIBLE_DEVICES=0` before torch import to hide the iGPU (Ryzen 9800X3D integrated Radeon)
- Numba/numpy 2.4 incompatibility: whisper and Cohere feature extractor import librosa→numba. Fix: fake numba module shim before import, or use soundfile instead of librosa where possible
- `device_map="auto"` should be avoided — it spreads across both GPUs. Use explicit `.to("cuda:0")`
- Memory leak fix: `del inputs, outputs; torch.cuda.empty_cache()` after each chunk generation

### Gemma4 video input (no torchcodec needed)
torchcodec is incompatible with the ROCm PyTorch dev build (`libc10_cuda.so` missing). Workaround: extract frames with ffmpeg as JPEG, load as PIL images, and pass as `{"type": "image", "image": <PIL.Image>}` in the message content list. This bypasses the video processor entirely.

### Scripts
All experiment scripts in `scripts/experiments/`:
- `transcribe_gemma4_audio.py` — Gemma4 4B audio-only
- `transcribe_gemma4_video.py` — Gemma4 2B video+audio (frames + audio)
- `transcribe_cohere_audio.py` — Cohere Transcribe 2B
- `transcribe_granite_audio.py` — Granite 4.0-1B Speech

## Recommendations

1. **Whisper pre-pass replacement**: Cohere Transcribe — 5x faster, similar coverage, cleaner output
2. **Second-opinion / gap-fill**: Gemma4 4B audio — only local model matching Gemini coverage
3. **Video+audio transcription**: promising but needs >16GB VRAM for the 4B model. Revisit with quantized 4B or GPU upgrade
4. **Not recommended for this use case**: Granite 1B (too small for multi-speaker)
