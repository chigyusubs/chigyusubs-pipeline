# Transcription Model Comparison

Benchmark results from testing alternative transcription models against the
Gemini Flash video transcription pipeline on Great Escape S02E01 (2344s).

## Results Summary

| Model | Type | Norm chars | Wall time | RTF | Failures | Hallucination |
|---|---|---|---|---|---|---|
| **Gemini 2.5 Flash** | API, video | 9,407 | API | API | 0 | Low |
| **Whisper large-v3** | Local, audio | 8,805 | 182s | 0.08x | 0 | Low |
| **Qwen3-ASR 1.7B** | Local, audio | 8,535 | 541s | 0.23x | 0 | Low |
| **Qwen3-Omni 30B-A3B** | Local, audio | 7,653 (clean) | 676s | 0.29x | 0 | Mild (9/69 tail repeats) |
| **MiMo v2 Omni** (video) | API, video | 7,079 (clean) | 1,044s | 0.45x | 13/69 loops | Severe |
| **MiMo v2 Omni** (audio) | API, audio | 11,230 (clean) | 1,150s | 0.49x | 9/69 loops | Severe |

Normalized chars = after stripping punctuation, speaker markers, visual markers.

## Xiaomi MiMo v2 Omni

**API**: OpenAI-compatible at `https://api.xiaomimimo.com/v1`, model `mimo-v2-omni`.
Requires `XIAOMI_API_KEY` in `.env`.

### Video mode (`scripts/transcribe_mimo_video.py`)

- Sends base64-encoded compressed video chunks (1fps, crf36, 24k audio)
- `frequency_penalty=1.0` to combat hallucination loops
- `--no-thinking` flag required — reasoning mode consumes all completion tokens
- Chunk target 30s / max 45s
- 13/69 chunks hit hallucination loops (repeated phrases 600+ times)
- Content filter triggered on some chunks with thinking enabled
- `_truncate_repetition()` post-processing catches exact consecutive matches
- Audio tokens ~6.25/sec; video tokens are much higher (expensive)

### Audio mode (`scripts/transcribe_mimo_audio.py`)

- Sends base64-encoded MP3 chunks at 64kbps
- `frequency_penalty=0.5`
- Same `--no-thinking` requirement
- 9/69 chunks hit hallucination loops (slightly better than video)
- Occasional 308s API latency spikes on individual chunks

### Verdict

Not viable for Japanese variety show transcription. Hallucination loops are
a fundamental issue — the model generates repeated phrases hundreds of times
when it loses track of the audio. Neither frequency penalty, shorter chunks,
nor audio-only mode fully resolves this. Cost is also high due to API token
usage per chunk.

## Qwen3-ASR 1.7B

**Package**: `qwen-asr` from HuggingFace. Script: `scripts/transcribe_qwen_asr.py`.

- Pure ASR model, ~1.7B params, runs on GPU in bfloat16
- Faithful phonetic transcription, similar to Whisper
- No speaker turns, no punctuation structure
- Outputs as single continuous text block (no line breaks)
- Chunk target 90s / max 120s works well (model handles longer audio)
- `max_new_tokens=2048` needed for full-length chunks
- 0.23x RTF — slower than Whisper but no failures

### Notes

- Using `vad_chunks.json` (250-270s chunks) causes collapse/repetition on
  long chunks. Use VAD segments directly with shorter boundaries.
- Runs on system python3.12 with ROCm-enabled PyTorch.

## Qwen3-Omni 30B-A3B (GGUF via llama.cpp)

**Architecture**: 30B total params, 3B active (128-expert MoE). Q4_K_M quant
is 18.6GB. Runs via experimental llama.cpp fork with audio support.

**Setup**:
```bash
# Build from experimental branch
git clone --branch feature/qwen3-omni --depth 1 \
  https://github.com/TrevorS/llama.cpp.git /tmp/llama-cpp-qwen3-omni
cd /tmp/llama-cpp-qwen3-omni
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS="gfx1201" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Download model files
huggingface-cli download TrevorJS/Qwen3-Omni-30B-A3B-GGUF \
  thinker-q4_k_m.gguf mmproj-f16.gguf --local-dir ~/models/qwen3-omni-30b-a3b
```

**Files needed**:
- `thinker-q4_k_m.gguf` (18.6GB) — main model
- `mmproj-f16.gguf` (2.39GB) — audio+vision projector
- `talker-f16.gguf` (7.09GB) — TTS, not needed for transcription

**Memory layout** (RX 9070 XT, 16GB VRAM + 32GB RAM):
- 30/48 layers on GPU (~10.7GB VRAM)
- 18 layers on CPU via mmap (~7GB)
- Audio encoder ~553MB VRAM, vision encoder ~360MB VRAM
- MoE architecture handles CPU offload well — only 3B active params per token

**Performance**:
- Prompt processing: 240-511 tok/s
- Generation: 34.9 tok/s (audio-only), 26.2 tok/s (audio+image)
- RTF: 0.29x overall including model load per chunk

### Script: `scripts/transcribe_qwen3_omni.py`

```bash
python3.12 scripts/transcribe_qwen3_omni.py \
  --video "samples/episodes/<slug>/source/<video>.mp4" \
  --vad-json "samples/episodes/<slug>/transcription/silero_vad_segments.json" \
  --output "samples/episodes/<slug>/transcription/<slug>_qwen3_omni_raw.json" \
  --temp 0.0
```

Key flags:
- `--ngl 30` — GPU layers (default 30, fits 16GB VRAM with audio encoder)
- `--ctx-size 2048` — context size (default 2048)
- `--n-predict 512` — max generation tokens (default 512)
- `--temp 0.2` — sampling temperature (default 0.2; temp=0.0 tested but increases repetition loops from 9/69 to 15/69)
- `--target-chunk-s 30 --max-chunk-s 45` — chunk boundaries

### Audio+Image mode

The model supports simultaneous audio and image input via `--audio` and `--image`
flags to `llama-mtmd-cli`. Each image adds ~2040 tokens. Potential use: feed a
midpoint keyframe per chunk for telop reading. Not yet scripted for batch use.

### Quality characteristics

**Strengths**:
- Zero catastrophic failures (vs MiMo's 13-19% loop rate)
- Natural Japanese sentence structure with proper punctuation
- LLM world knowledge helps with coherence and context
- MoE makes it fast despite 30B total params

**Weaknesses**:
- LLM "correction" behavior changes what was actually said
  - おかん (mom, Kansai dialect) → 奥さん (wife)
  - Proper nouns get substituted with plausible-sounding alternatives
- Mild tail repetition on 9/69 chunks at temp=0.2 (caught by `_truncate_repetition()`)
- Repetition worsens at temp=0.0 (15/69 chunks) — greedy decoding cannot escape loops
- Less faithful to audio than pure ASR models — problematic for CTC alignment
- Fewer captured utterances than Whisper (~81% of Gemini's normalized content)

### Verdict

Interesting model but not suitable as primary transcription source for the
alignment pipeline. The LLM correction behavior means CTC forced alignment
would align wrong text, producing bad timestamps or zero-duration words.

Potential alternative roles:
- Second-pass coherence reviewer on Whisper/Gemini output
- Audio+image telop reading (untested at batch scale)
- Transcription for cases where alignment is not needed

## Recommendations

For the current pipeline (OCR → glossary → ASR → reflow → diarization → translation):

1. **Primary**: Gemini Flash (video) — best quality, telops, speaker turns
2. **Local fallback**: Whisper large-v3 via faster-whisper — most faithful ASR,
   fastest local (0.08x RTF), best for CTC alignment
3. **Not recommended**: MiMo (hallucination loops), Qwen3-Omni (LLM corrections
   break alignment), Qwen3-ASR (viable but slower than Whisper with no advantage)

## Test environment

- GPU: AMD RX 9070 XT (16GB VRAM, RDNA 4 / gfx1201, ROCm 7.2)
- CPU: Ryzen 9 5900X (12c/24t, cores 2/14 isolated due to MCE)
- RAM: 32GB
- Episode: Great Escape S02E01 (2344s)
- All local models used 30s target / 45s max chunks from Silero VAD segments
- faster-whisper needs `LD_LIBRARY_PATH="/opt/rocm-7.2.0/lib"` and
  `CT2_CUDA_ALLOCATOR=cub_caching` set before Python starts
