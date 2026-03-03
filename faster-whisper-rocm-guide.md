# Running faster-whisper on AMD GPUs (ROCm)

## System

- **OS**: Fedora 43
- **GPU**: AMD Radeon RX 9070 XT (gfx1201 / RDNA 4)
- **ROCm**: 7.2.0 (`/opt/rocm-7.2.0`)
- **Python**: 3.12

## Install

### 1. Install ROCm ctranslate2 wheel

The standard ctranslate2 pip package is CUDA-only. You need the ROCm-built wheel from [GitHub releases](https://github.com/OpenNMT/CTranslate2).

```bash
# Download rocm-python-wheels-Linux.zip and extract it
unzip rocm-python-wheels-Linux.zip

# Install the wheel matching your Python version
pip install temp-linux/ctranslate2-4.7.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
```

### 2. Install faster-whisper

```bash
pip install faster-whisper
```

## Run

Two environment variables are **required**:

```bash
export LD_LIBRARY_PATH=/opt/rocm-7.2.0/lib:$LD_LIBRARY_PATH
export CT2_CUDA_ALLOCATOR=cub_caching
```

- `LD_LIBRARY_PATH` — ctranslate2 links against ROCm libs (libhiprand, libamdhip64, etc.) which aren't on the default library path.
- `CT2_CUDA_ALLOCATOR=cub_caching` — **Required or it will crash.** The default allocator doesn't work properly with ROCm/HIP.

### Transcribe a file

```bash
LD_LIBRARY_PATH=/opt/rocm-7.2.0/lib:$LD_LIBRARY_PATH \
CT2_CUDA_ALLOCATOR=cub_caching \
python3.12 -c "
from faster_whisper import WhisperModel

model = WhisperModel('large-v3', device='cuda', compute_type='float16')

segments, info = model.transcribe(
    'audio.mp3',
    beam_size=5,
    language='en',
    vad_filter=True,
)

for segment in segments:
    print(f'[{segment.start:.2f} -> {segment.end:.2f}] {segment.text}')
"
```

### Or as a script (`transcribe.sh`)

```bash
#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH=/opt/rocm-7.2.0/lib:${LD_LIBRARY_PATH:-}
export CT2_CUDA_ALLOCATOR=cub_caching

python3.12 - "$@" <<'PYTHON'
import sys
from faster_whisper import WhisperModel

audio = sys.argv[1]
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
segments, info = model.transcribe(audio, beam_size=5, language="en", vad_filter=True)

print(f"Language: {info.language} ({info.language_probability:.0%}), Duration: {info.duration:.0f}s\n")

for seg in segments:
    print(f"[{seg.start:.2f} -> {seg.end:.2f}] {seg.text}")
PYTHON
```

```bash
chmod +x transcribe.sh
./transcribe.sh "my-audio.mp3"
```

## Notes

- The model downloads on first run (~3 GB for `large-v3`) and is cached in `~/.cache/huggingface/`.
- `vad_filter=True` skips silent sections, which speeds up long files significantly.
- `compute_type='float16'` is the sweet spot for speed on RDNA 4. Available types: `float16`, `bfloat16`, `float32`, `int8`, `int8_float16`, `int8_bfloat16`, `int8_float32`.
- For non-English audio, remove `language='en'` to let it auto-detect, or set to the appropriate language code.
- A ~3-hour podcast transcribes fully on this system.

## Troubleshooting

| Error | Fix |
|-------|-----|
| `ImportError: libhiprand.so.1: cannot open shared object file` | Set `LD_LIBRARY_PATH=/opt/rocm-7.2.0/lib` |
| Crash / segfault during inference | Set `CT2_CUDA_ALLOCATOR=cub_caching` |
| `No module named 'ctranslate2'` | Install the ROCm wheel, not the pip default |
