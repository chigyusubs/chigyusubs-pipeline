# Running VibeVoice-ASR (4-bit)

This guide explains how to set up and run the 4-bit quantized version of Microsoft's VibeVoice ASR model (`scerz/VibeVoice-ASR-4bit`). This version allows the model to run on GPUs with less VRAM (12GB - 16GB) while retaining the core features of the 7B parameter model.

## Prerequisites

The model requires a Python environment with PyTorch installed (e.g., your existing ROCm setup in `.venv-kotoba`). 

You will also need `bitsandbytes` for 4-bit loading and the official `vibevoice` library from Microsoft's GitHub.

### 1. Activate your environment
```bash
source .venv-kotoba/bin/activate
```

### 2. Install Dependencies
Install `bitsandbytes` and core Hugging Face dependencies:
```bash
pip install bitsandbytes accelerate sentencepiece
```

### 3. Install VibeVoice
The model relies on Microsoft's custom modular codebase. Install it directly from their GitHub repository:
```bash
pip install git+https://github.com/microsoft/VibeVoice.git
```

## Running the Model

A custom inference script has been created at `scripts/test_vibevoice_4bit.py`. This script automatically loads the 4-bit model, processes the audio, and outputs both raw text and structured JSON (which includes timestamps and speaker tags).

### Usage

Run the script by passing the path to an audio file:

```bash
python scripts/test_vibevoice_4bit.py --audio_path samples/experiments/intro/intro_test.wav
```

### Arguments
* `--audio_path`: (Required) The path to the audio file you want to transcribe (e.g., `.wav`, `.mp3`).
* `--model_path`: (Optional) The Hugging Face model ID or local path. Defaults to `"scerz/VibeVoice-ASR-4bit"`.

## Notes and Caveats

- **ROCm Compatibility:** The script is configured to use `attn_implementation="sdpa"` (Scaled Dot-Product Attention) which works smoothly on AMD/ROCm GPUs without requiring Flash Attention 2.
- **Hallucinations on Silence:** Because VibeVoice is built on top of a Large Language Model (Qwen2.5-7B), it may aggressively hallucinate repetitive text when encountering long periods of silence, intro music, or non-speech noise. You may need to preprocess the audio (e.g., with Voice Activity Detection or Demucs) to isolate speech before feeding it to the model.
- **First Run:** The first time you run the script, it will download the ~7.5GB quantized model weights and tokenizer configurations from Hugging Face to your local `.cache`. Subsequent runs will be much faster.