# NVIDIA NeMo Speaker Diarization on AMD ROCm

This guide details how to run NVIDIA NeMo's speaker diarization on AMD GPUs using ROCm, bypassing the deep dependency conflicts that often occur when trying to use NeMo's full LLM stack or other tools like Pyannote.

## The Strategy

NeMo's LLM components require NVIDIA-specific hardware libraries (Apex, Transformer Engine, Flash Attention). However, **NeMo's ASR (Speech) domain**—which includes diarization—relies almost entirely on standard PyTorch operations and `torchaudio`. 

By strictly installing `nemo_toolkit[asr]` into an isolated environment, we can utilize NeMo's advanced `MarbleNet` (VAD) and `TitaNet` (Speaker Embeddings) natively on AMD hardware. Furthermore, by using the **NeuralDiarizer** class, we gain access to the **Multi-Scale Diarization Decoder (MSDD)**, which uses neural sequencing to detect overlapping speech (crucial for Japanese TV "Aizuchi" reactions).

*Note: NeMo ASR text generation (transcription) via beam search is broken on ROCm due to low-level tensor kernel exceptions (`gatherTopK`). However, the math used for VAD and Diarization is stable and lightning-fast on ROCm.*

## Prerequisites

1.  **Audio Requirements:** NeMo's VAD and Clustering models expect **16kHz mono WAV** files. Feed it anything else and the pipeline will fail or produce garbage.
2.  **UV Package Manager:** We use `uv` for extremely fast, reliable virtual environment creation.

## Setup Instructions

Run these commands from the root of your pipeline to create an isolated `.venv-nemo` environment.

```bash
# 1. Create a Python 3.12 environment
# (Python 3.14 breaks the tokenizers rust build)
uv venv .venv-nemo --python 3.12

# 2. Install PyTorch ROCm 7.2 wheels
uv pip install --python .venv-nemo --extra-index-url https://download.pytorch.org/whl/nightly/rocm7.2 torch torchvision torchaudio

# 3. Install strictly the ASR components of NeMo to bypass LLM dependency hell
uv pip install --python .venv-nemo --no-deps "git+https://github.com/NVIDIA/NeMo.git#egg=nemo_toolkit[asr]"

# 4. Install the required dependencies for NeMo ASR and Kotoba Whisper
uv pip install --python .venv-nemo wget text-unidecode scipy scikit-learn omegaconf pytorch-lightning librosa "transformers<4.40" hydra-core wrapt pydantic "lightning" accelerate soundfile 
```

## Running the Pipeline

A dedicated drop-in replacement script has been created: `scripts/kotoba_test/run_nemo_diarized.py`. 

It expects the `diarizer_config.yaml` to be present in your root directory (this contains NeMo's internal hyperparameters for clustering and windowing).

### 1. Pre-process the audio
Convert your source audio to 16kHz mono:

```bash
ffmpeg -i input_video.mp4 -ar 16000 -ac 1 temp_audio_16k.wav
```

### 2. Run Diarization + Transcription

```bash
.venv-nemo/bin/python scripts/kotoba_test/run_nemo_diarized.py 
  --audio temp_audio_16k.wav 
  --out-vtt output_diarized.vtt
```

## How it works

The script executes in three phases:

1.  **Phase 1 (NeMo):** Generates a temporary `nemo_manifest.json` pointing to your audio file. It loads the `diarizer_config.yaml` to initialize NeMo's `NeuralDiarizer` and runs MarbleNet (VAD) -> TitaNet (Embeddings) -> MSDD. It outputs standard `.rttm` files to a `nemo_output` directory.
2.  **Phase 2 (Kotoba):** Initializes standard `transformers` pipeline for Kotoba Whisper and transcribes the audio, generating the 15-second timestamp chunks.
3.  **Phase 3 (Merge):** Parses NeMo's RTTM output and assigns the generated speaker labels based on the temporal midpoint of your Whisper chunks.

### Tuning for Japanese Variety Shows

If your NeMo diarizer only finds **1 speaker** across an entire episode, you have fallen into the BGM clustering trap.

**1. Aggressive VAD is Mandatory**
In perfectly clean telephone recordings, NeMo's default `onset: 0.1` VAD threshold is fine. On a variety show, the constant background music and laugh tracks will trick a low-threshold VAD into classifying the entire episode as one continuous block of speech.
When the embedding extractor processes this, the dominant acoustic signature is the music, not the voices, causing the clustering algorithm to group everything into "Speaker 1".
*   **Fix:** Edit `diarizer_config.yaml` and set `onset: 0.8` and `offset: 0.6` under the `vad.parameters` block.

**2. Use NeuralDiarizer, not ClusteringDiarizer**
The NeMo `diarizer_config.yaml` includes an `msdd_model` block. However, if your Python script uses `from nemo.collections.asr.models import ClusteringDiarizer`, it will completely ignore the MSDD block and use basic single-speaker clustering.
*   **Fix:** You must instantiate `NeuralDiarizer(cfg=config)`. Also, ensure `save_embeddings: True` is set in your config, as MSDD relies on reading the intermediate clustering vectors from the disk to compute overlap probabilities.

### Troubleshooting

*   **SyntaxError in script:** Make sure you are using `.venv-nemo/bin/python` (Python 3.12).
*   **Garbage output:** Ensure your input audio was downsampled to exactly `16000 Hz` and is `mono` (1 channel).
*   **Missing `[UNKNOWN]` segments:** In highly chaotic panel shows with overlapping audio, Kotoba may generate text chunks that span across gaps where NeMo found no dominant speaker.