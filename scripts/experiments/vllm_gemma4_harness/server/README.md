# vLLM ROCm server recipes

Running multimodal models on the RX 9070 XT via vLLM's OpenAI-compatible
endpoint.

## Image

The base `vllm/vllm-openai-rocm:nightly` image ships with transformers 4.57
which does NOT recognize `model_type=gemma4`. This directory's Dockerfile
builds a thin layer that:

1. Installs `vllm[audio]` extras (pulls av/librosa/soundfile/resampy/soxr
   — needed for the `input_audio` content-part path). This pins
   transformers<5 as a side effect.
2. Upgrades transformers to 5.5.3 which recognizes `gemma4`.

**Install order matters.** Doing it in reverse downgrades transformers
back to 4.57 and audio requests return `HTTP 500 "Please install
vllm[audio]"`. There is a dep-resolver warning about the vllm<5 pin
being violated; runtime paths (model load + audio generation) work fine
despite the warning.

## Build

```bash
cd scripts/experiments/vllm_gemma4_harness/server
docker build -t vllm-gemma4-rocm:local .
```

## Run

```bash
# E2B, ~12.5 GiB VRAM footprint, fits comfortably on 16 GB
docker run -d --name vllm_gemma4 \
  --device /dev/kfd --device /dev/dri \
  --group-add video --ipc host \
  -e HF_HUB_CACHE=/mnt/models/huggingface \
  -e ROCR_VISIBLE_DEVICES=0 \
  -v /mnt/models/huggingface:/mnt/models/huggingface \
  -p 127.0.0.1:8000:8000 \
  vllm-gemma4-rocm:local \
  --model google/gemma-4-E2B-it \
  --max-model-len 8192 \
  --enable-prefix-caching \
  --attention-backend TRITON_ATTN
```

Notes:
- `ROCR_VISIBLE_DEVICES=0` hides the Raphael iGPU (gfx1036). Without
  it vLLM spreads the model across both GPUs and crashes.
- `TRITON_ATTN` is required: Gemma 4's heterogeneous head dimensions
  (`head_dim=256`, `global_head_dim=512`) aren't supported by the
  default backend.
- Prefix caching is on by default in recent vLLM, but we set it
  explicitly so the config is self-documenting.
- Health check: `curl http://127.0.0.1:8000/v1/models`
- Metrics (prefix cache hit rate etc.): `curl http://127.0.0.1:8000/metrics`

## E4B

Won't fit in 16 GB bf16. See `docs/gemma4-quant-calibration.md` for the
quantization path that would unlock it.

## MiniCPM-o 4.5 AWQ

Validated on `2026-04-11` with the same image on the RX `9070 XT` (`16 GB`).

This worked:

```bash
docker run --rm --name vllm_minicpm_awq \
  --device /dev/kfd --device /dev/dri \
  --group-add video --ipc host \
  -e HF_HUB_CACHE=/mnt/models/huggingface \
  -e ROCR_VISIBLE_DEVICES=0 \
  -e VLLM_USE_TRITON_AWQ=1 \
  -v /mnt/models/huggingface:/mnt/models/huggingface \
  -p 127.0.0.1:8001:8000 \
  vllm-gemma4-rocm:local \
  --model openbmb/MiniCPM-o-4_5-awq \
  --quantization awq \
  --dtype float16 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --max-model-len 4096 \
  --served-model-name minicpm_awq \
  --attention-backend TRITON_ATTN
```

Notes:

- `--dtype float16` was required. `--dtype auto` failed with:
  `torch.bfloat16 is not supported for quantization method awq`.
- `VLLM_USE_TRITON_AWQ=1` was set explicitly. This image (`vllm
  0.19.1rc1.dev203+g0f3ce4c74`) successfully loaded the model with AWQ on
  ROCm and held it at roughly `10.9 GiB` VRAM after startup.
- The server came up cleanly on `http://127.0.0.1:8001/v1/models` and
  served `minicpm_awq`.
- First smoke-test quality on `Killah Kuts S01E01` was poor, so this is a
  runtime validation, not a recommendation to use the model for Japanese
  subtitle transcription.

Quick checks:

```bash
curl http://127.0.0.1:8001/v1/models
curl http://127.0.0.1:8001/metrics
```

The existing harness can target this server:

```bash
python3 scripts/experiments/vllm_gemma4_harness/run_bench.py \
  --spec scripts/experiments/vllm_gemma4_harness/eval_specs/killah_kuts_s01e01.json \
  --out /tmp/killah_kuts_s01e01_minicpm_awq_smoke.json \
  --model minicpm_awq \
  --base-url http://127.0.0.1:8001/v1/chat/completions \
  --configs A_audio_base
```
