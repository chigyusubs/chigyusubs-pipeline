#!/usr/bin/env python3
"""Quantize google/gemma-4-E4B-it to FP8 dynamic via llm-compressor.

Produces a compressed-tensors checkpoint that vLLM can serve directly.
Audio tower, vision tower, multimodal projectors, and lm_head stay in
bf16; only the LM backbone Linears are quantized to FP8.

No calibration data needed — FP8_DYNAMIC computes scales at save time.

Usage (from the llmc venv):
    /tmp/llmc-venv/bin/python \
        scripts/experiments/vllm_gemma4_harness/quantize_e4b_fp8.py

Output: /mnt/models/huggingface/gemma-4-E4B-it-FP8-Dynamic-local/
"""

import argparse
from pathlib import Path

MODEL_ID = "google/gemma-4-E4B-it"
OUTPUT_DIR = "/mnt/models/huggingface/gemma-4-E4B-it-FP8-Dynamic-local"

# Layers to keep in bf16 — must match leon-se's proven ignore pattern.
IGNORE = [
    "lm_head",
    "re:model.embed_audio.*",
    "re:model.embed_vision.*",
    "re:model.audio_tower.*",
    "re:model.vision_tower.*",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--output", default=OUTPUT_DIR)
    args = parser.parse_args()

    # Shim for transformers 5.x compat with llmcompressor 0.10
    import transformers.modeling_utils as _mu
    if not hasattr(_mu, "TORCH_INIT_FUNCTIONS"):
        _mu.TORCH_INIT_FUNCTIONS = {}

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from transformers import AutoProcessor, Gemma4ForConditionalGeneration

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} (bf16, CPU)...")
    model = Gemma4ForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="cpu",
    )

    recipe = QuantizationModifier(
        targets=["Linear"],
        ignore=IGNORE,
        scheme="FP8_DYNAMIC",
    )

    print("Applying FP8_DYNAMIC quantization via llm-compressor...")
    oneshot(model=model, recipe=recipe, output_dir=str(output))
    print(f"Quantized model saved to {output}")

    # Copy processor + tokenizer from the original google upload (clean).
    print("Saving processor + tokenizer...")
    processor = AutoProcessor.from_pretrained(args.model)
    processor.save_pretrained(str(output))
    print("Done.")


if __name__ == "__main__":
    main()
