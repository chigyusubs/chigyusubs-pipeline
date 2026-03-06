import torch
import sys
import argparse
import json
from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="scerz/VibeVoice-ASR-4bit")
    parser.add_argument("--audio_path", type=str, help="Path to audio file (optional)")
    args = parser.parse_args()

    print(f"Loading processor for {args.model_path}...")
    processor = VibeVoiceASRProcessor.from_pretrained(
        args.model_path,
        language_model_pretrained_name="Qwen/Qwen2.5-7B"
    )

    print(f"Loading model {args.model_path}...")
    
    # Allow model to load using bitsandbytes inherently if saved in 4-bit
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=True
    )
    model.eval()

    print("Model loaded successfully!")

    if not args.audio_path:
        print("Run with --audio_path <file> to transcribe.")
        return

    print(f"Transcribing {args.audio_path}...")
    inputs = processor(
        audio=[args.audio_path],
        sampling_rate=None,
        return_tensors="pt",
        padding=True,
        add_generation_prompt=True
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            pad_token_id=processor.pad_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            do_sample=False
        )

    input_length = inputs['input_ids'].shape[1]
    generated_ids = output_ids[0, input_length:]
    
    # Remove padding/eos
    eos_positions = (generated_ids == processor.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
    if len(eos_positions) > 0:
        generated_ids = generated_ids[:eos_positions[0] + 1]

    generated_text = processor.decode(generated_ids, skip_special_tokens=True)
    
    print("\n--- Raw Output ---")
    print(generated_text)
    
    try:
        segments = processor.post_process_transcription(generated_text)
        print("\n--- Structured Output ---")
        print(json.dumps(segments, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"\nFailed to parse structured output: {e}")

if __name__ == "__main__":
    main()
