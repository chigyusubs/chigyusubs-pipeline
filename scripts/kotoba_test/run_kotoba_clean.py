import torch
from transformers import pipeline
import time
import argparse
import os

def format_timestamp(seconds: float) -> str:
    """Convert seconds to VTT timestamp HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

def main():
    parser = argparse.ArgumentParser(description="Run Kotoba-Whisper-v2.2 with a glossary for transcription.")
    parser.add_argument("--audio", required=True, help="Path to the input audio file (WAV/MP3/etc.)")
    parser.add_argument("--out-vtt", required=True, help="Path to save the output VTT file")
    parser.add_argument("--initial-prompt-file", help="Path to text file containing the initial prompt/glossary")
    args = parser.parse_args()

    initial_prompt = None
    if args.initial_prompt_file and os.path.exists(args.initial_prompt_file):
        with open(args.initial_prompt_file, 'r', encoding='utf-8') as f:
            initial_prompt = f.read().strip()
        print(f"Loaded initial prompt ({len(initial_prompt)} chars): {initial_prompt[:50]}...")

    print("
--- Running Kotoba-Whisper-v2.2 ---")
    print("Loading model via transformers pipeline...")
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="kotoba-tech/kotoba-whisper-v2.2",
        torch_dtype=torch.float16,
        device="cuda",
        model_kwargs={"attn_implementation": "sdpa"}
    )
    
    start_time = time.time()
    
    # Configure generation parameters including the glossary if provided
    gen_kwargs = {"language": "japanese", "task": "transcribe"}
    if initial_prompt:
        gen_kwargs["prompt"] = initial_prompt

    print(f"Processing audio: {args.audio}")
    result = asr_pipe(
        args.audio, 
        chunk_length_s=15, # Recommended chunk size for v2.2
        batch_size=16, 
        return_timestamps=True, 
        generate_kwargs=gen_kwargs
    )
    
    elapsed = time.time() - start_time
    print(f"Transcription completed in {elapsed:.2f} seconds")

    # Format and save as VTT
    print("
--- Formatting Output ---")
    with open(args.out_vtt, "w", encoding="utf-8") as f:
        f.write("WEBVTT

")
        
        # Keep track of cues to avoid writing empty strings
        written_cues = 0
        for chunk in result.get("chunks", []):
            start = chunk['timestamp'][0]
            end = chunk['timestamp'][1]
            text = chunk['text'].strip()
            
            if text:
                f.write(f"{format_timestamp(start)} --> {format_timestamp(end)}
")
                f.write(f"{text}

")
                written_cues += 1

    print(f"Success! {written_cues} cues written to {args.out_vtt}")

if __name__ == "__main__":
    main()
