import torch
from transformers import pipeline
import time
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to audio file")
    args = parser.parse_args()

    model_id = "kotoba-tech/kotoba-whisper-v2.2"
    print(f"Loading {model_id} via transformers pipeline...")
    
    # Initialize the pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        torch_dtype=torch.float16,
        device="cuda",
        model_kwargs={"attn_implementation": "sdpa"}
    )
    
    print(f"Processing {args.audio}...")
    start_time = time.time()
    
    # Kotoba whisper specifically recommends chunk_length_s=15 and batch_size=16 for v2.2
    # It also requires return_timestamps=True for segment level timestamps
    result = pipe(
        args.audio,
        chunk_length_s=15,
        batch_size=16,
        return_timestamps=True,
        generate_kwargs={"language": "japanese", "task": "transcribe"}
    )
    
    end_time = time.time()
    print(f"\nTranscription completed in {end_time - start_time:.2f} seconds")
    
    print("\n--- Transcription ---")
    print(result["text"])
    
    print("\n--- Segments ---")
    if "chunks" in result:
        for chunk in result["chunks"]:
            print(f"[{chunk['timestamp'][0]:.2f} -> {chunk['timestamp'][1]:.2f}] {chunk['text']}")

if __name__ == "__main__":
    main()
