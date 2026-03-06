import torch
from transformers import pipeline
from pyannote.audio import Pipeline
import time
import argparse
import os
from dotenv import load_dotenv

load_dotenv()

def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--hf-token", help="Hugging Face token for Pyannote")
    parser.add_argument("--out-vtt", default="output_diarized.vtt", help="Output VTT file path")
    parser.add_argument("--initial-prompt-file", help="Path to text file containing the initial prompt/glossary")
    args = parser.parse_args()

    if not args.hf_token:
        args.hf_token = os.environ.get("HF_TOKEN")
        if not args.hf_token:
             print("ERROR: --hf-token is required.")
             return

    initial_prompt = None
    if args.initial_prompt_file and os.path.exists(args.initial_prompt_file):
        with open(args.initial_prompt_file, 'r', encoding='utf-8') as f:
            initial_prompt = f.read().strip()
        print(f"Loaded initial prompt ({len(initial_prompt)} chars): {initial_prompt[:50]}...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n--- Phase 1: Speaker Diarization ---")
    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=args.hf_token)
    diarization_pipeline.to(device)
    
    start_time = time.time()
    diarization = diarization_pipeline(args.audio)
    
    speakers = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speakers.append({"start": turn.start, "end": turn.end, "speaker": speaker})
    print(f"Diarization completed in {time.time() - start_time:.2f} seconds. Found {len(speakers)} segments.")

    del diarization_pipeline
    torch.cuda.empty_cache()

    print("\n--- Phase 2: Transcription ---")
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="kotoba-tech/kotoba-whisper-v2.2",
        torch_dtype=torch.float16,
        device="cuda",
        model_kwargs={"attn_implementation": "sdpa"}
    )
    
    start_time = time.time()
    gen_kwargs = {"language": "japanese", "task": "transcribe"}
    if initial_prompt:
        gen_kwargs["prompt"] = initial_prompt

    result = asr_pipe(
        args.audio, chunk_length_s=15, batch_size=16, return_timestamps=True, generate_kwargs=gen_kwargs
    )
    print(f"Transcription completed in {time.time() - start_time:.2f} seconds")

    print("\n--- Phase 3: Merging & Formatting ---")
    cues = []
    for chunk in result.get("chunks", []):
        start = chunk['timestamp'][0]
        end = chunk['timestamp'][1]
        text = chunk['text'].strip()
        
        chunk_midpoint = start + ((end - start) / 2)
        current_speaker = "UNKNOWN"
        for spk in speakers:
            if spk["start"] <= chunk_midpoint <= spk["end"]:
                current_speaker = spk["speaker"]
                break
                
        cues.append({"start": start, "end": end, "text": text, "speaker": current_speaker})

    with open(args.out_vtt, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        last_speaker = None
        for cue in cues:
            f.write(f"{format_timestamp(cue['start'])} --> {format_timestamp(cue['end'])}\n")
            prefix = f"- [{cue['speaker']}] " if cue["speaker"] != last_speaker else ""
            last_speaker = cue["speaker"]
            f.write(f"{prefix}{cue['text']}\n\n")

    print(f"Success! Output written to {args.out_vtt}")

if __name__ == "__main__":
    main()