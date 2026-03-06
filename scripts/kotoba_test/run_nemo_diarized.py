import torch
import time
import argparse
import os
import json
from transformers import pipeline
from omegaconf import OmegaConf
from nemo.collections.asr.models import NeuralDiarizer

def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

def parse_rttm(rttm_path):
    speakers = []
    with open(rttm_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                start = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                speakers.append({"start": start, "end": start + duration, "speaker": speaker})
    return speakers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to audio file (must be 16kHz mono wav ideally)")
    parser.add_argument("--out-vtt", default="output_diarized.vtt", help="Output VTT file path")
    parser.add_argument("--initial-prompt-file", help="Path to text file containing the initial prompt/glossary")
    args = parser.parse_args()

    audio_path = os.path.abspath(args.audio)
    if not os.path.exists(audio_path):
        print(f"ERROR: Audio file not found at {audio_path}")
        return

    initial_prompt = None
    if args.initial_prompt_file and os.path.exists(args.initial_prompt_file):
        with open(args.initial_prompt_file, 'r', encoding='utf-8') as f:
            initial_prompt = f.read().strip()
        print(f"Loaded initial prompt ({len(initial_prompt)} chars): {initial_prompt[:50]}...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n--- Phase 1: NeMo Speaker Diarization ---")
    start_time = time.time()

    # Create NeMo manifest
    manifest_path = os.path.abspath("nemo_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump({
            "audio_filepath": audio_path,
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "num_speakers": None,
            "rttm_filepath": None,
            "uem_filepath": None
        }, f)
        f.write('\n')
    
    # Load diarizer config
    out_dir = os.path.abspath("nemo_output")
    os.makedirs(out_dir, exist_ok=True)
    
    config = OmegaConf.load("diarizer_config.yaml")
    config.diarizer.manifest_filepath = manifest_path
    config.diarizer.out_dir = out_dir
    # MSDD requires embeddings to be saved to disk to function
    config.diarizer.speaker_embeddings.parameters.save_embeddings = True
    
    # Run Diarization
    diarizer = NeuralDiarizer(cfg=config)
    diarizer.diarize()
    
    # Parse RTTM
    audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
    rttm_path = os.path.join(out_dir, "pred_rttms", f"{audio_basename}.rttm")
    
    if not os.path.exists(rttm_path):
        # Fallback to the first available RTTM if exact basename isn't matched
        rttm_files = [f for f in os.listdir(os.path.join(out_dir, "pred_rttms")) if f.endswith('.rttm')]
        if not rttm_files:
            print("ERROR: Diarization failed to produce an RTTM file.")
            return
        rttm_path = os.path.join(out_dir, "pred_rttms", rttm_files[0])
        
    speakers = parse_rttm(rttm_path)
    print(f"Diarization completed in {time.time() - start_time:.2f} seconds. Found {len(speakers)} segments.")

    # Cleanup memory
    del diarizer
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
        audio_path, chunk_length_s=15, batch_size=16, return_timestamps=True, generate_kwargs=gen_kwargs
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