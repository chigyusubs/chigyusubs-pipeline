import argparse
import os
import json
import nemo.collections.asr as nemo_asr

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using reazonspeech-nemo-v2")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--out-dir", required=True, help="Directory to save the transcript")
    
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("Loading ReazonSpeech NeMo model...")
    asr_model = nemo_asr.models.ASRModel.from_pretrained("reazon-research/reazonspeech-nemo-v2")
    
    print(f"Transcribing {args.audio}...")
    
    hypotheses = asr_model.transcribe([args.audio], return_hypotheses=True)
    
    if isinstance(hypotheses, tuple):
        hypotheses = hypotheses[0]
        
    hypothesis = hypotheses[0]
    
    out_file = os.path.join(args.out_dir, "transcript.json")
    out_txt = os.path.join(args.out_dir, "transcript.txt")
    
    print(f"Transcription complete. Saving to {args.out_dir}...")
    
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(hypothesis.text)
        
    # Check if hypothesis has a timestamp field and handle it safely
    timestamps = []
    if hasattr(hypothesis, "timestamp"):
        try:
            # Depending on NeMo version, timestamp could be a list of ints or tensors.
            # Convert tensors to python ints if needed.
            ts_raw = hypothesis.timestamp
            timestamps = [int(t) for t in ts_raw]
        except Exception as e:
            print(f"Warning: could not process timestamps: {e}")
            
    data = {
        "text": hypothesis.text,
        "timestamps": timestamps
    }
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print("Done.")

if __name__ == "__main__":
    main()
