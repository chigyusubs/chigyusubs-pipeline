import argparse
import os
import json
import librosa
import numpy as np
import tempfile
import soundfile as sf
from tqdm import tqdm

import reazonspeech.nemo.asr as asr

def create_vtt(segments, out_path):
    def format_time(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            start = format_time(seg["start"])
            end = format_time(seg["end"])
            text = seg["text"].strip()
            if text:
                f.write(f"{start} --> {end}\n{text}\n\n")


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using reazonspeech-nemo-v2 with chunking and batching")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--out-dir", required=True, help="Directory to save the transcript")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for transcription")
    
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("Loading audio and splitting by silence...")
    y, sr = librosa.load(args.audio, sr=16000)
    intervals = librosa.effects.split(y, top_db=30)
    
    # Filter short intervals (e.g. < 0.2s) which crash RNN-T ALSD decoding
    min_samples = int(0.2 * sr)
    intervals = [inter for inter in intervals if (inter[1] - inter[0]) >= min_samples]
    
    print(f"Total chunks to process: {len(intervals)}")
    
    print("Loading ReazonSpeech NeMo model...")
    model = asr.load_model()
    
    # Change decoding to greedy to avoid ROCm kernel crashes with ALSD beam search
    from omegaconf import DictConfig
    model.change_decoding_strategy(DictConfig({"strategy": "greedy", "greedy": {"max_symbols_per_step": 10}}))
    
    tmp_dir = tempfile.mkdtemp()
    chunk_paths = []
    chunk_start_times = []
    
    print(f"Saving {len(intervals)} chunks to {tmp_dir}...")
    for i, (start_idx, end_idx) in enumerate(intervals):
        chunk_y = y[start_idx:end_idx]
        chunk_start_time = start_idx / sr
        
        path = os.path.join(tmp_dir, f"chunk_{i:04d}.wav")
        sf.write(path, chunk_y, sr)
        
        chunk_paths.append(path)
        chunk_start_times.append(chunk_start_time)
        
    print(f"Transcribing {len(chunk_paths)} chunks in batches of {args.batch_size}...")
    # model.transcribe is from nemo EncDecRNNTBPEModel
    # It returns a tuple if return_hypotheses=True: (hypotheses, all_hypotheses)
    result = model.transcribe(chunk_paths, batch_size=args.batch_size, return_hypotheses=True)
    
    hypotheses = result[0] if isinstance(result, tuple) else result
    
    all_segments = []
    
    for i, hyp in enumerate(hypotheses):
        base_time = chunk_start_times[i]
        
        # We use reazonspeech's decode_hypothesis to get properly formatted sub-segments if possible,
        # but since that requires internal model logic, we can also just use the global text for the chunk
        # as a single segment for simplicity, or we can use decode_hypothesis.
        # Let's use decode_hypothesis from reazonspeech internals to get proper sub-timestamps!
        # wait, decode_hypothesis is not exported.
        # Let's just create a single segment for the whole chunk!
        # Or estimate end time based on the original audio duration
        duration = librosa.get_duration(path=chunk_paths[i])
        
        text = hyp.text.strip()
        if text:
            all_segments.append({
                "start": base_time,
                "end": base_time + duration,
                "text": text
            })
            
    # Cleanup tmp dir
    for path in chunk_paths:
        try:
            os.remove(path)
        except:
            pass
    try:
        os.rmdir(tmp_dir)
    except:
        pass

    vtt_file = os.path.join(args.out_dir, "transcript.vtt")
    json_file = os.path.join(args.out_dir, "transcript.json")
    
    print(f"Saving transcripts to {args.out_dir}...")
    create_vtt(all_segments, vtt_file)
    
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump({"segments": all_segments}, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
