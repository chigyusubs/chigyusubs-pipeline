import os
import subprocess
import json
from pathlib import Path
import easyocr
import logging

# Suppress warnings from easyocr
logging.getLogger("easyocr").setLevel(logging.ERROR)

def extract_frames(video_path: str, output_dir: str, max_fps: int = 2):
    """Extract frames if they don't already exist."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if we already have a significant number of frames
    existing_frames = list(Path(output_dir).glob("*.jpg"))
    if len(existing_frames) > 100:
        print(f"Found {len(existing_frames)} existing frames in {output_dir}. Skipping extraction.")
        return True
        
    print(f"Extracting frames from {video_path} to {output_dir}...")
    command = [
        "ffmpeg", "-i", video_path,
        "-vf", "fps=1/2",
        "-fps_mode", "vfr", "-qscale:v", "2",
        "-hide_banner", "-loglevel", "warning",
        os.path.join(output_dir, "frame_%04d.jpg")
    ]
    try:
        subprocess.run(command, check=True)
        num_frames = len(list(Path(output_dir).glob("*.jpg")))
        print(f"Extraction complete. Found {num_frames} unique frames.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg extraction failed: {e}")
        return False

def get_processed_frames(jsonl_path: str):
    """Read the checkpoint file to see which frames are already done."""
    processed = set()
    if not os.path.exists(jsonl_path):
        return processed
        
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                processed.add(data["frame"])
            except json.JSONDecodeError:
                pass
    return processed

def run_ocr(frames_dir: str, jsonl_path: str):
    """Run EasyOCR on unprocessed frames and stream to JSONL."""
    frame_files = sorted([f.name for f in Path(frames_dir).glob("*.jpg")])
    if not frame_files:
        print("No frames found to process.")
        return

    processed_frames = get_processed_frames(jsonl_path)
    frames_to_process = [f for f in frame_files if f not in processed_frames]
    
    if not frames_to_process:
        print("All frames have already been processed.")
        return

    print(f"Loading EasyOCR model to process {len(frames_to_process)} remaining frames...")
    reader = easyocr.Reader(['ja', 'en'])
    
    # Open file in append mode to stream results safely
    with open(jsonl_path, 'a', encoding='utf-8') as f:
        for i, frame_name in enumerate(frames_to_process):
            if i % 50 == 0:
                print(f"OCR Progress: {i}/{len(frames_to_process)} remaining frames...")
                
            frame_path = os.path.join(frames_dir, frame_name)
            results = reader.readtext(frame_path)
            
            # Filter and save immediately
            frame_text = []
            for (bbox, text, prob) in results:
                if prob > 0.3:
                    frame_text.append({"text": text, "prob": float(prob)})
            
            # Write a record even if empty, so we know we checked it
            record = {"frame": frame_name, "results": frame_text}
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            f.flush() # Force write to disk

def compile_glossary(jsonl_path: str, output_txt: str):
    """Read the JSONL, deduplicate, and create the final Whisper prompt."""
    print(f"Compiling final glossary from {jsonl_path}...")
    if not os.path.exists(jsonl_path):
        print("No OCR data found to compile.")
        return

    raw_texts = set()
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                for item in data.get("results", []):
                    raw_texts.add(item["text"])
            except json.JSONDecodeError:
                pass

    # 1. Filter garbage
    filtered = []
    ignore_list = ["字幕", "WE", "W", "MC", "T"]
    for text in raw_texts:
        text = text.strip()
        if len(text) <= 1:
            continue
        if text in ignore_list:
            continue
        filtered.append(text)
        
    # 2. Substring collapsing
    filtered.sort(key=len, reverse=True)
    collapsed = []
    for text in filtered:
        is_substring = False
        for longer_text in collapsed:
            if text in longer_text:
                is_substring = True
                break
        if not is_substring:
            collapsed.append(text)

    # 3. Save
    glossary_string = ", ".join(collapsed)
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write(glossary_string)
        
    print(f"Reduced from {len(raw_texts)} raw reads to {len(collapsed)} unique glossary terms.")
    print(f"Glossary successfully saved to {output_txt}")

if __name__ == "__main__":
    sample_video = "samples/WEDNESDAY_DOWNTOWN_2024-01-24 _#363.mp4"
    frames_dir = "samples/frames_workdir"
    ocr_jsonl = "samples/ocr_results.jsonl"
    output_prompt = "samples/whisper_prompt.txt"
    
    print("--- Stage 1: Extraction ---")
    success = extract_frames(sample_video, frames_dir)
    
    if success:
        print("\n--- Stage 2: OCR ---")
        run_ocr(frames_dir, ocr_jsonl)
        
        print("\n--- Stage 3: Compilation ---")
        compile_glossary(ocr_jsonl, output_prompt)
