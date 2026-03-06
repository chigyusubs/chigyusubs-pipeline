import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from manga_ocr import MangaOcr

def extract_frames(video_path: str, output_dir: str, max_fps: int = 2):
    """
    Extracts frames using mpdecimate to drop duplicates and keep only visual changes.
    """
    print(f"Extracting frames from {video_path}...")
    
    # We use fps to cap the max framerate to avoid extracting 60fps during animations
    # mpdecimate drops identical frames
    # vsync vfr ensures we only write the frames that pass the filter
    # -qscale:v 2 ensures high JPEG quality
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps={max_fps},mpdecimate",
        "-vsync", "vfr",
        "-qscale:v", "2",
        "-hide_banner",
        "-loglevel", "warning",
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

def clean_ocr_text(raw_texts):
    """
    Implements Substring Collapsing to remove typewriter animation fragments.
    Keeps only the longest unique strings.
    """
    # 1. Filter out single characters that are likely garbage/animation start points
    # (Unless it's a specific Kanji, but standard Hiragana/Katakana usually aren't standalone words here)
    filtered = []
    for text in raw_texts:
        text = text.strip()
        if len(text) <= 1:
            continue
        filtered.append(text)
        
    # 2. Sort by length descending
    filtered.sort(key=len, reverse=True)
    
    collapsed = []
    
    # 3. Substring collapsing
    for text in filtered:
        is_substring = False
        for longer_text in collapsed:
            if text in longer_text:
                is_substring = True
                break
        if not is_substring:
            collapsed.append(text)
            
    return collapsed

def run_ocr_pipeline(video_path: str):
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return

    # Use a temporary directory for frames so we don't clutter the workspace
    with tempfile.TemporaryDirectory() as temp_dir:
        success = extract_frames(video_path, temp_dir)
        if not success:
            return
            
        print("Loading MangaOCR model (this may take a moment)...")
        # Initialize MangaOCR (it downloads the model on first run if needed)
        mocr = MangaOcr()
        
        frame_files = sorted(list(Path(temp_dir).glob("*.jpg")))
        
        print("Running OCR on extracted frames...")
        raw_results = []
        for i, frame_path in enumerate(frame_files):
            # We process every 10th frame just to show progress
            if i % 10 == 0:
                print(f"Processing frame {i+1}/{len(frame_files)}...")
                
            text = mocr(frame_path)
            if text and text.strip():
                raw_results.append(text)
                
        print("\n--- RAW OCR DUMP (Sample) ---")
        # Print first 20 raw results to see how messy it is
        for t in raw_results[:20]:
            print(f"- {t}")
            
        print("\n--- CLEANING TEXT ---")
        cleaned_results = clean_ocr_text(raw_results)
        
        print(f"Reduced from {len(raw_results)} raw reads to {len(cleaned_results)} unique strings.")
        
        print("\n--- FINAL GLOSSARY CANDIDATES ---")
        for t in cleaned_results:
            print(f"- {t}")

if __name__ == "__main__":
    # We will point this to the sample video
    sample_video = "samples/WEDNESDAY_DOWNTOWN_2024-01-24 _#363.mp4"
    run_ocr_pipeline(sample_video)
