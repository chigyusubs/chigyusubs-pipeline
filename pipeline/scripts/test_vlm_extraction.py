import os
import sys
from pathlib import Path
from google import genai
from google.genai import types

def test_vlm_extraction(frames_dir: str):
    print("Initializing Gemini API Client...")
    client = genai.Client()
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")

    dumb_prompt = "You are an OCR system. Read all Japanese text in this image exactly as it appears. Do not translate. Output one line per distinct text region."
    
    smart_prompt = """You are an OCR system for Japanese TV. Read the Japanese text in this image. 
Extract ONLY proper nouns, character names, locations, and unique terminology. 
Ignore common verbs, particles, and partial sentences. 
Output the results strictly as a comma-separated list. Do not translate."""

    frames = sorted(Path(frames_dir).glob("*.jpg"))
    if not frames:
        print(f"No frames found in {frames_dir}")
        sys.exit(1)
        
    print(f"Testing on {len(frames)} frames...\n")

    for frame_path in frames:
        print(f"--- Frame: {frame_path.name} ---")
        
        try:
            with open(frame_path, "rb") as f:
                image_bytes = f.read()
            
            # 1. Dumb OCR
            print("\n[DUMB OCR]")
            resp_dumb = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    dumb_prompt
                ],
                config=types.GenerateContentConfig(temperature=0.1)
            )
            print(resp_dumb.text.strip())
            
            # 2. Smart OCR
            print("\n[SMART OCR (Proper Nouns Only)]")
            resp_smart = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    smart_prompt
                ],
                config=types.GenerateContentConfig(temperature=0.1)
            )
            print(resp_smart.text.strip())
            print("\n" + "="*50 + "\n")
            
        except Exception as e:
            print(f"Error processing {frame_path.name}: {e}")

if __name__ == "__main__":
    test_vlm_extraction("samples/test_frames")
