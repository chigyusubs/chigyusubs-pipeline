import os
import subprocess
import argparse
from pathlib import Path
from chigyusubs.paths import find_latest_episode_video, infer_episode_dir_from_video

def run_whisper_with_glossary(video_path: str, glossary_path: str, output_dir: str):
    """
    Runs the local Whisper CLI using the OCR glossary and strict anti-looping parameters.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return
        
    if not os.path.exists(glossary_path):
        print(f"Error: Glossary not found at {glossary_path}. Run the OCR pipeline first.")
        return

    os.makedirs(output_dir, exist_ok=True)

    with open(glossary_path, 'r', encoding='utf-8') as f:
        glossary_text = f.read().strip()
        
    print(f"Loaded Glossary ({len(glossary_text)} chars): {glossary_text[:100]}...")

    command = [
        "whisper",
        video_path,
        "--model", "large-v3", 
        "--language", "ja",
        "--output_dir", output_dir,
        "--output_format", "vtt",
        "--initial_prompt", glossary_text,
        "--condition_on_previous_text", "False",
        "--carry_initial_prompt", "True",
        "--compression_ratio_threshold", "2.4",
        "--logprob_threshold", "-1.0",
        "--word_timestamps", "True"
    ]

    print("\nExecuting Whisper with Anti-Looping config...")
    print(" ".join(command))
    print("-" * 50)

    try:
        subprocess.run(command, check=True)
        print(f"\nTranscription complete! Output saved to {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"\nWhisper transcription failed with exit code {e.returncode}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Whisper with OCR Glossary")
    parser.add_argument("--video", default="", help="Input video path. Defaults to latest samples/episodes/*/source/*.mp4")
    parser.add_argument("--glossary", default="", help="Glossary path. Defaults to episode glossary/whisper_prompt_condensed.txt")
    parser.add_argument("--outdir", default="", help="Output dir. Defaults to episode transcription/")
    
    args = parser.parse_args()
    default_video = find_latest_episode_video()
    if not args.video:
        if not default_video:
            raise SystemExit("No default video found. Pass --video explicitly.")
        args.video = str(default_video)

    episode_dir = infer_episode_dir_from_video(Path(args.video))
    if not args.glossary:
        args.glossary = str(episode_dir / "glossary" / "whisper_prompt_condensed.txt")
    if not args.outdir:
        args.outdir = str(episode_dir / "transcription")

    run_whisper_with_glossary(args.video, args.glossary, args.outdir)
