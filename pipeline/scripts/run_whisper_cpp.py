import os
import subprocess
import argparse
from pathlib import Path
from episode_paths import find_latest_episode_video, infer_episode_dir_from_video

def run_whisper_cpp(video_path: str, glossary_path: str, output_path: str):
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return
        
    initial_prompt = ""
    if os.path.exists(glossary_path):
        with open(glossary_path, 'r', encoding='utf-8') as f:
            initial_prompt = f.read().strip()
        print(f"Loaded OCR Glossary: {initial_prompt[:100]}...")

    base_output_dir = os.path.dirname(output_path)
    os.makedirs(base_output_dir, exist_ok=True)
    output_prefix = os.path.join(base_output_dir, "whisper_cpp_temp")
    wav_path = os.path.join(base_output_dir, "temp_audio.wav")

    print("\nExtracting 16kHz WAV audio from video...")
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        "-hide_banner", "-loglevel", "warning",
        wav_path
    ]
    subprocess.run(ffmpeg_cmd, check=True)

    command = [
        "whisper-cli",
        "--language", "ja",
        "-m", "/home/npk/.local/share/whisper-cpp/models/ggml-large-v3.bin", 
        "--prompt", initial_prompt,
        "--carry-initial-prompt",
        "--vad",
        "-vm", "/home/npk/.local/share/whisper-cpp/models/ggml-silero-v6.2.0.bin",
        "-et", "2.4",
        # REMOVED -ml and -sow entirely. Let Whisper decide naturally.
        "-ovtt",
        "-of", output_prefix,
        wav_path
    ]

    print("\nExecuting whisper-cli (Vulkan backend) with VAD and NO forced chunking...")
    print(" ".join(command))
    print("-" * 50)

    try:
        subprocess.run(command, check=True)
        generated_vtt = f"{output_prefix}.vtt"
        if os.path.exists(generated_vtt):
            os.rename(generated_vtt, output_path)
            print(f"\nTranscription complete! Output saved to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"\nWhisper transcription failed with exit code {e.returncode}")
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="", help="Input video path. Defaults to latest samples/episodes/*/source/*.mp4")
    parser.add_argument("--glossary", default="", help="Glossary path. Defaults to episode glossary/whisper_prompt_condensed.txt")
    parser.add_argument("--out", default="", help="Output VTT path. Defaults to episode transcription/<video_stem>_whispercpp_natural.vtt")
    
    args = parser.parse_args()
    default_video = find_latest_episode_video()
    if not args.video:
        if not default_video:
            raise SystemExit("No default video found. Pass --video explicitly.")
        args.video = str(default_video)

    episode_dir = infer_episode_dir_from_video(Path(args.video))
    if not args.glossary:
        args.glossary = str(episode_dir / "glossary" / "whisper_prompt_condensed.txt")
    if not args.out:
        args.out = str(episode_dir / "transcription" / f"{Path(args.video).stem}_whispercpp_natural.vtt")

    run_whisper_cpp(args.video, args.glossary, args.out)
