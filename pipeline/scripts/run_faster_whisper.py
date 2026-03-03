import os
import json
import argparse
from pathlib import Path
from episode_paths import find_latest_episode_video, infer_episode_dir_from_video

def write_standard_vtt(segments, output_path: str):
    """Writes a standard VTT file from faster-whisper segment output."""
    from faster_whisper.utils import format_timestamp

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("WEBVTT\n\n")
        for segment in segments:
            start = format_timestamp(segment.start)
            end = format_timestamp(segment.end)
            f.write(f"{start} --> {end}\n")
            f.write(f"{segment.text.strip()}\n\n")

def write_word_timestamps_json(segments, output_path: str):
    """Writes detailed word timestamps to a JSON file."""
    data = []
    for segment in segments:
        seg_data = {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "words": []
        }
        if segment.words:
            for word in segment.words:
                seg_data["words"].append({
                    "start": word.start,
                    "end": word.end,
                    "word": word.word,
                    "probability": word.probability
                })
        data.append(seg_data)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def run_faster_whisper(
    video_path: str,
    glossary_path: str,
    output_path: str,
    model_name: str,
    compute_type: str,
    hotwords: str,
    hotwords_file: str,
):
    from faster_whisper import WhisperModel
    from faster_whisper.utils import format_timestamp

    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return
        
    initial_prompt = ""
    if os.path.exists(glossary_path):
        with open(glossary_path, 'r', encoding='utf-8') as f:
            initial_prompt = f.read().strip()
        print(f"Loaded OCR Glossary: {initial_prompt[:100]}...")

    hotwords_text = ""
    if hotwords and hotwords.strip():
        hotwords_text = hotwords.strip()
        print(f"Using inline hotwords: {hotwords_text[:100]}...")
    elif hotwords_file and os.path.exists(hotwords_file):
        with open(hotwords_file, 'r', encoding='utf-8') as f:
            hotwords_text = f.read().strip()
        print(f"Loaded hotwords file: {hotwords_text[:100]}...")

    print(f"\nLoading faster-whisper model: {model_name} (compute_type={compute_type})")
    model = WhisperModel(model_name, device="cuda", compute_type=compute_type)

    print("\nStarting transcription with Silero VAD & Word Timestamps...")
    
    segments, info = model.transcribe(
        video_path,
        language="ja",
        initial_prompt=initial_prompt,
        hotwords=hotwords_text or None,
        condition_on_previous_text=False,
        vad_filter=True, # Built-in Silero VAD
        vad_parameters=dict(min_silence_duration_ms=500),
        compression_ratio_threshold=2.4,
        word_timestamps=True # Enable word-level timestamps
    )

    print(f"Detected language: {info.language} (Probability: {info.language_probability:.2f})")

    segment_list = []
    for segment in segments:
        segment_list.append(segment)
        print(f"[{format_timestamp(segment.start)} -> {format_timestamp(segment.end)}] {segment.text}")

    print(f"\nWriting standard VTT to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_standard_vtt(segment_list, output_path)
    
    json_output_path = output_path.replace('.vtt', '_words.json')
    print(f"Writing Word Timestamps JSON to {json_output_path}...")
    write_word_timestamps_json(segment_list, json_output_path)
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="", help="Input video path. Defaults to latest samples/episodes/*/source/*.mp4")
    parser.add_argument("--glossary", default="", help="Glossary path. Defaults to episode glossary/whisper_prompt_condensed.txt")
    parser.add_argument("--out", default="", help="Output VTT path. Defaults to episode transcription/<video_stem>_faster_stock.vtt")
    parser.add_argument("--model", default="large-v3", help="faster-whisper model name or local path")
    parser.add_argument("--compute-type", default="float16", help="ctranslate2 compute type (e.g., float16, int8)")
    parser.add_argument("--hotwords", default="", help="Inline hotwords string")
    parser.add_argument("--hotwords-file", default="", help="Path to hotwords text file. Defaults to episode glossary/whisper_hotwords.txt")
    
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
        args.out = str(episode_dir / "transcription" / f"{Path(args.video).stem}_faster_stock.vtt")
    if not args.hotwords_file:
        default_hotwords = episode_dir / "glossary" / "whisper_hotwords.txt"
        if default_hotwords.exists():
            args.hotwords_file = str(default_hotwords)

    run_faster_whisper(
        args.video,
        args.glossary,
        args.out,
        args.model,
        args.compute_type,
        args.hotwords,
        args.hotwords_file,
    )
