import os
import json
import argparse
from pathlib import Path

from chigyusubs.paths import find_latest_episode_video, infer_episode_dir_from_video
from chigyusubs.reflow import reflow_words
from chigyusubs.vtt import write_standard_vtt, write_word_timestamps_json


def split_long_segments(segments, max_duration: float = 10.0):
    """Split segments longer than max_duration using word timestamps."""
    normalised = []
    for seg in segments:
        if isinstance(seg, dict):
            normalised.append(seg)
        else:
            words = []
            if seg.words:
                words = [{"start": w.start, "end": w.end, "word": w.word,
                          "probability": w.probability} for w in seg.words]
            normalised.append({
                "start": seg.start, "end": seg.end,
                "text": seg.text, "words": words,
            })

    result = []
    queue = list(normalised)
    while queue:
        seg = queue.pop(0)
        dur = seg["end"] - seg["start"]
        words = seg.get("words", [])
        if dur <= max_duration or len(words) < 2:
            result.append(seg)
            continue
        cut = seg["start"] + max_duration
        best_idx = 0
        for i in range(1, len(words)):
            if words[i]["start"] <= cut:
                best_idx = i
            else:
                break
        if best_idx == 0:
            best_idx = 1
        left_words = words[:best_idx]
        right_words = words[best_idx:]
        left = {
            "start": seg["start"],
            "end": left_words[-1]["end"],
            "text": "".join(w["word"] for w in left_words).strip(),
            "words": left_words,
        }
        right = {
            "start": right_words[0]["start"],
            "end": seg["end"],
            "text": "".join(w["word"] for w in right_words).strip(),
            "words": right_words,
        }
        queue.insert(0, right)
        queue.insert(0, left)
    return result

def run_faster_whisper(
    video_path: str,
    glossary_path: str,
    output_path: str,
    model_name: str,
    compute_type: str,
    hotwords: str,
    hotwords_file: str,
    min_silence_ms: int = 500,
    max_speech_s: float = 15.0,
    max_cue_s: float = 10.0,
    reflow: bool = False,
    reflow_pause_ms: int = 300,
    reflow_min_cue_s: float = 0.3,
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
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=min_silence_ms,
            max_speech_duration_s=max_speech_s,
        ),
        compression_ratio_threshold=2.4,
        word_timestamps=True,
    )

    print(f"Detected language: {info.language} (Probability: {info.language_probability:.2f})")

    segment_list = []
    for segment in segments:
        segment_list.append(segment)
        print(f"[{format_timestamp(segment.start)} -> {format_timestamp(segment.end)}] {segment.text}")

    # Always write word timestamps JSON first (before any splitting/reflow).
    json_output_path = output_path.replace('.vtt', '_words.json')
    print(f"Writing Word Timestamps JSON to {json_output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_word_timestamps_json(segment_list, json_output_path)

    if reflow:
        # Semantic reflow: re-segment based on natural pauses.
        before = len(segment_list)
        # Normalise to dicts for reflow_words().
        normalised = []
        for seg in segment_list:
            if isinstance(seg, dict):
                normalised.append(seg)
            else:
                words = []
                if seg.words:
                    words = [{"start": w.start, "end": w.end, "word": w.word,
                              "probability": w.probability} for w in seg.words]
                normalised.append({
                    "start": seg.start, "end": seg.end,
                    "text": seg.text, "words": words,
                })
        segment_list = reflow_words(
            normalised,
            pause_threshold=reflow_pause_ms / 1000.0,
            max_cue_s=max_cue_s if max_cue_s and max_cue_s > 0 else 10.0,
            min_cue_s=reflow_min_cue_s,
        )
        after = len(segment_list)
        print(f"\nReflowed segments: {before} -> {after} cues (pause={reflow_pause_ms}ms)")
    elif max_cue_s and max_cue_s > 0:
        before = len(segment_list)
        segment_list = split_long_segments(segment_list, max_duration=max_cue_s)
        after = len(segment_list)
        if after > before:
            print(f"\nSplit long segments: {before} -> {after} cues (max {max_cue_s}s)")

    print(f"\nWriting standard VTT to {output_path}...")
    write_standard_vtt(segment_list, output_path)

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
    parser.add_argument("--min-silence-ms", type=int, default=500, help="VAD min silence duration in ms to split segments (default: 500)")
    parser.add_argument("--max-speech-s", type=float, default=15.0, help="VAD max speech duration in seconds before forced split (default: 15)")
    parser.add_argument("--max-cue-s", type=float, default=10.0, help="Hard max cue duration — split longer cues at word boundaries (default: 10, 0 to disable)")
    parser.add_argument("--reflow", action="store_true", help="Use semantic reflow (gap-based) instead of mechanical split")
    parser.add_argument("--reflow-pause-ms", type=int, default=300, help="Reflow: pause threshold in ms to trigger cue break (default: 300)")
    parser.add_argument("--reflow-min-cue-s", type=float, default=0.3, help="Reflow: minimum cue duration in seconds (default: 0.3)")

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
        args.min_silence_ms,
        args.max_speech_s,
        args.max_cue_s,
        args.reflow,
        args.reflow_pause_ms,
        args.reflow_min_cue_s,
    )
