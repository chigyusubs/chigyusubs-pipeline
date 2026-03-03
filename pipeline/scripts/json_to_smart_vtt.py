import json
import argparse
from faster_whisper.utils import format_timestamp

def smart_chunk_vtt(json_path, output_path, max_pause=0.3):
    with open(json_path, 'r', encoding='utf-8') as f:
        segments = json.load(f)

    cues = []
    
    for segment in segments:
        words = segment.get("words", [])
        if not words:
            continue
            
        current_chunk_words = []
        current_chunk_start = words[0]["start"]
        
        for i, word in enumerate(words):
            current_chunk_words.append(word["word"])
            
            # Check if we should split here
            is_last_word = (i == len(words) - 1)
            
            if not is_last_word:
                next_word_start = words[i+1]["start"]
                current_word_end = word["end"]
                pause_duration = next_word_start - current_word_end
                
                if pause_duration > max_pause:
                    # The pause is long enough, end the chunk here
                    cues.append({
                        "start": current_chunk_start,
                        "end": current_word_end,
                        "text": "".join(current_chunk_words).strip()
                    })
                    # Reset for next chunk
                    current_chunk_words = []
                    current_chunk_start = next_word_start
            else:
                # End of the segment
                cues.append({
                    "start": current_chunk_start,
                    "end": word["end"],
                    "text": "".join(current_chunk_words).strip()
                })

    # Write to VTT
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("WEBVTT

")
        for cue in cues:
            # Skip empty cues
            if not cue["text"]: continue
            start = format_timestamp(cue["start"])
            end = format_timestamp(cue["end"])
            f.write(f"{start} --> {end}
")
            f.write(f"{cue['text']}

")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="samples/transcription_output/WEDNESDAY_DOWNTOWN_faster_word_timestamps_words.json")
    parser.add_argument("--out", default="samples/transcription_output/WEDNESDAY_DOWNTOWN_smart_chunked.vtt")
    args = parser.parse_args()
    
    smart_chunk_vtt(args.input, args.out)
    print(f"Generated Smart Chunked VTT: {args.out}")
