import json
import re

def normalize(t):
    return re.sub(r'\s+', '', t)

def main():
    chunks_file = "samples/episodes/wednesday_downtown_2025-02-05_406/transcription/406_gemini_raw_ocr_filtered_experiment_v5_chunks.json"
    words_file = "samples/episodes/wednesday_downtown_2025-02-05_406/transcription/406_gemini_raw_ocr_filtered_experiment_v5_words.json"
    out_file = "samples/episodes/wednesday_downtown_2025-02-05_406/transcription/406_gemini_raw_ocr_filtered_experiment_v5_words_with_turns.json"

    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    with open(words_file, "r", encoding="utf-8") as f:
        segments = json.load(f)

    global_clean_text = ""
    turn_starts = set()
    
    for chunk in chunks:
        lines = chunk["text"].split("\n")
        for line in lines:
            s_line = line.strip()
            if not s_line:
                continue
            is_turn = s_line.startswith("-")
            clean_s = s_line.lstrip("- ").strip()
            if not clean_s:
                continue
            
            if is_turn:
                turn_starts.add(len(global_clean_text))
            
            global_clean_text += normalize(clean_s)

    current_char_idx = 0
    for seg in segments:
        for w in seg["words"]:
            clean_w = normalize(w["word"])
            if not clean_w:
                continue
                
            if current_char_idx in turn_starts:
                w["word"] = "- " + w["word"].lstrip()
            
            current_char_idx += len(clean_w)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    print(f"Restored turns to {out_file}")

if __name__ == "__main__":
    main()
