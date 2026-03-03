import os
import json
import argparse
import time
from google import genai
from google.genai import types
from google.genai import errors
from faster_whisper.utils import format_timestamp

# Using the genai SDK as requested
client = genai.Client()
MODEL_ID = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro") # Good default for complex reasoning tasks

def write_vtt_chunk(cues, output_path: str, is_first: bool):
    """Appends reflowed cues to the VTT file."""
    mode = 'w' if is_first else 'a'
    with open(output_path, mode, encoding='utf-8') as f:
        if is_first:
            f.write("WEBVTT\n\n")
        for cue in cues:
            start = format_timestamp(cue["start"])
            end = format_timestamp(cue["end"])
            f.write(f"{start} --> {end}\n")
            f.write(f"{cue['text'].strip()}\n\n")


def write_vtt(cues, output_path: str):
    """Write all cues to a single VTT file."""
    write_vtt_chunk(cues, output_path, is_first=True)

def chunk_words(words, max_words=100):
    """Chunks the word list into manageable blocks for the LLM context window."""
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(words[i:i + max_words])
    return chunks

def reflow_chunk_with_vertex(chunk, max_retries=3):
    """Sends a chunk of words to Vertex AI to be reflowed into subtitle cues."""
    
    SYSTEM_PROMPT = """
You are a professional Japanese subtitle editor working on a fast-paced variety show.
I am giving you a JSON array of words/tokens with their exact `start` and `end` times in seconds.
Your job is to group these tokens into sensible, readable subtitle cues.

Rules:
1. Semantics: Group complete thoughts, phrases, or clauses. Do not split a noun from its particle (e.g. keep 私 and は together).
2. Readability: Aim for a comfortable reading length. Break long rapid speech into smaller chunks.
3. Pacing: Look at the timestamps. If there is a noticeable gap (e.g. >0.4s) between the `end` of one word and the `start` of the next, it's often a natural pause or speaker change—strongly consider splitting the subtitle there.
4. Output Format: Return ONLY a valid JSON array of objects. Each object must have:
   - `start`: The start time of the FIRST word in the group (float).
   - `end`: The end time of the LAST word in the group (float).
   - `text`: The combined text, with appropriate punctuation (commas, periods) added if it helps readability. Do not hallucinate words.
"""

    prompt = f"""
Input Data:
{json.dumps(chunk, ensure_ascii=False, indent=2)}

Output JSON Array:
"""

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=[
                    types.Content(role="user", parts=[
                        types.Part.from_text(text=SYSTEM_PROMPT + "\n\n" + prompt)
                    ])
                ],
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json",
                )
            )
            
            text = response.text.strip()
            # Handle cases where the model wraps the output in a markdown block
            if text.startswith("```json"):
                text = text[7:]
            elif text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
                
            return json.loads(text.strip())
        except errors.ClientError as e:
            if "429" in str(e):
                wait_time = (attempt + 1) * 15 # Exponential backoff: 15s, 30s, 45s
                print(f"Rate limit hit. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"ClientError during generation: {e}")
                break
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON from LLM: {e}")
            break
        except Exception as e:
            print(f"Error during generation: {e}")
            break
            
    print("Failed to process chunk after retries.")
    return []

def reflow_subtitles(input_json, output_vtt):
    print(f"Loading word timestamps from {input_json}...")
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_words = []
    for segment in data:
        if "words" in segment:
            all_words.extend(segment["words"])

    print(f"Total words to process: {len(all_words)}")
    
    # Process in chunks of ~80 words to keep context tight and reduce JSON schema breakage
    word_chunks = chunk_words(all_words, max_words=80)
    
    print(f"Initializing genai client with model: {MODEL_ID}...")
    
    all_reflowed_cues = []
    
    for i, chunk in enumerate(word_chunks):
        print(f"Processing chunk {i+1}/{len(word_chunks)} ({len(chunk)} words)...")
        reflowed_cues = reflow_chunk_with_vertex(chunk)
        if reflowed_cues:
           all_reflowed_cues.extend(reflowed_cues)
        else:
           print(f"Warning: Chunk {i+1} returned empty or failed to parse. Some data might be missing.")
        
    print(f"Writing smart reflowed VTT to {output_vtt}...")
    write_vtt(all_reflowed_cues, output_vtt)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="samples/transcription_output/WEDNESDAY_DOWNTOWN_faster_word_timestamps_words.json")
    parser.add_argument("--out", default="samples/transcription_output/WEDNESDAY_DOWNTOWN_vertex_reflowed.vtt")
    args = parser.parse_args()
    
    reflow_subtitles(args.input, args.out)
