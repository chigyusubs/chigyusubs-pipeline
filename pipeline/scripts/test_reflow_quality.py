import os
import json
from google import genai
from google.genai import types

client = genai.Client()
MODEL_ID = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")

with open("samples/transcription_output/WEDNESDAY_DOWNTOWN_faster_word_timestamps_words.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract first ~150 words from the first few segments
sample_words = []
for segment in data:
    if "words" in segment:
        sample_words.extend(segment["words"])
    if len(sample_words) > 150:
        break
        
chunk = sample_words[:150]

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

print(f"Sending {len(chunk)} words to Gemini...")
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

print("\n--- Raw Word JSON (First 2 segments to compare) ---")
print(json.dumps(data[0]["words"], ensure_ascii=False))
print(json.dumps(data[1]["words"], ensure_ascii=False))

print("\n--- LLM Reflowed Subtitles ---")
result = json.loads(response.text.strip())
for cue in result:
    print(f"[{cue['start']:.2f} -> {cue['end']:.2f}] {cue['text']}")
