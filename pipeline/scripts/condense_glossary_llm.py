import os
import json
import urllib.request
import urllib.error

def build_smart_glossary(input_txt_path: str, output_txt_path: str):
    print(f"Reading raw OCR dump from {input_txt_path}...")
    with open(input_txt_path, 'r', encoding='utf-8') as f:
        raw_glossary = f.read()

    system_prompt = """You are a helpful assistant. I have a messy list of Japanese text extracted from a TV show.
Please pick out the most important proper nouns (people's names, places, unique show segments, and specific foods).
Ignore full sentences and generic words.
Return your answer as a simple comma-separated list."""

    user_prompt = f"Here is the text:\n{raw_glossary[:4000]}...\n\nPlease extract the top 50 most important terms as a comma-separated list."

    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 1000
    }

    req = urllib.request.Request(
        "http://127.0.0.1:8080/v1/chat/completions",
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )

    try:
        with urllib.request.urlopen(req) as response:
            raw_response = response.read().decode('utf-8')
            print("Raw server response:", raw_response[:200], "...")
            result = json.loads(raw_response)
            
            final_glossary = result['choices'][0]['message']['content'].strip()
            print("\n--- LLM GLOSSARY GENERATED ---")
            print(final_glossary)
            
            with open(output_txt_path, 'w', encoding='utf-8') as f:
                f.write(final_glossary)
                
    except Exception as e:
        print(f"Failed to run LLM: {e}")

if __name__ == "__main__":
    build_smart_glossary("samples/whisper_prompt.txt", "samples/whisper_prompt_condensed.txt")
