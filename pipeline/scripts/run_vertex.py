import os
import argparse
import sys

# Attempt to load from local .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Error: The 'google-genai' library is required.", file=sys.stderr)
    print("Please install it: pip install google-genai", file=sys.stderr)
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Utility to easily call Vertex AI Gemini models using google-genai.",
        epilog="Examples:\n  python pipeline/scripts/run_vertex.py --prompt \"Tell me a joke\"\n  cat prompt.txt | python pipeline/scripts/run_vertex.py",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--prompt", type=str, help="The prompt to send. If not provided, reads from standard input (stdin).")
    parser.add_argument("--model", type=str, default=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"), help="The model to use (default: GEMINI_MODEL env var or gemini-2.5-flash)")
    parser.add_argument("--project", type=str, help="GCP Project ID. Falls back to GOOGLE_CLOUD_PROJECT env var.", default=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    parser.add_argument("--location", type=str, help="GCP Location. Falls back to GOOGLE_CLOUD_LOCATION env var.", default=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"))
    parser.add_argument("--system", type=str, help="Optional system instructions")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for generation (default: 1.0)")
    parser.add_argument("--out", type=str, help="Optional output file. If not provided, prints to stdout.")

    args = parser.parse_args()

    # We allow running without explicit project if GOOGLE_GENAI_USE_VERTEXAI is set and ADC is present,
    # but it's safer to ensure we have a project string.
    if not args.project and not os.environ.get("GOOGLE_GENAI_USE_VERTEXAI"):
        print("Error: GCP Project ID is required. Pass --project or set GOOGLE_CLOUD_PROJECT environment variable.", file=sys.stderr)
        sys.exit(1)

    # Initialize the Vertex AI client using the unified google-genai SDK.
    # It automatically respects GOOGLE_APPLICATION_CREDENTIALS for the service account
    client = genai.Client(
        vertexai=True,
        project=args.project,
        location=args.location
    )

    prompt = args.prompt
    if prompt is None:
        # Read from stdin if no prompt is passed directly
        if not sys.stdin.isatty():
            prompt = sys.stdin.read()
        else:
            print("Error: No prompt provided. Provide --prompt or pipe text via stdin.", file=sys.stderr)
            sys.exit(1)

    if not prompt.strip():
        print("Error: Prompt is empty.", file=sys.stderr)
        sys.exit(1)

    # Configure generation
    config_args = {
        "temperature": args.temperature,
    }
    if args.system:
        config_args["system_instruction"] = args.system

    config = types.GenerateContentConfig(**config_args)

    max_retries = 5
    base_delay = 2.0 # starting delay in seconds

    for attempt in range(max_retries):
        try:
            if attempt == 0:
                print(f"Calling {args.model} on Vertex AI ({args.location})...", file=sys.stderr)
            else:
                print(f"Retrying ({attempt + 1}/{max_retries}) on Vertex AI...", file=sys.stderr)
                
            response = client.models.generate_content(
                model=args.model,
                contents=prompt,
                config=config
            )
            
            result_text = response.text
            
            if args.out:
                with open(args.out, "w", encoding="utf-8") as f:
                    f.write(result_text)
                print(f"Response successfully written to {args.out}", file=sys.stderr)
            else:
                # Print directly to stdout for piping/agent consumption
                print(result_text)
                
            break # Success, exit retry loop
                
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "resource exhausted" in error_msg or "quota" in error_msg:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"Resource Exhausted (429). Retrying in {delay} seconds...", file=sys.stderr)
                    import time
                    time.sleep(delay)
                    continue
            
            # If it's not a retryable error, or we've run out of retries
            print(f"Error calling Vertex AI: {e}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()
