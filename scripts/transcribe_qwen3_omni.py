#!/usr/bin/env python3
"""Transcribe audio using Qwen3-Omni-30B via llama.cpp (mtmd-cli).

Sends 16kHz WAV chunks to the local llama.cpp multimodal CLI and captures
the text output.  Outputs Gemini-compatible raw JSON for comparison.

Requires: compiled llama-mtmd-cli with Qwen3-Omni support, plus the
thinker GGUF and mmproj GGUF files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chigyusubs.audio import get_duration
from chigyusubs.chunking import find_chunk_boundaries
from chigyusubs.metadata import finish_run, metadata_path, start_run, write_metadata
from chigyusubs.vad import run_silero_vad


def log(msg: str = "", end="\n"):
    print(msg, end=end, flush=True)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

MTMD_CLI = "/tmp/llama-cpp-qwen3-omni/build/bin/llama-mtmd-cli"
DEFAULT_MODEL = os.path.expanduser("~/models/qwen3-omni-30b-a3b/thinker-q4_k_m.gguf")
DEFAULT_MMPROJ = os.path.expanduser("~/models/qwen3-omni-30b-a3b/mmproj-f16.gguf")

PROMPT = (
    "Transcribe ALL spoken Japanese dialogue in this audio clip faithfully. "
    "Output ONLY plain text. Do NOT add timestamps or speaker labels. "
    "Use standard Japanese punctuation. Do NOT translate. "
    "Output each utterance on a new line. "
    "If the audio is silent or background music only, output nothing."
)


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------

def _extract_wav_chunk(video_path: str, start_s: float, duration_s: float, out_path: str):
    """Extract a 16kHz mono WAV chunk via ffmpeg."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(start_s),
            "-i", video_path,
            "-t", str(duration_s),
            "-vn", "-ac", "1", "-ar", "16000",
            "-f", "wav", out_path,
        ],
        capture_output=True, check=True,
    )


# ---------------------------------------------------------------------------
# VAD chunk loading
# ---------------------------------------------------------------------------

def _load_vad_chunk_bounds(chunk_json_path: str) -> list[tuple[float, float]]:
    with open(chunk_json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return [(float(c["start_sec"]), float(c["end_sec"])) for c in chunks]


# ---------------------------------------------------------------------------
# llama.cpp transcription
# ---------------------------------------------------------------------------

_PERF_RE = re.compile(
    r"eval time\s*=\s*[\d.]+\s*ms\s*/\s*(\d+)\s*runs\s*\(\s*([\d.]+)\s*ms per token"
)


def _truncate_repetition(
    text: str, min_phrase: int = 2, max_phrase: int = 50, max_keep: int = 2,
) -> tuple[str, bool]:
    """Detect and strip trailing repetition loops.

    Scans for the longest repeating phrase at the tail of *text*.
    If found (>= 3 consecutive repeats), keeps at most *max_keep* copies.
    Returns (cleaned_text, was_truncated).
    """
    if len(text) < min_phrase * 3:
        return text, False

    best_start = len(text)
    best_plen = 0
    best_count = 0

    for plen in range(min_phrase, min(max_phrase + 1, len(text) // 3 + 1)):
        phrase = text[-plen:]
        count = 0
        pos = len(text) - plen
        while pos >= 0 and text[pos : pos + plen] == phrase:
            count += 1
            pos -= plen

        if count >= 3 and count * plen > best_count * best_plen:
            best_count = count
            best_plen = plen
            best_start = len(text) - (count * plen)

    if best_count >= 3:
        keep = best_start + best_plen * max_keep
        return text[:keep].rstrip(), True

    return text, False


def _transcribe_chunk(
    cli_path: str,
    model_path: str,
    mmproj_path: str,
    wav_path: str,
    prompt: str,
    *,
    ngl: int,
    ctx_size: int,
    n_predict: int,
    temp: float = 0.0,
) -> dict:
    """Run llama-mtmd-cli on a single WAV file and parse output."""
    cmd = [
        cli_path,
        "-m", model_path,
        "--mmproj", mmproj_path,
        "--audio", wav_path,
        "-p", prompt,
        "-ngl", str(ngl),
        "--ctx-size", str(ctx_size),
        "-n", str(n_predict),
        "--temp", str(temp),
        "--no-warmup",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
    )

    stderr = result.stderr or ""
    stdout = result.stdout or ""

    # The transcription text is on stdout. The CLI prints the prompt first
    # then the generated text. With mtmd-cli the generated text goes to stdout
    # and logs go to stderr.
    text = stdout.strip()

    # Parse perf stats from stderr
    tokens_generated = 0
    ms_per_token = 0.0
    for m in _PERF_RE.finditer(stderr):
        tokens_generated = int(m.group(1))
        ms_per_token = float(m.group(2))

    return {
        "text": text,
        "tokens": tokens_generated,
        "ms_per_token": ms_per_token,
        "returncode": result.returncode,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    run = start_run("transcribe_qwen3_omni")

    parser = argparse.ArgumentParser(
        description="Transcribe audio with Qwen3-Omni-30B via llama.cpp."
    )
    parser.add_argument("--video", required=True, help="Source video/audio file.")
    parser.add_argument("--output", required=True, help="Output raw JSON path.")
    parser.add_argument("--vad-json", default="")
    parser.add_argument("--chunk-json", default="")
    parser.add_argument("--cli", default=MTMD_CLI, help="Path to llama-mtmd-cli binary.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to thinker GGUF.")
    parser.add_argument("--mmproj", default=DEFAULT_MMPROJ, help="Path to mmproj GGUF.")
    parser.add_argument("--ngl", type=int, default=30, help="GPU layers (default: 30).")
    parser.add_argument("--ctx-size", type=int, default=2048, help="Context size (default: 2048).")
    parser.add_argument("--n-predict", type=int, default=512, help="Max tokens to generate (default: 512).")
    parser.add_argument("--temp", type=float, default=0.2, help="Sampling temperature (default: 0.2; 0.0 increases repetition loops).")
    parser.add_argument("--target-chunk-s", type=float, default=30, help="Target chunk duration (default: 30).")
    parser.add_argument("--max-chunk-s", type=float, default=45, help="Max chunk duration (default: 45).")
    parser.add_argument("--max-chunks", type=int, default=0, help="Limit chunks for testing (0 = all).")
    parser.add_argument("--prompt", default=PROMPT, help="Transcription prompt.")
    args = parser.parse_args()

    # Validate paths
    if not Path(args.cli).exists():
        raise SystemExit(f"llama-mtmd-cli not found: {args.cli}")
    if not Path(args.model).exists():
        raise SystemExit(f"Model GGUF not found: {args.model}")
    if not Path(args.mmproj).exists():
        raise SystemExit(f"mmproj GGUF not found: {args.mmproj}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Chunk boundaries
    # -----------------------------------------------------------------------
    if args.chunk_json:
        log(f"Loading chunk boundaries from {args.chunk_json}")
        chunk_bounds = _load_vad_chunk_bounds(args.chunk_json)
    else:
        duration = get_duration(args.video)
        log(f"Duration: {duration:.1f}s")

        if args.vad_json and Path(args.vad_json).exists():
            log(f"Loading VAD segments from {args.vad_json}")
            vad_segments = json.loads(Path(args.vad_json).read_text(encoding="utf-8"))
        else:
            log("Running Silero VAD...")
            with tempfile.TemporaryDirectory() as workdir:
                vad_segments = run_silero_vad(args.video, workdir)
            log(f"  {len(vad_segments)} speech segments detected")

        chunk_bounds = find_chunk_boundaries(
            vad_segments,
            total_duration=duration,
            target_chunk_s=args.target_chunk_s,
            max_chunk_s=args.max_chunk_s,
        )

    total = len(chunk_bounds)
    if args.max_chunks > 0:
        chunk_bounds = chunk_bounds[:args.max_chunks]
    log(f"{len(chunk_bounds)}/{total} chunks to transcribe")

    # -----------------------------------------------------------------------
    # Transcribe
    # -----------------------------------------------------------------------
    results: list[dict] = []
    t_start = time.monotonic()

    with tempfile.TemporaryDirectory() as workdir:
        for idx, (start_s, end_s) in enumerate(chunk_bounds):
            dur = end_s - start_s
            log(f"Chunk {idx+1}/{len(chunk_bounds)}: {start_s:.1f}-{end_s:.1f}s ({dur:.1f}s)")

            wav_path = os.path.join(workdir, f"qwen3_omni_{idx}.wav")
            _extract_wav_chunk(args.video, start_s, dur, wav_path)
            wav_size_mb = os.path.getsize(wav_path) / (1024 * 1024)

            t0 = time.monotonic()
            try:
                r = _transcribe_chunk(
                    args.cli, args.model, args.mmproj, wav_path, args.prompt,
                    ngl=args.ngl, ctx_size=args.ctx_size, n_predict=args.n_predict,
                    temp=args.temp,
                )
                elapsed = time.monotonic() - t0

                if r["returncode"] != 0 and not r["text"]:
                    log(f"  FAILED ({elapsed:.1f}s): exit code {r['returncode']}")
                    results.append({
                        "chunk": idx, "chunk_start_s": start_s, "chunk_end_s": end_s,
                        "text": "", "chunk_size_mb": round(wav_size_mb, 3),
                        "elapsed_seconds": round(elapsed, 3),
                        "error": f"exit code {r['returncode']}",
                    })
                    continue

                text = r["text"]
                truncated = False
                if r["tokens"] >= args.n_predict - 1:
                    text, truncated = _truncate_repetition(text)

                status = f"  {len(text)} chars, {elapsed:.1f}s, {r['tokens']} tok, {r['ms_per_token']:.1f}ms/tok"
                if truncated:
                    status += " [repetition truncated]"
                log(status)

                results.append({
                    "chunk": idx,
                    "chunk_start_s": start_s,
                    "chunk_end_s": end_s,
                    "text": text,
                    "chunk_size_mb": round(wav_size_mb, 3),
                    "elapsed_seconds": round(elapsed, 3),
                    "tokens_generated": r["tokens"],
                    "ms_per_token": r["ms_per_token"],
                    **({"repetition_truncated": True} if truncated else {}),
                })
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - t0
                log(f"  TIMEOUT ({elapsed:.1f}s)")
                results.append({
                    "chunk": idx, "chunk_start_s": start_s, "chunk_end_s": end_s,
                    "text": "", "chunk_size_mb": round(wav_size_mb, 3),
                    "elapsed_seconds": round(elapsed, 3),
                    "error": "timeout",
                })
            except Exception as exc:
                elapsed = time.monotonic() - t0
                msg = str(exc).strip().splitlines()[0] if str(exc).strip() else repr(exc)
                log(f"  FAILED ({elapsed:.1f}s): {msg}")
                results.append({
                    "chunk": idx, "chunk_start_s": start_s, "chunk_end_s": end_s,
                    "text": "", "chunk_size_mb": round(wav_size_mb, 3),
                    "elapsed_seconds": round(elapsed, 3),
                    "error": msg,
                })

            # Incremental save after each chunk
            output_path.write_text(
                json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    total_time = time.monotonic() - t_start

    total_chars = sum(len(r.get("text", "")) for r in results)
    failed = sum(1 for r in results if "error" in r)
    truncated = sum(1 for r in results if r.get("repetition_truncated"))
    audio_dur = sum(r["chunk_end_s"] - r["chunk_start_s"] for r in results)

    log(f"\nWrote {len(results)} chunks to {output_path}")
    log(f"  Chars: {total_chars}, Failed: {failed}, Truncated: {truncated}")
    log(f"  Audio: {audio_dur:.0f}s, Wall: {total_time:.0f}s")
    if audio_dur > 0:
        log(f"  RTF: {total_time / audio_dur:.2f}x")

    metadata = finish_run(
        run,
        inputs={"video": args.video},
        outputs={"raw_json": str(output_path)},
        settings={
            "model": args.model,
            "mmproj": args.mmproj,
            "ngl": args.ngl,
            "ctx_size": args.ctx_size,
            "n_predict": args.n_predict,
            "target_chunk_s": args.target_chunk_s,
            "max_chunk_s": args.max_chunk_s,
        },
        stats={
            "chunks": len(results),
            "total_chars": total_chars,
            "failed_chunks": failed,
            "audio_duration_s": round(audio_dur, 1),
            "wall_time_s": round(total_time, 1),
        },
    )
    write_metadata(str(output_path), metadata)
    log(f"Metadata: {metadata_path(str(output_path))}")


if __name__ == "__main__":
    main()
