#!/usr/bin/env python3
"""CTC forced alignment using NTQAI/wav2vec2-large-japanese.

Uses torchaudio.functional.forced_align for character-level CTC alignment,
which is more robust than stable-ts cross-attention alignment for short
utterances and chunk edges.

Usage:
  python scripts/align_ctc.py \
    --video samples/episodes/dmm/source/dmm.webm \
    --chunks samples/episodes/dmm/transcription/dmm_video_only_v2_gemini_raw.json \
    --output-words samples/episodes/dmm/transcription/dmm_ctc_words.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chigyusubs.metadata import finish_run, metadata_path, start_run, write_metadata

SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "NTQAI/wav2vec2-large-japanese"

# Lines that should be stripped before alignment (visual-only context)
_VISUAL_RE = re.compile(r"^\[画面:.*\]$")
_SPEAKER_DASH_RE = re.compile(r"^--\s*")


def load_model():
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    print(f"Loading {MODEL_NAME}...")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
    model.eval().to(DEVICE)
    print(f"Model on {DEVICE}")
    return model, processor


def extract_audio_slice(video_path, start_s, duration_s, out_path):
    subprocess.run(
        [
            "ffmpeg", "-y", "-ss", str(start_s), "-i", video_path,
            "-t", str(duration_s), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
            "-f", "wav", out_path,
        ],
        capture_output=True, check=True,
    )


def clean_chunk_text(raw_text: str) -> list[str]:
    """Extract spoken lines from a chunk, stripping visual context and speaker dashes."""
    lines = []
    for line in raw_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if _VISUAL_RE.match(line):
            continue
        line = _SPEAKER_DASH_RE.sub("", line).strip()
        if line:
            lines.append(line)
    return lines


def align_chunk(model, processor, waveform: torch.Tensor, lines: list[str]) -> list[dict]:
    """Align spoken lines to audio using CTC forced alignment.

    Returns list of segment dicts with word-level timestamps.
    """
    if not lines:
        return []

    vocab = processor.tokenizer.get_vocab()
    pad_id = processor.tokenizer.pad_token_id

    # Get log probabilities from model
    with torch.no_grad():
        inputs = processor(
            waveform.squeeze().numpy(),
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )
        input_values = inputs.input_values.to(DEVICE)
        logits = model(input_values).logits
        log_probs = torch.log_softmax(logits, dim=-1).cpu()

    num_frames = log_probs.shape[1]
    audio_duration = waveform.shape[-1] / SAMPLE_RATE

    # Build full token sequence from all lines
    full_text = "".join(lines)
    token_ids = []
    chars = []
    for ch in full_text:
        tid = vocab.get(ch)
        if tid is not None and tid != pad_id:
            token_ids.append(tid)
            chars.append(ch)

    if not token_ids:
        return []

    # Run CTC forced alignment
    targets = torch.tensor([token_ids], dtype=torch.int32)
    input_lengths = torch.tensor([num_frames], dtype=torch.int32)
    target_lengths = torch.tensor([len(token_ids)], dtype=torch.int32)

    try:
        aligned_tokens, scores = torchaudio.functional.forced_align(
            log_probs, targets, input_lengths, target_lengths, blank=pad_id,
        )
    except Exception as e:
        print(f"  forced_align failed: {e}")
        return []

    # Merge consecutive aligned frames into token spans
    token_spans = torchaudio.functional.merge_tokens(aligned_tokens[0], scores[0])

    if len(token_spans) != len(chars):
        print(f"  Warning: token_spans ({len(token_spans)}) != characters ({len(chars)}), skipping")
        return []

    # Map frame indices to time
    frame_to_time = audio_duration / num_frames

    # Build character-level timestamps
    char_timestamps = []
    for span, char in zip(token_spans, chars):
        t_start = span.start * frame_to_time
        t_end = (span.end + 1) * frame_to_time
        char_timestamps.append({
            "char": char,
            "start": round(t_start, 3),
            "end": round(t_end, 3),
            "score": round(span.score, 4),
        })

    # Reconstruct segments from the original lines
    segments = []
    char_idx = 0
    for line in lines:
        line_chars = []
        for ch in line:
            if char_idx < len(char_timestamps) and char_timestamps[char_idx]["char"] == ch:
                line_chars.append(char_timestamps[char_idx])
                char_idx += 1

        if not line_chars:
            segments.append({
                "start": 0.0,
                "end": 0.0,
                "text": line,
                "words": [{"start": 0.0, "end": 0.0, "word": line, "probability": 0.0}],
            })
            continue

        seg_start = line_chars[0]["start"]
        seg_end = line_chars[-1]["end"]

        words = [{
            "start": seg_start,
            "end": seg_end,
            "word": line,
            "probability": sum(c["score"] for c in line_chars) / len(line_chars),
        }]

        segments.append({
            "start": seg_start,
            "end": seg_end,
            "text": line,
            "words": words,
        })

    return segments


def main():
    run = start_run("align_ctc")
    parser = argparse.ArgumentParser(description="CTC forced alignment using wav2vec2 Japanese.")
    parser.add_argument("--video", required=True, help="Input video/audio file.")
    parser.add_argument("--chunks", required=True, help="Gemini raw transcription JSON with chunks.")
    parser.add_argument("--output-words", required=True, help="Output JSON with aligned words.")
    args = parser.parse_args()

    with open(args.chunks, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    model, processor = load_model()

    all_segments = []

    with tempfile.TemporaryDirectory() as work_dir:
        for i, chunk in enumerate(chunks_data):
            c_start = chunk["chunk_start_s"]
            c_end = chunk["chunk_end_s"]
            c_dur = c_end - c_start
            raw_text = chunk.get("text", "")

            lines = clean_chunk_text(raw_text)
            print(f"\nChunk {i+1}/{len(chunks_data)} ({c_start:.1f}s\u2013{c_end:.1f}s, {len(lines)} lines)")

            if not lines:
                print("  Skipping empty chunk.")
                continue

            # Extract audio slice
            slice_wav = os.path.join(work_dir, f"slice_{i}.wav")
            extract_audio_slice(args.video, c_start, c_dur, slice_wav)

            data, file_sr = sf.read(slice_wav)
            waveform = torch.from_numpy(data).float().unsqueeze(0)
            if file_sr != SAMPLE_RATE:
                waveform = torchaudio.functional.resample(waveform, file_sr, SAMPLE_RATE)

            # Align
            segments = align_chunk(model, processor, waveform, lines)

            # Offset timestamps to episode time
            for seg in segments:
                seg["start"] = round(seg["start"] + c_start, 3)
                seg["end"] = round(seg["end"] + c_start, 3)
                for w in seg["words"]:
                    w["start"] = round(w["start"] + c_start, 3)
                    w["end"] = round(w["end"] + c_start, 3)

            all_segments.extend(segments)

            # Quick stats
            zero_dur = sum(1 for s in segments if s["start"] == s["end"])
            if zero_dur:
                print(f"  {zero_dur}/{len(segments)} zero-duration segments")

    # Write output
    total_words = sum(len(s["words"]) for s in all_segments)
    zero_segs = sum(1 for s in all_segments if s["start"] == s["end"])
    zero_words = sum(1 for s in all_segments for w in s["words"] if w["start"] == w["end"])

    print(f"\nResults: {len(all_segments)} segments, {total_words} words")
    print(f"Zero-duration: {zero_segs} segments ({zero_segs/max(len(all_segments),1)*100:.1f}%), "
          f"{zero_words} words ({zero_words/max(total_words,1)*100:.1f}%)")

    Path(args.output_words).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_words, "w", encoding="utf-8") as f:
        json.dump(all_segments, f, ensure_ascii=False, indent=2)
    print(f"Written to {args.output_words}")

    metadata = finish_run(
        run,
        inputs={"video": args.video, "chunks_json": args.chunks},
        outputs={"words_json": args.output_words},
        settings={"model": MODEL_NAME, "language": "ja"},
        stats={
            "chunks_loaded": len(chunks_data),
            "segments_written": len(all_segments),
            "words_written": total_words,
            "zero_duration_segments": zero_segs,
            "zero_duration_words": zero_words,
        },
    )
    write_metadata(args.output_words, metadata)
    print(f"Metadata written: {metadata_path(args.output_words)}")


if __name__ == "__main__":
    main()
