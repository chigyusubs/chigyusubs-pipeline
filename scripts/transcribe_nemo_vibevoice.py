#!/usr/bin/env python3
"""Transcribe audio using NeMo VAD/diarization + VibeVoice ASR.

Phase 1 (NeMo, .venv-nemo): MarbleNet VAD + TitaNet speaker diarization → RTTM
Phase 2 (VibeVoice, .venv-kotoba): Transcribe each speech segment → text
Phase 3 (merge): Combine speaker labels + transcribed text → JSON + VTT

Usage:
  python scripts/transcribe_nemo_vibevoice.py \
    --video samples/episodes/.../source/video.mp4 \
    --output samples/episodes/.../transcription/406_nemo_vv.json
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

NEMO_PYTHON = ".venv-nemo/bin/python"
VIBEVOICE_PYTHON = ".venv-kotoba/bin/python"


def get_duration(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def extract_16k_wav(video_path: str, output_path: str):
    """Extract 16kHz mono WAV for NeMo."""
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", output_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def extract_segment_wav(video_path: str, output_path: str, start: float, duration: float):
    """Extract a single segment as 24kHz mono WAV for VibeVoice."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", video_path,
        "-t", str(duration),
        "-vn", "-ac", "1", "-ar", "24000", "-f", "wav", output_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)


# ---------------------------------------------------------------------------
# Phase 1: NeMo diarization (runs in .venv-nemo)
# ---------------------------------------------------------------------------

NEMO_DIARIZE_SCRIPT = '''
import json
import os
import sys
import torch
from omegaconf import OmegaConf
from nemo.collections.asr.models import ClusteringDiarizer

audio_path = sys.argv[1]
out_dir = sys.argv[2]
rttm_out = sys.argv[3]

os.makedirs(out_dir, exist_ok=True)

# Create manifest
manifest_path = os.path.join(out_dir, "manifest.json")
with open(manifest_path, "w") as f:
    json.dump({
        "audio_filepath": os.path.abspath(audio_path),
        "offset": 0, "duration": None, "label": "infer",
        "text": "-", "num_speakers": None,
        "rttm_filepath": None, "uem_filepath": None
    }, f)
    f.write("\\n")

# Load config
config = OmegaConf.load("diarizer_config.yaml")
config.diarizer.manifest_filepath = os.path.abspath(manifest_path)
config.diarizer.out_dir = os.path.abspath(out_dir)
config.diarizer.speaker_embeddings.parameters.save_embeddings = False

# Run
diarizer = ClusteringDiarizer(cfg=config)
diarizer.diarize()

# Find RTTM
rttm_dir = os.path.join(out_dir, "pred_rttms")
rttm_files = [f for f in os.listdir(rttm_dir) if f.endswith(".rttm")]
if not rttm_files:
    print("ERROR: No RTTM produced", file=sys.stderr)
    sys.exit(1)

# Copy to expected output path
import shutil
shutil.copy(os.path.join(rttm_dir, rttm_files[0]), rttm_out)
print(f"RTTM written to {rttm_out}")
'''


def run_nemo_diarization(
    wav_16k_path: str, work_dir: str, nemo_python: str = NEMO_PYTHON,
) -> str:
    """Run NeMo diarization, return path to RTTM file."""
    rttm_path = os.path.join(work_dir, "diarization.rttm")
    nemo_out = os.path.join(work_dir, "nemo_out")

    script_path = os.path.join(work_dir, "nemo_diarize.py")
    with open(script_path, "w") as f:
        f.write(NEMO_DIARIZE_SCRIPT)

    print("Phase 1: Running NeMo diarization...")
    result = subprocess.run(
        [nemo_python, script_path, wav_16k_path, nemo_out, rttm_path],
        capture_output=True, text=True,
    )
    if result.stdout:
        # Print only key lines
        for line in result.stdout.strip().split("\n"):
            if any(kw in line.lower() for kw in ["rttm", "speaker", "segment", "error", "diariz"]):
                print(f"  [NeMo] {line}")
    if result.returncode != 0:
        print(f"NeMo failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    return rttm_path


def parse_rttm(rttm_path: str) -> list[dict]:
    """Parse RTTM into speaker segments."""
    segments = []
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                start = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                segments.append({
                    "start": start,
                    "end": start + duration,
                    "speaker": speaker,
                })
    segments.sort(key=lambda s: s["start"])
    return segments


# ---------------------------------------------------------------------------
# Phase 2: VibeVoice transcription (runs in .venv-kotoba)
# ---------------------------------------------------------------------------

VIBEVOICE_TRANSCRIBE_SCRIPT = '''
import json
import sys
import torch
from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

segments_path = sys.argv[1]
model_path = sys.argv[2]
output_path = sys.argv[3]

with open(segments_path) as f:
    segments = json.load(f)

print(f"Loading VibeVoice model: {model_path}")
processor = VibeVoiceASRProcessor.from_pretrained(
    model_path, language_model_pretrained_name="Qwen/Qwen2.5-7B"
)
model = VibeVoiceASRForConditionalGeneration.from_pretrained(
    model_path, device_map="auto", attn_implementation="sdpa", trust_remote_code=True
)
model.eval()
print("Model loaded")

results = []
for i, seg in enumerate(segments):
    wav_path = seg["wav_path"]
    duration = seg["end"] - seg["start"]
    # Scale max tokens by duration: ~20 tokens/sec for Japanese
    max_tokens = max(256, min(8192, int(duration * 25)))

    print(f"  [{i+1}/{len(segments)}] {seg['start']:.1f}-{seg['end']:.1f}s ({duration:.1f}s, speaker={seg['speaker']})")

    try:
        inputs = processor(
            audio=[wav_path], sampling_rate=None,
            return_tensors="pt", padding=True, add_generation_prompt=True
        )
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=max_tokens,
                pad_token_id=processor.pad_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                do_sample=False,
                repetition_penalty=1.2,
            )

        input_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, input_length:]
        eos_positions = (generated_ids == processor.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            generated_ids = generated_ids[:eos_positions[0] + 1]

        text = processor.decode(generated_ids, skip_special_tokens=True)

        # Try structured output
        try:
            structured = processor.post_process_transcription(text)
        except Exception:
            structured = None

        results.append({
            "start": seg["start"],
            "end": seg["end"],
            "speaker": seg["speaker"],
            "text": text.strip(),
            "structured": structured,
        })
        print(f"    -> {len(text)} chars")
    except Exception as e:
        print(f"    ERROR: {e}")
        results.append({
            "start": seg["start"],
            "end": seg["end"],
            "speaker": seg["speaker"],
            "text": "",
            "structured": None,
            "error": str(e),
        })

with open(output_path, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"Wrote {len(results)} transcribed segments to {output_path}")
'''


def merge_adjacent_segments(segments: list[dict], max_gap: float = 1.0, max_duration: float = 30.0) -> list[dict]:
    """Merge adjacent segments from the same speaker."""
    if not segments:
        return []

    merged = [segments[0].copy()]
    for seg in segments[1:]:
        prev = merged[-1]
        gap = seg["start"] - prev["end"]
        combined_dur = seg["end"] - prev["start"]
        if seg["speaker"] == prev["speaker"] and gap <= max_gap and combined_dur <= max_duration:
            prev["end"] = seg["end"]
        else:
            merged.append(seg.copy())
    return merged


def run_vibevoice_transcription(
    segments: list[dict],
    video_path: str,
    work_dir: str,
    model_path: str,
) -> list[dict]:
    """Extract audio per segment, run VibeVoice, return transcribed segments."""
    # Extract WAV for each segment
    print(f"\nPhase 2: Extracting {len(segments)} audio segments...")
    seg_wavs_dir = os.path.join(work_dir, "segment_wavs")
    os.makedirs(seg_wavs_dir, exist_ok=True)

    for i, seg in enumerate(segments):
        wav_path = os.path.join(seg_wavs_dir, f"seg_{i:04d}.wav")
        pad = 0.2  # small padding for context
        start = max(0, seg["start"] - pad)
        duration = (seg["end"] + pad) - start
        extract_segment_wav(video_path, wav_path, start, duration)
        seg["wav_path"] = wav_path

    # Write segments JSON for the subprocess
    segs_path = os.path.join(work_dir, "segments_for_vv.json")
    with open(segs_path, "w") as f:
        json.dump(segments, f, ensure_ascii=False)

    output_path = os.path.join(work_dir, "vv_results.json")
    script_path = os.path.join(work_dir, "vv_transcribe.py")
    with open(script_path, "w") as f:
        f.write(VIBEVOICE_TRANSCRIBE_SCRIPT)

    print(f"Phase 2: Running VibeVoice on {len(segments)} segments...")
    result = subprocess.run(
        [VIBEVOICE_PYTHON, script_path, segs_path, model_path, output_path],
        text=True,
    )
    if result.returncode != 0:
        print("VibeVoice failed", file=sys.stderr)
        sys.exit(1)

    with open(output_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Phase 3: Merge and output
# ---------------------------------------------------------------------------

def _format_ts(seconds: float) -> str:
    total_ms = round(seconds * 1000)
    h = total_ms // 3600000
    mi = (total_ms % 3600000) // 60000
    s = (total_ms % 60000) // 1000
    ms = total_ms % 1000
    return f"{h:02d}:{mi:02d}:{s:02d}.{ms:03d}"


def write_outputs(results: list[dict], output_path: str):
    """Write JSON, VTT, and plain text outputs."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # VTT with speaker labels
    vtt_path = output_path.replace(".json", ".vtt")
    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for r in results:
            if not r.get("text"):
                continue
            f.write(f"{_format_ts(r['start'])} --> {_format_ts(r['end'])}\n")
            speaker = r.get("speaker", "")
            text = r["text"].strip()
            if speaker:
                f.write(f"{speaker}: {text}\n\n")
            else:
                f.write(f"{text}\n\n")

    # Plain text (speech only, for alignment)
    txt_path = output_path.replace(".json", "_speech.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for r in results:
            if r.get("text", "").strip():
                f.write(r["text"].strip() + "\n")

    return vtt_path, txt_path


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe with NeMo VAD/diarization + VibeVoice ASR."
    )
    parser.add_argument("--video", required=True, help="Input video file.")
    parser.add_argument("--output", default="", help="Output JSON path.")
    parser.add_argument(
        "--model", default="scerz/VibeVoice-ASR-4bit",
        help="VibeVoice model path (default: scerz/VibeVoice-ASR-4bit).",
    )
    parser.add_argument(
        "--max-segment-s", type=float, default=30.0,
        help="Max merged segment duration for VibeVoice (default: 30s).",
    )
    parser.add_argument(
        "--merge-gap-s", type=float, default=1.0,
        help="Max gap to merge same-speaker segments (default: 1.0s).",
    )
    args = parser.parse_args()

    video_path = args.video
    if not os.path.exists(video_path):
        print(f"Error: Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    if not args.output:
        stem = Path(video_path).stem
        out_dir = Path(video_path).parent.parent / "transcription"
        args.output = str(out_dir / f"{stem}_nemo_vv.json")

    with tempfile.TemporaryDirectory() as work_dir:
        # Extract 16kHz WAV for NeMo
        wav_16k = os.path.join(work_dir, "audio_16k.wav")
        print(f"Extracting 16kHz audio...")
        extract_16k_wav(video_path, wav_16k)
        duration = get_duration(wav_16k)
        print(f"Duration: {duration:.0f}s ({duration/60:.1f} min)")

        # Phase 1: NeMo
        rttm_path = run_nemo_diarization(wav_16k, work_dir)
        segments = parse_rttm(rttm_path)
        unique_speakers = set(s["speaker"] for s in segments)
        print(f"\nDiarization: {len(segments)} segments, {len(unique_speakers)} speakers")
        for spk in sorted(unique_speakers):
            spk_segs = [s for s in segments if s["speaker"] == spk]
            total_dur = sum(s["end"] - s["start"] for s in spk_segs)
            print(f"  {spk}: {len(spk_segs)} segments, {total_dur:.1f}s total")

        # Merge adjacent same-speaker segments
        merged = merge_adjacent_segments(segments, max_gap=args.merge_gap_s, max_duration=args.max_segment_s)
        print(f"\nMerged: {len(segments)} -> {len(merged)} segments")

        # Phase 2: VibeVoice
        results = run_vibevoice_transcription(merged, video_path, work_dir, args.model)

    # Phase 3: Output
    vtt_path, txt_path = write_outputs(results, args.output)

    speech_results = [r for r in results if r.get("text", "").strip()]
    print(f"\nDone!")
    print(f"  JSON: {args.output}")
    print(f"  VTT:  {vtt_path}")
    print(f"  Text: {txt_path}")
    print(f"  {len(speech_results)} transcribed segments")


if __name__ == "__main__":
    main()
