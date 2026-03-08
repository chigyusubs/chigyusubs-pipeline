"""Silero VAD speech detection."""

import os
import struct

from chigyusubs.audio import extract_16k_wav


def run_silero_vad(
    audio_path: str,
    work_dir: str | None = None,
    *,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
) -> list[dict]:
    """Run Silero VAD on audio file, return speech segments as list of {start, end}."""
    import torch

    print("  Loading Silero VAD model...", flush=True)
    model, utils = torch.hub.load(
        "snakers4/silero-vad", "silero_vad", trust_repo=True,
    )
    get_speech_timestamps = utils[0]

    if work_dir:
        wav_path = os.path.join(work_dir, "vad_16k.wav")
    else:
        wav_path = audio_path + ".vad_16k.wav"
    print(f"  Extracting 16kHz WAV from {audio_path}...", flush=True)
    extract_16k_wav(audio_path, wav_path)

    print(f"  Loading WAV: {wav_path}", flush=True)
    with open(wav_path, "rb") as f:
        raw = f.read()
    n_samples = (len(raw) - 44) // 2
    samples = struct.unpack(f"<{n_samples}h", raw[44:44 + n_samples * 2])
    wav = torch.tensor(samples, dtype=torch.float32) / 32768.0
    sr = 16000
    print(f"  Audio: {len(wav) / sr:.1f}s @ {sr}Hz, mono", flush=True)

    print("  Running VAD inference...", flush=True)
    timestamps = get_speech_timestamps(
        wav, model,
        sampling_rate=sr,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
    )

    segments = [{"start": ts["start"] / sr, "end": ts["end"] / sr} for ts in timestamps]
    print(f"  Found {len(segments)} speech segments", flush=True)
    return segments
