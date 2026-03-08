"""Audio/video utilities: duration, chunk extraction, ffmpeg helpers."""

import subprocess
from typing import Optional


def get_duration(path: str) -> float:
    """Get audio/video duration in seconds via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def extract_audio_chunk(
    video_path: str, output_path: str,
    start_s: float = 0, duration_s: Optional[float] = None,
):
    """Extract mono MP3 audio chunk from video."""
    cmd = ["ffmpeg", "-y", "-ss", str(start_s), "-i", video_path]
    if duration_s is not None:
        cmd += ["-t", str(duration_s)]
    cmd += ["-vn", "-ac", "1", "-ab", "64k", "-f", "mp3", output_path]
    subprocess.run(cmd, capture_output=True, check=True)


def extract_16k_wav(input_path: str, output_path: str):
    """Extract 16kHz mono WAV via ffmpeg."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path,
         "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", output_path],
        capture_output=True, check=True,
    )


def extract_inline_video_chunk(
    video_path: str,
    output_path: str,
    start_s: float = 0,
    duration_s: Optional[float] = None,
    *,
    fps: float = 1.0,
    width: int = 640,
    audio_bitrate: str = "24k",
    crf: int = 36,
):
    """Extract a low-bitrate MP4 chunk suitable for inline multimodal requests."""
    cmd = ["ffmpeg", "-y", "-ss", str(start_s), "-i", video_path]
    if duration_s is not None:
        cmd += ["-t", str(duration_s)]
    cmd += [
        "-vf", f"fps={fps},scale={width}:-2:flags=lanczos",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-ac", "1",
        "-b:a", audio_bitrate,
        "-movflags", "+faststart",
        output_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
