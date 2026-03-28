"""Audio utilities for the v0 scaffold."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from src.tools.io_tools import ensure_dir


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def extract_audio(video_path: str, output_dir: str) -> str:
    """Extract a mono wav track from a video when ffmpeg is available."""
    ensure_dir(output_dir)
    target = Path(output_dir) / f"{Path(video_path).stem}.wav"
    if not ffmpeg_available():
        return ""

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(target),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return ""
    return str(target)


def slice_audio(audio_path: str, start_time: float, end_time: float, output_dir: str) -> str:
    """Slice an audio segment by timestamp when ffmpeg is available."""
    ensure_dir(output_dir)
    target = Path(output_dir) / (
        f"{Path(audio_path).stem}_{int(start_time):06d}_{int(end_time):06d}.wav"
    )
    if not ffmpeg_available():
        return ""

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        audio_path,
        "-ss",
        str(start_time),
        "-t",
        str(max(end_time - start_time, 0.1)),
        str(target),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return ""
    return str(target)
