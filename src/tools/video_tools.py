"""Video utilities for the v0 scaffold."""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

from src.schemas import ClipUnit
from src.tools.io_tools import ensure_dir


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def ffprobe_available() -> bool:
    return shutil.which("ffprobe") is not None


def probe_duration(video_path: str) -> float | None:
    """Return video duration in seconds when ffprobe is available."""
    if not ffprobe_available():
        return None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def _format_seconds(seconds: float) -> str:
    whole = int(max(seconds, 0))
    hours = whole // 3600
    minutes = (whole % 3600) // 60
    secs = whole % 60
    return f"{hours:02d}-{minutes:02d}-{secs:02d}"


def cut_video(video_path: str, clip_duration: int, output_dir: str) -> list[ClipUnit]:
    """Cut a raw video into fixed-duration clips when ffmpeg is available.

    Falls back to a single clip pointing at the original video when ffmpeg/ffprobe
    is unavailable. This keeps the v0 preprocessing chain usable for structure and
    trace validation.
    """
    video_file = Path(video_path)
    if not video_file.exists():
        raise FileNotFoundError(f"Video path does not exist: {video_path}")

    ensure_dir(output_dir)
    duration = probe_duration(video_path)
    video_name = video_file.stem

    if not ffmpeg_available() or duration is None or clip_duration <= 0:
        return [
            ClipUnit(
                clip_id=f"{video_name}_clip_0000",
                video_name=video_name,
                clip_path=str(video_file),
                start_time=0.0,
                end_time=duration or 0.0,
                start_time_str=_format_seconds(0.0),
                end_time_str=_format_seconds(duration or 0.0),
                quality_flags=["fallback_single_clip"],
            )
        ]

    clip_units: list[ClipUnit] = []
    num_clips = max(1, math.ceil(duration / clip_duration))

    for index in range(num_clips):
        start_time = float(index * clip_duration)
        end_time = min(duration, float((index + 1) * clip_duration))
        clip_filename = (
            f"{video_name}_clip_{index:04d}_{_format_seconds(start_time)}"
            f"_to_{_format_seconds(end_time)}.mp4"
        )
        clip_path = Path(output_dir) / clip_filename

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_file),
            "-ss",
            str(start_time),
            "-t",
            str(max(end_time - start_time, 0.1)),
            "-c",
            "copy",
            str(clip_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        quality_flags: list[str] = []
        if result.returncode != 0:
            quality_flags.append("ffmpeg_clip_failed")
            clip_path = video_file

        clip_units.append(
            ClipUnit(
                clip_id=f"{video_name}_clip_{index:04d}",
                video_name=video_name,
                clip_path=str(clip_path),
                start_time=start_time,
                end_time=end_time,
                start_time_str=_format_seconds(start_time),
                end_time_str=_format_seconds(end_time),
                quality_flags=quality_flags,
            )
        )

    return clip_units


def extract_frames(clip_path: str, fps: float, max_frames: int, output_dir: str) -> list[str]:
    """Extract representative frames from a clip.

    When ffmpeg is unavailable, returns an empty list and lets the caller record
    the fallback in trace metadata.
    """
    ensure_dir(output_dir)
    if not ffmpeg_available() or fps <= 0 or max_frames <= 0:
        return []

    clip_file = Path(clip_path)
    frame_pattern = str(Path(output_dir) / "frame_%04d.jpg")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(clip_file),
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",
        frame_pattern,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return []

    frames = sorted(str(path) for path in Path(output_dir).glob("frame_*.jpg"))
    return frames[:max_frames]
