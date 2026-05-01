"""
Run ``test_data_frame/uniform_sample_frames.py`` before the research loop when configured.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from video_dr_agent.env_loader import collect_image_paths_from_directory


def run_uniform_overview_sampling(
    repo_root: Path | str,
    video_path: str,
    num_frames: int,
    output_dir: Path | str,
) -> list[str]:
    """
    Invoke OpenCV-based uniform sampling; returns sorted image paths under ``output_dir``.
    """
    root = Path(repo_root).resolve()
    script = root / "test_data_frame" / "uniform_sample_frames.py"
    if not script.is_file():
        raise FileNotFoundError(f"抽帧脚本不存在: {script}")

    v = Path(video_path).expanduser().resolve()
    if not v.is_file():
        raise FileNotFoundError(f"视频不存在: {v}")

    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(script),
        str(v),
        str(int(num_frames)),
        "-o",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    paths = collect_image_paths_from_directory(out)
    if not paths:
        raise RuntimeError(f"均匀抽帧后未在目录中发现图片: {out}")
    return paths
