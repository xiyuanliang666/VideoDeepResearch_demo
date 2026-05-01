"""
Run ``test_data_frame/uniform_sample_frames.py`` before the research loop when configured.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from video_dr_agent.env_loader import collect_image_paths_from_directory

_META_NAME = "uniform_overview_meta.json"


def run_uniform_overview_sampling(
    repo_root: Path | str,
    video_path: str,
    num_frames: int,
    output_dir: Path | str,
) -> tuple[list[str], dict[str, Any] | None]:
    """
    Invoke OpenCV-based uniform sampling; returns sorted image paths under ``output_dir``
    and optional ``uniform_overview_meta.json`` payload (timestamps / frame indices).
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

    meta: dict[str, Any] | None = None
    meta_path = out / _META_NAME
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = None

    return paths, meta


def format_uniform_overview_planning_appendix(
    *,
    meta: dict[str, Any] | None,
    research_question: str,
) -> str:
    """
    English Host appendix: maps overview JPEG order to timestamps for the Planning phase.
    """
    if not meta or meta.get("kind") != "uniform_overview":
        return ""

    q = (research_question or "").strip()
    fps = meta.get("fps")
    total = meta.get("total_frames")
    frames = meta.get("frames")
    if not isinstance(frames, list):
        return ""

    lines: list[str] = [
        "[Host — Uniform overview schedule]",
        "",
        "The table below matches the **uniformly sampled overview images** in your **first** user message: "
        "they appear as consecutive `image_url` parts **in this row order** (after any leading `video_url`, "
        "if present), **before** the final task text part.",
        "",
        "**User task / question (for planning, repeat):**",
        q if q else "(empty)",
        "",
    ]
    if fps is not None or total is not None:
        extra = []
        if fps is not None:
            extra.append(f"container FPS ≈ {fps}")
        if total is not None:
            extra.append(f"reported frame count ≈ {total}")
        lines.append("Video stats (" + "; ".join(extra) + "). Times are **approximate** from `frame_index / fps`.")
        lines.append("")

    lines.append("| # (1-based) | source `frame_index` | approx `time_sec` | JPEG |")
    lines.append("|---:|---:|---:|---|")

    for j, row in enumerate(frames, start=1):
        if not isinstance(row, dict):
            continue
        idx = row.get("frame_index", "")
        ts = row.get("time_sec")
        if isinstance(ts, (int, float)):
            ts_s = f"{float(ts):.3f}"
        else:
            ts_s = "n/a"
        fn = row.get("filename", "")
        lines.append(f"| {j} | {idx} | {ts_s} | `{fn}` |")

    lines.extend(
        [
            "",
            "**FOCUS hint:** When you later call `focus_select_keyframes`, pass **`time_start_sec`** and "
            "**`time_end_sec`** to restrict search to the on-timeline window that covers the overview rows "
            "relevant to your sub-question (add a small padding of a few seconds when unsure).",
        ]
    )
    return "\n".join(lines).strip()
