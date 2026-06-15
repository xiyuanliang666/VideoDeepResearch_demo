"""video_frame_extract tool: extract frames from a time range at fixed FPS."""

from __future__ import annotations

import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any

from src.rsagent.protocols import ToolDispatcher

TOOL_NAME = "video_frame_extract"


class VideoFrameExtractDispatcher(ToolDispatcher):
    """Extract frames from video at specified time range and FPS."""

    def __init__(self, video_path: str, *, fallback: ToolDispatcher | None = None) -> None:
        self.video_path = str(Path(video_path).expanduser().resolve())
        self.fallback = fallback

    def dispatch(self, name: str, arguments: dict[str, Any], *, round_index: int | None = None) -> str:
        if name != TOOL_NAME:
            if self.fallback:
                return self.fallback.dispatch(name, arguments, round_index=round_index)
            return json.dumps({"ok": False, "error": f"Unknown tool: {name}"})
        return self._run(arguments)

    def _run(self, arguments: dict[str, Any]) -> str:
        time_start = arguments.get("time_start_sec")
        time_end = arguments.get("time_end_sec")
        if time_start is None or time_end is None:
            return json.dumps({"ok": False, "tool": TOOL_NAME, "error": "time_start_sec and time_end_sec are required"})

        time_start = float(time_start)
        time_end = float(time_end)
        if time_end <= time_start:
            return json.dumps({"ok": False, "tool": TOOL_NAME, "error": "time_end_sec must be > time_start_sec"})

        fps = float(arguments.get("fps", 2.0))
        max_frames = int(arguments.get("max_frames", 16))

        video_path = self.video_path
        arg_path = arguments.get("video_path") or arguments.get("video")
        if arg_path:
            p = Path(str(arg_path).strip()).expanduser().resolve()
            if p.is_file():
                video_path = str(p)

        if not Path(video_path).is_file():
            return json.dumps({"ok": False, "tool": TOOL_NAME, "error": f"Video not found: {video_path}"})

        try:
            frames = _extract_frames(video_path, time_start, time_end, fps, max_frames)
        except Exception as exc:
            return json.dumps({"ok": False, "tool": TOOL_NAME, "error": f"{type(exc).__name__}: {exc}"})

        return json.dumps({
            "ok": True,
            "tool": TOOL_NAME,
            "video_path": video_path,
            "time_range": [time_start, time_end],
            "fps_used": fps,
            "count": len(frames),
            "frames": frames,
        }, ensure_ascii=False)


def _extract_frames(
    video_path: str,
    time_start: float,
    time_end: float,
    fps: float,
    max_frames: int,
) -> list[dict[str, Any]]:
    """Extract frames using decord, return metadata (file paths saved to tmp)."""
    from decord import VideoReader, cpu
    from PIL import Image
    import numpy as np

    vr = VideoReader(video_path, ctx=cpu(0))
    video_fps = float(vr.get_avg_fps())
    total_frames = len(vr)

    frame_start = max(0, int(time_start * video_fps))
    frame_end = min(total_frames - 1, int(time_end * video_fps))

    if frame_end <= frame_start:
        return []

    duration = time_end - time_start
    n_frames = max(1, int(duration * fps))
    if n_frames > max_frames:
        n_frames = max_frames

    indices = np.linspace(frame_start, frame_end, n_frames, dtype=int).tolist()
    indices = sorted(set(indices))
    if len(indices) > max_frames:
        step = len(indices) / max_frames
        indices = [indices[int(i * step)] for i in range(max_frames)]

    max_side = _get_max_side()
    out_dir = Path(os.environ.get("VDR_FRAME_CACHE_DIR", "/tmp/vdr_frames"))
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for idx in indices:
        idx = max(0, min(idx, total_frames - 1))
        raw = vr[idx]
        if hasattr(raw, "asnumpy"):
            arr = raw.asnumpy()
        elif hasattr(raw, "numpy"):
            arr = raw.numpy()
        else:
            import torch
            arr = raw.detach().cpu().numpy() if isinstance(raw, torch.Tensor) else np.array(raw)

        im = Image.fromarray(arr)
        if max_side > 0:
            im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

        time_sec = round(idx / video_fps, 3)
        fname = f"frame_{time_sec:.3f}s_idx{idx}.jpg"
        fpath = out_dir / fname
        im.save(fpath, quality=90)

        frames.append({"file": str(fpath), "time_sec": time_sec, "frame_index": idx})

    return frames


def _get_max_side() -> int:
    raw = (os.environ.get("RESEARCH_IMAGE_MAX_SIDE") or "").strip()
    try:
        return max(0, int(raw)) if raw else 720
    except ValueError:
        return 720
