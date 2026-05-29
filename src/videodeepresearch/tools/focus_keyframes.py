"""Optional FOCUS keyframe selection tool wrapper.

FOCUS is not vendored into the main framework tree. Set FOCUS_PACKAGE_DIR to a
local FOCUS checkout when enabling this tool.
"""

from __future__ import annotations

import json
import os
import re
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .base import StubToolDispatcher, ToolDispatcher

FOCUS_SELECT_KEYFRAMES_TOOL_NAME = "focus_select_keyframes"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _focus_package_dir() -> Path:
    configured = os.getenv("FOCUS_PACKAGE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return _project_root() / "external" / "FOCUS"


def _import_select_keyframe():
    root = _focus_package_dir()
    if not root.is_dir():
        raise ImportError(f"FOCUS directory not found: {root}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from select_keyframe import frame_indices_to_times_sec, run_focus_on_video_file  # noqa: WPS433

    return run_focus_on_video_file, frame_indices_to_times_sec


def _slug(text: str, max_len: int = 64) -> str:
    slug = (text or "focus").strip()
    slug = re.sub(r"[^\w\u4e00-\u9fff\-]+", "_", slug, flags=re.UNICODE)
    slug = slug.strip("_") or "focus"
    return slug[:max_len]


def default_focus_arg_namespace(**overrides: Any) -> SimpleNamespace:
    """Defaults aligned with the historical FOCUS single-video mode."""
    base: dict[str, Any] = {
        "dataset_name": "longvideobench",
        "dataset_path": "./datasets/longvideobench",
        "output_dir": "unused",
        "num_keyframes": 8,
        "batch_size": 32,
        "blip_model": "large",
        "top_ratio": 0.2,
        "temperature": 0.06,
        "min_gap_sec": 1.0,
        "disable_gap_below_sec": 0.2,
        "gap_ratio_of_avg": 0.25,
        "coarse_every_sec": 16.0,
        "fine_every_sec": 1.0,
        "zoom_ratio": 0.25,
        "min_coarse_segments": 8,
        "min_zoom_segments": 4,
        "region_half_window_sec": None,
        "extra_samples_per_region": 2,
        "min_variance_threshold": 1e-6,
        "fine_uniform_ratio": 0.5,
        "interpolation_method": "nearest",
        "final_min_arms": 4,
        "final_max_arms": 32,
        "seed": 42,
        "limit": None,
        "offset": 0,
        "video_path": None,
        "query": None,
        "query_file": None,
        "time_start_sec": None,
        "time_end_sec": None,
    }
    for key, value in overrides.items():
        if value is not None:
            base[key] = value
    return SimpleNamespace(**base)


class FocusKeyframeToolDispatcher:
    """Tool dispatcher for `focus_select_keyframes`; delegates unknown tools."""

    def __init__(
        self,
        output_root: str | Path,
        *,
        device: str = "cuda:0",
        fallback: ToolDispatcher | None = None,
        focus_arg_defaults: dict[str, Any] | None = None,
        default_video_path: str | None = None,
    ) -> None:
        self.output_root = Path(output_root).expanduser().resolve()
        self.device = device
        self.fallback = fallback or StubToolDispatcher()
        self.focus_arg_defaults = focus_arg_defaults or {}
        self.default_video_path = (
            str(Path(default_video_path).expanduser().resolve())
            if default_video_path
            else None
        )

    def dispatch(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        round_index: int | None = None,
    ) -> str:
        if name != FOCUS_SELECT_KEYFRAMES_TOOL_NAME:
            return self.fallback.dispatch(name, arguments, round_index=round_index)
        try:
            return self._run_focus(arguments, round_index=round_index)
        except Exception as exc:  # noqa: BLE001
            payload: dict[str, Any] = {
                "ok": False,
                "tool": FOCUS_SELECT_KEYFRAMES_TOOL_NAME,
                "error": f"{type(exc).__name__}: {exc}",
            }
            if "decord" in payload["error"].lower():
                payload["install_hint"] = "Install decord and set FOCUS_PACKAGE_DIR to a local FOCUS checkout."
            return json.dumps(payload, ensure_ascii=False)

    def _run_focus(self, arguments: dict[str, Any], *, round_index: int | None) -> str:
        question = (arguments.get("question") or arguments.get("query") or "").strip()
        if not question:
            raise ValueError("arguments.question or arguments.query is required")

        video_file, used_default_fallback = self._resolve_video(arguments)
        caption = _slug(str(arguments.get("frame_caption") or "focus"))
        round_id = int(arguments.get("round") or round_index or 1)
        output_dir = self._build_output_dir(video_file, round_id, caption)

        ns_kwargs = {**self.focus_arg_defaults}
        for key in (
            "num_keyframes",
            "batch_size",
            "blip_model",
            "time_start_sec",
            "time_end_sec",
            "seed",
            "top_ratio",
            "temperature",
            "min_gap_sec",
            "coarse_every_sec",
            "fine_every_sec",
            "zoom_ratio",
        ):
            if key in arguments and arguments[key] is not None:
                ns_kwargs[key] = arguments[key]
        args_ns = default_focus_arg_namespace(**ns_kwargs)

        import numpy as np
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("FOCUS/BLIP requires CUDA; torch.cuda.is_available() is False")

        run_focus, frame_indices_to_times = _import_select_keyframe()
        rng = np.random.default_rng(int(getattr(args_ns, "seed", 42)))
        selected, sampling_details, budget_stats = run_focus(
            video_file,
            question,
            args_ns,
            self.device,
            rng,
        )
        fps = float(sampling_details["video_metadata"]["fps"])
        times_sec = frame_indices_to_times(selected, fps)
        frames = self._export_frame_jpgs(video_file, [int(x) for x in selected], times_sec, output_dir / "frames")

        meta = {
            "ok": True,
            "tool": FOCUS_SELECT_KEYFRAMES_TOOL_NAME,
            "output_dir": str(output_dir),
            "times_sec": times_sec,
            "frame_indices": [int(x) for x in selected],
            "fps": fps,
            "frames": frames,
            "question": question,
            "video_path": video_file,
            "video_path_in_arguments": str(arguments.get("video_path") or arguments.get("video") or "") or None,
            "used_default_video_fallback": used_default_fallback,
            "round": round_id,
            "frame_caption": caption,
            "budget_used": budget_stats.get("budget_used") if isinstance(budget_stats, dict) else None,
        }
        (output_dir / "keyframes.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return json.dumps(meta, ensure_ascii=False)

    def _resolve_video(self, arguments: dict[str, Any]) -> tuple[str, bool]:
        raw = arguments.get("video_path") or arguments.get("video")
        raw_text = str(raw).strip() if raw is not None else ""
        if raw_text:
            path = Path(raw_text).expanduser().resolve()
            if path.is_file():
                return str(path), False
        if self.default_video_path:
            fallback = Path(self.default_video_path).expanduser().resolve()
            if fallback.is_file():
                return str(fallback), bool(raw_text)
        raise FileNotFoundError(
            f"Could not resolve local video file from arguments={raw_text!r} or default={self.default_video_path!r}"
        )

    def _build_output_dir(self, video_file: str, round_id: int, caption: str) -> Path:
        stem = _slug(Path(video_file).stem, max_len=80)
        base = self.output_root / f"{stem}_r{round_id}_{caption}"
        if not base.exists():
            base.mkdir(parents=True, exist_ok=True)
            return base
        target = self.output_root / f"{base.name}_{uuid.uuid4().hex[:8]}"
        target.mkdir(parents=True, exist_ok=True)
        return target

    @staticmethod
    def _export_frame_jpgs(
        video_path: str,
        frame_indices: list[int],
        times_sec: list[float],
        frames_dir: Path,
    ) -> list[dict[str, Any]]:
        from decord import VideoReader, cpu
        from PIL import Image
        import torch

        vr = VideoReader(video_path, ctx=cpu(0))
        frames_dir.mkdir(parents=True, exist_ok=True)
        files: list[dict[str, Any]] = []
        for index, frame_index in enumerate(frame_indices):
            frame_index = max(0, min(int(frame_index), len(vr) - 1))
            raw = vr[frame_index]
            if hasattr(raw, "asnumpy"):
                arr = raw.asnumpy()
            elif isinstance(raw, torch.Tensor):
                arr = raw.detach().cpu().numpy()
            elif hasattr(raw, "numpy"):
                arr = raw.numpy()
            else:
                raise TypeError(f"Unexpected frame type from decord: {type(raw)!r}")
            filename = f"{index:03d}_frame{frame_index}.jpg"
            path = frames_dir / filename
            Image.fromarray(arr).save(path, quality=92)
            time_sec = times_sec[index] if index < len(times_sec) else 0.0
            files.append(
                {
                    "file": str(path),
                    "rel_path": f"frames/{filename}",
                    "frame_index": frame_index,
                    "time_sec": round(float(time_sec), 6),
                }
            )
        return files


def build_focus_dispatcher(
    output_root: str | Path,
    *,
    device: str = "cuda:0",
    focus_arg_defaults: dict[str, Any] | None = None,
    default_video_path: str | None = None,
    fallback: ToolDispatcher | None = None,
) -> ToolDispatcher:
    """Build the unified FOCUS dispatcher."""
    return FocusKeyframeToolDispatcher(
        output_root,
        device=device,
        fallback=fallback or StubToolDispatcher(),
        focus_arg_defaults=focus_arg_defaults,
        default_video_path=default_video_path,
    )
