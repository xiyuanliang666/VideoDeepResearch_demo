"""Ablation-mode helpers shared by workflow and agentic runners."""

from __future__ import annotations

SUPPORTED_ABLATION_MODES = {"full_vdr", "video_only", "web_only", "text_only"}


def normalize_ablation_mode(task_profile: dict | None = None) -> str:
    value = str((task_profile or {}).get("ablation_mode") or "full_vdr").strip().lower()
    return value if value in SUPPORTED_ABLATION_MODES else "full_vdr"


def uses_video(ablation_mode: str) -> bool:
    return ablation_mode in {"full_vdr", "video_only"}


def uses_web(ablation_mode: str) -> bool:
    return ablation_mode in {"full_vdr", "web_only"}


def is_text_only(ablation_mode: str) -> bool:
    return ablation_mode == "text_only"
