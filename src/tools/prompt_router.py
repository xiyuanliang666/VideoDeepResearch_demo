"""Prompt routing by media input type (video / image)."""

from __future__ import annotations

from pathlib import Path


def input_type_to_prompt_group(input_type: str | None) -> str:
    """Map sample.input_type to a prompt group: 'video' or 'image'."""
    value = (input_type or "").strip().lower()
    if value in {"video"}:
        return "video"
    if value in {"image", "image_set"}:
        return "image"
    if "video" in value:
        return "video"
    return "image"


def normalize_benchmark_name(benchmark_name: str | None) -> str:
    """Normalize a benchmark name for display/logging only (not for routing)."""
    value = (benchmark_name or "").strip().lower()
    if not value:
        return ""
    out = []
    last_sep = False
    for char in value:
        if char.isalnum():
            out.append(char)
            last_sep = False
            continue
        if not last_sep:
            out.append("_")
            last_sep = True
    return "".join(out).strip("_")


def resolve_prompt_group(prompt_group: str | None) -> str:
    """Resolve prompt_group to 'video' or 'image' with fallback."""
    value = (prompt_group or "").strip().lower()
    if value in {"video", "image"}:
        return value
    if "video" in value:
        return "video"
    return "image"


def load_prompt_text(path: Path, fallback: str) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return fallback


def load_prompt_pair(
    *,
    prompt_root: Path,
    prompt_group: str | None,
    fallback_system: str,
    fallback_user: str,
) -> tuple[str, str]:
    """Load (system, user) prompt pair from prompt_root/{prompt_group}/, falling back to prompt_root/."""
    group = resolve_prompt_group(prompt_group)

    candidates = [
        prompt_root / group,
        prompt_root,
    ]

    system_text = ""
    user_text = ""
    for candidate in candidates:
        if not system_text:
            system_path = candidate / "system.txt"
            if system_path.exists():
                system_text = load_prompt_text(system_path, "")
        if not user_text:
            user_path = candidate / "user.txt"
            if user_path.exists():
                user_text = load_prompt_text(user_path, "")
        if system_text and user_text:
            break

    if not system_text:
        system_text = fallback_system
    if not user_text:
        user_text = fallback_user
    return system_text, user_text
