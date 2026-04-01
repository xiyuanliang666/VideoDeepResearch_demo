"""Prompt routing helpers by benchmark family."""

from __future__ import annotations

from pathlib import Path


VIDEO_BENCHMARKS = {
    "videodr",
    "vdr_bench",
    "vdrbench",
}

IMAGE_BENCHMARKS = {
    "mmsearch_plus",
    "mmsearchplus",
    "browsecomp_vl",
    "browsecompvl",
    "mmdeepresearch_bench",
    "mmdeepresearchbench",
}


def normalize_benchmark_name(benchmark_name: str | None) -> str:
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


def resolve_prompt_group(benchmark_name: str | None) -> str:
    normalized = normalize_benchmark_name(benchmark_name)
    if normalized in VIDEO_BENCHMARKS:
        return "video"
    if normalized in IMAGE_BENCHMARKS:
        return "image"
    if "video" in normalized or normalized.startswith("vdr"):
        return "video"
    return "image"


def load_prompt_text(path: Path, fallback: str) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return fallback


def load_prompt_pair(
    *,
    prompt_root: Path,
    benchmark_name: str | None,
    fallback_system: str,
    fallback_user: str,
) -> tuple[str, str]:
    normalized = normalize_benchmark_name(benchmark_name)
    prompt_group = resolve_prompt_group(benchmark_name)

    candidates = []
    if normalized:
        candidates.append(prompt_root / "benchmarks" / normalized)
    candidates.append(prompt_root / prompt_group)
    candidates.append(prompt_root)

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
