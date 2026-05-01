"""
Default research (Qwen3-VL) system prompt resolution — aligns with Video DeepResearch plan:
system + first multimodal user; tool protocol lives in the system prompt file.
"""

from __future__ import annotations

import os
from pathlib import Path


def resolve_research_system_content(
    repo_root: Path | str,
    cli_value: str | None,
    *,
    allow_default_file: bool = True,
) -> str | None:
    """
    Resolve the research model system prompt.

    Priority:
    1. Non-empty ``cli_value`` (inline system text from ``--research-system``).
    2. ``RESEARCH_SYSTEM_PROMPT_PATH`` env (repo-relative or absolute path to a UTF-8 text file).
    3. If ``allow_default_file``: ``<repo>/prompts/research_system_prompt.txt``, then
       ``<repo>/research_system_prompt.txt`` (legacy).

    Returns ``None`` if nothing is configured (ResearchContext will omit system message).
    """
    inline = (cli_value or "").strip()
    if inline:
        return inline

    root = Path(repo_root).resolve()
    env_path = (os.environ.get("RESEARCH_SYSTEM_PROMPT_PATH") or "").strip()
    candidates: list[Path] = []
    if env_path:
        p = Path(env_path)
        candidates.append(p if p.is_absolute() else (root / p).resolve())
    if allow_default_file:
        candidates.append(root / "prompts" / "research_system_prompt.txt")
        candidates.append(root / "research_system_prompt.txt")

    for path in candidates:
        if path.is_file():
            return path.read_text(encoding="utf-8").strip() or None
    return None
