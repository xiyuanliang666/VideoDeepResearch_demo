"""Environment file helpers."""

from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: str | Path = ".env", *, override: bool = False) -> bool:
    """Load KEY=VALUE pairs from a dotenv-style file into os.environ."""
    env_path = Path(path)
    if not env_path.exists():
        return False

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
    return True


def normalize_openai_base_url(raw: str) -> str:
    """Normalize an OpenAI-compatible base URL to API-root style.

    Examples:
    - https://host -> https://host/v1
    - https://host/v1 -> https://host/v1
    - https://host/v1/chat/completions -> https://host/v1
    """
    value = str(raw or "").strip().rstrip("/")
    if not value:
        return ""
    if value.endswith("/chat/completions"):
        value = value[: -len("/chat/completions")]
    if not value.endswith("/v1"):
        value = f"{value}/v1"
    return value
