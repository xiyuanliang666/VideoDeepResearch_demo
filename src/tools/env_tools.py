"""Environment file helpers."""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass


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


@dataclass(frozen=True)
class OpenAIConfig:
    api_key: str
    base_url: str
    api_key_var: str
    base_url_var: str

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.base_url)

    @property
    def missing_message(self) -> str:
        return f"missing {self.api_key_var} or {self.base_url_var}"


def resolve_openai_config(role: str) -> OpenAIConfig:
    """Resolve the shared OpenAI-compatible gateway for any model role.

    All LLM/VLM stages intentionally share LLM_API_KEY and LLM_BASE_URL. Stage
    selection is controlled by model variables such as VISION_MODEL,
    STATE_CONSTRUCTOR_MODEL, and JUDGE_MODEL.
    """
    _ = role
    return OpenAIConfig(
        api_key=os.getenv("LLM_API_KEY", "").strip(),
        base_url=normalize_openai_base_url(os.getenv("LLM_BASE_URL", "")),
        api_key_var="LLM_API_KEY",
        base_url_var="LLM_BASE_URL",
    )
