"""Multimodal observation helpers via a unified OpenAI-compatible gateway."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

from .env_tools import normalize_openai_base_url
from .model_router import get_model


PROMPT_ROOT = Path(__file__).resolve().parents[2] / "prompts" / "observation_multimodal"
_LAST_MULTIMODAL_ERROR = ""


def get_last_multimodal_error() -> str:
    return _LAST_MULTIMODAL_ERROR


def _set_multimodal_error(message: str) -> None:
    global _LAST_MULTIMODAL_ERROR
    _LAST_MULTIMODAL_ERROR = message.strip()[:240]


def _looks_like_image(path: Path) -> bool:
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def _file_to_data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type:
        mime_type = "application/octet-stream"
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{data}"


def _extract_json_dict(text: str) -> dict[str, Any] | None:
    payload = text.strip()
    if not payload:
        return None
    try:
        data = json.loads(payload)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    left = payload.find("{")
    right = payload.rfind("}")
    if left < 0 or right <= left:
        return None
    try:
        data = json.loads(payload[left : right + 1])
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        return None
    return None


def _load_prompt_text(path: Path, fallback: str) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return fallback


def analyze_observation_multimodal(
    *,
    question: str,
    media_paths: list[str],
    model_profile: dict | None = None,
    max_tokens: int = 500,
) -> dict[str, Any] | None:
    """Analyze media and return observation hints through the unified gateway."""
    _set_multimodal_error("")
    api_key = os.getenv("LLM_API_KEY", "").strip()
    base_url = normalize_openai_base_url(os.getenv("LLM_BASE_URL", ""))
    if not api_key or not base_url:
        _set_multimodal_error("missing LLM_API_KEY or LLM_BASE_URL")
        return None

    try:
        from openai import OpenAI
    except ModuleNotFoundError:
        _set_multimodal_error("openai package is not installed")
        return None

    system_prompt = _load_prompt_text(
        PROMPT_ROOT / "system.txt",
        "You are a careful multimodal analyst. Return JSON only.",
    )
    user_template = _load_prompt_text(
        PROMPT_ROOT / "user.txt",
        "Question:\n{question}\n\nReturn JSON with scene_clues/speech_clues/candidate_entities/candidate_actions/confidence.",
    )
    user_text = user_template.replace("{question}", question)

    content_parts: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for item in media_paths:
        media_path = Path(item)
        if not media_path.exists() or not media_path.is_file() or not _looks_like_image(media_path):
            continue
        try:
            data_url = _file_to_data_url(media_path)
        except OSError:
            continue
        content_parts.append({"type": "image_url", "image_url": {"url": data_url}})

    if len(content_parts) <= 1:
        _set_multimodal_error("no valid image media_paths were found")
        return None

    timeout_seconds = float(os.getenv("TIMEOUT_SECONDS", "30") or 30)
    model_name = get_model("vision", model_profile)
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_parts},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
    except Exception as exc:  # noqa: BLE001
        _set_multimodal_error(f"chat.completions failed: {type(exc).__name__}: {exc}")
        return None

    content = ""
    if getattr(response, "choices", None):
        message = response.choices[0].message
        content = getattr(message, "content", "") or ""
    parsed = _extract_json_dict(content)
    if parsed is None:
        _set_multimodal_error("model response is not valid JSON object")
    return parsed
