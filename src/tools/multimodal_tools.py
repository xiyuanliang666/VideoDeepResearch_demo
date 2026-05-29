"""Multimodal observation helpers via a unified OpenAI-compatible gateway."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

from .env_tools import resolve_openai_config
from .model_router import get_model
from .prompt_router import load_prompt_pair


PROMPT_ROOT = Path(__file__).resolve().parents[2] / "prompts" / "observation_multimodal"
_LAST_MULTIMODAL_ERROR = ""


def get_last_multimodal_error() -> str:
    return _LAST_MULTIMODAL_ERROR


def _set_multimodal_error(message: str) -> None:
    global _LAST_MULTIMODAL_ERROR
    _LAST_MULTIMODAL_ERROR = message.strip()[:240]


def _looks_like_image(path: Path) -> bool:
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def _looks_like_audio(path: Path) -> bool:
    return path.suffix.lower() in {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}


def _audio_format(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    if suffix in {"mp3", "wav"}:
        return suffix
    return suffix or "wav"


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


def analyze_observation_multimodal(
    *,
    question: str,
    media_paths: list[str],
    audio_paths: list[str] | None = None,
    prompt_group: str = "",
    model_profile: dict | None = None,
    max_tokens: int = 500,
) -> dict[str, Any] | None:
    """Analyze media and return observation hints through the unified gateway."""
    _set_multimodal_error("")
    role = "audio_vision" if audio_paths else "vision"
    config = resolve_openai_config(role)
    if not config.available:
        _set_multimodal_error(config.missing_message)
        return None

    try:
        from openai import OpenAI
    except ModuleNotFoundError:
        _set_multimodal_error("openai package is not installed")
        return None

    system_prompt, user_template = load_prompt_pair(
        prompt_root=PROMPT_ROOT,
        prompt_group=prompt_group,
        fallback_system="You are a careful multimodal analyst. Return JSON only.",
        fallback_user=(
            "Question:\n{question}\n\nReturn JSON with "
            "scene_clues/speech_clues/candidate_entities/candidate_actions/confidence."
        ),
    )
    user_text = user_template.replace("{question}", question)

    content_parts: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    image_count = 0
    for item in media_paths:
        media_path = Path(item)
        if not media_path.exists() or not media_path.is_file() or not _looks_like_image(media_path):
            continue
        try:
            data_url = _file_to_data_url(media_path)
        except OSError:
            continue
        content_parts.append({"type": "image_url", "image_url": {"url": data_url}})
        image_count += 1

    audio_count = 0
    enable_audio = os.getenv("ENABLE_AUDIO_IN_MULTIMODAL", "true").strip().lower() in {"1", "true", "yes", "on"}
    if enable_audio:
        max_audio_files = int(os.getenv("MAX_AUDIO_FILES_PER_MULTIMODAL_CALL", "1") or 1)
        for item in (audio_paths or [])[:max_audio_files]:
            audio_path = Path(item)
            if not audio_path.exists() or not audio_path.is_file() or not _looks_like_audio(audio_path):
                continue
            try:
                audio_data = base64.b64encode(audio_path.read_bytes()).decode("utf-8")
            except OSError:
                continue
            content_parts.append(
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio_data, "format": _audio_format(audio_path)},
                }
            )
            audio_count += 1

    if image_count <= 0 and audio_count <= 0:
        _set_multimodal_error("no valid image or audio media_paths were found")
        return None

    timeout_seconds = float(os.getenv("TIMEOUT_SECONDS", "30") or 30)
    model_role = "audio_vision" if audio_count else "vision"
    config = resolve_openai_config(model_role)
    if not config.available:
        _set_multimodal_error(config.missing_message)
        return None
    model_name = get_model(model_role, model_profile)
    client = OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=timeout_seconds)

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
        if audio_count > 0 and image_count > 0:
            audio_error = f"audio multimodal failed: {type(exc).__name__}: {exc}"
            image_only_parts = [part for part in content_parts if part.get("type") != "input_audio"]
            try:
                response = client.chat.completions.create(
                    model=get_model("vision", model_profile),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": image_only_parts},
                    ],
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                _set_multimodal_error(audio_error + "; retried image-only")
            except Exception as retry_exc:  # noqa: BLE001
                _set_multimodal_error(
                    f"chat.completions failed: {type(exc).__name__}: {exc}; "
                    f"image-only retry failed: {type(retry_exc).__name__}: {retry_exc}"
                )
                return None
        else:
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
