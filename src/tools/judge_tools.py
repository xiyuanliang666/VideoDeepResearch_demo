"""Online LLM judge helpers via a unified OpenAI-compatible gateway."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .env_tools import normalize_openai_base_url
from .model_router import get_model


PROMPT_ROOT = Path(__file__).resolve().parents[2] / "prompts" / "judge"
_LAST_JUDGE_ERROR = ""


def get_last_judge_error() -> str:
    return _LAST_JUDGE_ERROR


def _set_judge_error(message: str) -> None:
    global _LAST_JUDGE_ERROR
    _LAST_JUDGE_ERROR = message.strip()[:240]


def _load_prompt_text(path: Path, fallback: str) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return fallback


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


def _build_prompts(judge_input: dict[str, Any]) -> tuple[str, str]:
    system_prompt = _load_prompt_text(
        PROMPT_ROOT / "system.txt",
        "Judge the model output and return JSON only.",
    )
    user_template = _load_prompt_text(
        PROMPT_ROOT / "user.txt",
        "Return JSON with overall_score, verdict, explanation, dimension_scores, failure_tags.\n\n{judge_input_json}",
    )
    payload = json.dumps(judge_input, ensure_ascii=False)
    user_prompt = user_template.replace("{judge_input_json}", payload)
    return system_prompt, user_prompt


def run_online_judge(
    *,
    judge_input: dict[str, Any],
    model_name: str,
    max_tokens: int = 800,
) -> dict[str, Any] | None:
    _set_judge_error("")
    if os.getenv("ENABLE_ONLINE_JUDGE", "false").strip().lower() not in {"1", "true", "yes", "on"}:
        _set_judge_error("ENABLE_ONLINE_JUDGE is disabled")
        return None

    api_key = os.getenv("LLM_API_KEY", "").strip()
    base_url = normalize_openai_base_url(os.getenv("LLM_BASE_URL", ""))
    if not api_key or not base_url:
        _set_judge_error("missing LLM_API_KEY or LLM_BASE_URL")
        return None

    try:
        from openai import OpenAI
    except ModuleNotFoundError:
        _set_judge_error("openai package is not installed")
        return None

    timeout_seconds = float(os.getenv("TIMEOUT_SECONDS", "30") or 30)
    resolved_model = get_model("judge", {"judge_model": model_name})
    system_prompt, user_prompt = _build_prompts(judge_input)
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)
    try:
        response = client.chat.completions.create(
            model=resolved_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
    except Exception as exc:  # noqa: BLE001
        _set_judge_error(f"chat.completions failed: {type(exc).__name__}: {exc}")
        return None

    content = ""
    if getattr(response, "choices", None):
        message = response.choices[0].message
        content = getattr(message, "content", "") or ""
    parsed = _extract_json_dict(content)
    if parsed is None:
        _set_judge_error("model response is not valid JSON object")
    return parsed
