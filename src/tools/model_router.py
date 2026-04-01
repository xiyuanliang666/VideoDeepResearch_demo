"""Centralized model-name routing for different runtime roles."""

from __future__ import annotations

import os


def _clean(value: object) -> str:
    return str(value or "").strip()


def get_model(role: str, model_profile: dict | None = None) -> str:
    """Resolve model name by runtime role.

    Strategy:
    - reasoning: prefer experiment/model_profile override first, then env.
      This allows benchmark experiments to sweep multiple reasoning models.
    - judge/vision: prefer env first for stable operational defaults.
    """
    model_profile = model_profile or {}
    role_key = role.strip().lower()

    if role_key == "reasoning":
        return (
            _clean(model_profile.get("reasoning_model"))
            or _clean(os.getenv("REASONING_MODEL"))
            or _clean(os.getenv("LLM_MODEL"))
            or _clean(model_profile.get("model"))
            or "openai/gpt-4.1"
        )

    if role_key == "judge":
        return (
            _clean(os.getenv("JUDGE_MODEL"))
            or _clean(model_profile.get("judge_model"))
            or _clean(os.getenv("LLM_MODEL"))
            or _clean(os.getenv("REASONING_MODEL"))
            or _clean(model_profile.get("reasoning_model"))
            or "openai/gpt-4.1-mini"
        )

    if role_key == "vision":
        return (
            _clean(os.getenv("VISION_MODEL"))
            or _clean(model_profile.get("vision_model"))
            or _clean(os.getenv("LLM_MODEL"))
            or _clean(os.getenv("REASONING_MODEL"))
            or _clean(model_profile.get("reasoning_model"))
            or "openai/gpt-4.1"
        )

    return (
        _clean(os.getenv("LLM_MODEL"))
        or _clean(model_profile.get("model"))
        or _clean(os.getenv("REASONING_MODEL"))
        or "openai/gpt-4.1"
    )
