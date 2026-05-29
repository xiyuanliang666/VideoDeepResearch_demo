"""Centralized model-name routing for different runtime roles."""

from __future__ import annotations

import os


def _clean(value: object) -> str:
    return str(value or "").strip()


def get_model(role: str, model_profile: dict | None = None) -> str:
    """Resolve model name by runtime role.

    Strategy:
    - Runtime .env variables are the primary operational defaults.
    - model_profile values are static fallbacks for offline configs or sweeps.
    """
    model_profile = model_profile or {}
    role_key = role.strip().lower()

    if role_key == "reasoning":
        return (
            _clean(os.getenv("REASONING_MODEL"))
            or _clean(os.getenv("LLM_MODEL"))
            or _clean(model_profile.get("reasoning_model"))
            or _clean(model_profile.get("model"))
            or "gpt-4o-mini"
        )

    reasoning_role_env = {
        "state_constructor": "STATE_CONSTRUCTOR_MODEL",
        "query_planner": "QUERY_PLANNER_MODEL",
        "hypothesis_generator": "HYPOTHESIS_GENERATOR_MODEL",
        "evidence_binder": "EVIDENCE_BINDER_MODEL",
        "candidate_evaluator": "CANDIDATE_EVALUATOR_MODEL",
        "coverage_checker": "COVERAGE_CHECKER_MODEL",
        "frame_reranker": "FRAME_RERANKER_MODEL",
    }
    if role_key in reasoning_role_env:
        return (
            _clean(os.getenv(reasoning_role_env[role_key]))
            or _clean(os.getenv("REASONING_MODEL"))
            or _clean(os.getenv("LLM_MODEL"))
            or _clean(model_profile.get(f"{role_key}_model"))
            or _clean(model_profile.get("reasoning_model"))
            or "gpt-4o-mini"
        )

    if role_key == "judge":
        return (
            _clean(os.getenv("JUDGE_MODEL"))
            or _clean(model_profile.get("judge_model"))
            or _clean(os.getenv("LLM_MODEL"))
            or _clean(os.getenv("REASONING_MODEL"))
            or _clean(model_profile.get("reasoning_model"))
            or "gpt-4o-mini"
        )

    if role_key in {"audio_vision", "vision_audio", "multimodal_audio"}:
        return (
            _clean(os.getenv("AUDIO_VISION_MODEL"))
            or _clean(model_profile.get("audio_vision_model"))
            or _clean(os.getenv("VLM_MODEL"))
            or _clean(os.getenv("MULTIMODAL_AUDIO_MODEL"))
            or "ge-2.5-pro"
        )

    vision_role_env = {
        "vision": "VISION_MODEL",
        "event_observer": "EVENT_OBSERVER_MODEL",
        "anchor_extractor": "ANCHOR_EXTRACTOR_MODEL",
    }
    if role_key in vision_role_env:
        return (
            _clean(os.getenv(vision_role_env[role_key]))
            or _clean(model_profile.get(f"{role_key}_model"))
            or _clean(model_profile.get("vision_model"))
            or _clean(os.getenv("VISION_MODEL"))
            or _clean(os.getenv("VLM_MODEL"))
            or _clean(os.getenv("LLM_MODEL"))
            or _clean(os.getenv("REASONING_MODEL"))
            or _clean(model_profile.get("reasoning_model"))
            or "gpt-4o-mini"
        )

    return (
        _clean(os.getenv("LLM_MODEL"))
        or _clean(model_profile.get("model"))
        or _clean(os.getenv("REASONING_MODEL"))
        or "gpt-4o-mini"
    )
