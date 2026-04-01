"""Online reasoning helpers via unified OpenAI-compatible gateway."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .env_tools import normalize_openai_base_url
from .model_router import get_model
from .prompt_router import load_prompt_pair


PROMPT_ROOT = Path(__file__).resolve().parents[2] / "prompts" / "final_reasoning"
_LAST_REASONING_ERROR = ""


def get_last_reasoning_error() -> str:
    return _LAST_REASONING_ERROR


def _set_reasoning_error(message: str) -> None:
    global _LAST_REASONING_ERROR
    _LAST_REASONING_ERROR = message.strip()[:240]


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


def run_online_reasoning(
    *,
    question: str,
    evidence_store: dict[str, Any],
    benchmark_name: str = "",
    model_profile: dict | None = None,
    max_tokens: int = 800,
) -> dict[str, Any] | None:
    _set_reasoning_error("")
    if os.getenv("ENABLE_ONLINE_REASONING", "true").strip().lower() not in {"1", "true", "yes", "on"}:
        _set_reasoning_error("ENABLE_ONLINE_REASONING is disabled")
        return None

    api_key = os.getenv("LLM_API_KEY", "").strip()
    base_url = normalize_openai_base_url(os.getenv("LLM_BASE_URL", ""))
    if not api_key or not base_url:
        _set_reasoning_error("missing LLM_API_KEY or LLM_BASE_URL")
        return None

    try:
        from openai import OpenAI
    except ModuleNotFoundError:
        _set_reasoning_error("openai package is not installed")
        return None

    system_prompt, user_template = load_prompt_pair(
        prompt_root=PROMPT_ROOT,
        benchmark_name=benchmark_name,
        fallback_system="You are a careful reasoning assistant. Return JSON only.",
        fallback_user=(
            "Question:\n{question}\n\nEvidence:\n{evidence_summary_json}\n\nReturn JSON with "
            "final_answer/confidence/supporting_video_evidence/supporting_web_evidence."
        ),
    )
    evidence_summary = {
        "anchors": (evidence_store.get("anchors") or [])[:5],
        "evidences": (evidence_store.get("evidences") or [])[:8],
        "bindings": (evidence_store.get("bindings") or [])[:8],
    }
    user_prompt = (
        user_template.replace("{question}", question).replace(
            "{evidence_summary_json}",
            json.dumps(evidence_summary, ensure_ascii=False),
        )
    )

    timeout_seconds = float(os.getenv("TIMEOUT_SECONDS", "30") or 30)
    model_name = get_model("reasoning", model_profile)
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
    except Exception as exc:  # noqa: BLE001
        _set_reasoning_error(f"chat.completions failed: {type(exc).__name__}: {exc}")
        return None

    content = ""
    if getattr(response, "choices", None):
        message = response.choices[0].message
        content = getattr(message, "content", "") or ""
    parsed = _extract_json_dict(content)
    if parsed is None:
        _set_reasoning_error("model response is not valid JSON object")
    return parsed
