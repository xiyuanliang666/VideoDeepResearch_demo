"""Online reasoning helpers via unified OpenAI-compatible gateway."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .env_tools import resolve_openai_config
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
    prompt_group: str = "",
    model_profile: dict | None = None,
    max_tokens: int = 2500,
) -> dict[str, Any] | None:
    _set_reasoning_error("")
    if os.getenv("ENABLE_ONLINE_REASONING", "true").strip().lower() not in {"1", "true", "yes", "on"}:
        _set_reasoning_error("ENABLE_ONLINE_REASONING is disabled")
        return None

    config = resolve_openai_config("reasoning")
    if not config.available:
        _set_reasoning_error(config.missing_message)
        return None

    try:
        from openai import OpenAI
    except ModuleNotFoundError:
        _set_reasoning_error("openai package is not installed")
        return None

    system_prompt, user_template = load_prompt_pair(
        prompt_root=PROMPT_ROOT,
        prompt_group=prompt_group,
        fallback_system="You are a careful reasoning assistant. Return JSON only.",
        fallback_user=(
            "Question:\n{question}\n\nEvidence:\n{evidence_summary_json}\n\nReturn JSON with "
            "final_answer/confidence/supporting_video_evidence/supporting_web_evidence."
        ),
    )
    evidence_summary = _build_reasoning_evidence_summary(question, evidence_store)
    user_prompt = (
        user_template.replace("{question}", question).replace(
            "{evidence_summary_json}",
            json.dumps(evidence_summary, ensure_ascii=False),
        )
    )

    timeout_seconds = float(os.getenv("TIMEOUT_SECONDS", "30") or 30)
    model_name = get_model("reasoning", model_profile)
    client = OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=timeout_seconds)
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


def _build_reasoning_evidence_summary(question: str, evidence_store: dict[str, Any]) -> dict[str, Any]:
    max_anchors = int(os.getenv("FINAL_REASONING_MAX_ANCHORS", "8") or 8)
    max_evidences = int(os.getenv("FINAL_REASONING_MAX_EVIDENCES", "32") or 32)
    max_bindings = int(os.getenv("FINAL_REASONING_MAX_BINDINGS", "32") or 32)
    anchors = evidence_store.get("anchors") or []
    evidences = evidence_store.get("evidences") or []
    bindings = evidence_store.get("bindings") or []
    selected_evidences = _select_relevant_items(
        evidences,
        question,
        max_items=max_evidences,
        text_fields=("content_summary", "raw_excerpt", "source_url", "source_ref"),
    )
    selected_ids = {
        str(item.get("evidence_id") or "")
        for item in selected_evidences
        if isinstance(item, dict)
    }
    selected_bindings = [
        item for item in bindings
        if not selected_ids or str(item.get("evidence_id") or "") in selected_ids
    ][:max_bindings]
    selected_anchor_ids = {
        str(item.get("anchor_id") or "")
        for item in selected_bindings
        if isinstance(item, dict)
    }
    selected_anchors = [
        item for item in anchors
        if not selected_anchor_ids or str(item.get("anchor_id") or "") in selected_anchor_ids
    ][:max_anchors]
    if not selected_anchors:
        selected_anchors = anchors[:max_anchors]
    return {
        "anchors": selected_anchors,
        "evidences": selected_evidences,
        "bindings": selected_bindings,
        "selection_policy": {
            "max_anchors": max_anchors,
            "max_evidences": max_evidences,
            "max_bindings": max_bindings,
            "question_aware": True,
        },
    }


def _select_relevant_items(
    items: list[Any],
    question: str,
    *,
    max_items: int,
    text_fields: tuple[str, ...],
) -> list[Any]:
    if len(items) <= max_items:
        return items
    question_tokens = _tokens(question)
    scored: list[tuple[float, int, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            scored.append((0.0, index, item))
            continue
        text = " ".join(str(item.get(field) or "") for field in text_fields)
        item_tokens = _tokens(text)
        overlap = len(question_tokens & item_tokens)
        score = float(overlap)
        scored.append((score, index, item))
    selected = sorted(scored, key=lambda entry: (-entry[0], entry[1]))[:max_items]
    return [item for _, _, item in sorted(selected, key=lambda entry: entry[1])]


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(token) > 2}
