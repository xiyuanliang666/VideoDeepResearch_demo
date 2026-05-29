"""Candidate hypothesis generation from states and anchors."""

from __future__ import annotations

import json
import os
from typing import Any

from src.schemas.state import TemporalState


class CandidateHypothesis:
    def __init__(self, **kwargs: Any) -> None:
        self.hypothesis_id: str = kwargs.get("hypothesis_id", "")
        self.explanation: str = kwargs.get("explanation", "")
        self.linked_state_ids: list[str] = kwargs.get("linked_state_ids", [])
        self.supporting_evidence_ids: list[str] = []
        self.refuting_evidence_ids: list[str] = []
        self.support_score: float = 0.0
        self.refute_score: float = 0.0
        self.status: str = "pending"
        self.generation_source: str = kwargs.get("generation_source", "model")
        self.fallback_reason: str = kwargs.get("fallback_reason", "")

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def generate_hypotheses(
    *,
    question: str,
    states: list[TemporalState],
    anchors: list[dict[str, Any]],
    web_candidates: list[Any] | None = None,
    model_profile: dict | None = None,
    max_hypotheses: int = 4,
    max_tokens: int = 2500,
) -> list[CandidateHypothesis]:
    """Generate candidate branches from verified runtime observations."""
    from src.tools.env_tools import resolve_openai_config
    from src.tools.model_router import get_model

    config = resolve_openai_config("hypothesis_generator")
    if not config.available:
        return _fallback_hypotheses(
            question=question,
            states=states,
            web_candidates=web_candidates or [],
            reason=config.missing_message,
        )

    try:
        from openai import OpenAI
    except ImportError:
        return _fallback_hypotheses(
            question=question,
            states=states,
            web_candidates=web_candidates or [],
            reason="openai_not_installed",
        )

    states_text = "\n".join(
        f"- {s.state_id}: {s.sub_question} → {s.sub_answer}" for s in states
    )
    anchors_text = "\n".join(
        f"- {a.get('anchor_id')}: {a.get('searchable_entities', [])}" for a in anchors
    )
    web_text = "\n".join(
        f"- [{_candidate_id(c)}] {_candidate_title(c)} | {_candidate_snippet(c)[:180]}"
        for c in (web_candidates or [])[:16]
        if not _is_fallback_candidate(c)
    )

    # Derive example IDs from actual state data so the model copies exact IDs
    example_ids = ", ".join(s.state_id for s in states[:3]) if states else "S1, S2, S3"

    user_text = (
        f"Question: {question}\n\n"
        f"Video states:\n{states_text}\n\n"
        f"Visual anchors:\n{anchors_text}\n\n"
        f"Retrieved web observations:\n{web_text or '(none)'}\n\n"
        f"Generate up to {max_hypotheses} candidate answer hypotheses only from video states and retrieved web observations. "
        "Each hypothesis should be a tentative explanation path, not a search query template. "
        "Do not invent named entities that are absent from the video states, anchors, or retrieved web observations.\n"
        f"CRITICAL: linked_state_ids MUST use the EXACT state IDs from the list above ({example_ids}). "
        "Do NOT invent new IDs — copy the state IDs exactly as shown.\n"
        "Return JSON array: [{hypothesis_id, explanation, linked_state_ids: []}]"
    )

    model_name = get_model("hypothesis_generator", model_profile)
    fallback_model_name = (
        os.getenv("HYPOTHESIS_GENERATOR_FALLBACK_MODEL", "").strip()
        or os.getenv("REASONING_FALLBACK_MODEL", "").strip()
    )
    model_attempts = [model_name]
    if fallback_model_name and fallback_model_name != model_name:
        model_attempts.append(fallback_model_name)
    max_attempts = max(1, int(os.getenv("HYPOTHESIS_GENERATOR_MAX_ATTEMPTS", "2") or 2))
    timeout = float(os.getenv("TIMEOUT_SECONDS", "30") or 30)
    client = OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=timeout)

    errors: list[str] = []
    data: list | None = None
    for attempt_model in model_attempts:
        for attempt_index in range(max_attempts):
            try:
                response = client.chat.completions.create(
                    model=attempt_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Return JSON only. Do not leave the final answer empty. "
                                "If you reason internally, still emit the final JSON array in message.content."
                            ),
                        },
                        {"role": "user", "content": _retry_user_text(user_text, attempt_index)},
                    ],
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                raw = _extract_response_text(response)
            except Exception as exc:
                errors.append(f"{attempt_model}:exception:{type(exc).__name__}:{exc}")
                continue
            if not raw:
                errors.append(f"{attempt_model}:empty_response")
                continue
            data = _parse_json_list(raw)
            if data is not None:
                break
            errors.append(f"{attempt_model}:invalid_json")
        if data is not None:
            break

    if data is None:
        return _fallback_hypotheses(
            question=question,
            states=states,
            web_candidates=web_candidates or [],
            reason=";".join(errors[-4:]) or "hypothesis_model_failed",
        )

    hypotheses = [
        CandidateHypothesis(
            hypothesis_id=str(item.get("hypothesis_id") or f"hyp_{i:03d}"),
            explanation=str(item.get("explanation") or ""),
            linked_state_ids=[str(s) for s in (item.get("linked_state_ids") or [])],
        )
        for i, item in enumerate(data)
        if isinstance(item, dict)
    ]
    return hypotheses or _fallback_hypotheses(
        question=question,
        states=states,
        web_candidates=web_candidates or [],
        reason="model_returned_no_hypothesis_objects",
    )


def _retry_user_text(user_text: str, attempt_index: int) -> str:
    if attempt_index <= 0:
        return user_text
    return (
        f"{user_text}\n\n"
        "Previous attempt produced an empty or unparsable response. "
        "Return only a compact JSON array now. No markdown. No prose."
    )


def _extract_response_text(response: Any) -> str:
    """Extract content from OpenAI-compatible responses, including common gateway variants."""
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    if message is None:
        return ""

    content = getattr(message, "content", "") or ""
    if isinstance(content, str):
        text = content.strip()
        if text:
            return text
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                value = item.get("text") or item.get("content") or ""
                if value:
                    parts.append(str(value))
            elif item:
                parts.append(str(item))
        text = "\n".join(parts).strip()
        if text:
            return text

    for attr in ("reasoning_content", "reasoning", "text", "output_text"):
        value = getattr(message, attr, "") or ""
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _fallback_hypotheses(
    *,
    question: str,
    states: list[TemporalState],
    web_candidates: list[Any],
    reason: str,
) -> list[CandidateHypothesis]:
    real_candidates = [candidate for candidate in web_candidates if not _is_fallback_candidate(candidate)]
    if real_candidates:
        hypotheses: list[CandidateHypothesis] = []
        for index, candidate in enumerate(real_candidates[:4]):
            title = _candidate_title(candidate)
            snippet = _candidate_snippet(candidate)
            state_ids = _best_state_ids(states, f"{title} {snippet}")
            hypotheses.append(
                CandidateHypothesis(
                    hypothesis_id=f"hyp_web_{index:03d}",
                    explanation=(
                        f"Retrieved observation may provide a candidate path: {title}. "
                        f"Snippet: {snippet[:180]}"
                    ),
                    linked_state_ids=state_ids,
                    generation_source="post_retrieval_fallback",
                    fallback_reason=reason,
                )
            )
        return hypotheses

    if not states:
        return [
            CandidateHypothesis(
                hypothesis_id="hyp_fallback_000",
                explanation=f"Fallback hypothesis for answering the question from available evidence: {question}",
                linked_state_ids=[],
                generation_source="fallback",
                fallback_reason=reason,
            )
        ]
    hypotheses: list[CandidateHypothesis] = []
    for index, state in enumerate(states[:4]):
        state_answer = str(getattr(state, "sub_answer", "") or "").strip()
        state_question = str(getattr(state, "sub_question", "") or "").strip()
        explanation = (
            f"Evaluate whether state {state.state_id} supports the question. "
            f"State question: {state_question}. "
            f"Observed answer: {state_answer}."
        ).strip()
        hypotheses.append(
            CandidateHypothesis(
                hypothesis_id=f"hyp_fallback_{index:03d}",
                explanation=explanation,
                linked_state_ids=[state.state_id],
                generation_source="fallback",
                fallback_reason=reason,
            )
        )
    return hypotheses


def _parse_json_list(raw: str) -> list | None:
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else None
    except json.JSONDecodeError:
        left, right = raw.find("["), raw.rfind("]")
        if left >= 0 and right > left:
            try:
                data = json.loads(raw[left:right + 1])
                return data if isinstance(data, list) else None
            except json.JSONDecodeError:
                pass
    return None


def _candidate_id(candidate: Any) -> str:
    if isinstance(candidate, dict):
        return str(candidate.get("candidate_id") or "")
    return str(getattr(candidate, "candidate_id", "") or "")


def _candidate_metadata(candidate: Any) -> dict[str, Any]:
    if isinstance(candidate, dict):
        metadata = candidate.get("metadata")
    else:
        metadata = getattr(candidate, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def _candidate_title(candidate: Any) -> str:
    return str(_candidate_metadata(candidate).get("title") or "")


def _candidate_snippet(candidate: Any) -> str:
    metadata = _candidate_metadata(candidate)
    return str(metadata.get("snippet") or metadata.get("content") or "")


def _is_fallback_candidate(candidate: Any) -> bool:
    if isinstance(candidate, dict):
        retriever = str(candidate.get("retriever_name") or "")
        source_ref = str(candidate.get("source_ref") or "")
    else:
        retriever = str(getattr(candidate, "retriever_name", "") or "")
        source_ref = str(getattr(candidate, "source_ref", "") or "")
    return retriever == "fallback_web_retriever" or source_ref.startswith("fallback://")


def _best_state_ids(states: list[TemporalState], text: str) -> list[str]:
    import re

    tokens = set(re.findall(r"[A-Za-z0-9一-鿿]+", text.lower()))
    scored: list[tuple[int, str]] = []
    for state in states:
        state_text = " ".join([
            state.sub_question,
            state.sub_answer,
            " ".join(state.required_predicates),
        ]).lower()
        state_tokens = set(re.findall(r"[A-Za-z0-9一-鿿]+", state_text))
        scored.append((len(tokens & state_tokens), state.state_id))
    ranked = [state_id for score, state_id in sorted(scored, reverse=True) if score > 0]
    return ranked[:2] or ([states[0].state_id] if states else [])
