"""State proposal: convert event observations into structured TemporalState objects."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from src.schemas.state import TemporalState


def propose_states(
    *,
    question: str,
    observations: list[dict[str, Any]],
    model_profile: dict | None = None,
    max_states: int = 6,
    max_tokens: int = 1200,
) -> list[TemporalState]:
    """Call reasoning model once to decompose Q + observations into TemporalState list."""
    from src.tools.env_tools import resolve_openai_config
    from src.tools.model_router import get_model

    config = resolve_openai_config("state_constructor")
    if not config.available:
        return _fallback_states(question=question, observations=observations, max_states=max_states)

    try:
        from openai import OpenAI
    except ImportError:
        return _fallback_states(question=question, observations=observations, max_states=max_states)

    obs_summary = "\n".join(
        f"[{i}] {o.get('start_sec', 0):.1f}s-{o.get('end_sec', 0):.1f}s: "
        f"{o.get('event_description', '')} | entities: {o.get('entities', [])} | "
        f"actions: {o.get('actions', [])}"
        for i, o in enumerate(observations)
        if o.get("event_description")
    )

    system_prompt = (
        "You are a structured video analyst. "
        "Return JSON only: a list of state objects."
    )
    user_text = (
        f"Question: {question}\n\n"
        f"Video observations:\n{obs_summary}\n\n"
        f"Decompose the question into {max_states} or fewer video-grounded temporal states needed to answer it. "
        "The model's job is decomposition, not external fact answering. "
        "Each state must be grounded in the video observations above. "
        "sub_answer must contain only what is directly visible/described in the observations for that state. "
        "Do not name external movies, artworks, people, websites, records, public facts, or final answer candidates "
        "unless that exact name is visible in the observations or already appears in the question. "
        "If a sub-question cannot be answered from video observations alone, leave sub_answer empty, "
        "set can_answer_from_video=false, and mark state_type=cross_modal or web_only. "
        "Return JSON array of objects with keys: "
        "state_id (str), sub_question (str), sub_answer (str), "
        "temporal_span ([start_sec, end_sec]), required_predicates ([str]), "
        "state_type (video_anchored|web_only|cross_modal), can_answer_from_video (bool), confidence (0-1)."
    )

    model_name = get_model("state_constructor", model_profile)
    timeout = float(os.getenv("TIMEOUT_SECONDS", "30") or 30)
    client = OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=timeout)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        raw = (response.choices[0].message.content or "").strip() if response.choices else ""
    except Exception:
        return _fallback_states(question=question, observations=observations, max_states=max_states)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        left = raw.find("[")
        right = raw.rfind("]")
        if left < 0 or right <= left:
            return _fallback_states(question=question, observations=observations, max_states=max_states)
        try:
            data = json.loads(raw[left:right + 1])
        except json.JSONDecodeError:
            return _fallback_states(question=question, observations=observations, max_states=max_states)

    if not isinstance(data, list):
        return _fallback_states(question=question, observations=observations, max_states=max_states)

    states: list[TemporalState] = []
    source_text = _source_text(question, observations)
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        sub_answer = _sanitize_sub_answer(
            answer=str(item.get("sub_answer") or ""),
            item=item,
            observations=observations,
            source_text=source_text,
        )
        can_answer = bool(item.get("can_answer_from_video", bool(sub_answer)))
        state_type = _safe_state_type(str(item.get("state_type") or ""), can_answer_from_video=can_answer)
        states.append(TemporalState(
            state_id=str(item.get("state_id") or f"state_{i:03d}"),
            sub_question=str(item.get("sub_question") or ""),
            sub_answer=sub_answer,
            temporal_span=_safe_span(item.get("temporal_span")),
            required_predicates=[str(p) for p in (item.get("required_predicates") or []) if p],
            state_type=state_type,
            can_answer_from_video=can_answer,
            confidence=max(0.0, min(1.0, float(item.get("confidence") or 0.0))),
        ))
    return states or _fallback_states(question=question, observations=observations, max_states=max_states)


def _fallback_states(
    *,
    question: str,
    observations: list[dict[str, Any]],
    max_states: int,
) -> list[TemporalState]:
    """Build coarse state proposals when the reasoning model is unavailable.

    The fallback keeps the hybrid pipeline active in offline/dry-run settings.
    It intentionally creates state-shaped video evidence targets instead of
    pretending to solve semantic decomposition.
    """
    usable = [
        obs for obs in observations
        if str(obs.get("event_description") or "").strip()
    ] or observations
    if not usable:
        return []

    buckets = _select_temporal_buckets(usable, max_states=max_states)
    states: list[TemporalState] = []
    question_terms = _keyword_predicates(question)
    for index, bucket in enumerate(buckets):
        start = _coerce_float(bucket.get("start_sec"), 0.0)
        end = _coerce_float(bucket.get("end_sec"), start)
        description = str(bucket.get("event_description") or "").strip()
        predicates = _keyword_predicates(description)[:8] or question_terms[:5]
        states.append(
            TemporalState(
                state_id=f"s{index + 1}",
                sub_question=(
                    f"Which visible evidence in {start:.1f}-{end:.1f}s is relevant to the question?"
                ),
                sub_answer=description[:240],
                temporal_span=[start, max(end, start)],
                required_predicates=predicates,
                state_type="video_anchored",
                can_answer_from_video=bool(description),
                confidence=0.25 if description else 0.1,
            )
        )
    return states


def _select_temporal_buckets(
    observations: list[dict[str, Any]],
    *,
    max_states: int,
) -> list[dict[str, Any]]:
    if len(observations) <= max_states:
        return observations
    if max_states <= 1:
        return [observations[len(observations) // 2]]
    indices = {
        round(i * (len(observations) - 1) / (max_states - 1))
        for i in range(max_states)
    }
    return [observations[i] for i in sorted(indices)]


def _keyword_predicates(text: str) -> list[str]:
    stop = {
        "the", "and", "or", "in", "of", "to", "a", "an", "with", "for", "by",
        "is", "are", "was", "were", "which", "what", "who", "does", "from",
        "this", "that", "video", "question",
    }
    tokens = [
        token.lower()
        for token in re.findall(r"[A-Za-z0-9一-鿿]+", text)
        if len(token) > 2 and token.lower() not in stop
    ]
    seen: set[str] = set()
    predicates: list[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        predicates.append(token)
    return predicates[:12]


def _safe_span(value: Any) -> list[float]:
    if isinstance(value, list) and len(value) >= 2:
        try:
            return [float(value[0]), float(value[1])]
        except (TypeError, ValueError):
            pass
    return []


def _safe_state_type(value: str, *, can_answer_from_video: bool) -> str:
    value = value if value in {"video_anchored", "web_only", "cross_modal"} else ""
    if value:
        return value
    return "video_anchored" if can_answer_from_video else "cross_modal"


def _source_text(question: str, observations: list[dict[str, Any]]) -> str:
    parts = [question]
    for obs in observations:
        parts.append(str(obs.get("event_description") or ""))
        parts.extend(str(item) for item in (obs.get("entities") or []))
        parts.extend(str(item) for item in (obs.get("actions") or []))
    return " ".join(parts)


def _sanitize_sub_answer(
    *,
    answer: str,
    item: dict[str, Any],
    observations: list[dict[str, Any]],
    source_text: str,
) -> str:
    answer = re.sub(r"\s+", " ", answer).strip()
    if not answer:
        return ""
    if _has_ungrounded_named_phrase(answer, source_text):
        replacement = _nearest_observation_summary(item, observations)
        return replacement[:240]
    return answer[:240]


def _has_ungrounded_named_phrase(answer: str, source_text: str) -> bool:
    source_norm = source_text.lower()
    phrases = re.findall(
        r"\b[A-Z][A-Za-z0-9'&-]+(?:\s+[A-Z][A-Za-z0-9'&-]+){1,5}\b",
        answer,
    )
    quoted = re.findall(r"\"([^\"]+)\"|'([^']+)'", answer)
    phrases.extend(" ".join(part for part in pair if part) for pair in quoted)
    for phrase in phrases:
        phrase = re.sub(r"\s+", " ", phrase).strip()
        if len(phrase) >= 4 and phrase.lower() not in source_norm:
            return True
    return False


def _nearest_observation_summary(item: dict[str, Any], observations: list[dict[str, Any]]) -> str:
    span = _safe_span(item.get("temporal_span"))
    if span:
        target = (span[0] + span[1]) / 2
        scored: list[tuple[float, str]] = []
        for obs in observations:
            desc = str(obs.get("event_description") or "").strip()
            if not desc:
                continue
            start = _coerce_float(obs.get("start_sec"), target)
            end = _coerce_float(obs.get("end_sec"), start)
            center = (start + end) / 2
            scored.append((abs(center - target), desc))
        if scored:
            return min(scored, key=lambda row: row[0])[1]
    for obs in observations:
        desc = str(obs.get("event_description") or "").strip()
        if desc:
            return desc
    return ""


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
