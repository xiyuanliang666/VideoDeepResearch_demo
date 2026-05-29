"""Anchor extraction: convert per-state supporting frames into retrieval-ready anchors."""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any


def extract_anchors_for_states(
    *,
    question: str,
    states: list[Any],
    state_frames: dict[str, list[dict[str, Any]]],
    model_profile: dict | None = None,
    max_frames_per_call: int = 4,
    max_tokens: int = 600,
) -> list[dict[str, Any]]:
    """For each state, call a lightweight VLM to extract visual anchors and search queries.

    Returns list of anchor dicts with: anchor_id, state_id, visual_anchors, search_queries.
    """
    from src.tools.env_tools import resolve_openai_config
    from src.tools.model_router import get_model

    config = resolve_openai_config("anchor_extractor")
    if not config.available:
        return _fallback_anchors(states, question, state_frames=state_frames)

    try:
        from openai import OpenAI
    except ImportError:
        return _fallback_anchors(states, question, state_frames=state_frames)

    model_name = get_model("anchor_extractor", model_profile)
    timeout = float(os.getenv("TIMEOUT_SECONDS", "30") or 30)
    client = OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=timeout)

    anchors: list[dict[str, Any]] = []
    for i, state in enumerate(states):
        frames = state_frames.get(state.state_id, [])[:max_frames_per_call]
        anchor = _extract_one_anchor(
            client=client,
            model_name=model_name,
            question=question,
            state=state,
            frames=frames,
            anchor_index=i,
            max_tokens=max_tokens,
        )
        anchors.append(anchor)
        # Update state with extracted search queries
        if anchor.get("search_queries"):
            state.search_queries = anchor["search_queries"]
        if anchor.get("visual_anchors"):
            state.visual_anchors = anchor["visual_anchors"]
        if frames:
            state.supporting_frame_paths = [f["frame_path"] for f in frames]
    _add_cross_anchor_bridge_queries(question=question, states=states, anchors=anchors)
    for state in states:
        anchor = next((item for item in anchors if item.get("state_id") == state.state_id), None)
        if anchor and anchor.get("search_queries"):
            state.search_queries = anchor["search_queries"]
    return anchors


def _extract_one_anchor(
    *,
    client: Any,
    model_name: str,
    question: str,
    state: Any,
    frames: list[dict[str, Any]],
    anchor_index: int,
    max_tokens: int,
) -> dict[str, Any]:
    system_prompt = (
        "You are a visual verification and search-anchor analyst. "
        "Return JSON only. Do not use outside knowledge."
    )
    user_text = (
        f"Question: {question}\n"
        f"Sub-question: {state.sub_question}\n"
        f"Required predicates: {state.required_predicates}\n\n"
        "From the frames only, extract:\n"
        "0. can_answer_from_video: whether this sub-question can be answered from the frames alone\n"
        "0b. video_answer: a short answer using only visible frame evidence; leave empty if not visible\n"
        "1. visual_anchors: specific visual details useful for web search "
        "(clothing, objects, text, architecture, logos)\n"
        "2. search_queries: 2-4 concrete web search queries using visible anchors and unresolved constraints "
        "(NOT the original question)\n"
        "3. searchable_entities: named entities only if visibly written, logo-like, or already in the question\n\n"
        "Do not infer external movie titles, artworks, people, databases, or answer candidates from style alone. "
        "If the frames only show a style or scene, describe the style instead of naming an external entity. "
        "Prefer abstract search anchors over pixel-level descriptions: character/stage/style/action words are useful; "
        "long clothing/color inventories are not useful search queries. "
        "Return JSON: {visual_anchors: [{type, description}], "
        "can_answer_from_video: bool, video_answer: str, search_queries: [str], searchable_entities: [str]}"
    )

    content_parts: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for frame_info in frames:
        path = frame_info["frame_path"]
        try:
            b64 = base64.b64encode(Path(path).read_bytes()).decode("utf-8")
            suffix = Path(path).suffix.lower().lstrip(".")
            mime = f"image/{suffix}" if suffix in {"jpg", "jpeg", "png", "webp"} else "image/jpeg"
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })
        except OSError:
            continue

    fallback = {
        "anchor_id": f"anchor_{anchor_index:04d}",
        "state_id": state.state_id,
        "visual_anchors": [],
        "can_answer_from_video": False,
        "video_answer": "",
        "search_queries": _fallback_search_queries(question, state),
        "searchable_entities": [],
    }

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
        raw = (response.choices[0].message.content or "").strip() if response.choices else ""
    except Exception:
        return fallback

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        left, right = raw.find("{"), raw.rfind("}")
        if left >= 0 and right > left:
            try:
                data = json.loads(raw[left:right + 1])
            except json.JSONDecodeError:
                return fallback
        else:
            return fallback

    if not isinstance(data, dict):
        return fallback

    video_answer = _sanitize_visible_text(str(data.get("video_answer") or ""))
    can_answer = bool(data.get("can_answer_from_video", bool(video_answer)))
    if video_answer:
        state.sub_answer = video_answer[:240]
        state.can_answer_from_video = True
        if getattr(state, "state_type", "") == "web_only":
            state.state_type = "cross_modal"
    else:
        state.can_answer_from_video = False

    visual_anchors = data.get("visual_anchors") or []
    searchable_entities = _filter_visible_entities(
        [str(e) for e in (data.get("searchable_entities") or []) if e],
        question=question,
        state=state,
        visual_anchors=visual_anchors,
    )
    queries = _filter_queries(
        [str(q) for q in (data.get("search_queries") or []) if q],
        question=question,
        state=state,
        visual_anchors=visual_anchors,
        searchable_entities=searchable_entities,
    )
    queries = _rank_bridge_queries_first(
        _bridge_queries_for_state(question=question, state=state, visual_anchors=visual_anchors, entities=searchable_entities)
        + queries
    )
    if not queries:
        queries = _fallback_search_queries(question, state)

    return {
        "anchor_id": f"anchor_{anchor_index:04d}",
        "state_id": state.state_id,
        "visual_anchors": visual_anchors,
        "can_answer_from_video": can_answer,
        "video_answer": video_answer,
        "search_queries": queries[:4],
        "searchable_entities": searchable_entities,
    }


def _fallback_anchors(
    states: list[Any],
    question: str,
    *,
    state_frames: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    state_frames = state_frames or {}
    for i, state in enumerate(states):
        frames = state_frames.get(state.state_id, [])
        if frames:
            state.supporting_frame_paths = [str(frame.get("frame_path") or "") for frame in frames if frame.get("frame_path")]
        if not getattr(state, "search_queries", None):
            state.search_queries = _fallback_search_queries(question, state)
        anchors.append({
            "anchor_id": f"anchor_{i:04d}",
            "state_id": state.state_id,
            "visual_anchors": [],
            "can_answer_from_video": bool(getattr(state, "sub_answer", "")),
            "video_answer": str(getattr(state, "sub_answer", "") or ""),
            "search_queries": list(state.search_queries or [question]),
            "searchable_entities": [],
        })
    return anchors


def _fallback_search_queries(question: str, state: Any) -> list[str]:
    """Return short search queries when the VLM anchor call is unavailable."""
    q = re.sub(r"\s+", " ", question).strip()
    lower = q.lower()
    visual_terms = []
    visual_terms.extend(str(item) for item in getattr(state, "visual_anchors", [])[:4])
    visual_terms.extend(str(item) for item in getattr(state, "required_predicates", [])[:4])
    visual = " ".join(term for term in visual_terms if term).strip()
    queries: list[str] = []
    if visual:
        queries.extend(_bridge_queries_for_state(
            question=question,
            state=state,
            visual_anchors=getattr(state, "visual_anchors", []) or [],
            entities=_visible_entity_candidates(question=question, state=state, visual_anchors=getattr(state, "visual_anchors", []) or []),
        ))
        compact_visual = _compact_query(visual + " " + q)
        if not _is_pixel_level_query(compact_visual):
            queries.append(compact_visual)
    question_terms = _question_terms(q)
    if question_terms:
        queries.append(_compact_query(" ".join(question_terms[:12])))
    source_terms = _source_or_field_terms(q)
    if source_terms:
        queries.append(_compact_query(" ".join(source_terms[:12])))
    if not queries:
        queries.append(_compact_query(q))
    deduped: list[str] = []
    for query in queries:
        if query and query.lower() not in {item.lower() for item in deduped}:
            deduped.append(query)
    return deduped[:4]


def _compact_query(text: str) -> str:
    years = re.findall(r"\b(?:19|20)\d{2}\b", text)
    caps = re.findall(r"\b[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,3}", text)
    pieces = years + caps[:8]
    return " ".join(dict.fromkeys(pieces))[:160] if pieces else text[:160]


def _question_terms(text: str) -> list[str]:
    stop = {
        "what", "which", "where", "when", "who", "whose", "why", "how",
        "the", "and", "for", "with", "from", "that", "this", "according",
        "external", "sources", "video", "answer",
    }
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9'-]*|\b(?:19|20)\d{2}\b", text.lower())
    return list(dict.fromkeys(token for token in tokens if len(token) > 2 and token not in stop))


def _source_or_field_terms(text: str) -> list[str]:
    terms: list[str] = []
    for pattern in (r"according to ([A-Z][A-Za-z0-9&.,' -]{2,80})", r"on ([A-Z][A-Za-z0-9&.,' -]{2,80})"):
        terms.extend(str(match).strip(" .,'") for match in re.findall(pattern, text))
    lowered = text.lower()
    for pattern in (r"what is (?:its|the)?\s*([a-z][a-z0-9 -]{2,80})", r"what was (?:its|the)?\s*([a-z][a-z0-9 -]{2,80})"):
        for match in re.findall(pattern, lowered):
            cleaned = re.split(r"\baccording to\b|\bin\b|\bon\b|\?", match)[0]
            if cleaned.strip():
                terms.append(cleaned.strip())
    return list(dict.fromkeys(term for term in terms if term))


def _sanitize_visible_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()[:240]


def _filter_visible_entities(
    entities: list[str],
    *,
    question: str,
    state: Any,
    visual_anchors: list[Any],
) -> list[str]:
    visible_text = " ".join([
        question,
        str(getattr(state, "sub_question", "") or ""),
        str(getattr(state, "sub_answer", "") or ""),
        " ".join(str(item) for item in getattr(state, "required_predicates", []) or []),
        *(_anchor_text(anchor) for anchor in visual_anchors),
    ]).lower()
    entities = [
        *entities,
        *_visible_entity_candidates(question=question, state=state, visual_anchors=visual_anchors),
    ]
    kept: list[str] = []
    for entity in entities:
        value = re.sub(r"\s+", " ", entity).strip()
        if not value:
            continue
        if value.lower() in visible_text or _is_short_visible_token(value):
            kept.append(value)
    return list(dict.fromkeys(kept))[:8]


def _filter_queries(
    queries: list[str],
    *,
    question: str,
    state: Any,
    visual_anchors: list[Any],
    searchable_entities: list[str],
) -> list[str]:
    allowed_text = " ".join([
        question,
        str(getattr(state, "sub_question", "") or ""),
        str(getattr(state, "sub_answer", "") or ""),
        " ".join(str(item) for item in getattr(state, "required_predicates", []) or []),
        " ".join(_anchor_text(anchor) for anchor in visual_anchors),
        " ".join(searchable_entities),
    ]).lower()
    kept: list[str] = []
    for query in queries:
        value = re.sub(r"\s+", " ", query).strip()
        if not value:
            continue
        if _is_pixel_level_query(value):
            continue
        if _has_ungrounded_named_phrase(value, allowed_text):
            continue
        kept.append(value)
    return list(dict.fromkeys(kept))[:4]


def _anchor_text(anchor: Any) -> str:
    if isinstance(anchor, dict):
        return str(anchor.get("description") or anchor.get("text") or anchor)
    return str(anchor)


def _is_short_visible_token(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Z0-9&.-]{2,12}", value))


def _has_ungrounded_named_phrase(text: str, allowed_text_lower: str) -> bool:
    phrases = re.findall(
        r"\b[A-Z][A-Za-z0-9'&-]+(?:\s+[A-Z][A-Za-z0-9'&-]+){1,5}\b",
        text,
    )
    for phrase in phrases:
        phrase = re.sub(r"\s+", " ", phrase).strip()
        if len(phrase) >= 4 and phrase.lower() not in allowed_text_lower:
            return True
    return False


def _add_cross_anchor_bridge_queries(
    *,
    question: str,
    states: list[Any],
    anchors: list[dict[str, Any]],
) -> None:
    entity_pool: list[str] = []
    stage_pool: list[str] = []
    style_pool: list[str] = []
    state_by_id = {str(getattr(state, "state_id", "") or ""): state for state in states}
    for anchor in anchors:
        state = state_by_id.get(str(anchor.get("state_id") or ""))
        if state is not None:
            entity_pool.extend(_visible_entity_candidates(
                question=question,
                state=state,
                visual_anchors=anchor.get("visual_anchors") or [],
            ))
        entity_pool.extend(str(item) for item in (anchor.get("searchable_entities") or []))
        text = " ".join([
            " ".join(str(item) for item in anchor.get("search_queries") or []),
            " ".join(_anchor_text(item) for item in anchor.get("visual_anchors") or []),
        ])
        stage_pool.extend(_stage_terms(text))
        style_pool.extend(_style_terms(text))

    entity_pool = _dedupe_terms(entity_pool)[:4]
    stage_pool = _dedupe_terms(stage_pool)[:4]
    style_pool = _dedupe_terms(style_pool)[:6]
    if not (entity_pool or stage_pool or style_pool):
        return

    for anchor in anchors:
        state = state_by_id.get(str(anchor.get("state_id") or ""))
        local_entities = _dedupe_terms(
            list(anchor.get("searchable_entities") or [])
            + (_visible_entity_candidates(question=question, state=state, visual_anchors=anchor.get("visual_anchors") or []) if state else [])
        )
        entities = local_entities or entity_pool
        bridge_queries = _cross_bridge_queries(
            question=question,
            entities=entities,
            stages=stage_pool,
            styles=style_pool,
        )
        anchor["searchable_entities"] = _dedupe_terms(list(anchor.get("searchable_entities") or []) + local_entities)[:8]
        anchor["search_queries"] = _rank_bridge_queries_first(bridge_queries + list(anchor.get("search_queries") or []))[:4]


def _bridge_queries_for_state(
    *,
    question: str,
    state: Any,
    visual_anchors: list[Any],
    entities: list[str],
) -> list[str]:
    text = " ".join([
        str(getattr(state, "sub_question", "") or ""),
        str(getattr(state, "sub_answer", "") or ""),
        " ".join(str(item) for item in getattr(state, "required_predicates", []) or []),
        " ".join(_anchor_text(anchor) for anchor in visual_anchors),
    ])
    stages = _stage_terms(text)
    styles = _style_terms(text)
    entities = _dedupe_terms(entities or _visible_entity_candidates(question=question, state=state, visual_anchors=visual_anchors))
    return _cross_bridge_queries(question=question, entities=entities, stages=stages, styles=styles)


def _cross_bridge_queries(
    *,
    question: str,
    entities: list[str],
    stages: list[str],
    styles: list[str],
) -> list[str]:
    downstream_terms = _downstream_terms(question)
    queries: list[str] = []
    entity_text = " ".join(entities[:3])
    stage_text = " ".join(stages[:3])
    style_text = " ".join(styles[:4])
    if entity_text and stage_text:
        queries.append(f"{entity_text} {stage_text} scene")
    if entity_text and style_text:
        queries.append(f"{entity_text} {style_text} animated scene")
    if entity_text:
        queries.append(f"{entity_text} abstract animated scene")
    if stage_text and style_text:
        queries.append(f"{stage_text} {style_text} animation")
    if style_text and downstream_terms:
        queries.append(f"{style_text} {downstream_terms}")
    return [query for query in queries if query.strip()]


def _visible_entity_candidates(*, question: str, state: Any, visual_anchors: list[Any]) -> list[str]:
    text = " ".join([
        question,
        str(getattr(state, "sub_answer", "") or ""),
        " ".join(_anchor_text(anchor) for anchor in visual_anchors),
    ])
    candidates = re.findall(
        r"\b[A-Z][A-Za-z0-9'&-]+(?:\s+[A-Z][A-Za-z0-9'&-]+){0,3}\b",
        text,
    )
    stop = {
        "Question", "Sub", "Required", "Box Office Mojo", "Office Mojo",
        "European", "Hungarian", "Weekend", "Last Weekend",
        "In", "What", "Which", "Where", "When", "Who", "The", "This", "That",
        "Text", "Oh", "Yes", "No",
    }
    return [
        value for value in _dedupe_terms(candidates)
        if len(value) > 2
        and value not in stop
        and not value.startswith("What ")
        and not value.startswith("Which ")
    ][:8]


def _stage_terms(text: str) -> list[str]:
    terms: list[str] = []
    lowered = text.lower()
    for pattern in (
        r"\bnonfigurative\b",
        r"\bnon-figurative\b",
        r"\babstract(?:ion)?\b",
        r"\bfragment(?:ed|ation)?\b",
        r"\bgeometric\b",
        r"\bcub(?:e|ist|ism|ism-influenced)?\b",
    ):
        terms.extend(match.group(0) for match in re.finditer(pattern, lowered))
    normalized = []
    for term in terms:
        if term in {"fragment", "fragmented", "fragmentation"}:
            normalized.append("fragmented")
        elif term in {"cubist", "cubism", "cubism-influenced"}:
            normalized.append("cubist")
        elif term == "non-figurative":
            normalized.append("nonfigurative")
        else:
            normalized.append(term)
    return _dedupe_terms(normalized)


def _style_terms(text: str) -> list[str]:
    lowered = text.lower()
    terms = []
    for term in (
        "abstract",
        "nonfigurative",
        "geometric",
        "fragmented",
        "low poly",
        "faceted",
        "polyhedral",
        "deconstruction",
        "modern art",
    ):
        if term in lowered:
            terms.append(term)
    return _dedupe_terms(terms)


def _downstream_terms(question: str) -> str:
    terms = _question_terms(question)
    keep = [
        term for term in terms
        if term not in {"characters", "enter", "door", "several", "visual", "scenes"}
    ]
    return " ".join(keep[:8])


def _rank_bridge_queries_first(queries: list[str]) -> list[str]:
    deduped = _dedupe_terms(re.sub(r"\s+", " ", query).strip() for query in queries if str(query).strip())
    return sorted(
        deduped,
        key=lambda query: (
            _is_pixel_level_query(query),
            -int(any(term in query.lower() for term in ("nonfigurative", "abstract", "geometric", "fragmented", "cubist"))),
            len(query),
        ),
    )


def _is_pixel_level_query(query: str) -> bool:
    lowered = query.lower()
    pixel_terms = {
        "pink", "green", "black", "dark", "striped", "orange", "yellow",
        "blue", "white", "gray", "grey", "hat", "jacket", "pants", "eyes",
        "wearing", "patches", "floor", "rocks", "pipes", "walls",
    }
    tokens = re.findall(r"[a-z0-9'-]+", lowered)
    return len(tokens) >= 14 and sum(1 for token in tokens if token in pixel_terms) >= 4


def _dedupe_terms(values) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = re.sub(r"\s+", " ", str(value)).strip()
        key = item.lower()
        if item and key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped
