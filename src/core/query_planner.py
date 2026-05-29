"""Retrieval intent planning and query composition."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from src.schemas.state import TemporalState


class RetrievalIntent:
    def __init__(self, **kwargs: Any) -> None:
        self.intent_id: str = kwargs.get("intent_id", "")
        self.slot_id: str = kwargs.get("slot_id") or kwargs.get("state_id", "")
        self.state_id: str = self.slot_id
        self.slot_name: str = kwargs.get("slot_name") or kwargs.get("state_id", "")
        self.target_type: str = kwargs.get("target_type", "state")
        self.target_id: str = kwargs.get("target_id", "")
        self.support_target: str = kwargs.get("support_target", "final_answer")
        self.linked_candidate_id: str = kwargs.get("linked_candidate_id", "")
        self.source_anchor_ids: list[str] = kwargs.get("source_anchor_ids", [])
        self.query_templates: list[str] = kwargs.get("query_templates", [])
        self.search_domains: list[str] = kwargs.get("search_domains", [])
        self.expected_answer_type: str = kwargs.get("expected_answer_type", "")
        self.current_uncertainty: list[str] = kwargs.get("current_uncertainty", [])
        self.unresolved_constraints: dict[str, Any] = kwargs.get("unresolved_constraints", kwargs.get("hard_constraints", {}))
        self.hard_constraints: dict[str, Any] = self.unresolved_constraints
        self.candidate_conflicts: list[str] = kwargs.get("candidate_conflicts", [])
        self.missing_evidence: list[str] = kwargs.get("missing_evidence", [])
        self.soft_clues: list[str] = kwargs.get("soft_clues", [])
        self.consistency_check: str = kwargs.get("consistency_check", kwargs.get("verification_rule", ""))
        self.verification_rule: str = self.consistency_check
        self.retrieval_rationale: str = kwargs.get("retrieval_rationale", "")
        self.depends_on: list[str] = kwargs.get("depends_on", [])
        self.planner_phase: str = kwargs.get("planner_phase", "uncertainty_aware_v1")
        self.priority: int = kwargs.get("priority", 1)
        self.status: str = "pending"
        self.retrieved_result_ids: list[str] = []

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class RetrievalSlot:
    def __init__(self, **kwargs: Any) -> None:
        self.slot_id: str = kwargs.get("slot_id") or kwargs.get("state_id", "")
        self.state_id: str = self.slot_id
        self.slot_name: str = kwargs.get("slot_name") or kwargs.get("state_id", "")
        self.support_target: str = kwargs.get("support_target", "final_answer")
        self.expected_answer_type: str = kwargs.get("expected_answer_type", "")
        self.source_state_ids: list[str] = kwargs.get("source_state_ids", [])
        self.source_anchor_ids: list[str] = kwargs.get("source_anchor_ids", [])
        self.depends_on: list[str] = kwargs.get("depends_on", [])
        self.current_uncertainty: list[str] = kwargs.get("current_uncertainty", [])
        self.unresolved_constraints: dict[str, Any] = kwargs.get("unresolved_constraints", kwargs.get("hard_constraints", {}))
        self.hard_constraints: dict[str, Any] = self.unresolved_constraints
        self.candidate_conflicts: list[str] = kwargs.get("candidate_conflicts", [])
        self.missing_evidence: list[str] = kwargs.get("missing_evidence", [])
        self.soft_clues: list[str] = kwargs.get("soft_clues", [])
        self.consistency_check: str = kwargs.get("consistency_check", kwargs.get("verification_rule", ""))
        self.verification_rule: str = self.consistency_check
        self.retrieval_rationale: str = kwargs.get("retrieval_rationale", "")
        self.query_candidates: list[str] = kwargs.get("query_candidates", [])
        self.priority: int = kwargs.get("priority", 1)
        self.status: str = kwargs.get("status", "pending")
        self.planner_phase: str = kwargs.get("planner_phase", "uncertainty_aware_v1")

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def plan_retrieval_intents(
    *,
    question: str,
    states: list[TemporalState],
    anchors: list[dict[str, Any]],
    hypotheses: list[Any],
    model_profile: dict | None = None,
    max_intents: int = 8,
    max_tokens: int = 1000,
) -> list[RetrievalIntent]:
    """Plan runtime retrieval needs, then project them to retrieval intents."""
    from src.tools.env_tools import resolve_openai_config
    from src.tools.model_router import get_model

    config = resolve_openai_config("query_planner")
    if not config.available:
        return _fallback_intents(states, question, anchors=anchors, max_intents=max_intents)

    try:
        from openai import OpenAI
    except ImportError:
        return _fallback_intents(states, question, anchors=anchors, max_intents=max_intents)

    anchors_text = "\n".join(
        "- "
        + str(anchor.get("state_id") or anchor.get("anchor_id") or "")
        + ": visual="
        + str(anchor.get("visual_anchors") or [])[:500]
        + " | entities="
        + str(anchor.get("searchable_entities") or [])[:240]
        + " | queries="
        + str(anchor.get("search_queries") or [])[:500]
        for anchor in anchors[:12]
    )
    states_text = "\n".join(
        f"- {s.state_id}: {s.sub_question} | answer: {s.sub_answer} | "
        f"visual_anchors: {s.visual_anchors[:4]} | state_queries: {s.search_queries[:2]}"
        for s in states
    )
    hyp_text = "\n".join(
        f"- {h.hypothesis_id}: {h.explanation}" for h in hypotheses
    ) if hypotheses else "(none)"

    user_text = (
        f"Question: {question}\n\n"
        f"States:\n{states_text}\n\n"
        f"Extracted visual anchors:\n{anchors_text or '(none)'}\n\n"
        f"Hypotheses:\n{hyp_text}\n\n"
        f"Plan up to {max_intents} runtime retrieval actions that progressively reduce uncertainty using "
        "available video observations, anchors, hypotheses, and retrieval history. "
        "Retrieval states are temporary runtime uncertainty states, not benchmark schemas, question categories, "
        "or fixed retrieval templates. "
        "Each retrieval state should describe current uncertainty, missing evidence, unresolved constraints, "
        "candidate conflicts and competing interpretations. "
        "Do not turn retrieval needs into a question-type taxonomy or fixed domain recipe. "
        "Do NOT copy the whole question as a query and avoid degenerate retrieval actions that do not reduce uncertainty. "
        "Avoid prematurely collapsing to highly specific candidate entities before sufficient retrieval evidence supports them. "
        "When candidate identity is still uncertain, avoid retrieval actions that assume a specific final entity. "
        "Specific candidate names may be used after they emerge from runtime retrieval observations or verified intermediate hypotheses. "
        "Prefer short query candidates grounded in the current uncertainty and available constraints. "
        "Return JSON array: [{state_id: str, source_state_ids: [str], source_anchor_ids: [str], "
        "current_uncertainty: [str], unresolved_constraints: object, candidate_conflicts: [str], "
        "missing_evidence: [str], query_candidates: [str], retrieval_rationale: str, priority: 1-3}]"
    )

    model_name = get_model("query_planner", model_profile)
    timeout = float(os.getenv("TIMEOUT_SECONDS", "30") or 30)
    client = OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=timeout)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a search strategist. Return JSON only."},
                {"role": "user", "content": user_text},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        raw = (response.choices[0].message.content or "").strip() if response.choices else ""
    except Exception:
        return _fallback_intents(states, question, anchors=anchors, max_intents=max_intents)

    data = _parse_json_list(raw)
    if not data:
        return _fallback_intents(states, question, anchors=anchors, max_intents=max_intents)

    slots = _slots_from_model_data(data, question=question, states=states, anchors=anchors)
    return _slots_to_intents(slots, max_intents=max_intents) or _fallback_intents(
        states,
        question,
        anchors=anchors,
        max_intents=max_intents,
    )


def compose_queries(intents: list[RetrievalIntent]) -> list[tuple[RetrievalIntent, str]]:
    """Expand each intent's first query template into a concrete query string."""
    return [
        (intent, intent.query_templates[0])
        for intent in intents
        if intent.query_templates
    ]


def _fallback_intents(
    states: list[TemporalState],
    question: str,
    anchors: list[dict[str, Any]] | None = None,
    max_intents: int = 8,
) -> list[RetrievalIntent]:
    return _slots_to_intents(
        _fallback_slots(states, question, anchors=anchors or []),
        max_intents=max_intents,
    )


def slots_from_intents(intents: list[Any]) -> list[dict[str, Any]]:
    """Recover unique slot records from projected retrieval intents."""
    slots: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, intent in enumerate(intents):
        slot_id = str(getattr(intent, "slot_id", "") or getattr(intent, "intent_id", "") or f"slot_{index:03d}")
        if slot_id in seen:
            continue
        seen.add(slot_id)
        slots.append({
            "state_id": slot_id,
            "slot_id": slot_id,
            "slot_name": str(getattr(intent, "slot_name", "") or ""),
            "support_target": str(getattr(intent, "support_target", "") or "final_answer"),
            "expected_answer_type": str(getattr(intent, "expected_answer_type", "") or ""),
            "source_state_ids": [str(getattr(intent, "target_id", "") or "")] if str(getattr(intent, "target_type", "") or "") == "state" and str(getattr(intent, "target_id", "") or "") else [],
            "source_anchor_ids": [str(item) for item in (getattr(intent, "source_anchor_ids", []) or [])],
            "depends_on": [str(item) for item in (getattr(intent, "depends_on", []) or [])],
            "current_uncertainty": [str(item) for item in (getattr(intent, "current_uncertainty", []) or [])],
            "unresolved_constraints": getattr(intent, "unresolved_constraints", getattr(intent, "hard_constraints", {})) or {},
            "candidate_conflicts": [str(item) for item in (getattr(intent, "candidate_conflicts", []) or [])],
            "missing_evidence": [str(item) for item in (getattr(intent, "missing_evidence", []) or [])],
            "soft_clues": [str(item) for item in (getattr(intent, "soft_clues", []) or [])],
            "consistency_check": str(getattr(intent, "consistency_check", getattr(intent, "verification_rule", "")) or ""),
            "retrieval_rationale": str(getattr(intent, "retrieval_rationale", "") or ""),
            "query_candidates": [str(item) for item in (getattr(intent, "query_templates", []) or [])],
            "priority": int(getattr(intent, "priority", 1) or 1),
            "status": str(getattr(intent, "status", "pending") or "pending"),
            "planner_phase": str(getattr(intent, "planner_phase", "uncertainty_aware_v1") or "uncertainty_aware_v1"),
        })
    return slots


def _fallback_slots(
    states: list[TemporalState],
    question: str,
    anchors: list[dict[str, Any]] | None = None,
) -> list[RetrievalSlot]:
    decomposed = _heuristic_multihop_queries(question, anchors or [])
    slots: list[RetrievalSlot] = []
    if decomposed:
        targets = states or []
        for i, plan in enumerate(decomposed[: max(len(decomposed), 1)]):
            state = _select_source_state_for_plan(plan, targets, i) if targets else None
            slots.append(RetrievalSlot(
                slot_id=f"slot_{i:03d}",
                slot_name=str(plan.get("slot_name") or f"slot_{i:03d}"),
                support_target=_normalize_support_target(str(plan.get("support_target") or "")),
                expected_answer_type=str(plan.get("expected_answer_type") or ""),
                source_state_ids=[state.state_id] if state else [],
                source_anchor_ids=[],
                depends_on=[f"slot_{i - 1:03d}"] if i > 0 else [],
                current_uncertainty=[str(item) for item in (plan.get("current_uncertainty") or [])],
                unresolved_constraints=plan.get("unresolved_constraints") if isinstance(plan.get("unresolved_constraints"), dict) else {},
                candidate_conflicts=[str(item) for item in (plan.get("candidate_conflicts") or [])],
                missing_evidence=[str(item) for item in (plan.get("missing_evidence") or [])],
                soft_clues=[str(item) for item in (plan.get("soft_clues") or [])],
                consistency_check=str(plan.get("consistency_check") or ""),
                retrieval_rationale=str(plan.get("retrieval_rationale") or ""),
                query_candidates=[str(q) for q in (plan.get("queries") or [])],
                priority=int(plan.get("priority") or 1),
                planner_phase="uncertainty_aware_v1_fallback",
            ))
        return slots

    for i, state in enumerate(states):
        state_queries = _clean_query_templates(
            list(state.search_queries or []),
            question=question,
            states=states,
            anchors=anchors or [],
        )
        query = state_queries[0] if state_queries else _compact_question_query(question)
        support_target = _infer_support_target(state, question)
        slots.append(RetrievalSlot(
            slot_id=f"slot_{i:03d}",
            slot_name=f"state_{state.state_id}_web_support",
            support_target=support_target,
            expected_answer_type="evidence",
            source_state_ids=[state.state_id],
            source_anchor_ids=[],
            current_uncertainty=["Need external evidence for the selected video-derived state."],
            unresolved_constraints=_constraints_from_text(" ".join([question, state.sub_question, state.sub_answer])),
            candidate_conflicts=[],
            missing_evidence=["external evidence relevant to the current uncertainty"],
            soft_clues=list(state.visual_anchors or [])[:4],
            consistency_check="Observation should reduce uncertainty and remain consistent with known constraints.",
            retrieval_rationale="Use the state-level video observation as a search anchor, then verify whether web evidence reduces the unresolved uncertainty.",
            query_candidates=[query],
            priority=1,
        ))
    if not slots:
        slots.append(RetrievalSlot(
            slot_id="slot_000",
            slot_name="question_web_support",
            support_target="final_answer",
            expected_answer_type="evidence",
            source_state_ids=[],
            source_anchor_ids=[],
            current_uncertainty=["Need external evidence relevant to the unresolved question."],
            unresolved_constraints=_constraints_from_text(question),
            candidate_conflicts=[],
            missing_evidence=["external evidence relevant to the question"],
            soft_clues=[],
            consistency_check="Observation should reduce uncertainty and remain consistent with known constraints.",
            retrieval_rationale="Start from compact question constraints because no state-specific retrieval plan is available.",
            query_candidates=[_compact_question_query(question)],
            priority=1,
        ))
    return slots


def _select_source_state_for_plan(
    plan: dict[str, Any],
    states: list[TemporalState],
    index: int,
) -> TemporalState | None:
    if not states:
        return None
    slot_text = " ".join(
        str(plan.get(key) or "")
        for key in ("slot_name", "support_target", "expected_answer_type", "consistency_check")
    ).lower()
    wants_visual_evidence = any(term in slot_text for term in ("visual", "appearance", "image", "scene"))
    if wants_visual_evidence:
        scored: list[tuple[int, TemporalState]] = []
        for state in states:
            state_text = " ".join([
                str(state.sub_question),
                str(state.sub_answer),
                " ".join(str(item) for item in state.required_predicates),
            ]).lower()
            score = sum(
                1 for term in ("visual", "appearance", "scene", "image", "object", "text", "shape", "color")
                if term in state_text
            )
            scored.append((score, state))
        best_score, best_state = max(scored, key=lambda item: item[0])
        if best_score > 0:
            return best_state
    return states[min(index, len(states) - 1)]


def _constraints_from_text(text: str) -> dict[str, Any]:
    constraints: dict[str, Any] = {}
    years = re.findall(r"\b(?:19|20)\d{2}\b", text)
    if years:
        constraints["years"] = list(dict.fromkeys(years))
    quoted = [
        " ".join(part for part in pair if part)
        for pair in re.findall(r"\"([^\"]+)\"|'([^']+)'", text)
    ]
    if quoted:
        constraints["quoted_terms"] = list(dict.fromkeys(item for item in quoted if item))
    source_hints = _source_hints(text)
    if source_hints:
        constraints["source_hints"] = source_hints
    field_hints = _field_hints(text)
    if field_hints:
        constraints["requested_fields"] = field_hints
    return constraints


def _slots_to_intents(slots: list[RetrievalSlot], *, max_intents: int) -> list[RetrievalIntent]:
    intents: list[RetrievalIntent] = []
    for index, slot in enumerate(slots[:max_intents]):
        templates = [str(query) for query in slot.query_candidates if str(query).strip()]
        if not templates:
            continue
        target_id = slot.source_state_ids[0] if slot.source_state_ids else "question"
        intents.append(RetrievalIntent(
            intent_id=f"intent_{index:03d}_{slot.slot_id or f'slot_{index:03d}'}",
            slot_id=slot.slot_id or f"slot_{index:03d}",
            slot_name=slot.slot_name,
            target_type="state" if slot.source_state_ids else "question",
            target_id=target_id,
            support_target=slot.support_target,
            linked_candidate_id="",
            source_anchor_ids=slot.source_anchor_ids,
            query_templates=templates[:4],
            expected_answer_type=slot.expected_answer_type,
            current_uncertainty=slot.current_uncertainty,
            unresolved_constraints=slot.unresolved_constraints,
            hard_constraints=slot.unresolved_constraints,
            candidate_conflicts=slot.candidate_conflicts,
            missing_evidence=slot.missing_evidence,
            soft_clues=slot.soft_clues,
            consistency_check=slot.consistency_check,
            verification_rule=slot.consistency_check,
            retrieval_rationale=slot.retrieval_rationale,
            depends_on=slot.depends_on,
            priority=slot.priority,
            planner_phase=slot.planner_phase,
        ))
    return intents


def _clean_query_templates(
    templates: list[str],
    *,
    question: str,
    states: list[TemporalState],
    anchors: list[dict[str, Any]],
) -> list[str]:
    cleaned: list[str] = []
    for query in templates:
        value = re.sub(r"\s+", " ", str(query)).strip()
        if not value:
            continue
        value = _remove_unverified_identifier_leaks(value, question)
        if _is_degenerate_query(value, question):
            continue
        if len(value) > 180:
            value = _compact_question_query(value)
        if value and value.lower() not in {item.lower() for item in cleaned}:
            cleaned.append(value)
    if cleaned:
        return cleaned[:4]
    plans = _heuristic_multihop_queries(question, anchors)
    if plans:
        return plans[0]["queries"]
    state_terms = []
    for state in states[:2]:
        state_terms.extend(state.visual_anchors[:3])
        state_terms.extend(state.required_predicates[:3])
    fallback = " ".join(str(term) for term in state_terms if term).strip()
    return [fallback[:160]] if fallback else []


def _slots_from_model_data(
    data: list,
    *,
    question: str,
    states: list[TemporalState],
    anchors: list[dict[str, Any]],
) -> list[RetrievalSlot]:
    slots: list[RetrievalSlot] = []
    state_ids = {str(state.state_id) for state in states}
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        raw_queries = item.get("query_candidates")
        if raw_queries is None:
            raw_queries = item.get("query_templates")
        queries = _clean_query_templates(
            [str(query) for query in (raw_queries or []) if query],
            question=question,
            states=states,
            anchors=anchors,
        )
        if not queries:
            continue
        source_state_ids = [
            str(value) for value in (item.get("source_state_ids") or [])
            if str(value) in state_ids
        ]
        if not source_state_ids and str(item.get("target_type") or "") == "state":
            target_id = str(item.get("target_id") or "")
            if target_id in state_ids:
                source_state_ids.append(target_id)
        slot_id = str(item.get("state_id") or item.get("slot_id") or f"slot_{index:03d}")
        slots.append(RetrievalSlot(
            slot_id=slot_id,
            slot_name=str(item.get("slot_name") or item.get("intent_id") or item.get("state_id") or slot_id),
            support_target=_normalize_support_target(str(item.get("support_target") or "")),
            expected_answer_type=str(item.get("expected_answer_type") or ""),
            source_state_ids=source_state_ids,
            source_anchor_ids=[str(value) for value in (item.get("source_anchor_ids") or []) if value],
            depends_on=[str(value) for value in (item.get("depends_on") or []) if value],
            current_uncertainty=[str(value) for value in (item.get("current_uncertainty") or []) if value],
            unresolved_constraints=(
                item.get("unresolved_constraints")
                if isinstance(item.get("unresolved_constraints"), dict)
                else item.get("hard_constraints")
                if isinstance(item.get("hard_constraints"), dict)
                else {}
            ),
            candidate_conflicts=[str(value) for value in (item.get("candidate_conflicts") or []) if value],
            missing_evidence=[str(value) for value in (item.get("missing_evidence") or []) if value],
            soft_clues=[str(value) for value in (item.get("soft_clues") or []) if value],
            consistency_check=str(item.get("consistency_check") or item.get("verification_rule") or ""),
            retrieval_rationale=str(item.get("retrieval_rationale") or ""),
            query_candidates=queries,
            priority=int(item.get("priority") or 1),
            planner_phase="uncertainty_aware_v1_model",
        ))
    return slots


def _is_degenerate_query(query: str, question: str) -> bool:
    q_norm = _norm(query)
    question_norm = _norm(question)
    if not q_norm:
        return True
    if q_norm == question_norm:
        return True
    if len(query) > 220:
        return True
    q_tokens = set(re.findall(r"[a-z0-9]+", q_norm))
    question_tokens = set(re.findall(r"[a-z0-9]+", question_norm))
    if len(q_tokens) >= 18 and question_tokens:
        overlap = len(q_tokens & question_tokens) / max(len(q_tokens), 1)
        return overlap > 0.82
    return False


def _heuristic_multihop_queries(question: str, anchors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    text = question.strip()
    anchor_terms = _anchor_terms(anchors)
    constraints = _constraints_from_text(text)
    question_terms = _content_terms(text)
    plans: list[dict[str, Any]] = []

    if anchor_terms or _looks_visual_question(text):
        visual_query = anchor_terms[0] if anchor_terms else _join_query(question_terms[:8])
        alternate_visual_query = anchor_terms[1] if len(anchor_terms) > 1 else _join_query(question_terms[:10])
        plans.append({
            "slot_name": "visual_anchor_uncertainty",
            "support_target": "candidate_attribute",
            "expected_answer_type": "evidence",
            "current_uncertainty": ["Which external clue or candidate best explains the video-derived visual evidence."],
            "unresolved_constraints": constraints,
            "candidate_conflicts": [],
            "missing_evidence": ["external evidence that explains or names the video-derived visual clue"],
            "soft_clues": anchor_terms[:6],
            "consistency_check": "Observation should reduce visual-anchor uncertainty and remain consistent with video observations.",
            "retrieval_rationale": "Search from visual anchors before assuming a final downstream entity.",
            "queries": [
                visual_query,
                alternate_visual_query,
            ],
            "priority": 1,
        })

    if question_terms:
        discovery_terms = _join_query(_prioritized_terms(question_terms, constraints)[:10])
        if discovery_terms:
            plans.append({
                "slot_name": "candidate_or_clue_uncertainty",
                "support_target": "candidate_identity",
                "expected_answer_type": "entity_or_clue",
                "current_uncertainty": ["Which candidate or intermediate clue should be carried into the next retrieval step."],
                "unresolved_constraints": constraints,
                "candidate_conflicts": [],
                "missing_evidence": ["candidate, clue, or intermediate fact needed for the next retrieval step"],
                "soft_clues": anchor_terms[:4],
                "consistency_check": "Observation should identify a candidate or intermediate clue without conflicting with known constraints.",
                "retrieval_rationale": "Use compact constraints to discover candidate clues, then let later retrieval verify attributes.",
                "queries": [
                    discovery_terms,
                    anchor_terms[0] if anchor_terms else _join_query(question_terms[:10]),
                ],
                "priority": 1,
            })

    source_hints = constraints.get("source_hints") or []
    field_hints = constraints.get("requested_fields") or []
    if source_hints or field_hints:
        lookup_terms = _join_query((source_hints + field_hints + ["verified candidate"])[:10])
        plans.append({
            "slot_name": "record_or_attribute_uncertainty",
            "support_target": "event_record",
            "expected_answer_type": "record_or_attribute_evidence",
            "current_uncertainty": ["Which requested record or attribute value belongs to a verified candidate."],
            "unresolved_constraints": {
                **constraints,
                "candidate": "requires_verified_candidate",
            },
            "candidate_conflicts": ["Avoid using this lookup before a candidate is verified."],
            "missing_evidence": ["requested field or record value for a verified candidate"],
            "soft_clues": [],
            "consistency_check": "Observation should provide the requested field for a verified candidate and not drift to a stale retrieval path.",
            "retrieval_rationale": "Delay record lookup until candidate evidence is available, while preserving source and field constraints.",
            "queries": [
                lookup_terms,
                _join_query((field_hints + source_hints)[:10]),
            ],
            "priority": 1,
        })

    if plans:
        return plans[:6]

    named_bits = re.findall(r"\"([^\"]+)\"|'([^']+)'|\b[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,3}", text)
    queries = []
    for item in named_bits:
        value = " ".join(part for part in item if part) if isinstance(item, tuple) else str(item)
        value = value.strip()
        if value and value.lower() not in {"in", "for", "as"}:
            queries.append(value)
    compact = _compact_question_query(text)
    if compact:
        queries.append(compact)
    return [{
        "slot_name": "open_uncertainty",
        "support_target": "final_answer",
        "expected_answer_type": "evidence",
        "current_uncertainty": ["Need external evidence that reduces the unresolved question."],
        "unresolved_constraints": constraints,
        "candidate_conflicts": [],
        "missing_evidence": ["external evidence relevant to the unresolved question"],
        "soft_clues": anchor_terms[:4],
        "consistency_check": "Observation should reduce uncertainty and remain consistent with known constraints.",
        "retrieval_rationale": "Use compact named and constraint-bearing terms because no stronger runtime anchor is available.",
        "queries": queries[:3] or [compact],
        "priority": 1,
    }]


def _anchor_terms(anchors: list[dict[str, Any]]) -> list[str]:
    terms: list[str] = []
    for anchor in anchors:
        for query in anchor.get("search_queries") or []:
            value = str(query).strip()
            if value:
                terms.append(value)
        for entity in anchor.get("searchable_entities") or []:
            terms.append(str(entity))
        for visual in anchor.get("visual_anchors") or []:
            if isinstance(visual, dict):
                value = str(visual.get("description") or visual.get("text") or "")
            else:
                value = str(visual)
            abstracted = _abstract_visual_term(value)
            if abstracted:
                terms.append(abstracted)
    return _dedupe_texts(term for term in terms if term.strip())[:12]


def _abstract_visual_term(text: str) -> str:
    lowered = text.lower()
    terms: list[str] = []
    for term in (
        "nonfigurative",
        "non-figurative",
        "abstract",
        "geometric",
        "fragmented",
        "low-poly",
        "low poly",
        "faceted",
        "polyhedral",
        "cubist",
        "modern art",
        "animated scene",
    ):
        if term in lowered:
            terms.append("nonfigurative" if term == "non-figurative" else term)
    if terms:
        return " ".join(_dedupe_texts(terms))
    return ""


def _dedupe_texts(values) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = re.sub(r"\s+", " ", str(value)).strip()
        key = item.lower()
        if item and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _remove_unverified_identifier_leaks(query: str, question: str) -> str:
    """Remove opaque IDs that were not present in the task input."""
    question_ids = set(_opaque_ids(question))
    value = query
    for identifier in _opaque_ids(query):
        if identifier.lower() not in question_ids:
            value = re.sub(re.escape(identifier), "", value, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", value).strip()


def _opaque_ids(text: str) -> list[str]:
    return re.findall(r"\b[a-z]{1,4}\d{5,}\b|\b\d{7,}\b", text, flags=re.IGNORECASE)


def _looks_visual_question(text: str) -> bool:
    lowered = text.lower()
    return any(
        term in lowered
        for term in (
            "video",
            "visual",
            "scene",
            "frame",
            "image",
            "appears",
            "resembles",
            "looks like",
        )
    )


def _content_terms(text: str) -> list[str]:
    lowered = text.lower()
    stop = {
        "what", "which", "where", "when", "who", "whose", "why", "how",
        "the", "a", "an", "and", "or", "of", "in", "on", "to", "for",
        "with", "from", "by", "as", "is", "are", "was", "were", "be",
        "this", "that", "these", "those", "according", "external", "sources",
        "video", "question", "answer", "current", "later", "another",
    }
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9'-]*|\b(?:19|20)\d{2}\b", lowered)
    terms = [token for token in tokens if len(token) > 2 and token not in stop]
    quoted = [
        " ".join(part for part in pair if part)
        for pair in re.findall(r"\"([^\"]+)\"|'([^']+)'", text)
    ]
    return list(dict.fromkeys([*quoted, *terms]))


def _prioritized_terms(terms: list[str], constraints: dict[str, Any]) -> list[str]:
    priority: list[str] = []
    for key in ("years", "quoted_terms", "requested_fields", "source_hints"):
        value = constraints.get(key)
        if isinstance(value, list):
            priority.extend(str(item) for item in value)
    priority.extend(terms)
    return list(dict.fromkeys(item for item in priority if str(item).strip()))


def _source_hints(text: str) -> list[str]:
    hints: list[str] = []
    patterns = [
        r"according to ([A-Z][A-Za-z0-9&.,' -]{2,80})",
        r"on ([A-Z][A-Za-z0-9&.,' -]{2,80})",
        r"in ([A-Z][A-Za-z0-9&.,' -]{2,80}) database",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, text):
            cleaned = re.sub(r"\s+", " ", str(match)).strip(" .,'")
            if cleaned:
                hints.append(cleaned)
    return list(dict.fromkeys(hints[:3]))


def _field_hints(text: str) -> list[str]:
    lowered = text.lower()
    hints: list[str] = []
    patterns = [
        r"what is (?:its|the|their)?\s*([a-z][a-z0-9 -]{2,80})",
        r"what was (?:its|the|their)?\s*([a-z][a-z0-9 -]{2,80})",
        r"which ([a-z][a-z0-9 -]{2,80})",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, lowered):
            cleaned = re.split(r"\baccording to\b|\bin\b|\bon\b|\?", match)[0]
            cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,'")
            if cleaned:
                hints.append(cleaned)
    return list(dict.fromkeys(hints[:3]))


def _join_query(terms: list[str]) -> str:
    return " ".join(str(term).strip() for term in terms if str(term).strip())[:180]


def _compact_question_query(question: str) -> str:
    text = re.sub(r"\s+", " ", question).strip()
    quoted = re.findall(r"\"([^\"]+)\"|'([^']+)'", text)
    pieces = [" ".join(item for item in pair if item) for pair in quoted]
    years = re.findall(r"\b(?:19|20)\d{2}\b", text)
    caps = re.findall(r"\b[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,3}", text)
    pieces.extend(years)
    pieces.extend(caps[:8])
    if pieces:
        return " ".join(dict.fromkeys(piece for piece in pieces if piece))[:180]
    return text[:160]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _normalize_support_target(value: str) -> str:
    allowed = {
        "final_answer",
        "candidate_identity",
        "candidate_attribute",
        "candidate_difference",
        "event_record",
        "temporal_condition",
        "mechanism_condition",
        "rule_condition",
        "candidate_elimination",
    }
    return value if value in allowed else "final_answer"


def _infer_support_target(state: TemporalState, question: str) -> str:
    text = " ".join([question, state.sub_question, state.sub_answer, " ".join(state.required_predicates)]).lower()
    if any(term in text for term in ("minute", "date", "stage", "score", "match", "record", "event")):
        return "event_record"
    if any(term in text for term in ("difference", "better", "which clip", "compare", "rather than")):
        return "candidate_difference"
    if any(term in text for term in ("because", "mechanism", "condition", "rule", "allowing", "failing")):
        return "mechanism_condition"
    if any(term in text for term in ("who", "person", "actor", "scorer", "creator", "owner")):
        return "candidate_identity"
    return "final_answer"


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
