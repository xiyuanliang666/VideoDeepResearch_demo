"""Structured evidence binding: web results → states/hypotheses via reasoning model."""

from __future__ import annotations

import json
import os
from typing import Any

from src.schemas import Binding, Evidence, RetrievalCandidate
from src.schemas.state import TemporalState

SUPPORT_TARGET_MAP = {
    "final_answer", "candidate_identity", "candidate_attribute",
    "candidate_difference", "event_record", "temporal_condition",
    "mechanism_condition", "rule_condition", "candidate_elimination",
}


def bind_evidence_to_states(
    *,
    question: str,
    states: list[TemporalState],
    hypotheses: list[Any],
    web_candidates: list[RetrievalCandidate],
    model_profile: dict | None = None,
    max_tokens: int = 1200,
) -> tuple[list[Evidence], list[Binding]]:
    """Call reasoning model once to produce structured evidence bindings.

    Falls back to token-overlap grounding when the LLM call fails.
    """
    from src.tools.env_tools import resolve_openai_config
    from src.tools.model_router import get_model

    real_candidates = [
        candidate for candidate in web_candidates
        if not _is_fallback_candidate(candidate)
    ]
    if not real_candidates:
        return [], []

    config = resolve_openai_config("evidence_binder")

    if not config.available:
        return _token_overlap_fallback(states, real_candidates)

    try:
        from openai import OpenAI
    except ImportError:
        return _token_overlap_fallback(states, real_candidates)

    results_text = "\n".join(
        f"[{c.candidate_id}] {c.metadata.get('title','')} | {c.metadata.get('snippet','')[:120]} | {c.metadata.get('url','')}"
        for c in real_candidates[:20]
    )
    states_text = "\n".join(f"- {s.state_id}: {s.sub_question}" for s in states)
    hyp_text = "\n".join(f"- {h.hypothesis_id}: {h.explanation}" for h in hypotheses) if hypotheses else "(none)"

    user_text = (
        f"Question: {question}\n\n"
        f"States:\n{states_text}\n\n"
        f"Hypotheses:\n{hyp_text}\n\n"
        f"Web results:\n{results_text}\n\n"
        "For each relevant web result, produce a binding to the most relevant state or hypothesis.\n"
        "CRITICAL: bound_to_id MUST use the EXACT state_id (e.g., S1, S2) or hypothesis_id "
        "(e.g., hyp_0, hyp_1) from the lists above. Do NOT invent new IDs.\n"
        "Return JSON array: [{evidence_id (=candidate_id), bound_to_type (state|hypothesis|final_answer), "
        "bound_to_id, relation (supports|refutes|uncertain), snippet (str), confidence (0-1)}]"
    )

    model_name = get_model("evidence_binder", model_profile)
    timeout = float(os.getenv("TIMEOUT_SECONDS", "30") or 30)
    client = OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=timeout)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a structured evidence analyst. Return JSON only."},
                {"role": "user", "content": user_text},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        raw = (response.choices[0].message.content or "").strip() if response.choices else ""
    except Exception:
        return _token_overlap_fallback(states, real_candidates)

    data = _parse_json_list(raw)
    if not data:
        return _token_overlap_fallback(states, real_candidates)

    candidate_by_id = {c.candidate_id: c for c in real_candidates}
    evidences: list[Evidence] = []
    bindings: list[Binding] = []

    for item in data:
        if not isinstance(item, dict):
            continue
        cand_id = str(item.get("evidence_id") or "")
        candidate = candidate_by_id.get(cand_id)
        if candidate is None:
            continue
        confidence = max(0.0, min(1.0, float(item.get("confidence") or 0.3)))
        relation = str(item.get("relation") or "uncertain")
        snippet = str(item.get("snippet") or candidate.metadata.get("snippet") or "")
        bound_to_id = str(item.get("bound_to_id") or "")
        support_target = _map_support_target(
            str(item.get("bound_to_type") or candidate.metadata.get("target_type") or ""),
            relation,
            fallback=str(candidate.metadata.get("support_target") or ""),
        )
        linked_candidate_id = str(
            item.get("linked_candidate_id")
            or candidate.metadata.get("linked_candidate_id")
            or (bound_to_id if str(item.get("bound_to_type") or "") == "hypothesis" else "")
        )

        ev = Evidence(
            evidence_id=f"ev_{cand_id}",
            evidence_type="web",
            source_ref=candidate.source_ref,
            content_summary=candidate.metadata.get("title", "")[:200],
            confidence=confidence,
            source_url=candidate.metadata.get("url", ""),
            raw_excerpt=snippet[:300],
        )
        ev.metadata = {
            "support_target": support_target,
            "bound_to_id": bound_to_id,
            "bound_to_type": str(item.get("bound_to_type") or ""),
            "linked_candidate_id": linked_candidate_id,
            "source_candidate_id": cand_id,
            "is_fallback": False,
        }
        binding = Binding(
            binding_id=f"binding_{cand_id}_{bound_to_id}",
            anchor_id=bound_to_id,
            evidence_id=ev.evidence_id,
            relation=relation,
            confidence=confidence,
            reason=f"LLM binding: {relation} → {bound_to_id}",
            judgeable_claim=f"Web result {cand_id} {relation} {bound_to_id}",
        )
        evidences.append(ev)
        bindings.append(binding)

    if evidences:
        return evidences, bindings
    return _token_overlap_fallback(states, real_candidates)


def _map_support_target(bound_to_type: str, relation: str, *, fallback: str = "") -> str:
    if fallback in SUPPORT_TARGET_MAP:
        return fallback
    if bound_to_type == "final_answer":
        return "final_answer"
    if relation == "refutes":
        return "candidate_elimination"
    if bound_to_type in {"state", "hypothesis"}:
        return "event_record"
    return "final_answer"


def _token_overlap_fallback(
    states: list[TemporalState],
    candidates: list[RetrievalCandidate],
) -> tuple[list[Evidence], list[Binding]]:
    """Simple token-overlap fallback (same logic as grounding.py)."""
    import re

    def tokenize(text: str) -> set[str]:
        return set(re.findall(r"[A-Za-z0-9一-鿿]+", text.lower()))

    state_tokens = {s.state_id: tokenize(s.sub_question) for s in states}
    evidences: list[Evidence] = []
    bindings: list[Binding] = []

    for candidate in candidates:
        if _is_fallback_candidate(candidate):
            continue
        title = candidate.metadata.get("title", "")
        snippet = candidate.metadata.get("snippet", "")
        cand_tokens = tokenize(f"{title} {snippet}")
        best_state_id = ""
        best_overlap = 0
        for state_id, tokens in state_tokens.items():
            overlap = len(tokens & cand_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_state_id = state_id

        confidence = min(1.0, 0.1 + 0.05 * best_overlap)
        ev = Evidence(
            evidence_id=f"ev_{candidate.candidate_id}",
            evidence_type="web",
            source_ref=candidate.source_ref,
            content_summary=title[:200],
            confidence=confidence,
            source_url=candidate.metadata.get("url", ""),
            raw_excerpt=snippet[:300],
        )
        support_target = str(candidate.metadata.get("support_target") or "final_answer")
        if support_target not in SUPPORT_TARGET_MAP:
            support_target = "final_answer"
        linked_candidate_id = str(candidate.metadata.get("linked_candidate_id") or "")
        ev.metadata = {
            "support_target": support_target,
            "bound_to_id": best_state_id,
            "bound_to_type": "state" if best_state_id else "question",
            "linked_candidate_id": linked_candidate_id,
            "source_candidate_id": candidate.candidate_id,
            "is_fallback": False,
        }
        binding = Binding(
            binding_id=f"binding_{candidate.candidate_id}_{best_state_id}",
            anchor_id=best_state_id or "question",
            evidence_id=ev.evidence_id,
            relation="supports" if best_overlap > 0 else "uncertain",
            confidence=confidence,
            reason=f"Token overlap: {best_overlap} terms",
            judgeable_claim=f"Web result {candidate.candidate_id} bound to {best_state_id or 'question'}",
        )
        evidences.append(ev)
        bindings.append(binding)
    return evidences, bindings


def _is_fallback_candidate(candidate: RetrievalCandidate) -> bool:
    return (
        candidate.retriever_name == "fallback_web_retriever"
        or str(candidate.source_ref).startswith("fallback://")
    )


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
