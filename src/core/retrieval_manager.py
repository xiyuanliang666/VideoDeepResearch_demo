"""Iterative web retrieval with budget control."""

from __future__ import annotations

import os
import re
from typing import Any

from src.core.retrieval import _fallback_candidates, _search_serper, _search_tavily
from src.schemas import QueryUnit, RetrievalCandidate


def run_retrieval_for_intents(
    intents: list[Any],
    *,
    topk: int = 5,
    max_rounds: int = 2,
    return_memory: bool = False,
) -> tuple[list[QueryUnit], list[RetrievalCandidate]] | tuple[list[QueryUnit], list[RetrievalCandidate], dict[str, Any]]:
    """Execute web search for each intent, with one refinement round if needed.

    Returns (all_query_units, all_candidates), or includes retrieval memory when requested.
    """
    serper_key = os.getenv("SERPER_API_KEY", "").strip()
    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
    provider = "serper" if serper_key else "tavily" if tavily_key else ""

    all_queries: list[QueryUnit] = []
    all_candidates: list[RetrievalCandidate] = []
    memory = _new_memory(provider=provider or "fallback")
    global_rank_offset = 0

    for step, intent in enumerate(intents):
        intent_runtime = _intent_runtime_metadata(intent)
        templates = list(dict.fromkeys(
            str(query).strip()
            for query in (getattr(intent, "query_templates", []) or [])
            if str(query).strip()
        ))
        if not templates:
            continue

        intent_candidate_ids: list[str] = []
        tried_queries: list[str] = []
        rounds = max(1, int(max_rounds or 1))

        for round_index in range(rounds):
            query_text = templates[round_index] if round_index < len(templates) else _refine_query(
                intent=intent,
                failed_queries=tried_queries,
                memory=memory,
            )
            query_text = re.sub(r"\s+", " ", query_text).strip()
            if query_text.lower() in {query.lower() for query in tried_queries}:
                query_text = _refine_query(
                    intent=intent,
                    failed_queries=tried_queries,
                    memory=memory,
                )
                query_text = re.sub(r"\s+", " ", query_text).strip()
            if not query_text or query_text.lower() in {query.lower() for query in tried_queries}:
                break

            query_unit = QueryUnit(
                query_id=f"web_query_{step:04d}_{intent.intent_id}_r{round_index}",
                anchor_id=getattr(intent, "target_id", ""),
                query_text=query_text,
                query_type="web_search" if round_index == 0 else "web_search_refine",
                step_index=step,
                motivation=_query_motivation(intent, round_index),
                query_source="retrieval_intent" if round_index == 0 else "observation_driven_refinement",
                rewrite_history=list(tried_queries),
                support_target=getattr(intent, "support_target", "final_answer"),
                linked_candidate_id=getattr(intent, "linked_candidate_id", ""),
                target_type=getattr(intent, "target_type", "state"),
                target_id=getattr(intent, "target_id", ""),
                **intent_runtime,
            )
            all_queries.append(query_unit)
            tried_queries.append(query_text)

            candidates = _execute_search(
                query_text=query_text,
                query_unit=query_unit,
                provider=provider,
                serper_key=serper_key,
                tavily_key=tavily_key,
                topk=topk,
                rank_offset=global_rank_offset,
            )
            global_rank_offset += len(candidates)
            all_candidates.extend(candidates)
            intent_candidate_ids.extend(candidate.candidate_id for candidate in candidates)

            decision, decision_reason = _control_decision(candidates)
            _update_memory(
                memory=memory,
                intent=intent,
                query_unit=query_unit,
                candidates=candidates,
                decision=decision,
                decision_reason=decision_reason,
                round_index=round_index,
            )
            if decision == "advance":
                break

        intent.retrieved_result_ids = intent_candidate_ids
        intent.status = "satisfied" if _intent_has_partial_evidence(memory, intent) else "exhausted"

    if return_memory:
        return all_queries, all_candidates, memory
    return all_queries, all_candidates


def _intent_runtime_metadata(intent: Any) -> dict[str, Any]:
    unresolved = getattr(intent, "unresolved_constraints", getattr(intent, "hard_constraints", {})) or {}
    consistency_check = str(getattr(intent, "consistency_check", getattr(intent, "verification_rule", "")) or "")
    return {
        "slot_id": str(getattr(intent, "slot_id", "") or ""),
        "state_id": str(getattr(intent, "state_id", getattr(intent, "slot_id", "")) or ""),
        "slot_name": str(getattr(intent, "slot_name", "") or ""),
        "expected_answer_type": str(getattr(intent, "expected_answer_type", "") or ""),
        "current_uncertainty": [str(item) for item in (getattr(intent, "current_uncertainty", []) or [])],
        "unresolved_constraints": unresolved,
        "hard_constraints": unresolved,
        "candidate_conflicts": [str(item) for item in (getattr(intent, "candidate_conflicts", []) or [])],
        "missing_evidence": [str(item) for item in (getattr(intent, "missing_evidence", []) or [])],
        "soft_clues": [str(item) for item in (getattr(intent, "soft_clues", []) or [])],
        "consistency_check": consistency_check,
        "verification_rule": consistency_check,
        "retrieval_rationale": str(getattr(intent, "retrieval_rationale", "") or ""),
        "depends_on": [str(item) for item in (getattr(intent, "depends_on", []) or [])],
        "planner_phase": str(getattr(intent, "planner_phase", "uncertainty_aware_v1") or "uncertainty_aware_v1"),
    }


def _execute_search(
    *,
    query_text: str,
    query_unit: QueryUnit,
    provider: str,
    serper_key: str,
    tavily_key: str,
    topk: int,
    rank_offset: int,
) -> list[RetrievalCandidate]:
    if not provider:
        return _fallback_candidates(query_unit, query_text, topk, reason="no search API key")

    try:
        if provider == "serper":
            raw = _search_serper(query_text, api_key=serper_key, topk=topk)
        else:
            raw = _search_tavily(query_text, api_key=tavily_key, topk=topk)
    except Exception as exc:
        return _fallback_candidates(query_unit, query_text, topk, reason=str(exc))

    candidates: list[RetrievalCandidate] = []
    for i, item in enumerate(raw[:topk]):
        url = str(item.get("url") or "")
        title = str(item.get("title") or "")
        snippet = str(item.get("content") or item.get("snippet") or "")
        observation_label, observation_reason = _classify_observation(
            query_text=query_text,
            title=title,
            snippet=snippet,
        )
        candidates.append(RetrievalCandidate(
            candidate_id=f"web_{rank_offset + i:04d}",
            candidate_type="web_result",
            source_query_id=query_unit.query_id,
            source_ref=url or f"web_result_{rank_offset + i}",
            score=float(item.get("score") or 0.0),
            rank=rank_offset + i + 1,
            metadata={
                "title": title,
                "snippet": snippet[:400],
                "url": url,
                "query": query_text,
                "support_target": getattr(query_unit, "support_target", "final_answer"),
                "linked_candidate_id": getattr(query_unit, "linked_candidate_id", ""),
                "target_type": getattr(query_unit, "target_type", ""),
                "target_id": getattr(query_unit, "target_id", ""),
                "slot_id": getattr(query_unit, "slot_id", ""),
                "state_id": getattr(query_unit, "state_id", getattr(query_unit, "slot_id", "")),
                "slot_name": getattr(query_unit, "slot_name", ""),
                "expected_answer_type": getattr(query_unit, "expected_answer_type", ""),
                "current_uncertainty": getattr(query_unit, "current_uncertainty", []) or [],
                "unresolved_constraints": getattr(query_unit, "unresolved_constraints", getattr(query_unit, "hard_constraints", {})) or {},
                "hard_constraints": getattr(query_unit, "unresolved_constraints", getattr(query_unit, "hard_constraints", {})) or {},
                "candidate_conflicts": getattr(query_unit, "candidate_conflicts", []) or [],
                "missing_evidence": getattr(query_unit, "missing_evidence", []) or [],
                "soft_clues": getattr(query_unit, "soft_clues", []) or [],
                "consistency_check": getattr(query_unit, "consistency_check", getattr(query_unit, "verification_rule", "")),
                "verification_rule": getattr(query_unit, "consistency_check", getattr(query_unit, "verification_rule", "")),
                "retrieval_rationale": getattr(query_unit, "retrieval_rationale", ""),
                "depends_on": getattr(query_unit, "depends_on", []) or [],
                "planner_phase": getattr(query_unit, "planner_phase", "uncertainty_aware_v1"),
                "observation_label": observation_label,
                "observation_reason": observation_reason,
            },
            retriever_name=f"{provider}_web_retriever",
            normalized_score=max(0.0, 1.0 - 0.1 * i),
            keep_label="keep",
            keep_reason="Retrieved via retrieval intent",
        ))
    return candidates or _fallback_candidates(query_unit, query_text, topk, reason="empty results")


def _classify_observation(*, query_text: str, title: str, snippet: str) -> tuple[str, str]:
    """Lightweight abstract signal for Phase 2; no domain-specific branching."""
    query_tokens = _tokens(query_text)
    result_tokens = _tokens(f"{title} {snippet}")
    if not title and not snippet:
        return "empty_result", "No title or snippet returned."
    if not query_tokens or not result_tokens:
        return "partial_evidence", "Sparse query/result text."
    overlap = query_tokens & result_tokens
    overlap_ratio = len(overlap) / max(len(query_tokens), 1)
    if overlap_ratio >= 0.35:
        return "partial_evidence", "Result overlaps with the current retrieval need."
    if overlap_ratio > 0:
        return "low_relevance", "Result has weak overlap with the current retrieval need."
    return "mismatch", "Result has no lexical overlap with the current retrieval need."


def _tokens(text: str) -> set[str]:
    stop = {"the", "and", "for", "with", "from", "that", "this", "what", "which", "where", "when", "who"}
    return {
        token
        for token in (part.lower() for part in re.findall(r"[A-Za-z0-9]+", text))
        if len(token) > 2 and token not in stop
    }


def _new_memory(*, provider: str) -> dict[str, Any]:
    return {
        "schema_version": "retrieval_memory.v1",
        "provider": provider,
        "candidate_ledger": {},
        "failed_candidates": [],
        "failed_constraints": [],
        "missing_evidence": [],
        "retrieval_history": [],
        "verified_outputs": {},
        "rewrite_count": 0,
    }


def _query_motivation(intent: Any, round_index: int) -> str:
    uncertainty = "; ".join(str(item) for item in (getattr(intent, "current_uncertainty", []) or [])[:3])
    missing = "; ".join(str(item) for item in (getattr(intent, "missing_evidence", []) or [])[:3])
    if round_index == 0:
        return f"Reduce runtime uncertainty for {getattr(intent, 'intent_id', '')}: {uncertainty or missing}".strip()
    return f"Refine after low-value observation; missing evidence: {missing or uncertainty}".strip()


def _control_decision(candidates: list[RetrievalCandidate]) -> tuple[str, str]:
    if not candidates:
        return "rewrite", "empty_result"
    labels = [
        str((candidate.metadata or {}).get("observation_label") or "")
        for candidate in candidates
    ]
    has_real = any(candidate.retriever_name != "fallback_web_retriever" for candidate in candidates)
    if has_real and "partial_evidence" in labels:
        return "advance", "partial_evidence reduced uncertainty"
    if all(label == "empty_result" for label in labels):
        return "rewrite", "empty_result"
    if any(label in {"mismatch", "low_relevance"} for label in labels) and "partial_evidence" not in labels:
        return "rewrite", "low_relevance_or_mismatch"
    return ("advance", "usable observation") if has_real else ("rewrite", "fallback_only")


def _update_memory(
    *,
    memory: dict[str, Any],
    intent: Any,
    query_unit: QueryUnit,
    candidates: list[RetrievalCandidate],
    decision: str,
    decision_reason: str,
    round_index: int,
) -> None:
    labels = []
    candidate_ids = []
    for candidate in candidates:
        metadata = candidate.metadata or {}
        label = str(metadata.get("observation_label") or "")
        candidate_ids.append(candidate.candidate_id)
        labels.append(label)
        status = "partial_evidence" if label == "partial_evidence" else "failed"
        memory["candidate_ledger"][candidate.candidate_id] = {
            "status": status,
            "observation_label": label,
            "query_id": candidate.source_query_id,
            "query": metadata.get("query") or query_unit.query_text,
            "title": metadata.get("title") or "",
            "url": metadata.get("url") or candidate.source_ref,
            "slot_id": getattr(query_unit, "slot_id", ""),
            "intent_id": getattr(intent, "intent_id", ""),
        }
        if label in {"empty_result", "low_relevance", "mismatch"}:
            memory["failed_candidates"].append({
                "candidate_id": candidate.candidate_id,
                "reason": metadata.get("observation_reason") or label,
                "query_id": candidate.source_query_id,
                "label": label,
            })

    if decision == "rewrite":
        memory["rewrite_count"] = int(memory.get("rewrite_count") or 0) + 1
        for item in getattr(intent, "missing_evidence", []) or []:
            value = str(item)
            if value and value not in memory["missing_evidence"]:
                memory["missing_evidence"].append(value)

    memory["retrieval_history"].append({
        "step_id": f"retrieval_step_{len(memory['retrieval_history']):04d}",
        "intent_id": getattr(intent, "intent_id", ""),
        "slot_id": getattr(intent, "slot_id", ""),
        "state_id": getattr(intent, "state_id", getattr(intent, "slot_id", "")),
        "round_index": round_index,
        "query_id": query_unit.query_id,
        "query": query_unit.query_text,
        "candidate_ids": candidate_ids,
        "observation_labels": labels,
        "control_decision": decision,
        "decision_reason": decision_reason,
    })


def _intent_has_partial_evidence(memory: dict[str, Any], intent: Any) -> bool:
    intent_id = str(getattr(intent, "intent_id", "") or "")
    return any(
        isinstance(row, dict)
        and row.get("intent_id") == intent_id
        and row.get("control_decision") == "advance"
        for row in memory.get("retrieval_history", [])
    )


def _refine_query(*, intent: Any, failed_queries: list[str], memory: dict[str, Any]) -> str:
    terms: list[str] = []
    for value in getattr(intent, "missing_evidence", []) or []:
        terms.extend(sorted(_tokens(str(value))))
    for value in getattr(intent, "current_uncertainty", []) or []:
        terms.extend(sorted(_tokens(str(value))))
    constraints = getattr(intent, "unresolved_constraints", {}) or {}
    if isinstance(constraints, dict):
        for value in constraints.values():
            if isinstance(value, list):
                for item in value:
                    terms.extend(sorted(_tokens(str(item))))
            else:
                terms.extend(sorted(_tokens(str(value))))
    for value in getattr(intent, "soft_clues", []) or []:
        terms.extend(sorted(_tokens(str(value))))
    for value in memory.get("missing_evidence", []) or []:
        terms.extend(sorted(_tokens(str(value))))

    failed_tokens = set()
    for query in failed_queries:
        failed_tokens.update(_tokens(query))
    unique_terms = [
        term for term in dict.fromkeys(terms)
        if term not in failed_tokens or len(terms) <= 4
    ]
    if not unique_terms:
        unique_terms = list(dict.fromkeys(terms))
    return " ".join(unique_terms[:12])
