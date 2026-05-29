"""Unified VideoDeepResearch runner.

This module is the integration layer between the existing method-first
implementation and future agentic/tool-rich implementations. The public
contract is intentionally small:

    run(sample, task_profile, model_profile, mode="workflow")

The returned object has one normalized structure for all modes and keeps the
raw trace under ``raw_trace`` for debugging and migration.
"""

from __future__ import annotations

from typing import Any

from src.core.pipeline import run_pipeline
from src.videodeepresearch.agentic.runner import run_unified_agentic_pipeline
from src.schemas import Sample, utc_now_iso

SUPPORTED_MODES = {"workflow", "agentic"}
UNIFIED_RUN_SCHEMA_VERSION = "videodeepresearch.run.v1"


def normalize_mode(mode: str | None, task_profile: dict | None = None) -> str:
    """Resolve the execution mode from CLI/config with a conservative default."""
    requested = str(mode or "").strip().lower()
    if requested in SUPPORTED_MODES:
        return requested
    configured = str((task_profile or {}).get("mode", "")).strip().lower()
    if configured in SUPPORTED_MODES:
        return configured
    return "workflow"


def run_raw(
    *,
    sample: Sample,
    task_profile: dict,
    model_profile: dict,
    mode: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Run one sample and return the legacy/raw trace plus resolved mode."""
    execution_mode = normalize_mode(mode, task_profile)
    pipeline_fn = run_unified_agentic_pipeline if execution_mode == "agentic" else run_pipeline
    return pipeline_fn(
        sample=sample,
        task_profile=task_profile,
        model_profile=model_profile,
    ), execution_mode


def run(
    *,
    sample: Sample,
    task_profile: dict,
    model_profile: dict,
    mode: str | None = None,
    include_raw_trace: bool = True,
) -> dict[str, Any]:
    """Run one sample and return the unified framework output."""
    raw_trace, execution_mode = run_raw(
        sample=sample,
        task_profile=task_profile,
        model_profile=model_profile,
        mode=mode,
    )
    return normalize_raw_trace(
        raw_trace=raw_trace,
        execution_mode=execution_mode,
        include_raw_trace=include_raw_trace,
    )


def normalize_raw_trace(
    *,
    raw_trace: dict[str, Any],
    execution_mode: str,
    include_raw_trace: bool = True,
) -> dict[str, Any]:
    """Convert a legacy/raw trace into the stable framework output."""
    final_answer = _normalize_final_answer(raw_trace.get("final_answer") or {})
    out = {
        "schema_version": UNIFIED_RUN_SCHEMA_VERSION,
        "created_at": raw_trace.get("created_at") or utc_now_iso(),
        "mode": execution_mode,
        "status": raw_trace.get("status", ""),
        "sample": raw_trace.get("sample") or {},
        "question": raw_trace.get("question", ""),
        "final_answer": final_answer,
        "used_video_evidence": _normalize_video_evidence(raw_trace),
        "retrieved_web_results": _normalize_retrieved_web_results(raw_trace),
        "used_web_evidence": _normalize_web_evidence(raw_trace),
        "evidence_links": _normalize_evidence_links(raw_trace),
        "dependency_graph": _normalize_dependency_graph(raw_trace),
        "tool_trace": raw_trace.get("tool_trace") or [],
        "judge_result": raw_trace.get("judge_result"),
        "diagnostics": _build_diagnostics(raw_trace),
    }
    if include_raw_trace:
        out["raw_trace"] = raw_trace
    return out


def _normalize_final_answer(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        payload = {}
    answer_text = (
        payload.get("answer_text")
        or payload.get("final_answer")
        or payload.get("answer")
        or ""
    )
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "answer_text": str(answer_text),
        "answer_type": str(payload.get("answer_type") or ""),
        "confidence": max(0.0, min(1.0, confidence)),
        "supporting_video_evidence": _list_of_strings(payload.get("supporting_video_evidence")),
        "supporting_web_evidence": _list_of_strings(payload.get("supporting_web_evidence")),
    }


def _normalize_video_evidence(raw_trace: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # Hybrid path: project each state-supporting frame set into one interval evidence unit.
    states = _as_list(raw_trace.get("states"))
    if states:
        ev_index = 0
        for state in states:
            if not isinstance(state, dict):
                continue
            state_id = str(state.get("state_id") or "")
            span = state.get("temporal_span") or []
            frames = _list_of_strings(state.get("supporting_frame_paths"))
            predicates = _list_of_strings(state.get("required_predicates"))
            visual_anchors = _as_list(state.get("visual_anchors"))
            anchor_text = "; ".join(
                str(anchor.get("description") or anchor.get("text") or anchor)
                for anchor in visual_anchors[:4]
            )
            description_parts = [
                str(state.get("sub_question") or "").strip(),
                str(state.get("sub_answer") or "").strip(),
                "Predicates: " + "; ".join(predicates[:8]) if predicates else "",
                "Visual anchors: " + anchor_text if anchor_text else "",
            ]
            start = _coerce_float(span[0] if len(span) >= 1 else None, 0.0)
            end = _coerce_float(span[1] if len(span) >= 2 else None, start)
            evidence_suffix = state_id if state_id else f"{ev_index:04d}"
            rows.append({
                "evidence_id": f"video_state_{evidence_suffix}",
                "source_type": "video",
                "start_sec": start,
                "end_sec": max(end, start),
                "description": " ".join(part for part in description_parts if part),
                "frame_paths": frames,
                "linked_state_ids": [state_id] if state_id else [],
                "confidence": _coerce_float(state.get("confidence"), 0.3),
            })
            ev_index += 1
        if rows:
            return rows

    # Legacy path: from observations
    for index, observation in enumerate(_as_list(raw_trace.get("observations"))):
        if not isinstance(observation, dict):
            continue
        start = _coerce_float(observation.get("timestamp_start"), 0.0)
        end = _coerce_float(observation.get("timestamp_end"), start)
        rows.append({
            "evidence_id": str(observation.get("observation_id") or f"video_obs_{index:04d}"),
            "source_type": str(observation.get("source_type") or "video"),
            "source_path": str(observation.get("source_path") or ""),
            "start_sec": start,
            "end_sec": end,
            "description": _observation_description(observation),
            "frame_paths": _list_of_strings(observation.get("frame_paths")),
            "transcript_text": str(observation.get("transcript_text") or ""),
            "metadata": observation.get("metadata") if isinstance(observation.get("metadata"), dict) else {},
            "confidence": _coerce_float(observation.get("confidence"), 0.0),
        })
    return rows


def _normalize_retrieved_web_results(raw_trace: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    query_by_id = {
        str(query.get("query_id")): query
        for query in _as_list(raw_trace.get("web_queries"))
        if isinstance(query, dict)
    }
    for index, candidate in enumerate(_as_list(raw_trace.get("web_candidates"))):
        if not isinstance(candidate, dict):
            continue
        metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
        query = query_by_id.get(str(candidate.get("source_query_id")), {})
        rows.append(
            {
                "rank": int(candidate.get("rank") or index + 1),
                "query": str(query.get("query_text") or metadata.get("query") or ""),
                "hop_id": str(candidate.get("source_query_id") or ""),
                "url": str(metadata.get("url") or candidate.get("source_ref") or ""),
                "title": str(metadata.get("title") or ""),
                "snippet": str(metadata.get("snippet") or metadata.get("content") or ""),
                "support_target": str(metadata.get("support_target") or query.get("support_target") or ""),
                "linked_state_ids": [str(metadata.get("target_id") or query.get("target_id") or "")]
                if str(metadata.get("target_type") or query.get("target_type") or "") == "state"
                and str(metadata.get("target_id") or query.get("target_id") or "")
                else [],
                "linked_candidate_id": str(metadata.get("linked_candidate_id") or query.get("linked_candidate_id") or ""),
                "score": _coerce_float(candidate.get("normalized_score") or candidate.get("score"), 0.0),
                "metadata": metadata,
            }
        )
    return rows


def _normalize_web_evidence(raw_trace: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, evidence in enumerate(_as_list(raw_trace.get("evidences"))):
        if not isinstance(evidence, dict):
            continue
        meta = evidence.get("metadata") if isinstance(evidence.get("metadata"), dict) else {}
        if bool(meta.get("is_fallback")) or str(evidence.get("source_ref") or "").startswith("fallback://"):
            continue
        support_target = str(meta.get("support_target") or "final_answer")
        bound_to_id = str(meta.get("bound_to_id") or "")
        bound_to_type = str(meta.get("bound_to_type") or "")
        linked_state_ids = [bound_to_id] if bound_to_id and bound_to_type in {"state", ""} else []
        linked_candidate_id = str(meta.get("linked_candidate_id") or "")
        if not linked_candidate_id and bound_to_type == "hypothesis":
            linked_candidate_id = bound_to_id
        rows.append({
            "evidence_id": str(evidence.get("evidence_id") or f"web_ev_{index:04d}"),
            "url": str(evidence.get("source_url") or evidence.get("source_ref") or ""),
            "title": str(evidence.get("title") or evidence.get("content_summary") or "")[:200],
            "evidence_snippet": str(evidence.get("raw_excerpt") or evidence.get("content_summary") or ""),
            "support_target": support_target,
            "linked_candidate_id": linked_candidate_id or None,
            "linked_state_ids": linked_state_ids,
            "confidence": _coerce_float(evidence.get("confidence"), 0.0),
        })
    return rows


def _normalize_evidence_links(raw_trace: dict[str, Any]) -> list[dict[str, Any]]:
    links: list[dict[str, Any]] = []
    for binding in _as_list(raw_trace.get("bindings")):
        if not isinstance(binding, dict):
            continue
        evidence_id = str(binding.get("evidence_id") or "")
        anchor_id = str(binding.get("anchor_id") or "")
        if not evidence_id or not anchor_id:
            continue
        states = _as_list(raw_trace.get("states"))
        state_ids = {str(s.get("state_id")) for s in states if isinstance(s, dict)}
        hypotheses = _as_list(raw_trace.get("hypotheses"))
        hypothesis_ids = {str(h.get("hypothesis_id")) for h in hypotheses if isinstance(h, dict)}
        if anchor_id in state_ids:
            target_type = "state"
        elif anchor_id in hypothesis_ids:
            target_type = "candidate"
        elif anchor_id == "final_answer":
            target_type = "final_answer"
        else:
            target_type = "web_evidence"
        links.append({
            "source_type": "web_evidence",
            "source_id": evidence_id,
            "target_type": target_type,
            "target_id": anchor_id,
            "relation": str(binding.get("relation") or "supports"),
        })
    return links


def _normalize_dependency_graph(raw_trace: dict[str, Any]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    # Phase B: state → intent → candidate chain
    for slot in _as_list(raw_trace.get("retrieval_slots")):
        if not isinstance(slot, dict):
            continue
        slot_id = str(slot.get("slot_id") or "")
        if not slot_id:
            continue
        for state_id in _as_list(slot.get("source_state_ids")):
            if state_id:
                edges.append({"from": str(state_id), "to": slot_id, "relation": "opens_retrieval_slot"})
        for dep_id in _as_list(slot.get("depends_on")):
            if dep_id:
                edges.append({"from": str(dep_id), "to": slot_id, "relation": "slot_dependency"})
    for intent in _as_list(raw_trace.get("intents")):
        if not isinstance(intent, dict):
            continue
        slot_id = str(intent.get("slot_id") or "")
        target_id = str(intent.get("target_id") or "")
        intent_id = str(intent.get("intent_id") or "")
        if slot_id and intent_id:
            edges.append({"from": slot_id, "to": intent_id, "relation": "planned_intent"})
        if target_id and intent_id:
            edges.append({"from": target_id, "to": intent_id, "relation": "planned_intent"})
        for cand_id in _as_list(intent.get("retrieved_result_ids")):
            if intent_id and cand_id:
                edges.append({"from": intent_id, "to": str(cand_id), "relation": "retrieved"})

    # Legacy: anchor → query → candidate chain
    for query in _as_list(raw_trace.get("web_queries")):
        if not isinstance(query, dict):
            continue
        anchor_id = str(query.get("anchor_id") or "")
        query_id = str(query.get("query_id") or "")
        if anchor_id and query_id:
            edges.append({"from": anchor_id, "to": query_id, "relation": "planned_query"})
    for candidate in _as_list(raw_trace.get("web_candidates")):
        if not isinstance(candidate, dict):
            continue
        query_id = str(candidate.get("source_query_id") or "")
        candidate_id = str(candidate.get("candidate_id") or "")
        if query_id and candidate_id:
            edges.append({"from": query_id, "to": candidate_id, "relation": "retrieved"})
    for binding in _as_list(raw_trace.get("bindings")):
        if not isinstance(binding, dict):
            continue
        evidence_id = str(binding.get("evidence_id") or "")
        anchor_id = str(binding.get("anchor_id") or "")
        if evidence_id and anchor_id:
            edges.append({"from": evidence_id, "to": anchor_id, "relation": str(binding.get("relation") or "binds")})
    return edges


def _build_diagnostics(raw_trace: dict[str, Any]) -> dict[str, Any]:
    observations = _as_list(raw_trace.get("observations"))
    web_candidates = _as_list(raw_trace.get("web_candidates"))
    web_queries = _as_list(raw_trace.get("web_queries"))
    fallback_candidates = [
        candidate for candidate in web_candidates
        if isinstance(candidate, dict)
        and (
            str(candidate.get("retriever_name") or "") == "fallback_web_retriever"
            or str(candidate.get("source_ref") or "").startswith("fallback://")
        )
    ]
    degenerate_queries = [
        query for query in web_queries
        if isinstance(query, dict)
        and _looks_like_full_question(
            str(query.get("query_text") or ""),
            str(raw_trace.get("question") or ""),
        )
    ]
    observation_errors: list[str] = []
    for observation in observations:
        if not isinstance(observation, dict):
            continue
        observation_errors.extend(str(item) for item in _as_list(observation.get("uncertainty_notes")))
    visual_anchor_empty_count = sum(
        1 for state in _as_list(raw_trace.get("states"))
        if isinstance(state, dict) and not _as_list(state.get("visual_anchors"))
    )
    retrieval_slots = _as_list(raw_trace.get("retrieval_slots"))
    retrieval_memory = raw_trace.get("retrieval_memory") if isinstance(raw_trace.get("retrieval_memory"), dict) else {}
    retrieval_history = _as_list(raw_trace.get("retrieval_history") or retrieval_memory.get("retrieval_history"))
    slot_ids_in_queries = {
        str(query.get("slot_id") or "")
        for query in web_queries
        if isinstance(query, dict) and str(query.get("slot_id") or "")
    }
    observation_label_counts: dict[str, int] = {}
    for candidate in web_candidates:
        if not isinstance(candidate, dict):
            continue
        metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
        label = str(metadata.get("observation_label") or "unknown")
        observation_label_counts[label] = observation_label_counts.get(label, 0) + 1
    return {
        "clip_count": len(_as_list(raw_trace.get("clips"))),
        "observation_count": len(observations),
        "anchor_count": len(_as_list(raw_trace.get("anchors"))),
        "retrieval_slot_count": len(retrieval_slots),
        "web_queries_with_slot_count": len(slot_ids_in_queries),
        "retrieval_slot_names": [
            str(slot.get("slot_name") or slot.get("slot_id") or "")
            for slot in retrieval_slots[:8]
            if isinstance(slot, dict)
        ],
        "web_query_count": len(web_queries),
        "web_candidate_count": len(web_candidates),
        "fallback_web_candidate_count": len(fallback_candidates),
        "real_web_candidate_count": len(web_candidates) - len(fallback_candidates),
        "retrieval_history_count": len(retrieval_history),
        "retrieval_rewrite_count": int(retrieval_memory.get("rewrite_count") or 0) if isinstance(retrieval_memory, dict) else 0,
        "failed_candidate_count": len(_as_list(retrieval_memory.get("failed_candidates"))) if isinstance(retrieval_memory, dict) else 0,
        "candidate_ledger_count": len(retrieval_memory.get("candidate_ledger") or {}) if isinstance(retrieval_memory, dict) else 0,
        "observation_label_counts": observation_label_counts,
        "web_evidence_count": len(_as_list(raw_trace.get("evidences"))),
        "binding_count": len(_as_list(raw_trace.get("bindings"))),
        "visual_anchor_empty_count": visual_anchor_empty_count,
        "degenerate_query_count": len(degenerate_queries),
        "blocking_errors": _blocking_errors(
            fallback_candidates=fallback_candidates,
            observation_errors=observation_errors,
            degenerate_queries=degenerate_queries,
        ),
        "observation_error_samples": observation_errors[:8],
        "coverage_report": raw_trace.get("coverage_report") if isinstance(raw_trace.get("coverage_report"), dict) else {},
        "has_agent_trace": bool(raw_trace.get("agent_trace")),
        "planner_errors": raw_trace.get("planner_errors") or [],
    }


def _observation_description(observation: dict[str, Any]) -> str:
    for key in ("scene_summary", "summary", "content_summary", "description"):
        value = str(observation.get(key) or "").strip()
        if value:
            return value
    scene_clues = observation.get("scene_clues")
    if isinstance(scene_clues, list) and scene_clues:
        return " ".join(str(item) for item in scene_clues[:8])
    transcript = str(observation.get("transcript_text") or "").strip()
    if transcript:
        return transcript[:300]
    return ""


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _list_of_strings(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _looks_like_full_question(query: str, question: str) -> bool:
    import re

    query_norm = re.sub(r"\s+", " ", query).strip().lower()
    question_norm = re.sub(r"\s+", " ", question).strip().lower()
    if not query_norm or not question_norm:
        return False
    if query_norm == question_norm:
        return True
    q_tokens = set(re.findall(r"[a-z0-9]+", query_norm))
    question_tokens = set(re.findall(r"[a-z0-9]+", question_norm))
    return len(q_tokens) >= 18 and len(q_tokens & question_tokens) / max(len(q_tokens), 1) > 0.82


def _blocking_errors(
    *,
    fallback_candidates: list[Any],
    observation_errors: list[str],
    degenerate_queries: list[Any],
) -> list[str]:
    errors: list[str] = []
    fallback_reasons = []
    for candidate in fallback_candidates:
        if not isinstance(candidate, dict):
            continue
        metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
        reason = str(metadata.get("fallback_reason") or "").strip()
        if reason and reason not in fallback_reasons:
            fallback_reasons.append(reason)
    for reason in fallback_reasons[:3]:
        errors.append(f"web_retrieval_fallback:{reason}")
    if any("event_observer_error" in item for item in observation_errors):
        errors.append("event_observer_failed")
    if any("paddleocr" in item for item in observation_errors):
        errors.append("paddleocr_failed")
    if any("yolo" in item for item in observation_errors):
        errors.append("yolo_failed")
    if degenerate_queries:
        errors.append("query_planner_degenerated_to_full_question")
    return errors
