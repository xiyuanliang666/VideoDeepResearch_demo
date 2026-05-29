"""Rule-based coverage checks for hybrid state/evidence runs."""

from __future__ import annotations

from typing import Any

# General evidence-type taxonomy shared across multi-hop QA tasks.
# These are not tied to any specific benchmark schema.
SUPPORT_TARGETS = {
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


def check_state_frame_coverage(
    *,
    states: list[Any],
    state_frames: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Check whether proposed states have time spans, predicates, and frames."""
    missing_supporting_frames: list[str] = []
    missing_predicates: list[str] = []
    missing_temporal_span: list[str] = []
    covered_state_ids: list[str] = []

    for state in states:
        state_id = _get(state, "state_id")
        if not state_id:
            continue
        frames = state_frames.get(state_id) or _get(state, "supporting_frame_paths", default=[])
        predicates = _get(state, "required_predicates", default=[])
        span = _get(state, "temporal_span", default=[])
        state_type = str(_get(state, "state_type", default="video_anchored") or "video_anchored")
        if frames:
            covered_state_ids.append(state_id)
        else:
            missing_supporting_frames.append(state_id)
        if not predicates:
            missing_predicates.append(state_id)
        if state_type != "web_only" and (not isinstance(span, list) or len(span) < 2):
            missing_temporal_span.append(state_id)

    total = len([s for s in states if _get(s, "state_id")])
    return {
        "stage": "state_frame_coverage",
        "state_count": total,
        "covered_state_ids": covered_state_ids,
        "missing_supporting_frames": missing_supporting_frames,
        "missing_predicates": missing_predicates,
        "missing_temporal_span": missing_temporal_span,
        "state_coverage_ratio": (len(covered_state_ids) / total) if total else 0.0,
        "status": (
            "pass"
            if total and not missing_supporting_frames and not missing_predicates and not missing_temporal_span
            else "needs_attention"
        ),
    }


def check_evidence_coverage(
    *,
    states: list[Any],
    intents: list[Any],
    evidences: list[Any],
    bindings: list[Any],
    required_support_targets: list[str] | None = None,
) -> dict[str, Any]:
    """Check support-target and state binding coverage after retrieval/binding."""
    state_ids = {_get(state, "state_id") for state in states if _get(state, "state_id")}
    intent_targets = {
        _normalize_support_target(_get(intent, "support_target"))
        for intent in intents
        if _normalize_support_target(_get(intent, "support_target"))
    }
    real_evidences = [
        evidence for evidence in evidences
        if not bool((_get(evidence, "metadata", default={}) or {}).get("is_fallback"))
    ]
    evidence_targets = {
        _normalize_support_target((_get(evidence, "metadata", default={}) or {}).get("support_target"))
        for evidence in real_evidences
        if _normalize_support_target((_get(evidence, "metadata", default={}) or {}).get("support_target"))
    }
    required_targets = {
        _normalize_support_target(target)
        for target in (required_support_targets or [])
        if _normalize_support_target(target)
    }
    if not required_targets:
        required_targets = intent_targets or {"final_answer"}

    real_evidence_ids = {_get(evidence, "evidence_id") for evidence in real_evidences}
    bound_state_ids = {
        _get(binding, "anchor_id")
        for binding in bindings
        if _get(binding, "anchor_id") in state_ids
        and _get(binding, "evidence_id") in real_evidence_ids
    }
    missing_state_bindings = sorted(state_ids - bound_state_ids) if real_evidences else sorted(state_ids)
    missing_support_targets = sorted(required_targets - evidence_targets)
    unused_intent_targets = sorted(intent_targets - evidence_targets)

    return {
        "stage": "evidence_coverage",
        "required_support_targets": sorted(required_targets),
        "planned_support_targets": sorted(intent_targets),
        "covered_support_targets": sorted(evidence_targets),
        "missing_support_targets": missing_support_targets,
        "unused_intent_targets": unused_intent_targets,
        "bound_state_ids": sorted(bound_state_ids),
        "missing_state_bindings": missing_state_bindings,
        "binding_count": len(bindings),
        "evidence_count": len(evidences),
        "real_evidence_count": len(real_evidences),
        "fallback_evidence_count": len(evidences) - len(real_evidences),
        "status": "pass" if not missing_support_targets and not missing_state_bindings else "needs_attention",
    }


def merge_coverage_reports(*reports: dict[str, Any]) -> dict[str, Any]:
    """Return a compact aggregate coverage report for diagnostics."""
    valid_reports = [report for report in reports if isinstance(report, dict) and report]
    return {
        "status": "pass" if valid_reports and all(r.get("status") == "pass" for r in valid_reports) else "needs_attention",
        "reports": valid_reports,
        "missing_support_targets": sorted({
            target
            for report in valid_reports
            for target in report.get("missing_support_targets", [])
        }),
        "missing_state_ids": sorted({
            state_id
            for report in valid_reports
            for key in ("missing_supporting_frames", "missing_state_bindings", "missing_temporal_span")
            for state_id in report.get(key, [])
        }),
    }


def _normalize_support_target(value: Any) -> str:
    text = str(value or "")
    return text if text in SUPPORT_TARGETS else ""


def _get(obj: Any, key: str, default: Any = "") -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
