"""Candidate hypothesis support/refute evaluation."""

from __future__ import annotations

import json
import os
from typing import Any


def evaluate_candidates(
    *,
    question: str,
    hypotheses: list[Any],
    evidences: list[Any],
    bindings: list[Any],
    model_profile: dict | None = None,
    max_tokens: int = 2500,
) -> list[Any]:
    """Score each hypothesis with support/refute evidence. Mutates hypotheses in-place and returns them."""
    if not hypotheses:
        return hypotheses

    # Build evidence lookup from bindings
    ev_by_id = {
        (e.evidence_id if hasattr(e, "evidence_id") else e.get("evidence_id", "")): e
        for e in evidences
    }
    for hyp in hypotheses:
        linked = getattr(hyp, "linked_state_ids", []) or []
        supporting = [
            b for b in bindings
            if _binding_anchor(b) in linked and _binding_relation(b) == "supports"
        ]
        refuting = [
            b for b in bindings
            if _binding_anchor(b) in linked and _binding_relation(b) == "refutes"
        ]
        hyp.supporting_evidence_ids = [_binding_ev_id(b) for b in supporting]
        hyp.refuting_evidence_ids = [_binding_ev_id(b) for b in refuting]
        hyp.support_score = min(1.0, 0.3 * len(supporting))
        hyp.refute_score = min(1.0, 0.3 * len(refuting))

    # One LLM call to finalize verdicts when API is available
    _llm_verdict(
        question=question,
        hypotheses=hypotheses,
        ev_by_id=ev_by_id,
        model_profile=model_profile,
        max_tokens=max_tokens,
    )
    return hypotheses


def _llm_verdict(
    *,
    question: str,
    hypotheses: list[Any],
    ev_by_id: dict,
    model_profile: dict | None,
    max_tokens: int,
) -> None:
    from src.tools.env_tools import resolve_openai_config
    from src.tools.model_router import get_model

    config = resolve_openai_config("candidate_evaluator")
    if not config.available:
        return

    try:
        from openai import OpenAI
    except ImportError:
        return

    hyp_text = "\n".join(
        f"- {h.hypothesis_id}: {h.explanation} "
        f"(support_ids={h.supporting_evidence_ids}, refute_ids={h.refuting_evidence_ids})"
        for h in hypotheses
    )
    user_text = (
        f"Question: {question}\n\nHypotheses:\n{hyp_text}\n\n"
        "For each hypothesis, assign status: supported | refuted | uncertain. "
        "Return JSON array: [{hypothesis_id, status, support_score (0-1), refute_score (0-1)}]"
    )

    model_name = get_model("candidate_evaluator", model_profile)
    timeout = float(os.getenv("TIMEOUT_SECONDS", "30") or 30)
    client = OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=timeout)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": user_text},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        raw = (response.choices[0].message.content or "").strip() if response.choices else ""
    except Exception:
        return

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        left, right = raw.find("["), raw.rfind("]")
        if left < 0 or right <= left:
            return
        try:
            data = json.loads(raw[left:right + 1])
        except json.JSONDecodeError:
            return

    if not isinstance(data, list):
        return

    hyp_by_id = {h.hypothesis_id: h for h in hypotheses}
    for item in data:
        if not isinstance(item, dict):
            continue
        hyp = hyp_by_id.get(str(item.get("hypothesis_id") or ""))
        if hyp is None:
            continue
        hyp.status = str(item.get("status") or "uncertain")
        try:
            hyp.support_score = max(0.0, min(1.0, float(item.get("support_score") or hyp.support_score)))
            hyp.refute_score = max(0.0, min(1.0, float(item.get("refute_score") or hyp.refute_score)))
        except (TypeError, ValueError):
            pass


def _binding_anchor(b: Any) -> str:
    return b.anchor_id if hasattr(b, "anchor_id") else b.get("anchor_id", "")


def _binding_relation(b: Any) -> str:
    return b.relation if hasattr(b, "relation") else b.get("relation", "")


def _binding_ev_id(b: Any) -> str:
    return b.evidence_id if hasattr(b, "evidence_id") else b.get("evidence_id", "")
