"""Lightweight evidence store helpers for the compact core."""

from __future__ import annotations

from src.schemas import EvidenceStore


def build_evidence_store(
    *,
    store_id: str,
    observations: list[dict] | None = None,
    anchors: list[dict] | None = None,
    queries: list[dict] | None = None,
    retrieval_candidates: list[dict] | None = None,
    evidences: list[dict] | None = None,
    bindings: list[dict] | None = None,
    claims: list[dict] | None = None,
    open_questions: list[str] | None = None,
    entity_aliases: dict | None = None,
    step_summaries: list[str] | None = None,
) -> EvidenceStore:
    cleaned_open_questions = [item for item in (open_questions or []) if item]
    return EvidenceStore(
        store_id=store_id,
        observations=observations or [],
        anchors=anchors or [],
        queries=queries or [],
        retrieval_candidates=retrieval_candidates or [],
        evidences=evidences or [],
        bindings=bindings or [],
        claims=claims or [],
        open_questions=cleaned_open_questions,
        entity_aliases=entity_aliases or {},
        step_summaries=step_summaries or [],
    )
