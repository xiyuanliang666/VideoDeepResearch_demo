"""Evidence and memory schemas."""

from __future__ import annotations

from typing import Optional

from pydantic import Field

from .base import V0BaseModel


class Evidence(V0BaseModel):
    evidence_id: str
    evidence_type: str
    source_ref: str
    content_summary: str
    span: Optional[list[float]] = None
    confidence: float = 0.0
    source_url: str = ""
    source_clip_id: str = ""
    source_timestamp: Optional[float] = None
    raw_excerpt: str = ""
    embedding_ref: str = ""


class Binding(V0BaseModel):
    binding_id: str
    anchor_id: str
    evidence_id: str
    relation: str
    matched_entities: list[str] = Field(default_factory=list)
    matched_actions: list[str] = Field(default_factory=list)
    matched_temporal_clues: list[str] = Field(default_factory=list)
    reason: str = ""
    confidence: float = 0.0
    judgeable_claim: str = ""


class Claim(V0BaseModel):
    claim_id: str
    statement: str
    supporting_bindings: list[str] = Field(default_factory=list)
    status: str = "tentative"
    confidence: float = 0.0
    last_updated_step: int = 0
    conflict_evidence_ids: list[str] = Field(default_factory=list)


class EvidenceStore(V0BaseModel):
    store_id: str
    anchors: list[dict] = Field(default_factory=list)
    observations: list[dict] = Field(default_factory=list)
    queries: list[dict] = Field(default_factory=list)
    retrieval_candidates: list[dict] = Field(default_factory=list)
    evidences: list[dict] = Field(default_factory=list)
    bindings: list[dict] = Field(default_factory=list)
    claims: list[dict] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    entity_aliases: dict = Field(default_factory=dict)
    step_summaries: list[str] = Field(default_factory=list)
