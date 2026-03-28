"""Retrieval schemas."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from .base import V0BaseModel


class QueryUnit(V0BaseModel):
    query_id: str
    anchor_id: str
    query_text: str
    query_type: str = "entity_focused"
    step_index: int = 0
    motivation: str = ""
    query_source: str = ""
    query_embedding_ref: str = ""
    rewrite_history: list[str] = Field(default_factory=list)


class RetrievalCandidate(V0BaseModel):
    candidate_id: str
    candidate_type: str
    source_query_id: str
    source_ref: str
    score: float = 0.0
    rank: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    retriever_name: str = ""
    normalized_score: float = 0.0
    keep_label: str = "unknown"
    keep_reason: str = ""
