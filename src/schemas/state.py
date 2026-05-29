"""TemporalState schema for structured video-to-web grounding."""

from __future__ import annotations

from pydantic import Field

from .base import V0BaseModel


class TemporalState(V0BaseModel):
    state_id: str
    sub_question: str
    sub_answer: str
    temporal_span: list[float] = Field(default_factory=list)  # [start_sec, end_sec]
    required_predicates: list[str] = Field(default_factory=list)
    min_evidence_count: int = 1
    state_type: str = "video_anchored"  # video_anchored | web_only | cross_modal
    relation_to_other_states: dict[str, str] = Field(default_factory=dict)
    supporting_frame_paths: list[str] = Field(default_factory=list)
    visual_anchors: list[dict] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    confidence: float = 0.0
