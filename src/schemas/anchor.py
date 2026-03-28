"""Anchor schemas."""

from __future__ import annotations

from typing import Optional

from pydantic import Field

from .base import V0BaseModel


class Anchor(V0BaseModel):
    anchor_id: str
    source_clip_ids: list[str] = Field(default_factory=list)
    source_observation_ids: list[str] = Field(default_factory=list)
    time_span: Optional[list[float]] = None
    anchor_type: str = "event"
    entities: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    scene_summary: str = ""
    ocr_clues: list[str] = Field(default_factory=list)
    subtitle_clues: list[str] = Field(default_factory=list)
    speech_clues: list[str] = Field(default_factory=list)
    temporal_clues: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    status: str = "tentative"
    priority_score: float = 0.0
    evidence_ids: list[str] = Field(default_factory=list)
    open_slots: list[str] = Field(default_factory=list)
