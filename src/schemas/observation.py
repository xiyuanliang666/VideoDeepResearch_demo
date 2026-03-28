"""Observation-related schemas."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import Field

from .base import V0BaseModel


class ObservationUnit(V0BaseModel):
    observation_id: str
    source_type: str
    source_path: str
    order_index: int = 0
    timestamp_start: Optional[float] = None
    timestamp_end: Optional[float] = None
    frame_paths: list[str] = Field(default_factory=list)
    subtitle_text: str = ""
    transcript_text: str = ""
    ocr_text: str = ""
    scene_clues: list[str] = Field(default_factory=list)
    speech_clues: list[str] = Field(default_factory=list)
    candidate_entities: list[str] = Field(default_factory=list)
    candidate_actions: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    uncertainty_notes: list[str] = Field(default_factory=list)
    audio_path: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    quality_flags: list[str] = Field(default_factory=list)
    embedding_refs: list[str] = Field(default_factory=list)


class ClipUnit(V0BaseModel):
    clip_id: str
    video_name: str
    clip_path: str
    start_time: float
    end_time: float
    start_time_str: str = ""
    end_time_str: str = ""
    frame_dir: str = ""
    frame_count: int = 0
    fps: float = 0.0
    audio_segment_path: str = ""
    subtitle_segment: str = ""
    transcript_segment: str = ""
    embedding_ref: str = ""
    quality_flags: list[str] = Field(default_factory=list)
