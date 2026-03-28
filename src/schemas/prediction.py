"""Unified prediction schema."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from .base import V0BaseModel


class Prediction(V0BaseModel):
    sample_id: str
    output_mode: str = "short_answer"
    answer: str = ""
    report: str = ""
    evidence_chain: list[dict[str, Any]] = Field(default_factory=list)
    judge_result: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
