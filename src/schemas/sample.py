"""Benchmark-agnostic sample schema."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from .base import V0BaseModel


class Sample(V0BaseModel):
    sample_id: str
    benchmark_name: str
    input_type: str
    media_paths: list[str] = Field(default_factory=list)
    question: str
    reference_answer: str = ""
    output_mode: str = "short_answer"
    metadata: dict[str, Any] = Field(default_factory=dict)
