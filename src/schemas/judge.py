"""Judge schemas."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from .base import V0BaseModel


class JudgeInputBundle(V0BaseModel):
    judge_input_id: str
    task_input: dict[str, Any]
    model_output: dict[str, Any]
    reference_answer: str = ""
    reference_materials: list[dict[str, Any]] = Field(default_factory=list)
    reasoning_trace: list[dict[str, Any]] = Field(default_factory=list)
    evidence_chain: list[dict[str, Any]] = Field(default_factory=list)
    tool_trace: list[dict[str, Any]] = Field(default_factory=list)
    rubric: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class JudgeResult(V0BaseModel):
    judge_result_id: str
    judge_model: str
    judge_prompt_version: str
    task_type: str
    overall_score: float = 0.0
    dimension_scores: dict[str, float] = Field(default_factory=dict)
    verdict: str = ""
    explanation: str = ""
    failure_tags: list[str] = Field(default_factory=list)
    raw_judge_output: str = ""
