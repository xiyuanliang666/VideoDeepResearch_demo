"""Output schemas."""

from __future__ import annotations

from pydantic import Field

from .base import V0BaseModel


class ToolCallRecord(V0BaseModel):
    tool_call_id: str
    tool_name: str
    tool_input: dict | str
    tool_output_ref: str = ""
    start_time: str = ""
    end_time: str = ""
    status: str = "pending"
    error_message: str = ""


class ReasoningTrace(V0BaseModel):
    trace_id: str
    step_index: int
    active_anchor_ids: list[str] = Field(default_factory=list)
    selected_queries: list[str] = Field(default_factory=list)
    retrieval_summary: str = ""
    binding_summary: str = ""
    claim_updates: list[str] = Field(default_factory=list)
    notes: str = ""


class FinalAnswerBundle(V0BaseModel):
    task_id: str
    question: str
    final_answer: str
    confidence: float = 0.0
    supporting_video_evidence: list[str] = Field(default_factory=list)
    supporting_web_evidence: list[str] = Field(default_factory=list)
    tool_trace_refs: list[str] = Field(default_factory=list)
    reasoning_trace_refs: list[str] = Field(default_factory=list)
