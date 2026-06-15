"""Shared datatypes for rsagent-v4."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParsedToolCall:
    """One <tool_call> block after extraction (may fail JSON parse)."""
    raw_text: str
    name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    parse_error: str | None = None


@dataclass
class ToolRun:
    """Single dispatched tool invocation and its raw string result."""
    name: str
    arguments: dict[str, Any]
    raw_output: str
    parse_error: str | None = None


@dataclass
class SubQuestion:
    """A decomposed sub-question with evidence tracking."""
    text: str
    visual_evidence_plan: str = ""
    web_evidence_plan: str = ""
    visual_evidence: list[dict[str, Any]] = field(default_factory=list)
    web_evidence: list[dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    sufficient: bool = False


@dataclass
class CoordinatorAction:
    """Parsed action from the coordinator LLM."""
    action: str  # "work_on" | "adjust_plan" | "all_resolved"
    sq_id: int | None = None
    directive: str = ""
    tasks: list[dict[str, Any]] = field(default_factory=list)  # parallel tasks
    add_sub_questions: list[dict[str, Any]] = field(default_factory=list)
    merge_ids: list[int] = field(default_factory=list)


@dataclass
class RelevanceVerdict:
    """LLM judgment on whether a tool result is relevant."""
    tool_run_index: int
    keep: bool
    reason: str = ""
