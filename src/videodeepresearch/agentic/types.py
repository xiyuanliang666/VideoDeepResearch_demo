"""Datatypes for agentic tool-call execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParsedToolCall:
    """One parsed `<tool_call>` block."""

    raw_text: str
    name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    parse_error: str | None = None


@dataclass
class ToolRun:
    """One dispatched tool invocation."""

    name: str
    arguments: dict[str, Any]
    raw_output: str
    parse_error: str | None = None
