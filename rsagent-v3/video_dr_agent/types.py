"""Shared datatypes for the video deep-research loop."""

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
