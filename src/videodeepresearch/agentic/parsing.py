"""Parse `<tool_call>` blocks emitted by agentic research models."""

from __future__ import annotations

import json
import re
from typing import Any

from .types import ParsedToolCall


def extract_tag_block(text: str, tag: str) -> str | None:
    """Return the first `<tag>...</tag>` block body."""
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    if start_tag in text and end_tag in text:
        return text.split(start_tag, 1)[1].split(end_tag, 1)[0].strip()
    return None


def extract_all_tag_blocks(text: str, tag: str) -> list[str]:
    """Return non-overlapping `<tag>...</tag>` block bodies."""
    pattern = re.compile(
        re.escape(f"<{tag}>") + r"(.*?)" + re.escape(f"</{tag}>"),
        re.DOTALL,
    )
    return [match.group(1).strip() for match in pattern.finditer(text)]


def parse_tool_calls_from_assistant(text: str) -> list[ParsedToolCall]:
    """Parse all VDR-style tool calls from assistant text."""
    out: list[ParsedToolCall] = []
    for raw in extract_all_tag_blocks(text, "tool_call"):
        try:
            data = _loads_tool_json(raw)
            name = data.get("name")
            if not isinstance(name, str):
                name = None
            arguments = data.get("arguments")
            if not isinstance(arguments, dict):
                arguments = {}
            out.append(ParsedToolCall(raw_text=raw, name=name, arguments=arguments))
        except Exception as exc:  # noqa: BLE001
            out.append(
                ParsedToolCall(
                    raw_text=raw,
                    name=None,
                    arguments={},
                    parse_error=f"{type(exc).__name__}: {exc}",
                )
            )
    return out


def has_tool_calls(text: str) -> bool:
    """Return whether text contains at least one complete tool call tag."""
    return "<tool_call>" in text and "</tool_call>" in text


def _loads_tool_json(raw: str) -> dict[str, Any]:
    """Load strict JSON first, then json5 when available."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        try:
            import json5
        except ImportError:
            raise
        payload = json5.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("tool_call body must decode to a JSON object")
    return payload
