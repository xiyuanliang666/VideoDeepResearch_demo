"""Parse <tool_call> blocks from assistant text."""

from __future__ import annotations

import re
from typing import List

import json5

from src.rsagent.types import ParsedToolCall


def extract_all_tag_blocks(text: str, tag: str) -> List[str]:
    pattern = re.compile(
        re.escape(f"<{tag}>") + r"(.*?)" + re.escape(f"</{tag}>"),
        re.DOTALL,
    )
    return [m.group(1).strip() for m in pattern.finditer(text)]


def parse_tool_calls_from_assistant(text: str) -> List[ParsedToolCall]:
    blocks = extract_all_tag_blocks(text, "tool_call")
    out: List[ParsedToolCall] = []
    for raw in blocks:
        try:
            data = json5.loads(raw)
            name = data.get("name")
            if not isinstance(name, str):
                name = None
            args = data.get("arguments")
            if not isinstance(args, dict):
                args = {}
            out.append(ParsedToolCall(raw_text=raw, name=name, arguments=args))
        except Exception as exc:
            out.append(ParsedToolCall(raw_text=raw, name=None, arguments={}, parse_error=f"{type(exc).__name__}: {exc}"))
    return out


def has_tool_calls(text: str) -> bool:
    return "<tool_call>" in text and "</tool_call>" in text
