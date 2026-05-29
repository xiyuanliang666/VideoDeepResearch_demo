"""Execute parsed agentic tool calls through the unified dispatcher."""

from __future__ import annotations

from typing import Any

from src.videodeepresearch.tools import ToolDispatcher

from .parsing import parse_tool_calls_from_assistant
from .types import ToolRun


def run_tool_round(
    *,
    assistant_text: str,
    dispatcher: ToolDispatcher,
    round_index: int,
) -> list[ToolRun]:
    """Parse assistant tool calls and dispatch them.

    Raw tool outputs are returned to the caller. The caller decides whether a
    critic summarizes them, whether selected images are injected, and whether
    the research model sees the raw payload.
    """
    runs: list[ToolRun] = []
    for parsed in parse_tool_calls_from_assistant(assistant_text):
        if parsed.parse_error or not parsed.name:
            runs.append(
                ToolRun(
                    name=parsed.name or "_unparsed",
                    arguments=parsed.arguments,
                    raw_output=parsed.parse_error or "[tool_call_parse_error]",
                    parse_error=parsed.parse_error,
                )
            )
            continue
        try:
            output = dispatcher.dispatch(
                parsed.name,
                parsed.arguments,
                round_index=round_index,
            )
        except Exception as exc:  # noqa: BLE001
            output = f"[ToolDispatcher error] {type(exc).__name__}: {exc}"
        runs.append(
            ToolRun(
                name=parsed.name,
                arguments=parsed.arguments,
                raw_output=output,
                parse_error=None,
            )
        )
    return runs
