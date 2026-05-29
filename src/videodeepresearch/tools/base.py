"""Common tool protocols and fallback implementations."""

from __future__ import annotations

import json
from typing import Any, Protocol


class ToolDispatcher(Protocol):
    """Execute a named tool and return an opaque string payload for the caller."""

    def dispatch(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        round_index: int | None = None,
    ) -> str:
        """Execute tool by name."""


class StubToolDispatcher:
    """No-op fallback used when a tool is unavailable or intentionally disabled."""

    def dispatch(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        round_index: int | None = None,
    ) -> str:
        payload = {
            "ok": False,
            "tool": name,
            "round_index": round_index,
            "error": "tool_not_available",
            "arguments_preview": arguments,
        }
        return json.dumps(payload, ensure_ascii=False)
