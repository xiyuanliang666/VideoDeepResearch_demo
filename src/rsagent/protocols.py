"""Protocol interfaces for rsagent-v4."""

from __future__ import annotations

from typing import Any, Protocol


class ToolDispatcher(Protocol):
    def dispatch(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        round_index: int | None = None,
    ) -> str:
        """Execute tool by name; return JSON string result."""
