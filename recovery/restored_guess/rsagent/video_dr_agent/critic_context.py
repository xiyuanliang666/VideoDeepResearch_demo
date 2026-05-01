"""Critic-side history: system + critic assistants only (no persistent tool-output user turns)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, List

from video_dr_agent.research_context import redact_messages_for_log


class CriticContext:
    """
    Persistent messages: [system, assistant_1, assistant_2, ...].
    Each API call uses: persistent + [user_payload] (payload is not stored).
    """

    def __init__(self, *, system_content: str = "") -> None:
        self._messages: List[dict[str, Any]] = []
        if system_content.strip():
            self._messages.append({"role": "system", "content": system_content})

    def messages_for_api(self, user_payload: str) -> List[dict[str, Any]]:
        return [*self._messages, {"role": "user", "content": user_payload}]

    def append_assistant(self, content: str) -> None:
        self._messages.append({"role": "assistant", "content": content})

    def persistent_snapshot_redacted(self) -> List[dict[str, Any]]:
        return redact_messages_for_log(self._messages)

    def deepcopy_persistent(self) -> List[dict[str, Any]]:
        return deepcopy(self._messages)
