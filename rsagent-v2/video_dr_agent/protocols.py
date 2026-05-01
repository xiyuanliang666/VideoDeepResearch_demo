"""
Protocols for Video DeepResearch (see repo plan: research/critic separation).

- ``ToolDispatcher``: execute tools; outputs go to critic payload only, not research messages.
- ``CriticPromptBuilder``: assemble this round's user payload (may embed tool runs).
- ``CriticAgent``: ``complete_round(user_payload) -> str`` feedback for research's next user turn.
"""

from __future__ import annotations

from typing import Any, List, Protocol, Sequence

from video_dr_agent.types import ToolRun


class ToolDispatcher(Protocol):
    def dispatch(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        round_index: int | None = None,
    ) -> str:
        """Execute tool by name; return opaque string for critic payload.

        round_index: current research loop round (1-based), for naming outputs etc.
        """


class CriticPromptBuilder(Protocol):
    def build(
        self,
        *,
        round_index: int,
        research_assistant_text: str,
        tool_runs: Sequence[ToolRun],
        extra: Any = None,
    ) -> str:
        """Assemble this round's user payload for the critic (may embed tool outputs)."""


class CriticAgent(Protocol):
    def complete_round(self, user_payload: str) -> str:
        """
        Call OpenAI-compatible chat.completions with internal critic history + user_payload.
        Returns feedback string appended to research_messages as the next user turn.
        """
