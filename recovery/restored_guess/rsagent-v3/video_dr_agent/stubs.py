"""Placeholder tool / critic implementations."""

from __future__ import annotations

import json
from typing import Any, Sequence

from video_dr_agent.protocols import CriticPromptBuilder, CriticAgent, ToolDispatcher
from video_dr_agent.types import ToolRun


class StubToolDispatcher(ToolDispatcher):
    def dispatch(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        round_index: int | None = None,
    ) -> str:
        _ = round_index
        return (
            f"[stub_tool_output] name={name!r} arguments="
            f"{json.dumps(arguments, ensure_ascii=False)[:500]}"
        )


class StubCriticPromptBuilder(CriticPromptBuilder):
    def build(
        self,
        *,
        round_index: int,
        research_assistant_text: str,
        tool_runs: Sequence[ToolRun],
        extra: Any = None,
    ) -> str:
        lines = [
            f"[stub critic user payload] round={round_index}",
            "--- research assistant (trunc) ---",
            research_assistant_text[:2000],
            "--- tool runs ---",
        ]
        for tr in tool_runs:
            pe = f" parse_error={tr.parse_error!r}" if tr.parse_error else ""
            lines.append(
                f"- {tr.name}: {tr.raw_output[:1500]!s}{pe}"
            )
        return "\n".join(lines)


class StubCriticAgent(CriticAgent):
    """No network; echoes a short prefix + payload head (for dry runs)."""

    def __init__(self, critic_ctx: Any = None) -> None:
        self.ctx = critic_ctx

    def complete_round(self, user_payload: str) -> str:
        feedback = "[stub_critic_feedback]\n" + user_payload[:4000]
        if self.ctx is not None:
            self.ctx.append_assistant(feedback)
        return feedback
