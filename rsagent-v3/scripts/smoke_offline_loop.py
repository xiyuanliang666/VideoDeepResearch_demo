#!/usr/bin/env python3
"""
Offline smoke: core ``run_loop`` + stub tools/critic + trace JSON (no vLLM).

Run from ``rsagent/``:
  python scripts/smoke_offline_loop.py
"""

from __future__ import annotations

import base64
import shutil
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from video_dr_agent.loop import run_loop
from video_dr_agent.research_context import ResearchContext
from video_dr_agent.qwen3vl_client import build_initial_user_content
from video_dr_agent.stubs import (
    StubCriticAgent,
    StubCriticPromptBuilder,
    StubToolDispatcher,
)

_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


class OfflineMockResearchClient:
    def __init__(self) -> None:
        self._turn = 0

    def complete(self, messages: list) -> str:  # noqa: ARG002
        self._turn += 1
        if self._turn == 1:
            return (
                "Offline smoke — Research Plan:\n"
                "Step 1: run smoke_probe. Step 2: finish.\n"
                "(No tool_call in planning turn.)"
            )
        if self._turn == 2:
            return (
                '<tool_call>\n'
                '{"name": "smoke_probe", "arguments": {"round": 1}}\n'
                "</tool_call>"
            )
        return "Offline smoke: OK — no further tool calls."


def main() -> None:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(_TINY_PNG)
        png_path = f.name

    trace_dir: str | None = None
    try:
        research = ResearchContext(
            system_content="(offline smoke)",
            initial_user_content=build_initial_user_content(
                question="Smoke test question.",
                frame_paths=[png_path],
                video_path=None,
            ),
        )
        trace_dir = tempfile.mkdtemp(prefix="vdr_smoke_")
        result = run_loop(
            research=research,
            research_client=OfflineMockResearchClient(),
            tool_dispatcher=StubToolDispatcher(),
            critic_builder=StubCriticPromptBuilder(),
            critic_agent=StubCriticAgent(None),
            max_rounds=8,
            run_dir=trace_dir,
            research_question_for_critic="Smoke test question.",
            planning_phase_enabled=True,
        )
        round_files = sorted(Path(trace_dir).glob("round_*.json"))
        planning_path = Path(trace_dir) / "planning.json"
        assert planning_path.is_file(), "planning.json missing"
    finally:
        Path(png_path).unlink(missing_ok=True)
        if trace_dir:
            shutil.rmtree(trace_dir, ignore_errors=True)

    assert result.termination == "no_tool_calls", result.termination
    assert result.rounds == 2, result.rounds
    assert len(round_files) == 2, round_files
    print("smoke_offline_loop: OK")
    print("  termination:", result.termination)
    print("  rounds:", result.rounds)


if __name__ == "__main__":
    main()
