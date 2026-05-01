#!/usr/bin/env python3
"""
Lightweight checks that the Video DeepResearch stack imports and wires correctly:

- Default research prompt file resolves
- Deep critic prompts import
- Serper + FOCUS tool names
- Offline loop smoke (no network)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    from video_dr_agent import (
        CRITIC_SYSTEM_PROMPT_DEEP,
        DEEP_RESEARCH_WEB_SEARCH_TOOL_NAME,
        FOCUS_SELECT_KEYFRAMES_TOOL_NAME,
        resolve_research_system_content,
        run_loop,
    )

    assert "Critic" in CRITIC_SYSTEM_PROMPT_DEEP or "critic" in CRITIC_SYSTEM_PROMPT_DEEP.lower()
    assert DEEP_RESEARCH_WEB_SEARCH_TOOL_NAME == "deep_research_web_search"
    assert FOCUS_SELECT_KEYFRAMES_TOOL_NAME == "focus_select_keyframes"

    text = resolve_research_system_content(_ROOT, "", allow_default_file=True)
    assert text and len(text) > 100, "prompts/research_system_prompt.txt should load"
    assert "tool_call" in text.lower() or "<tool_call>" in text

    assert callable(run_loop)

    smoke = _ROOT / "scripts" / "smoke_offline_loop.py"
    r = subprocess.run(
        [sys.executable, str(smoke)],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        raise SystemExit(r.returncode)
    print(r.stdout.strip())
    print("architecture_checks: OK")


if __name__ == "__main__":
    main()
