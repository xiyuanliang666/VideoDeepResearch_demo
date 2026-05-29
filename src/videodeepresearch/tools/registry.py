"""Tool dispatcher factory for agentic VideoDeepResearch runs."""

from __future__ import annotations

from pathlib import Path

from .base import StubToolDispatcher, ToolDispatcher
from .focus_keyframes import build_focus_dispatcher
from .web_search import maybe_wrap_serper_search


def build_default_tool_dispatcher(
    *,
    output_root: str | Path = "outputs/tool_runs",
    default_video_path: str | None = None,
    focus_enabled: bool = False,
    focus_device: str = "cuda:0",
    serper_api_key: str | None = None,
    serper_results_per_worker: int = 2,
    evidence_domain: str | None = None,
) -> ToolDispatcher:
    """Build the unified tool chain used by agentic runs.

    Tool execution is name-routed, not sequential. Web search and FOCUS are
    siblings in a dispatcher chain; the selected tool is entirely determined by
    the model's tool name.
    """
    dispatcher: ToolDispatcher = StubToolDispatcher()
    if focus_enabled:
        dispatcher = build_focus_dispatcher(
            output_root,
            device=focus_device,
            default_video_path=default_video_path,
            fallback=dispatcher,
        )
    dispatcher = maybe_wrap_serper_search(
        dispatcher,
        api_key=serper_api_key,
        results_per_worker=serper_results_per_worker,
        default_evidence_domain=evidence_domain,
    )
    return dispatcher
