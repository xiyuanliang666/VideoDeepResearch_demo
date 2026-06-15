"""Unified tool dispatcher for rsagent-v4."""

from __future__ import annotations

import json
import os
from typing import Any

from src.rsagent.protocols import ToolDispatcher
from src.rsagent.tools.video_frame_extract import VideoFrameExtractDispatcher, TOOL_NAME as VFE_TOOL
from src.rsagent.tools.web_search import WebSearchDispatcher, TOOL_NAME as WS_TOOL


class StubDispatcher(ToolDispatcher):
    def dispatch(self, name: str, arguments: dict[str, Any], *, round_index: int | None = None) -> str:
        return json.dumps({"ok": False, "error": f"Unknown tool: {name}"})


def build_dispatcher(
    *,
    video_path: str | None = None,
    serper_api_key: str | None = None,
    jina_api_key: str | None = None,
    llm_client: Any = None,
    results_per_worker: int = 3,
    evidence_domain: str | None = None,
) -> ToolDispatcher:
    """Build the tool dispatcher chain: web_search -> video_frame_extract -> stub."""
    stub = StubDispatcher()

    if video_path:
        vfe = VideoFrameExtractDispatcher(video_path, fallback=stub)
    else:
        vfe = stub  # type: ignore

    ws = WebSearchDispatcher(
        api_key=serper_api_key,
        jina_api_key=jina_api_key,
        llm_client=llm_client,
        fallback=vfe,
        results_per_worker=results_per_worker,
        default_evidence_domain=evidence_domain,
    )
    return ws
