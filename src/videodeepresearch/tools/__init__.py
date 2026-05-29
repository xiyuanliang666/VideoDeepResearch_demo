"""Tooling layer for the unified VideoDeepResearch framework."""

from .base import StubToolDispatcher, ToolDispatcher
from .focus_keyframes import (
    FOCUS_SELECT_KEYFRAMES_TOOL_NAME,
    FocusKeyframeToolDispatcher,
    build_focus_dispatcher,
)
from .jina_reader import jina_reader_fetch, normalize_http_url, truncate_for_llm
from .registry import build_default_tool_dispatcher
from .web_search import (
    DEEP_RESEARCH_WEB_SEARCH_TOOL_NAME,
    DeepResearchWebSearchToolDispatcher,
    maybe_wrap_serper_search,
)

__all__ = [
    "DEEP_RESEARCH_WEB_SEARCH_TOOL_NAME",
    "FOCUS_SELECT_KEYFRAMES_TOOL_NAME",
    "DeepResearchWebSearchToolDispatcher",
    "FocusKeyframeToolDispatcher",
    "StubToolDispatcher",
    "ToolDispatcher",
    "build_default_tool_dispatcher",
    "build_focus_dispatcher",
    "jina_reader_fetch",
    "maybe_wrap_serper_search",
    "normalize_http_url",
    "truncate_for_llm",
]
