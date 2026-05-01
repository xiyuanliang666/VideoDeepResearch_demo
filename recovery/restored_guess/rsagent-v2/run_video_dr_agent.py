"""Video DeepResearch agent framework (vLLM Qwen3-VL research + OpenAI-compatible critic)."""

from video_dr_agent.critic_context import CriticContext
from video_dr_agent.critic_deep_research import (
    DeepResearchCriticAgent,
    DeepResearchCriticConfig,
    DeepResearchCriticPromptBuilder,
)
from video_dr_agent.critic_prompts import (
    CRITIC_SYSTEM_PROMPT_DEEP,
    CRITIC_USER_INTEGRATE_TEMPLATE,
    CRITIC_USER_RANK_URLS_TEMPLATE,
)
from video_dr_agent.focus_keyframe_tool import (
    FOCUS_SELECT_KEYFRAMES_TOOL_NAME,
    FocusKeyframeToolDispatcher,
    build_focus_dispatcher,
    default_focus_arg_namespace,
)
from video_dr_agent.serper_web_search_tool import (
    DEEP_RESEARCH_WEB_SEARCH_TOOL_NAME,
    DEEP_RESEARCH_WORKER_CONFIGS,
    DeepResearchWebSearchToolDispatcher,
    maybe_wrap_serper_search,
    run_deep_research_parallel_serper,
    serper_search,
)
from video_dr_agent.loop import LoopResult, run_loop
from video_dr_agent.openai_critic import OpenAICriticAgent, OpenAICriticConfig
from video_dr_agent.protocols import CriticAgent, CriticPromptBuilder, ToolDispatcher
from video_dr_agent.qwen3vl_client import (
    VLLMResearchConfig,
    Qwen3VLClient,
    build_initial_user_content,
)
from video_dr_agent.jina_reader import jina_reader_fetch, normalize_http_url, truncate_for_llm
from video_dr_agent.research_context import ResearchContext, extract_research_question_from_context
from video_dr_agent.research_prompt import resolve_research_system_content
from video_dr_agent.stubs import StubCriticAgent, StubCriticPromptBuilder, StubToolDispatcher
from video_dr_agent.tools_example import EXAMPLE_NOOP_TOOL_NAME, ExampleToolDispatcher
from video_dr_agent.types import ParsedToolCall, ToolRun

__all__ = [
    "CRITIC_SYSTEM_PROMPT_DEEP",
    "CRITIC_USER_INTEGRATE_TEMPLATE",
    "CRITIC_USER_RANK_URLS_TEMPLATE",
    "DeepResearchCriticAgent",
    "DeepResearchCriticConfig",
    "DeepResearchCriticPromptBuilder",
    "DEEP_RESEARCH_WEB_SEARCH_TOOL_NAME",
    "DEEP_RESEARCH_WORKER_CONFIGS",
    "DeepResearchWebSearchToolDispatcher",
    "maybe_wrap_serper_search",
    "run_deep_research_parallel_serper",
    "serper_search",
    "FOCUS_SELECT_KEYFRAMES_TOOL_NAME",
    "FocusKeyframeToolDispatcher",
    "build_focus_dispatcher",
    "default_focus_arg_namespace",
    "EXAMPLE_NOOP_TOOL_NAME",
    "ExampleToolDispatcher",
    "CriticAgent",
    "CriticContext",
    "CriticPromptBuilder",
    "LoopResult",
    "OpenAICriticAgent",
    "OpenAICriticConfig",
    "ParsedToolCall",
    "Qwen3VLClient",
    "ResearchContext",
    "resolve_research_system_content",
    "StubCriticAgent",
    "StubCriticPromptBuilder",
    "StubToolDispatcher",
    "ToolDispatcher",
    "ToolRun",
    "VLLMResearchConfig",
    "build_initial_user_content",
    "extract_research_question_from_context",
    "jina_reader_fetch",
    "normalize_http_url",
    "run_loop",
    "truncate_for_llm",
]
