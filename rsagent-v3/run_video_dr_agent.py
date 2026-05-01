#!/usr/bin/env python3
"""
CLI entry for video_dr_agent (**rsagent-v3**).

v3 adds **evidence-tier** rules and **structured critic verdicts** (see ``critic_prompts``,
``evidence_policy``, and ``prompts/research_system_prompt.txt``) so Research does not treat
travel/UGC snippets as UNESCO/ISO **registry** proof.

Recommended environment (add only missing deps, avoid upgrading unrelated packages):
  conda activate AKS
  pip install -r requirements-video-dr-agent.txt --upgrade-strategy only-if-needed

Config: place API keys and ``TEST_*`` defaults in ``video_dr_agent/.env`` (see ``video_dr_agent/env_loader.py`` for aliases).
Optional ``RESEARCH_EVIDENCE_DOMAIN`` (or ``TEST_QUESTION_DOMAIN``) biases ``deep_research_web_search`` site filters and the deep Critic.

Run from repo root `rsagent/`:
  python run_video_dr_agent.py --help
  python run_video_dr_agent.py --from-env --out-dir ./runs/exp1

Uniform overview (optional): set ``TEST_UNIFORM_OVERVIEW_FRAMES`` and ``TEST_UNIFORM_OVERVIEW_DIR`` in ``video_dr_agent/.env``,
or pass ``--uniform-overview-frames N`` / ``--uniform-overview-dir DIR``, to run ``test_data_frame/uniform_sample_frames.py``
before the loop. Use ``RESEARCH_INITIAL_INCLUDE_FULL_VIDEO=0`` or ``--no-initial-full-video`` to send only overview frames + question
on the first turn (keep ``--video`` / ``FOCUS_SOURCE_VIDEO`` for ``focus_select_keyframes``).

Round-1 no-tool nudge (optional): ``RESEARCH_REJECT_ROUND1_NO_TOOLS`` or ``--reject-round1-no-tools`` / ``--no-reject-round1-no-tools``;
optional custom nudge text via ``RESEARCH_ROUND1_NO_TOOLS_FEEDBACK`` or ``--round1-no-tools-feedback``.

Mandatory planning (default on): one Host turn asks for a written Research Plan (no tools), then a handoff before the tool loop.
Disable with ``RESEARCH_PLANNING_PHASE=0`` or ``--no-planning-phase``. Optional ``RESEARCH_PLANNING_USER_PATH`` or ``--planning-user-file``.

Research system prompt: unless ``--research-system`` or ``--no-default-research-prompt`` is set,
loads ``prompts/research_system_prompt.txt`` (deep-research workflow + ``<tool_call>`` protocol).
Critic (deep mode): ``--deep-critic`` uses ``critic_prompts.CRITIC_SYSTEM_PROMPT_DEEP`` + staged templates
(structured ``### Structured evidence verdict`` block on Jina integrate turns; FOCUS-only rounds get a template verdict).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from video_dr_agent.env_loader import (
    collect_frames_from_test_env,
    get_default_source_video_path,
    get_research_evidence_domain_from_env,
    get_research_round1_no_tools_feedback_from_env,
    get_test_question_from_env,
    get_uniform_overview_frame_count_from_env,
    get_uniform_overview_output_dir_from_env,
    load_video_dr_agent_dotenv,
    research_initial_include_full_video_from_env,
    research_inject_focus_frames_from_env,
    research_max_focus_images_from_env,
    research_reject_round1_no_tools_from_env,
    research_planning_phase_enabled_from_env,
    get_research_planning_user_path_from_env,
)
from video_dr_agent.evidence_policy import normalize_evidence_domain
from video_dr_agent.uniform_overview_prep import (
    format_uniform_overview_planning_appendix,
    run_uniform_overview_sampling,
)
from video_dr_agent.research_prompt import resolve_research_system_content
from video_dr_agent.critic_context import CriticContext
from video_dr_agent.critic_deep_research import (
    DeepResearchCriticAgent,
    DeepResearchCriticConfig,
    DeepResearchCriticPromptBuilder,
)
from video_dr_agent.critic_prompts import CRITIC_SYSTEM_PROMPT_DEEP
from video_dr_agent.loop import run_loop
from video_dr_agent.openai_critic import OpenAICriticAgent, OpenAICriticConfig
from video_dr_agent.qwen3vl_client import (
    VLLMResearchConfig,
    Qwen3VLClient,
    build_initial_user_content,
)
from video_dr_agent.research_context import ResearchContext
from video_dr_agent.focus_keyframe_tool import build_focus_dispatcher
from video_dr_agent.serper_web_search_tool import maybe_wrap_serper_search
from video_dr_agent.stubs import StubCriticAgent, StubCriticPromptBuilder, StubToolDispatcher
from video_dr_agent.tools_example import ExampleToolDispatcher


def main() -> None:
    load_video_dr_agent_dotenv(_ROOT)

    p = argparse.ArgumentParser(description="Video DeepResearch loop (vLLM + critic)")
    p.add_argument(
        "--frame",
        action="append",
        default=[],
        metavar="PATH",
        help="Frame image path (repeatable). At least one of --frame or --video required.",
    )
    p.add_argument(
        "--video",
        default=None,
        help=(
            "Local video path for vLLM video_url and FOCUS default_video_path. "
            "If omitted, uses FOCUS_SOURCE_VIDEO or VIDEO_PATH from video_dr_agent/.env."
        ),
    )
    p.add_argument(
        "--question",
        default=None,
        help="Task / question text. Omit if using --from-env (reads TEST_QUESTION from video_dr_agent/.env).",
    )
    p.add_argument(
        "--from-env",
        action="store_true",
        help=(
            "Load TEST_QUESTION from video_dr_agent/.env; frames from TEST_FRAME_PATH and/or "
            "TEST_UNIFORM_OVERVIEW_FRAMES + TEST_UNIFORM_OVERVIEW_DIR (see --uniform-overview-frames)."
        ),
    )
    p.add_argument(
        "--evidence-domain",
        default=None,
        metavar="PROFILE",
        help=(
            "Host evidence profile for search + deep critic (e.g. registry, academic). "
            "When omitted, use RESEARCH_EVIDENCE_DOMAIN or TEST_QUESTION_DOMAIN from video_dr_agent/.env."
        ),
    )
    p.add_argument(
        "--uniform-overview-frames",
        type=int,
        default=None,
        metavar="N",
        help=(
            "If > 0, run test_data_frame/uniform_sample_frames.py on the effective video before the loop. "
            "Overrides TEST_UNIFORM_OVERVIEW_FRAMES; use 0 to disable when env would otherwise enable."
        ),
    )
    p.add_argument(
        "--uniform-overview-dir",
        default=None,
        metavar="DIR",
        help="Output directory for --uniform-overview-frames (overrides TEST_UNIFORM_OVERVIEW_DIR).",
    )
    p.add_argument(
        "--no-initial-full-video",
        action="store_true",
        help=(
            "Do not attach the full video to the first user message (only frames + question). "
            "Still pass --video / env for focus_select_keyframes. Overrides RESEARCH_INITIAL_INCLUDE_FULL_VIDEO."
        ),
    )
    p.add_argument(
        "--no-inject-focus-frames",
        action="store_true",
        help=(
            "After critic feedback, do not append FOCUS keyframe images as multimodal user content. "
            "Overrides RESEARCH_INJECT_FOCUS_FRAMES."
        ),
    )
    p.add_argument(
        "--max-focus-images",
        type=int,
        default=None,
        metavar="N",
        help="Max FOCUS JPEGs to inject per round (default: env RESEARCH_MAX_FOCUS_IMAGES or 32).",
    )
    p.add_argument(
        "--reject-round1-no-tools",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "If enabled: when round 1 has no <tool_call>, append a nudge user message and request one more turn. "
            "Default: from RESEARCH_REJECT_ROUND1_NO_TOOLS when omitted."
        ),
    )
    p.add_argument(
        "--round1-no-tools-feedback",
        default=None,
        metavar="TEXT",
        help=(
            "Custom nudge text when rejecting round-1 no-tool answers. "
            "Default: RESEARCH_ROUND1_NO_TOOLS_FEEDBACK or built-in English prompt."
        ),
    )
    p.add_argument(
        "--planning-phase",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Mandatory Research Plan Host turn + execution handoff before tools (default: on; "
            "env RESEARCH_PLANNING_PHASE=0 disables when CLI omitted)."
        ),
    )
    p.add_argument(
        "--planning-user-file",
        default=None,
        metavar="PATH",
        help=(
            "UTF-8 file whose contents replace the default English planning Host message "
            "(overrides RESEARCH_PLANNING_USER_PATH)."
        ),
    )
    p.add_argument(
        "--vllm-base-url",
        default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        help="vLLM OpenAI base URL (with /v1)",
    )
    p.add_argument(
        "--vllm-model",
        default=os.environ.get("VLLM_MODEL", "Qwen3-VL-8B-Instruct"),
        help="Served model name",
    )
    p.add_argument("--vllm-api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-rounds", type=int, default=16)
    p.add_argument(
        "--out-dir",
        default=None,
        help="Directory for round_*.json traces",
    )
    p.add_argument(
        "--critic-base-url",
        default=os.environ.get("CRITIC_BASE_URL"),
        help="If set with --critic-api-key, use live critic; else stub critic.",
    )
    p.add_argument("--critic-api-key", default=os.environ.get("CRITIC_API_KEY"))
    p.add_argument("--critic-model", default=os.environ.get("CRITIC_MODEL", "gpt-4o-mini"))
    p.add_argument(
        "--research-system",
        default="",
        help=(
            "Inline system prompt for research (vLLM). If empty, load from "
            "RESEARCH_SYSTEM_PROMPT_PATH or prompts/research_system_prompt.txt (see --no-default-research-prompt)."
        ),
    )
    p.add_argument(
        "--no-default-research-prompt",
        action="store_true",
        help="Do not fall back to prompts/research_system_prompt.txt when --research-system is empty.",
    )
    p.add_argument(
        "--critic-system",
        default="You are a critic. Summarize tool results for the research model. (stub)",
        help="Critic system message.",
    )
    p.add_argument(
        "--use-example-tools",
        action="store_true",
        help="Use ExampleToolDispatcher (example_noop + unknown fallback) instead of StubToolDispatcher.",
    )
    p.add_argument(
        "--focus-frame-set-dir",
        default=os.environ.get("FOCUS_FRAME_SET_DIR"),
        metavar="DIR",
        help=(
            "If set (CLI or env FOCUS_FRAME_SET_DIR), enable focus_select_keyframes (FOCUS); "
            "outputs under DIR as {video_stem}_r{round}_{frame_caption}/. Needs FOCUS deps + CUDA. "
            "--deep-critic alone does NOT enable FOCUS. Combine with --use-example-tools for example_noop."
        ),
    )
    p.add_argument(
        "--focus-device",
        default=os.environ.get("FOCUS_DEVICE", "cuda:0"),
        help="CUDA device for FOCUS/BLIP (e.g. cuda:0).",
    )
    p.add_argument(
        "--serper-api-key",
        default=os.environ.get("SERPER_API_KEY"),
        help="Serper API key for tool deep_research_web_search (or set SERPER_API_KEY).",
    )
    p.add_argument(
        "--serper-results-per-worker",
        type=int,
        default=2,
        help="Max organic-style results per worker (each of 4 workers calls Serper once).",
    )
    p.add_argument(
        "--deep-critic",
        action="store_true",
        help=(
            "Use DeepResearchCritic: FOCUS success triggers Top-3 frame ranking for injection; "
            "web search uses snippet ranking, Jina Top-3 fetch, then synthesized answer. "
            "Requires --critic-base-url and --critic-api-key; JINA_API_KEY recommended."
        ),
    )
    p.add_argument(
        "--jina-api-key",
        default=os.environ.get("JINA_API_KEY"),
        help="Jina Reader API key for deep critic (or JINA_API_KEY).",
    )
    p.add_argument("--critic-max-tokens-rank", type=int, default=1200)
    p.add_argument("--critic-max-tokens-integrate", type=int, default=4096)
    p.add_argument(
        "--search-failure-ratio",
        type=float,
        default=0.5,
        help="Deep critic: if this fraction of Serper rows look like errors, skip Jina.",
    )
    args = p.parse_args()

    if args.from_env:
        q = get_test_question_from_env()
        if not q:
            p.error("--from-env requires TEST_QUESTION in video_dr_agent/.env")
        args.question = q
    elif not args.question:
        p.error("--question is required unless --from-env is set")

    if args.evidence_domain is not None:
        evidence_domain_raw = (args.evidence_domain or "").strip() or None
    else:
        evidence_domain_raw = get_research_evidence_domain_from_env()
    evidence_domain_norm = (
        normalize_evidence_domain(evidence_domain_raw)
        if evidence_domain_raw
        else "general"
    )
    critic_evidence_domain = (
        evidence_domain_norm if evidence_domain_norm != "general" else None
    )

    effective_video = (args.video or "").strip() or get_default_source_video_path()

    if args.uniform_overview_frames is not None:
        u_count = max(0, int(args.uniform_overview_frames))
    else:
        u_count = get_uniform_overview_frame_count_from_env()

    u_dir_raw = (args.uniform_overview_dir or "").strip() if args.uniform_overview_dir else None
    if u_dir_raw:
        uniform_out_dir: Path | None = Path(u_dir_raw)
    else:
        env_udir = get_uniform_overview_output_dir_from_env()
        uniform_out_dir = Path(env_udir) if env_udir else None

    overview_paths: list[str] = []
    overview_meta: dict | None = None
    if u_count > 0:
        if not effective_video:
            p.error(
                "均匀概览抽帧需要本地视频路径：请使用 --video 或在 video_dr_agent/.env 中设置 "
                "FOCUS_SOURCE_VIDEO / VIDEO_PATH"
            )
        vp = Path(effective_video).expanduser()
        try:
            vp = vp.resolve()
        except OSError:
            pass
        if not vp.is_file():
            p.error(f"均匀概览抽帧：视频文件不存在或不可读: {effective_video}")
        out_dir = uniform_out_dir if uniform_out_dir is not None else (
            _ROOT / "test_data_frame" / f"overview_{vp.stem}"
        )
        try:
            overview_paths, overview_meta = run_uniform_overview_sampling(
                _ROOT, str(vp), u_count, out_dir
            )
        except Exception as exc:
            p.error(f"均匀概览抽帧失败: {exc}")

    frame_paths: list[str] = list(args.frame)
    if args.from_env:
        if overview_paths:
            frame_paths = overview_paths + frame_paths
        else:
            env_frames = collect_frames_from_test_env()
            if not env_frames:
                p.error(
                    "--from-env requires TEST_FRAME_PATH (目录或单张图), "
                    "或设置 TEST_UNIFORM_OVERVIEW_FRAMES>0 以自动生成概览帧"
                )
            frame_paths = env_frames + frame_paths
    elif overview_paths:
        frame_paths = overview_paths + frame_paths

    include_initial_video = research_initial_include_full_video_from_env()
    if args.no_initial_full_video:
        include_initial_video = False

    if not frame_paths and not effective_video:
        p.error(
            "Provide at least one --frame or --video, or use --from-env with TEST_FRAME_PATH, "
            "or set FOCUS_SOURCE_VIDEO / VIDEO_PATH in video_dr_agent/.env, "
            "or enable uniform overview (TEST_UNIFORM_OVERVIEW_FRAMES / --uniform-overview-frames)"
        )
    if not include_initial_video and not frame_paths:
        p.error(
            "首轮已关闭整段视频（--no-initial-full-video 或 RESEARCH_INITIAL_INCLUDE_FULL_VIDEO=0），"
            "但未提供任何图片帧：请启用均匀概览抽帧、--frame 或 TEST_FRAME_PATH"
        )

    research_system = resolve_research_system_content(
        _ROOT,
        args.research_system,
        allow_default_file=not args.no_default_research_prompt,
    )

    user_content = build_initial_user_content(
        question=args.question or "",
        frame_paths=frame_paths or None,
        video_path=effective_video,
        include_video=include_initial_video,
        evidence_domain=evidence_domain_raw,
    )
    research = ResearchContext(
        system_content=research_system,
        initial_user_content=user_content,
    )
    vcfg = VLLMResearchConfig(
        base_url=args.vllm_base_url,
        api_key=args.vllm_api_key,
        model=args.vllm_model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    qwen = Qwen3VLClient(vcfg)

    if args.deep_critic:
        if not args.critic_base_url or not args.critic_api_key:
            p.error("--deep-critic requires both --critic-base-url and --critic-api-key")
        critic_ctx = CriticContext(system_content=CRITIC_SYSTEM_PROMPT_DEEP)
        dcfg = DeepResearchCriticConfig(
            base_url=args.critic_base_url,
            api_key=args.critic_api_key,
            model=args.critic_model,
            max_tokens_rank=args.critic_max_tokens_rank,
            max_tokens_integrate=args.critic_max_tokens_integrate,
            search_failure_ratio_threshold=float(args.search_failure_ratio),
        )
        critic = DeepResearchCriticAgent(
            dcfg,
            critic_ctx,
            jina_api_key=args.jina_api_key,
        )
        critic_builder = DeepResearchCriticPromptBuilder()
    else:
        critic_ctx = CriticContext(system_content=args.critic_system)
        critic_builder = StubCriticPromptBuilder()
        if args.critic_base_url and args.critic_api_key:
            ccfg = OpenAICriticConfig(
                base_url=args.critic_base_url,
                api_key=args.critic_api_key,
                model=args.critic_model,
            )
            critic = OpenAICriticAgent(ccfg, critic_ctx)
        else:
            critic = StubCriticAgent(critic_ctx)

    # No fixed pipeline between tools: each round routes by a single tool name from research.
    # Outer Serper wrap is a chain: name==deep_research_web_search → search; else delegate
    # to inner (FOCUS / example_noop / stub) for other names.
    if args.focus_frame_set_dir:
        dispatcher = build_focus_dispatcher(
            args.focus_frame_set_dir,
            device=args.focus_device,
            include_example_noop=args.use_example_tools,
            default_video_path=effective_video,
        )
    elif args.use_example_tools:
        dispatcher = ExampleToolDispatcher()
    else:
        dispatcher = StubToolDispatcher()

    dispatcher = maybe_wrap_serper_search(
        dispatcher,
        api_key=args.serper_api_key,
        results_per_worker=max(1, int(args.serper_results_per_worker)),
        default_evidence_domain=evidence_domain_raw,
    )

    inject_focus = research_inject_focus_frames_from_env()
    if args.no_inject_focus_frames:
        inject_focus = False
    max_focus_img = (
        args.max_focus_images
        if args.max_focus_images is not None
        else research_max_focus_images_from_env()
    )

    reject_round1_no_tools = (
        args.reject_round1_no_tools
        if args.reject_round1_no_tools is not None
        else research_reject_round1_no_tools_from_env()
    )
    if args.round1_no_tools_feedback is not None:
        _rf = args.round1_no_tools_feedback.strip()
        round1_no_tools_feedback = _rf or None
    else:
        round1_no_tools_feedback = get_research_round1_no_tools_feedback_from_env()

    if args.planning_phase is None:
        planning_on = research_planning_phase_enabled_from_env()
    else:
        planning_on = args.planning_phase

    planning_user_msg: str | None = None
    _puf = (args.planning_user_file or "").strip() or (
        get_research_planning_user_path_from_env() or ""
    )
    if _puf:
        _pp = Path(_puf).expanduser()
        if not _pp.is_file():
            p.error(f"Planning user file not found: {_pp}")
        planning_user_msg = _pp.read_text(encoding="utf-8")

    planning_suffix = ""
    if overview_meta:
        planning_suffix = format_uniform_overview_planning_appendix(
            meta=overview_meta,
            research_question=args.question or "",
        )
    planning_user_suffix = planning_suffix.strip() or None

    result = run_loop(
        research=research,
        tool_dispatcher=dispatcher,
        critic_builder=critic_builder,
        critic_agent=critic,
        max_rounds=args.max_rounds,
        run_dir=args.out_dir,
        research_question_for_critic=args.question,
        research_evidence_domain=critic_evidence_domain,
        research_client=qwen,
        inject_focus_frames=inject_focus,
        max_focus_images_per_round=max_focus_img,
        reject_round1_no_tools=reject_round1_no_tools,
        round1_no_tools_feedback=round1_no_tools_feedback,
        planning_phase_enabled=planning_on,
        planning_user_message=planning_user_msg,
        planning_user_suffix=planning_user_suffix,
    )
    print("termination:", result.termination)
    print("rounds:", result.rounds)
    print()
    print("=" * 72)
    print("Final research model answer (last assistant turn)")
    print("=" * 72)
    if result.termination == "max_rounds_reached":
        print(
            "(Stopped at max_rounds; last turn may still contain <tool_call> or be incomplete.)"
        )
    body = (result.last_assistant or "").strip()
    print(body if body else "(empty)")


if __name__ == "__main__":
    main()
