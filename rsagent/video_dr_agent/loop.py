"""
Main research loop (Video DeepResearch plan).

Flow: **vLLM research** → parse ``<tool_call>`` → **ToolDispatcher** (raw tool JSON 不直接进 research) →
**CriticPromptBuilder** + **CriticAgent** → append **critic 文本**；若本轮含成功的 ``focus_select_keyframes``，
可再追加一条 **多模态 user**（关键帧 ``image_url``），让模型直接看到画面。

Termination: assistant text contains **no** ``<tool_call>`` blocks, or ``max_rounds`` reached.
Raw tool outputs are persisted in ``round_*.json`` only; research context stays clean per plan.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List, Sequence

from video_dr_agent.critic_context import CriticContext
from video_dr_agent.parsing import parse_tool_calls_from_assistant
from video_dr_agent.protocols import CriticAgent, CriticPromptBuilder, ToolDispatcher
from video_dr_agent.focus_keyframe_tool import FOCUS_SELECT_KEYFRAMES_TOOL_NAME
from video_dr_agent.qwen3vl_client import Qwen3VLClient, image_content_part
from video_dr_agent.research_context import (
    ResearchContext,
    extract_research_question_from_context,
)
from video_dr_agent.types import ToolRun

DEFAULT_ROUND1_NO_TOOLS_NUDGE = (
    "[Host] 你的上一则回复没有包含任何 <tool_call>。"
    "请先至少调用一个已声明的工具（例如 focus_select_keyframes 或 deep_research_web_search），"
    "再基于工具结果继续推理；请勿在无工具调用的情况下结束探索。"
    "请重新生成本轮 assistant，且必须包含至少一个合法的 <tool_call>。"
)


@dataclass
class LoopResult:
    termination: str
    rounds: int
    last_assistant: str = ""
    research_messages_redacted: List[dict[str, Any]] = field(default_factory=list)


def _collect_tool_runs(
    parsed: Sequence[Any],
    dispatcher: ToolDispatcher,
    *,
    round_index: int,
) -> List[ToolRun]:
    runs: List[ToolRun] = []
    for pc in parsed:
        if pc.parse_error or not pc.name:
            runs.append(
                ToolRun(
                    name=pc.name or "_unparsed",
                    arguments=pc.arguments,
                    raw_output=pc.parse_error or "[Json Parse Error]",
                    parse_error=pc.parse_error,
                )
            )
            continue
        try:
            out = dispatcher.dispatch(
                pc.name, pc.arguments, round_index=round_index
            )
        except Exception as exc:  # noqa: BLE001
            out = f"[ToolDispatcher error] {type(exc).__name__}: {exc}"
        runs.append(
            ToolRun(
                name=pc.name,
                arguments=pc.arguments,
                raw_output=out,
                parse_error=None,
            )
        )
    return runs


def _write_round(
    run_dir: Path,
    round_index: int,
    record: dict[str, Any],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"round_{round_index:04d}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def _critic_trace(critic_agent: CriticAgent) -> List[dict[str, Any]]:
    ctx = getattr(critic_agent, "ctx", None)
    if isinstance(ctx, CriticContext):
        return ctx.persistent_snapshot_redacted()
    return []


def _collect_focus_frame_paths(
    tool_runs: Sequence[ToolRun],
    *,
    max_images: int,
) -> list[str]:
    """从成功的 ``focus_select_keyframes`` JSON 输出中收集存在的本地图片路径（去重、保序）。"""
    out: list[str] = []
    seen: set[str] = set()
    for tr in tool_runs:
        if tr.parse_error or tr.name != FOCUS_SELECT_KEYFRAMES_TOOL_NAME:
            continue
        try:
            data = json.loads(tr.raw_output)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict) or not data.get("ok"):
            continue
        frames = data.get("frames")
        if not isinstance(frames, list):
            continue
        for rec in frames:
            if len(out) >= max_images:
                return out
            if not isinstance(rec, dict):
                continue
            fp = (rec.get("file") or "").strip()
            if not fp:
                continue
            p = Path(fp)
            if not p.is_file():
                continue
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
    return out


def run_loop(
    *,
    research: ResearchContext,
    research_client: Qwen3VLClient,
    tool_dispatcher: ToolDispatcher,
    critic_builder: CriticPromptBuilder,
    critic_agent: CriticAgent,
    max_rounds: int = 32,
    run_dir: Path | str | None = None,
    research_question_for_critic: str | None = None,
    inject_focus_frames: bool = True,
    max_focus_images_per_round: int = 32,
    reject_round1_no_tools: bool = False,
    round1_no_tools_feedback: str | None = None,
) -> LoopResult:
    """
    Terminates when the research model produces no <tool_call> blocks, or max_rounds hit.
    Critic 反馈以文本 user 消息追加；若 ``inject_focus_frames`` 为 True，随后将本轮 FOCUS 成功输出中的
    关键帧以多模态 user 消息（image_url）追加，供 vLLM 直接读图。

    ``reject_round1_no_tools``: 若第 1 轮 assistant 中解析不到任何 ``<tool_call>``，追加一条 user（提示必须调用工具）
    并再请求一次模型；若第二次仍无工具调用，则 ``termination`` 为 ``no_tool_calls``（并可在轨迹中见首轮尝试）。

    research_question_for_critic:
        传入 Critic 的「原始研究问题」锚点；为 None 时从首条 user 多模态文本自动提取。
    """
    out_dir = Path(run_dir) if run_dir else None
    termination = "max_rounds_reached"
    last_assistant = ""
    critic_rq = (research_question_for_critic or "").strip() or (
        extract_research_question_from_context(research)
    )
    # 主循环
    for r in range(1, max_rounds + 1):
        round1_nudge_applied = False
        assistant_first_attempt = ""
        # 步骤 A — Research 推理
        messages = research.messages()
        assistant = research_client.complete(messages)
        last_assistant = assistant
        # 步骤 B — 解析工具调用
        parsed = parse_tool_calls_from_assistant(assistant)
        research.append_assistant(assistant)

        if not parsed and r == 1 and reject_round1_no_tools:
            assistant_first_attempt = assistant
            nudge = (round1_no_tools_feedback or "").strip() or DEFAULT_ROUND1_NO_TOOLS_NUDGE
            research.append_user_content(nudge)
            assistant = research_client.complete(research.messages())
            last_assistant = assistant
            parsed = parse_tool_calls_from_assistant(assistant)
            research.append_assistant(assistant)
            round1_nudge_applied = True

        if not parsed:
            termination = "no_tool_calls"
            if out_dir:
                rec_early: dict[str, Any] = {
                    "round": r,
                    "termination": termination,
                    "research_assistant_raw": assistant,
                    "parsed_tool_calls": [],
                    "tool_runs": [],
                    "critic_user_payload": None,
                    "critic_assistant": None,
                    "research_feedback": None,
                    "research_messages_redacted": research.snapshot_redacted(),
                    "critic_messages_redacted": _critic_trace(critic_agent),
                }
                if round1_nudge_applied:
                    rec_early["round1_no_tools_nudge_applied"] = True
                    rec_early["research_assistant_raw_first_attempt"] = assistant_first_attempt
                    rec_early["round1_no_tools_user_text"] = (
                        (round1_no_tools_feedback or "").strip()
                        or DEFAULT_ROUND1_NO_TOOLS_NUDGE
                    )
                _write_round(out_dir, r, rec_early)
            return LoopResult(
                termination=termination,
                rounds=r,
                last_assistant=assistant,
                research_messages_redacted=research.snapshot_redacted(),
            )
        # 执行工具调用
        tool_runs = _collect_tool_runs(
            parsed, tool_dispatcher, round_index=r
        )
        # 步骤 C — 构建批评者用户负载
        critic_user_payload = critic_builder.build(
            round_index=r,
            research_assistant_text=assistant,
            tool_runs=tool_runs,
            extra={"research_question": critic_rq},
        )
        # 步骤 D — 批评者推理
        feedback = critic_agent.complete_round(critic_user_payload)
        research.append_critic_feedback(feedback)

        focus_injected_paths: list[str] = []
        if inject_focus_frames and max_focus_images_per_round > 0:
            focus_injected_paths = _collect_focus_frame_paths(
                tool_runs, max_images=max_focus_images_per_round
            )
            if focus_injected_paths:
                intro = (
                    f"以下为 `focus_select_keyframes` 在本轮（round {r}）输出的关键帧图像，"
                    "顺序与工具 JSON 中 `frames` 一致；请结合上一条 Critic 报告中的时间与索引信息阅读画面。"
                )
                parts: list[dict[str, Any]] = [{"type": "text", "text": intro}]
                for fp in focus_injected_paths:
                    try:
                        parts.append(image_content_part(fp))
                    except OSError:
                        continue
                if len(parts) > 1:
                    research.append_focus_keyframe_injection(parts, round_index=r)

        if out_dir:
            round_record: dict[str, Any] = {
                "round": r,
                "termination": None,
                "research_assistant_raw": assistant,
                "parsed_tool_calls": [asdict(x) for x in parsed],
                "tool_runs": [asdict(x) for x in tool_runs],
                "critic_user_payload": critic_user_payload,
                "critic_assistant": feedback,
                "research_feedback": feedback,
                "focus_injected_image_paths": focus_injected_paths,
                "research_messages_redacted": research.snapshot_redacted(),
                "critic_messages_redacted": _critic_trace(critic_agent),
            }
            if r == 1 and round1_nudge_applied:
                round_record["round1_no_tools_nudge_applied"] = True
                round_record["research_assistant_raw_first_attempt"] = assistant_first_attempt
                round_record["round1_no_tools_user_text"] = (
                    (round1_no_tools_feedback or "").strip() or DEFAULT_ROUND1_NO_TOOLS_NUDGE
                )
            _write_round(out_dir, r, round_record)

    return LoopResult(
        termination=termination,
        rounds=max_rounds,
        last_assistant=last_assistant,
        research_messages_redacted=research.snapshot_redacted(),
    )
