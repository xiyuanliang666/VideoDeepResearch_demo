"""
Main research loop (Video DeepResearch plan).

Flow: optional **mandatory planning** (no tools) → **vLLM research** → parse ``<tool_call>`` →
**ToolDispatcher** (raw tool JSON does not go directly to research) → **CriticPromptBuilder** +
**CriticAgent** → append **critic text**; if the round includes successful ``focus_select_keyframes``,
append a **multimodal user** (keyframe ``image_url``) — paths prefer **Critic Top-3** when available.

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
    "[Host] Your last message contained no <tool_call> blocks. "
    "Execute the next step of your written Research Plan: issue at least one valid `<tool_call>` "
    "for the tool(s) that step declares (for example `deep_research_web_search` and/or "
    "`focus_select_keyframes` when your plan needs extra video evidence), then continue from the evidence. "
    "Do not end exploration without tools. Regenerate this turn with at least one valid <tool_call>."
)

DEFAULT_PLANNING_USER_MESSAGE = """[Host — Planning phase]

Before any tool calls in later turns, write a **detailed Research Plan** using the **uniform overview frames**, any **Host schedule text** appended below (timestamps / frame indices), and the **user task** already in context.

**Hard rule for this turn:** output **plain text only** — do **not** include any `<tool_call>` blocks.

Your plan **must** include:
1. **Goal restatement** and what counts as a complete answer.
2. **What you can already infer** from the uniform frames (and the schedule, if provided) vs **information gaps**.
3. **Ordered steps table**: for each step — sub-question, **intended tool** (`focus_select_keyframes` and/or `deep_research_web_search` when useful), **evidence you expect**, link to the main question.
4. **Video tooling:** If a step needs **finer-grained local evidence** than the overview strip, and **`focus_select_keyframes`** is enabled on the host, you may plan that step with a **short BLIP-style query draft** plus an **approximate time window** (from the overview schedule) to pass as `time_start_sec` / `time_end_sec` in execution. If the task is **purely external** and does not depend on this footage, state an explicit **reason** steps can rely on web-only evidence.
5. **Web steps**: separate, focused search steps with keyword ideas (not one giant query).
6. **Pre-final checklist**: what visual claims and which web sources must be satisfied before you stop calling tools.

Be concrete enough that the following execution rounds can follow this plan step by step."""

DEFAULT_EXECUTION_HANDOFF_USER_MESSAGE = """[Host — Execution phase]

Your Research Plan is recorded above. **Execution rules:**
- Each tool round: in your reasoning, state **which plan step** you are executing (e.g. Step 2).
- Use `<tool_call>` to carry out the plan **in order**. After critic feedback, you may **briefly revise** later steps if evidence demands it — say so at the start of the next assistant turn, then continue.
- **Use tools** until you have **enough evidence** for the user task (video and/or web, as your plan requires); do not stop after a single shallow step if the checklist is unfinished.
- The run ends only when you reply **with no `<tool_call>`** (final answer). Until the checklist in your plan is satisfied, keep issuing tools when more evidence is needed."""

DEFAULT_FOCUS_INJECTION_INTRO = (
    "Below are **Critic-selected** keyframe images from `focus_select_keyframes` for this loop round ({round_idx}). "
    "Read them together with the critic report above."
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


def _write_planning_record(run_dir: Path, record: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "planning.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


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
    """Collect local image paths from successful ``focus_select_keyframes`` JSON (dedupe, order preserved)."""
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


def _critic_focus_inject_paths(critic_agent: CriticAgent) -> list[str]:
    raw = getattr(critic_agent, "focus_inject_paths", None)
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for p in raw:
        if not isinstance(p, str):
            continue
        fp = p.strip()
        if not fp:
            continue
        try:
            pr = Path(fp).resolve()
        except OSError:
            continue
        if not pr.is_file():
            continue
        k = str(pr)
        if k in seen:
            continue
        seen.add(k)
        out.append(k)
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
    planning_phase_enabled: bool = True,
    planning_user_message: str | None = None,
    planning_user_suffix: str | None = None,
    execution_handoff_user_message: str | None = None,
) -> LoopResult:
    """
    Terminates when the research model produces no <tool_call> blocks, or max_rounds hit.

    When ``planning_phase_enabled`` (default True), runs one planning turn (no tools) plus a host
    handoff before the main tool loop. Does not consume a ``max_rounds`` slot.

    Critic feedback is appended as a text user message; when ``inject_focus_frames`` is True,
    paths from ``DeepResearchCriticAgent.focus_inject_paths`` are preferred for image injection;
    otherwise falls back to the first ``max_focus_images_per_round`` frames from tool JSON.

    ``reject_round1_no_tools``: if round 1 has no ``<tool_call>``, append a nudge and retry once.

    ``planning_user_suffix``: optional English text appended to the planning Host message (after the default
    or ``planning_user_message`` file body), e.g. uniform-overview timestamps for the research model.

    research_question_for_critic:
        Original research question for the critic; if None, extracted from the first user message.
    """
    out_dir = Path(run_dir) if run_dir else None
    termination = "max_rounds_reached"
    last_assistant = ""
    critic_rq = (research_question_for_critic or "").strip() or (
        extract_research_question_from_context(research)
    )

    if planning_phase_enabled:
        pu = (planning_user_message or "").strip() or DEFAULT_PLANNING_USER_MESSAGE
        suf = (planning_user_suffix or "").strip()
        if suf:
            pu = pu.rstrip() + "\n\n" + suf
        eh = (execution_handoff_user_message or "").strip() or DEFAULT_EXECUTION_HANDOFF_USER_MESSAGE
        research.append_user_content(pu)
        plan_assistant = research_client.complete(research.messages())
        last_assistant = plan_assistant
        research.append_assistant(plan_assistant)
        research.append_user_content(eh)
        if out_dir:
            rec_plan: dict[str, Any] = {
                "planning_user": pu,
                "assistant_plan": plan_assistant,
                "execution_handoff_user": eh,
            }
            if suf:
                rec_plan["planning_user_suffix_only"] = suf
            _write_planning_record(out_dir, rec_plan)

    for r in range(1, max_rounds + 1):
        round1_nudge_applied = False
        assistant_first_attempt = ""
        messages = research.messages()
        assistant = research_client.complete(messages)
        last_assistant = assistant
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
        tool_runs = _collect_tool_runs(
            parsed, tool_dispatcher, round_index=r
        )
        critic_user_payload = critic_builder.build(
            round_index=r,
            research_assistant_text=assistant,
            tool_runs=tool_runs,
            extra={"research_question": critic_rq},
        )
        feedback = critic_agent.complete_round(critic_user_payload)
        research.append_critic_feedback(feedback)

        focus_injected_paths: list[str] = []
        if inject_focus_frames and max_focus_images_per_round > 0:
            critic_paths = _critic_focus_inject_paths(critic_agent)
            if critic_paths:
                focus_injected_paths = critic_paths[:max_focus_images_per_round]
            else:
                fb_cap = min(3, max_focus_images_per_round)
                focus_injected_paths = _collect_focus_frame_paths(
                    tool_runs,
                    max_images=fb_cap if fb_cap > 0 else max_focus_images_per_round,
                )
            if focus_injected_paths:
                intro = DEFAULT_FOCUS_INJECTION_INTRO.format(round_idx=r)
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
