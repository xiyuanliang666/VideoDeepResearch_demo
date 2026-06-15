"""Core 5-stage pipeline orchestrator for rsagent-v4."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.rsagent.llm_client import LLMClient, image_content_part
from src.rsagent.main_context import MainContext
from src.rsagent.parsing import parse_tool_calls_from_assistant
from src.rsagent.protocols import ToolDispatcher
from src.rsagent.relevance import evaluate_relevance
from src.rsagent.research_context import ResearchContext
from src.rsagent.sub_question_context import SubQuestionContext
from src.rsagent.types import CoordinatorAction, SubQuestion, ToolRun


def run_pipeline(
    *,
    video_path: str,
    question: str,
    llm: LLMClient,
    dispatcher: ToolDispatcher,
    system_prompt: str,
    planning_prompt: str,
    evidence_loop_prompt: str,
    binding_prompt: str,
    final_answer_prompt: str,
    overview_frames: list[dict[str, Any]],
    preprocessing_result: dict[str, Any] | None = None,
    max_rounds: int = 12,
    max_inner_rounds: int = 5,
    out_dir: Path | None = None,
    coordinator_prompt: str = "",
    worker_prompt: str = "",
) -> dict[str, Any]:
    """Run the full 5-stage pipeline. Returns final result dict."""

    trace: dict[str, Any] = {"stages": {}}

    # --- Stage 2: Planning ---
    sub_questions = _stage_planning(
        llm=llm, question=question, system_prompt=system_prompt,
        planning_prompt=planning_prompt, overview_frames=overview_frames,
        preprocessing_result=preprocessing_result, trace=trace, out_dir=out_dir,
    )

    # --- Stage 3: Evidence Loop (dual context) ---
    main_ctx, sq_contexts = _stage_evidence_loop(
        llm=llm, question=question,
        coordinator_prompt=coordinator_prompt or system_prompt,
        worker_prompt=worker_prompt or system_prompt,
        sub_questions=sub_questions, dispatcher=dispatcher,
        overview_frames=overview_frames, preprocessing_result=preprocessing_result,
        max_rounds=max_rounds, max_inner_rounds=max_inner_rounds,
        trace=trace, out_dir=out_dir,
    )

    # --- Stage 4: Evidence Binding + Sub-question Answering ---
    binding_result = _stage_binding(
        llm=llm, question=question, system_prompt=system_prompt,
        binding_prompt=binding_prompt, sub_questions=main_ctx.sub_questions,
        sq_contexts=sq_contexts, trace=trace, out_dir=out_dir,
    )

    # --- Stage 5: Final Answer ---
    final_answer = _stage_final_answer(
        llm=llm, question=question, system_prompt=system_prompt,
        final_answer_prompt=final_answer_prompt,
        binding_result=binding_result, trace=trace, out_dir=out_dir,
    )

    result = {
        "question": question,
        "sub_questions": [asdict(sq) for sq in main_ctx.sub_questions],
        "final_answer": final_answer,
        "trace": trace,
    }
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------

def _stage_planning(
    *, llm: LLMClient, question: str, system_prompt: str, planning_prompt: str,
    overview_frames: list[dict[str, Any]], preprocessing_result: dict[str, Any] | None,
    trace: dict[str, Any], out_dir: Path | None,
) -> list[SubQuestion]:
    """Stage 2: decompose question into sub-questions with evidence plans."""
    ctx = ResearchContext(system=system_prompt)

    parts: list[dict[str, Any]] = []
    for frame in overview_frames:
        parts.append(image_content_part(frame["file"]))

    context_text = ""
    if preprocessing_result:
        context_text = f"## Video Preprocessing Result\n```json\n{json.dumps(preprocessing_result, ensure_ascii=False, indent=2)}\n```\n\n"

    user_text = f"{context_text}{planning_prompt}\n\n## Question\n{question}"
    parts.append({"type": "text", "text": user_text})
    ctx.append_user(parts)

    response = llm.complete(ctx.messages())
    sub_questions = _parse_sub_questions(response)

    trace["stages"]["planning"] = {"response": response, "sub_questions": [asdict(sq) for sq in sub_questions]}
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "planning.json").write_text(json.dumps(trace["stages"]["planning"], ensure_ascii=False, indent=2), encoding="utf-8")

    return sub_questions


def _stage_evidence_loop(
    *, llm: LLMClient, question: str, coordinator_prompt: str, worker_prompt: str,
    sub_questions: list[SubQuestion], dispatcher: ToolDispatcher,
    overview_frames: list[dict[str, Any]], preprocessing_result: dict[str, Any] | None,
    max_rounds: int, max_inner_rounds: int,
    trace: dict[str, Any], out_dir: Path | None,
) -> tuple[MainContext, dict[int, SubQuestionContext]]:
    """Stage 3: dual-context evidence collection loop with parallel worker support."""
    import concurrent.futures

    main_ctx = MainContext(system=coordinator_prompt, question=question, sub_questions=sub_questions)
    sq_contexts: dict[int, SubQuestionContext] = {
        i: SubQuestionContext(
            system=worker_prompt,
            sub_question=sq,
            overview_frames=overview_frames,
            preprocessing_result=preprocessing_result,
        )
        for i, sq in enumerate(sub_questions)
    }

    rounds_log: list[dict[str, Any]] = []
    sq_attempts: dict[int, int] = {i: 0 for i in range(len(sub_questions))}

    for global_round in range(1, max_rounds + 1):
        # Coordinator decides next action
        coord_response = llm.complete(main_ctx.build_messages())
        main_ctx.append_coordinator_response(coord_response)
        action = _parse_coordinator_action(coord_response)

        round_entry: dict[str, Any] = {"round": global_round, "coordinator_action": asdict(action)}

        if main_ctx.all_resolved():
            rounds_log.append(round_entry)
            break

        if action.action == "all_resolved":
            main_ctx.inject_error("Cannot declare all_resolved until every sub-question is marked supported or insufficient in the status table.")
            rounds_log.append(round_entry)
            continue

        if action.action == "adjust_plan":
            main_ctx.inject_plan_adjustment(action)
            for new_sq in main_ctx.sub_questions[len(sq_contexts):]:
                idx = len(sq_contexts)
                sq_contexts[idx] = SubQuestionContext(
                    system=worker_prompt,
                    sub_question=new_sq,
                    overview_frames=overview_frames,
                    preprocessing_result=preprocessing_result,
                )
                sq_attempts[idx] = 0
            rounds_log.append(round_entry)
            continue

        # Get task list (parallel tasks from coordinator)
        tasks = action.tasks or ([{"sq_id": action.sq_id or 0, "directive": action.directive}])
        # Filter valid tasks
        valid_tasks = []
        for t in tasks:
            sid = t.get("sq_id", 0)
            if sid < 0 or sid >= len(sq_contexts):
                main_ctx.inject_error(f"Invalid sq_id={sid}. Valid range: 0-{len(sq_contexts)-1}.")
                continue
            if sq_contexts[sid].resolved:
                continue
            valid_tasks.append(t)

        if not valid_tasks:
            rounds_log.append(round_entry)
            continue

        def _run_worker(task: dict) -> dict:
            sid = task["sq_id"]
            directive = task.get("directive", "")
            sq_ctx = sq_contexts[sid]

            sq_attempts[sid] = sq_attempts.get(sid, 0) + 1
            if sq_attempts[sid] > 3 and not sq_ctx.resolved:
                if sq_ctx.has_kept_evidence():
                    sq_ctx.mark_insufficient(
                        "Evidence: INSUFFICIENT_EVIDENCE: attempt limit reached before a supported answer.\n"
                        f"Kept evidence:\n{sq_ctx.kept_evidence_text()}\n"
                        "Answer: INSUFFICIENT_EVIDENCE\nConfidence: low"
                    )
                else:
                    sq_ctx.mark_insufficient(sq_ctx.export_summary() or "INSUFFICIENT_EVIDENCE: attempt limit reached without supported evidence")
                return {"sq_id": sid, "inner_rounds": [], "force_resolved": True, "sufficient": False}

            known_facts = main_ctx.known_facts_text(exclude_sq_id=sid)
            if known_facts:
                sq_ctx.inject_known_facts(known_facts)
            if directive:
                sq_ctx.inject_directive(directive)

            inner_logs = []
            for inner_round in range(1, max_inner_rounds + 1):
                worker_response = llm.complete(sq_ctx.messages())
                sq_ctx.append_assistant(worker_response)

                parsed = parse_tool_calls_from_assistant(worker_response)
                if not parsed:
                    summary = _extract_worker_summary(worker_response)
                    sufficient = _summary_has_supported_answer(summary)
                    if sufficient:
                        sq_ctx.mark_resolved(summary)
                    else:
                        sq_ctx.mark_insufficient(summary or "INSUFFICIENT_EVIDENCE: worker stopped without a supported answer")
                    inner_logs.append({"inner_round": inner_round, "resolved": True, "sufficient": sufficient})
                    break

                tool_runs: list[ToolRun] = []
                for pc in parsed:
                    if pc.parse_error or not pc.name:
                        tool_runs.append(ToolRun(name=pc.name or "_error", arguments=pc.arguments, raw_output=pc.parse_error or "", parse_error=pc.parse_error))
                        continue
                    try:
                        out = dispatcher.dispatch(pc.name, pc.arguments, round_index=global_round)
                    except Exception as exc:
                        out = json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
                    tool_runs.append(ToolRun(name=pc.name, arguments=pc.arguments, raw_output=out))

                feedback_parts = _build_tool_feedback(tool_runs)
                sq_ctx.append_tool_feedback(feedback_parts)

                verdicts = evaluate_relevance(llm, sq_ctx.sub_question, tool_runs)
                sq_ctx.apply_verdicts(verdicts, tool_runs)

                # Merge parsed tool calls with their raw outputs for trace completeness
                tools_with_output = []
                for ti, pc in enumerate(parsed):
                    entry = asdict(pc)
                    tr = tool_runs[ti] if ti < len(tool_runs) else None
                    entry["raw_output"] = tr.raw_output if tr else ""
                    tools_with_output.append(entry)
                inner_logs.append({"inner_round": inner_round, "tools": tools_with_output, "verdicts": [asdict(v) for v in verdicts]})

            if not sq_ctx.resolved and sq_ctx.has_kept_evidence():
                sq_ctx.inject_force_summary_request()
                worker_response = llm.complete(sq_ctx.messages())
                sq_ctx.append_assistant(worker_response)
                parsed = parse_tool_calls_from_assistant(worker_response)
                if parsed:
                    summary = (
                        "Evidence: INSUFFICIENT_EVIDENCE: worker tried to call tools after forced summary request.\n"
                        f"Kept evidence:\n{sq_ctx.kept_evidence_text()}\n"
                        "Answer: INSUFFICIENT_EVIDENCE\nConfidence: low"
                    )
                    sufficient = False
                else:
                    summary = _extract_worker_summary(worker_response)
                    sufficient = _summary_has_supported_answer(summary)
                if sufficient:
                    sq_ctx.mark_resolved(summary)
                else:
                    sq_ctx.mark_insufficient(summary or "INSUFFICIENT_EVIDENCE: forced summary did not contain a supported answer")
                inner_logs.append({"inner_round": max_inner_rounds + 1, "forced_summary": True, "resolved": True, "sufficient": sufficient})

            return {"sq_id": sid, "inner_rounds": inner_logs, "force_resolved": False}

        # Execute workers in parallel
        if len(valid_tasks) == 1:
            results = [_run_worker(valid_tasks[0])]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(valid_tasks)) as executor:
                results = list(executor.map(_run_worker, valid_tasks))

        # Feed results back to main context
        for res in results:
            sid = res["sq_id"]
            sq_ctx = sq_contexts[sid]
            if sq_ctx.resolved:
                main_ctx.inject_resolution(sid, sq_ctx.export_summary(), sufficient=sq_ctx.sub_question.sufficient)
            else:
                main_ctx.update_partial_progress(sid)

        round_entry["tasks"] = [{"sq_id": r["sq_id"], "inner_rounds": r["inner_rounds"]} for r in results]
        rounds_log.append(round_entry)

    trace["stages"]["evidence_loop"] = {"rounds": rounds_log, "total_rounds": len(rounds_log)}
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        edir = out_dir / "evidence_rounds"
        edir.mkdir(exist_ok=True)
        for rd in rounds_log:
            (edir / f"round_{rd['round']:03d}.json").write_text(json.dumps(rd, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    return main_ctx, sq_contexts


def _stage_binding(
    *, llm: LLMClient, question: str, system_prompt: str, binding_prompt: str,
    sub_questions: list[SubQuestion], sq_contexts: dict[int, SubQuestionContext],
    trace: dict[str, Any], out_dir: Path | None,
) -> str:
    """Stage 4: bind evidence to sub-questions and answer each."""
    ctx = ResearchContext(system=system_prompt)

    evidence_sections = []
    for i, sq in enumerate(sub_questions):
        summary = sq_contexts[i].export_summary() if i in sq_contexts else ""
        status = "SUPPORTED" if sq.sufficient else "INSUFFICIENT"
        evidence_sections.append(
            f"### Sub-Q{i+1}: {sq.text}\n"
            f"Status: {status}\n"
            f"Evidence summary:\n{summary or 'INSUFFICIENT_EVIDENCE: no evidence gathered'}"
        )

    evidence_text = "\n\n".join(evidence_sections)
    user_text = f"{binding_prompt}\n\n## Original Question\n{question}\n\n## Sub-questions with Evidence Status\n{evidence_text}\n\nBind evidence to each sub-question. Do not upgrade INSUFFICIENT sub-questions to supported answers."
    ctx.append_user(user_text)

    response = llm.complete(ctx.messages())
    if "<tool_call>" in response:
        response = _insufficient_binding_result(question, sub_questions, sq_contexts, "binding attempted to call tools after evidence collection")
    trace["stages"]["binding"] = {"response": response}
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "binding.json").write_text(json.dumps({"response": response}, ensure_ascii=False, indent=2), encoding="utf-8")
    return response


def _stage_final_answer(
    *, llm: LLMClient, question: str, system_prompt: str, final_answer_prompt: str,
    binding_result: str, trace: dict[str, Any], out_dir: Path | None,
) -> str:
    """Stage 5: synthesize final answer from sub-question answers."""
    ctx = ResearchContext(system=system_prompt)
    evidence_gate = ""
    if "INSUFFICIENT_EVIDENCE" in binding_result:
        evidence_gate = "\n\n## Evidence Gate\nAt least one required sub-question is INSUFFICIENT_EVIDENCE. The final answer must not be definitive; state that the original question cannot be determined from the gathered evidence."
    user_text = f"{final_answer_prompt}\n\n## Original Question\n{question}\n\n## Sub-question Answers\n{binding_result}{evidence_gate}\n\nSynthesize the final answer under these evidence constraints."
    ctx.append_user(user_text)

    if "INSUFFICIENT_EVIDENCE" in binding_result:
        response = _insufficient_final_answer(question, "one or more required sub-questions lacked discriminating evidence")
    else:
        response = llm.complete(ctx.messages())
        if "<tool_call>" in response:
            response = _insufficient_final_answer(question, "final synthesis attempted to call tools after evidence collection")
    trace["stages"]["final_answer"] = {"response": response}
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "final_answer.json").write_text(json.dumps({"response": response}, ensure_ascii=False, indent=2), encoding="utf-8")
    return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _insufficient_binding_result(
    question: str,
    sub_questions: list[SubQuestion],
    sq_contexts: dict[int, SubQuestionContext],
    reason: str,
) -> str:
    """Build a deterministic insufficient binding result when synthesis violates constraints."""
    sections = [f"Original question: {question}", f"Binding status: INSUFFICIENT_EVIDENCE ({reason})"]
    for i, sq in enumerate(sub_questions):
        summary = sq_contexts[i].export_summary() if i in sq_contexts else ""
        status = "SUPPORTED" if sq.sufficient else "INSUFFICIENT"
        answer = sq.answer if sq.sufficient and sq.answer else "INSUFFICIENT_EVIDENCE"
        sections.append(
            f"- Sub-question: {sq.text}\n"
            f"  Status: {status}\n"
            f"  Bound evidence: {summary or 'INSUFFICIENT_EVIDENCE: no evidence gathered'}\n"
            f"  Candidate check: INSUFFICIENT_EVIDENCE\n"
            f"  Answer: {answer}\n"
            f"  Confidence: {'medium' if sq.sufficient else 'low'}"
        )
    return "\n".join(sections)


def _insufficient_final_answer(question: str, reason: str) -> str:
    """Build a deterministic final answer when final synthesis violates constraints."""
    return (
        "Final answer: Cannot determine from the gathered evidence.\n"
        f"Evidence basis: INSUFFICIENT_EVIDENCE for the original question: {question}\n"
        f"Limitations: {reason}."
    )


def _parse_sub_questions(response: str) -> list[SubQuestion]:
    """Parse planning response into SubQuestion list."""
    try:
        start = response.find("[")
        end = response.rfind("]") + 1
        if start >= 0 and end > start:
            data = json.loads(response[start:end])
            if isinstance(data, list):
                return [
                    SubQuestion(
                        text=item.get("sub_question", item.get("text", "")),
                        visual_evidence_plan=item.get("visual_evidence_plan", ""),
                        web_evidence_plan=item.get("web_evidence_plan", ""),
                    )
                    for item in data if isinstance(item, dict)
                ]
    except (json.JSONDecodeError, ValueError):
        pass
    return [SubQuestion(text="Answer the full question directly")]


def _parse_coordinator_action(response: str) -> CoordinatorAction:
    """Parse coordinator JSON response into CoordinatorAction."""
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(response[start:end])
            tasks = data.get("tasks", [])
            # Backward compat: single sq_id → wrap in tasks list
            if not tasks and data.get("sq_id") is not None:
                tasks = [{"sq_id": data["sq_id"], "directive": data.get("directive", "")}]
            return CoordinatorAction(
                action=data.get("action", "work_on"),
                sq_id=data.get("sq_id"),
                directive=data.get("directive", ""),
                tasks=tasks,
                add_sub_questions=data.get("add", []),
                merge_ids=data.get("merge", []),
            )
    except (json.JSONDecodeError, ValueError):
        pass
    return CoordinatorAction(action="work_on", sq_id=0, tasks=[{"sq_id": 0, "directive": ""}])


def _extract_worker_summary(response: str) -> str:
    """Extract summary from worker response."""
    # Look for SUMMARY: marker
    marker = "SUMMARY:"
    idx = response.find(marker)
    if idx >= 0:
        return response[idx + len(marker):].strip()
    # Fallback: use last paragraph
    lines = response.strip().split("\n")
    return "\n".join(lines[-5:]) if lines else response


def _summary_has_supported_answer(summary: str) -> bool:
    """Return whether a worker summary contains a supported answer, not an abstention."""
    text = summary.strip().lower()
    if not text:
        return False
    negative_markers = (
        "insufficient_evidence",
        "insufficient evidence",
        "no evidence",
        "no relevant evidence",
        "cannot yet",
        "cannot determine",
        "cannot identify",
        "not enough evidence",
        "not identifiable",
        "unknown",
        "wait for the evidence",
        "please provide the results",
        "general knowledge",
        "not on direct evidence",
        "not directly verified",
        "confidence: low",
        "low confidence",
    )
    if any(marker in text for marker in negative_markers):
        return False
    return "answer:" in text and "evidence:" in text


def _build_tool_feedback(tool_runs: list[ToolRun]) -> list[dict[str, Any]]:
    """Build multimodal feedback parts from tool results."""
    parts: list[dict[str, Any]] = []
    text_results: list[str] = []

    for tr in tool_runs:
        if tr.name == "video_frame_extract" and not tr.parse_error:
            try:
                data = json.loads(tr.raw_output)
                if data.get("ok") and data.get("frames"):
                    text_results.append(f"[video_frame_extract] Extracted {data['count']} frames from {data['time_range'][0]:.1f}s-{data['time_range'][1]:.1f}s")
                    for frame in data["frames"]:
                        fpath = frame.get("file", "")
                        if fpath and Path(fpath).is_file():
                            parts.append(image_content_part(fpath))
                else:
                    text_results.append(f"[video_frame_extract] {data.get('error', 'failed')}")
            except (json.JSONDecodeError, KeyError):
                text_results.append(f"[video_frame_extract] Parse error: {tr.raw_output[:200]}")
        elif tr.name == "deep_research_web_search" and not tr.parse_error:
            try:
                data = json.loads(tr.raw_output)
                if data.get("ok"):
                    text_results.append(f"[web_search] query='{data.get('query', '')}' results:")
                    for worker in data.get("workers", []):
                        for res in worker.get("results", []):
                            text_results.append(f"  - [{worker.get('worker_id', '')}] {res.get('title', '')} | {res.get('snippet', '')[:120]} | {res.get('url', '')}")
                    if data.get("summary"):
                        text_results.append(f"  [LLM Summary] {data['summary'][:500]}")
                else:
                    text_results.append(f"[web_search] Error: {data.get('error', '')}")
            except json.JSONDecodeError:
                text_results.append("[web_search] Parse error")
        else:
            text_results.append(f"[{tr.name}] {tr.raw_output[:300]}")

    if text_results:
        parts.append({"type": "text", "text": "## Tool Results\n" + "\n".join(text_results)})
    return parts if parts else [{"type": "text", "text": "## Tool Results\nNo results."}]
