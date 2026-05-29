"""Unified agentic pipeline.

This is the first consolidated agentic runner. It uses the existing
preprocessing/observation/anchor/evidence-store components, but routes external
actions through the unified `<tool_call>` parser and ToolDispatcher layer.
"""

from __future__ import annotations

import json
import os
from typing import Any

from src.core.anchors import build_anchors
from src.core.grounding import bind_web_candidates
from src.core.observation import build_low_risk_observations
from src.core.preprocessing import build_observation_candidates
from src.core.query_planner import slots_from_intents
from src.core.reasoning import build_final_answer, build_judge_result
from src.core.retrieval import run_local_retrieval
from src.core.store import build_evidence_store
from src.schemas import Anchor, QueryUnit, RetrievalCandidate, Sample, ToolCallRecord, utc_now_iso
from src.videodeepresearch.agentic.tool_loop import run_tool_round
from src.videodeepresearch.agentic.types import ToolRun
from src.videodeepresearch.ablation import normalize_ablation_mode, uses_video, uses_web
from src.videodeepresearch.tools import (
    DEEP_RESEARCH_WEB_SEARCH_TOOL_NAME,
    FOCUS_SELECT_KEYFRAMES_TOOL_NAME,
    build_default_tool_dispatcher,
)


def run_unified_agentic_pipeline(sample: Sample, task_profile: dict, model_profile: dict) -> dict[str, Any]:
    """Execute the unified agentic mode.

    Current scope:
    - deterministic host planning over anchors;
    - VDR-style `<tool_call>` execution through unified dispatchers;
    - conversion of web search tool payloads into shared evidence objects;
    - final answer and judge via existing shared reasoning utilities.

    Future scope:
    - replace deterministic host planning with an LLM planning/research loop;
    - insert structured critic feedback and FOCUS image injection between rounds.
    """
    topk_local = int(task_profile.get("topk_local", 5))
    max_steps = int(task_profile.get("max_steps", 3))
    max_agent_iterations = int(task_profile.get("max_agent_iterations", max_steps))
    ablation_mode = normalize_ablation_mode(task_profile)
    prompt_group = "video" if (sample.input_type or "").strip().lower() == "video" else "image"

    clips = []
    observations = []
    transcript_payload = {"audio_path": "", "segments": [], "aligned_by_clip": {}}
    local_query = _question_query(sample)
    local_candidates = []
    anchors: list[Anchor] = []

    states: list = []
    hypotheses: list = []
    intents: list = []
    raw_frames: list = []
    enable_state_proposal = bool(task_profile.get("enable_state_proposal", False))

    if uses_video(ablation_mode):
        if enable_state_proposal and sample.input_type == "video":
            from src.core.preprocessing import build_dense_observations
            from src.core.observation import build_event_observations
            from src.core.state_constructor import propose_states
            primary_media = sample.media_paths[0] if sample.media_paths else ""
            raw_frames, observations = build_dense_observations(primary_media, sample.sample_id, task_profile)
            observations = build_event_observations(
                observations, sample.question,
                prompt_group=prompt_group,
                task_profile=task_profile, model_profile=model_profile,
            )
            obs_dicts = [
                {
                    "start_sec": obs.timestamp_start or 0.0,
                    "end_sec": obs.timestamp_end or 0.0,
                    "event_description": " ".join(obs.scene_clues),
                    "entities": obs.candidate_entities,
                    "actions": obs.candidate_actions,
                }
                for obs in observations
            ]
            states = propose_states(
                question=sample.question, observations=obs_dicts,
                model_profile=model_profile,
                max_states=int(task_profile.get("max_states", 6)),
            )
            if states and raw_frames:
                from src.core.anchor_extractor import extract_anchors_for_states
                from src.core.frame_selector import select_frames_for_states
                state_frames = select_frames_for_states(
                    states=states, all_frames=raw_frames,
                    top_k=int(task_profile.get("max_images_per_multimodal_call", 4)),
                )
                extract_anchors_for_states(
                    question=sample.question, states=states,
                    state_frames=state_frames, model_profile=model_profile,
                )
                for state in states:
                    for obs in observations:
                        span = state.temporal_span
                        if (
                            len(span) >= 2
                            and (obs.timestamp_start or 0.0) >= span[0]
                            and (obs.timestamp_end or 0.0) <= span[1]
                            and state.search_queries
                        ):
                            obs.metadata.setdefault("search_queries", state.search_queries)
        else:
            clips, observations, transcript_payload = build_observation_candidates(sample, task_profile)
            observations = build_low_risk_observations(
                observations, sample.question,
                prompt_group=prompt_group,
                task_profile=task_profile, model_profile=model_profile,
            )
        local_query, local_candidates = run_local_retrieval(
            sample.question, observations, topk=topk_local, step_index=0,
        )
        anchors = build_anchors(
            observations=observations, local_candidates=local_candidates, question=sample.question,
        )
    elif ablation_mode == "web_only":
        anchors = [_question_anchor(sample)]

    observation_by_clip = {
        observation.metadata.get("clip_id", observation.observation_id): observation
        for observation in observations
    }

    web_queries: list[QueryUnit] = []
    web_candidates: list[RetrievalCandidate] = []
    evidences = []
    bindings = []
    tool_trace: list[ToolCallRecord] = []
    agent_trace: list[dict[str, Any]] = []
    planner_errors: list[str] = []
    retrieval_memory: dict[str, Any] = {}

    if uses_web(ablation_mode):
        if enable_state_proposal and states:
            from src.core.hypothesis_generator import generate_hypotheses
            from src.core.query_planner import plan_retrieval_intents
            from src.core.retrieval_manager import run_retrieval_for_intents
            from src.core.evidence_binder import bind_evidence_to_states
            from src.core.candidate_evaluator import evaluate_candidates

            intents = plan_retrieval_intents(
                question=sample.question, states=states,
                anchors=[a.to_dict() for a in anchors], hypotheses=[],
                model_profile=model_profile,
                max_intents=int(task_profile.get("retrieval_budget", 5)),
            )
            web_queries, web_candidates, retrieval_memory = run_retrieval_for_intents(
                intents,
                topk=int(task_profile.get("topk_web", 5)),
                max_rounds=2,
                return_memory=True,
            )
            hypotheses = generate_hypotheses(
                question=sample.question,
                states=states,
                anchors=[a.to_dict() for a in anchors],
                web_candidates=web_candidates,
                model_profile=model_profile,
            )
            evidences, bindings = bind_evidence_to_states(
                question=sample.question, states=states, hypotheses=hypotheses,
                web_candidates=web_candidates, model_profile=model_profile,
            )
            evaluate_candidates(
                question=sample.question, hypotheses=hypotheses,
                evidences=evidences, bindings=bindings, model_profile=model_profile,
            )
            tool_trace.extend([
                ToolCallRecord(
                    tool_call_id=f"tool_web_search_{i:04d}",
                    tool_name="search_web",
                    tool_input={"query": q.query_text, "intent_id": q.anchor_id},
                    tool_output_ref=q.query_id,
                    start_time=utc_now_iso(), end_time=utc_now_iso(), status="completed",
                )
                for i, q in enumerate(web_queries)
            ])
            agent_trace.append({"mode": "structured_phase_b", "intent_count": len(intents)})
        else:
            dispatcher = build_default_tool_dispatcher(
                output_root=task_profile.get("tool_output_dir", "outputs/tool_runs"),
                default_video_path=(sample.media_paths[0] if sample.media_paths else None),
                focus_enabled=bool(task_profile.get("enable_focus_tool", False)),
                focus_device=str(task_profile.get("focus_device", "cuda:0")),
                serper_api_key=os.getenv("SERPER_API_KEY", ""),
                serper_results_per_worker=int(task_profile.get("serper_results_per_worker", 2)),
                evidence_domain=task_profile.get("evidence_domain") or sample.metadata.get("evidence_domain"),
            )
            for step_index, anchor in enumerate(anchors[:max_agent_iterations]):
                assistant_text = _build_host_tool_call_text(
                    anchor=anchor, question=sample.question,
                    step_index=step_index, task_profile=task_profile,
                )
                tool_runs = run_tool_round(
                    assistant_text=assistant_text, dispatcher=dispatcher, round_index=step_index + 1,
                )
                agent_trace.append({
                    "step_index": step_index,
                    "mode": "deterministic_host_plan",
                    "anchor_id": anchor.anchor_id,
                    "assistant_text": assistant_text,
                    "tool_runs": [_tool_run_to_dict(run) for run in tool_runs],
                })
                for run_index, tool_run in enumerate(tool_runs):
                    trace_id = f"tool_{step_index:04d}_{run_index:04d}_{tool_run.name}"
                    tool_trace.append(ToolCallRecord(
                        tool_call_id=trace_id, tool_name=tool_run.name,
                        tool_input=tool_run.arguments, tool_output_ref=trace_id,
                        start_time=utc_now_iso(), end_time=utc_now_iso(),
                        status="failed" if tool_run.parse_error else "completed",
                        error_message=tool_run.parse_error or "",
                    ))
                    if tool_run.name == DEEP_RESEARCH_WEB_SEARCH_TOOL_NAME and not tool_run.parse_error:
                        query_unit, candidates = _web_tool_run_to_candidates(
                            anchor=anchor, tool_run=tool_run, step_index=step_index,
                        )
                        web_queries.append(query_unit)
                        web_candidates.extend(candidates)
                        step_evidences, step_bindings = bind_web_candidates(
                            anchor, candidates, observation_by_clip,
                        )
                        evidences.extend(step_evidences)
                        bindings.extend(step_bindings)
                    elif tool_run.parse_error:
                        planner_errors.append(tool_run.parse_error)

    evidence_store = build_evidence_store(
        store_id=f"store_{sample.sample_id}",
        observations=[observation.to_dict() for observation in observations],
        anchors=[anchor.to_dict() for anchor in anchors],
        queries=[local_query.to_dict()] + [query.to_dict() for query in web_queries],
        retrieval_candidates=[candidate.to_dict() for candidate in local_candidates]
        + [candidate.to_dict() for candidate in web_candidates],
        evidences=[evidence.to_dict() for evidence in evidences],
        bindings=[binding.to_dict() for binding in bindings],
        claims=[_hypothesis_to_claim(hypothesis) for hypothesis in hypotheses],
        open_questions=["Agentic tool loop found no strong web evidence." if not bindings else ""],
        step_summaries=[
            f"Ablation mode: {ablation_mode}.",
            f"Built {len(observations)} observations.",
            f"Built {len(anchors)} anchors from local retrieval.",
            f"Ran {len(tool_trace)} agentic tool calls and collected {len(evidences)} web evidences.",
        ],
    )
    final_answer = build_final_answer(
        sample.question,
        evidence_store,
        prompt_group=prompt_group,
        model_profile=model_profile,
    )
    judge_input, judge_result = build_judge_result(
        task_input={
            "sample_id": sample.sample_id,
            "benchmark_name": sample.benchmark_name,
            "input_type": sample.input_type,
            "question": sample.question,
            "media_paths": sample.media_paths,
            "reference_answer": sample.reference_answer,
            "metadata": sample.metadata,
        },
        final_answer=final_answer,
        evidence_chain=evidence_store.evidences,
        tool_trace=[record.to_dict() for record in tool_trace],
        task_profile=task_profile,
        model_profile=model_profile,
    )

    return {
        "created_at": utc_now_iso(),
        "sample": sample.to_dict(),
        "task_profile": task_profile,
        "model_profile": model_profile,
        "status": "answer_ready",
        "execution_mode": "agentic",
        "ablation_mode": ablation_mode,
        "agentic_runner": "src.videodeepresearch.agentic.runner.run_unified_agentic_pipeline",
        "clips": [clip.to_dict() for clip in clips],
        "states": [s.to_dict() for s in states],
        "hypotheses": [h.to_dict() for h in hypotheses],
        "retrieval_slots": slots_from_intents(intents),
        "intents": [i.to_dict() for i in intents],
        "retrieval_memory": retrieval_memory,
        "retrieval_history": retrieval_memory.get("retrieval_history", []) if isinstance(retrieval_memory, dict) else [],
        "observations": [observation.to_dict() for observation in observations],
        "transcript": transcript_payload,
        "local_query": local_query.to_dict(),
        "local_candidates": [candidate.to_dict() for candidate in local_candidates],
        "anchors": [anchor.to_dict() for anchor in anchors],
        "web_queries": [query.to_dict() for query in web_queries],
        "web_candidates": [candidate.to_dict() for candidate in web_candidates],
        "evidences": [evidence.to_dict() for evidence in evidences],
        "bindings": [binding.to_dict() for binding in bindings],
        "evidence_store": evidence_store.to_dict(),
        "final_answer": final_answer.to_dict(),
        "judge_input": judge_input.to_dict(),
        "judge_result": judge_result.to_dict() if judge_result else None,
        "tool_trace": [record.to_dict() for record in tool_trace],
        "agent_trace": agent_trace,
        "planner_errors": planner_errors,
        "question": sample.question,
    }



def _question_query(sample: Sample) -> QueryUnit:
    return QueryUnit(
        query_id="agentic_question_query_0000",
        anchor_id="question",
        query_text=sample.question,
        query_type="question_only",
        step_index=0,
        motivation="Use only the question text for this ablation setting.",
        query_source="question",
    )


def _hypothesis_to_claim(hypothesis) -> dict:
    return {
        "claim_id": str(getattr(hypothesis, "hypothesis_id", "") or ""),
        "statement": str(getattr(hypothesis, "explanation", "") or ""),
        "supporting_bindings": list(getattr(hypothesis, "supporting_evidence_ids", []) or []),
        "status": str(getattr(hypothesis, "status", "pending") or "pending"),
        "confidence": float(getattr(hypothesis, "support_score", 0.0) or 0.0),
        "metadata": {
            "linked_state_ids": list(getattr(hypothesis, "linked_state_ids", []) or []),
            "refuting_evidence_ids": list(getattr(hypothesis, "refuting_evidence_ids", []) or []),
            "generation_source": str(getattr(hypothesis, "generation_source", "") or ""),
        },
    }


def _question_anchor(sample: Sample) -> Anchor:
    return Anchor(
        anchor_id="anchor_question_0000",
        source_clip_ids=[],
        source_observation_ids=[],
        time_span=None,
        anchor_type="question_only",
        scene_summary=sample.question,
        search_queries=[sample.question],
        confidence=0.1,
        status="tentative",
        priority_score=0.1,
        open_slots=["web_grounding"],
    )


def _build_host_tool_call_text(*, anchor: Anchor, question: str, step_index: int, task_profile: dict) -> str:
    query = _anchor_query(anchor, question)
    calls = [
        {
            "name": DEEP_RESEARCH_WEB_SEARCH_TOOL_NAME,
            "arguments": {
                "query": query,
                "evidence_domain": task_profile.get("evidence_domain", "general"),
            },
        }
    ]
    if bool(task_profile.get("enable_focus_tool", False)):
        start = None
        end = None
        if anchor.time_span and len(anchor.time_span) >= 2:
            start = max(0.0, float(anchor.time_span[0]) - 2.0)
            end = max(start, float(anchor.time_span[1]) + 2.0)
        calls.append(
            {
                "name": FOCUS_SELECT_KEYFRAMES_TOOL_NAME,
                "arguments": {
                    "question": anchor.scene_summary or question,
                    "frame_caption": f"anchor_{step_index:04d}",
                    "time_start_sec": start,
                    "time_end_sec": end,
                    "num_keyframes": int(task_profile.get("focus_num_keyframes", 6)),
                },
            }
        )
    return "\n".join(
        f"<tool_call>{json.dumps(call, ensure_ascii=False)}</tool_call>" for call in calls
    )


def _anchor_query(anchor: Anchor, question: str) -> str:
    if anchor.search_queries:
        return anchor.search_queries[0]
    parts = [question, anchor.scene_summary, " ".join(anchor.entities[:4])]
    return " ".join(part for part in parts if part).strip() or question


def _web_tool_run_to_candidates(*, anchor: Anchor, tool_run: ToolRun, step_index: int) -> tuple[QueryUnit, list[RetrievalCandidate]]:
    payload = _loads_tool_payload(tool_run.raw_output)
    query_text = str(payload.get("query") or tool_run.arguments.get("query") or tool_run.arguments.get("question") or "")
    query_unit = QueryUnit(
        query_id=f"agentic_web_query_{step_index:04d}_{anchor.anchor_id}",
        anchor_id=anchor.anchor_id,
        query_text=query_text,
        query_type="agentic_web_search",
        step_index=step_index,
        motivation="Agentic tool-call web search grounded by selected anchor.",
        query_source="agentic_tool_call",
    )
    candidates: list[RetrievalCandidate] = []
    if not payload.get("ok"):
        error = str(payload.get("error") or "tool returned ok=false")
        candidates.append(
            RetrievalCandidate(
                candidate_id=f"agentic_web_error_{step_index:04d}_0000",
                candidate_type="web_result",
                source_query_id=query_unit.query_id,
                source_ref="fallback://agentic_web_error",
                score=0.0,
                rank=1,
                metadata={
                    "title": "Agentic web search unavailable",
                    "snippet": error,
                    "url": "fallback://agentic_web_error",
                    "raw_tool_payload": payload,
                },
                retriever_name="agentic_tool_dispatcher",
                normalized_score=0.01,
                keep_label="weak_keep",
                keep_reason="Tool failed or unavailable; preserved for traceability.",
            )
        )
        return query_unit, candidates

    rank = 0
    for worker in payload.get("workers", []) or []:
        if not isinstance(worker, dict):
            continue
        for result in worker.get("results", []) or []:
            if not isinstance(result, dict):
                continue
            rank += 1
            url = str(result.get("url") or "")
            title = str(result.get("title") or "")
            snippet = str(result.get("snippet") or "")
            candidates.append(
                RetrievalCandidate(
                    candidate_id=f"agentic_web_{step_index:04d}_{rank:04d}",
                    candidate_type="web_result",
                    source_query_id=query_unit.query_id,
                    source_ref=url or f"agentic_web_result_{rank}",
                    score=0.0,
                    rank=rank,
                    metadata={
                        "title": title,
                        "snippet": snippet[:500],
                        "url": url,
                        "worker_id": worker.get("worker_id", ""),
                        "worker_label": worker.get("label", ""),
                        "search_profile_resolved": payload.get("search_profile_resolved", ""),
                    },
                    retriever_name="agentic_serper_web_search",
                    normalized_score=max(0.01, 1.0 - 0.05 * (rank - 1)),
                    keep_label="keep" if url else "weak_keep",
                    keep_reason="Returned by agentic deep_research_web_search.",
                )
            )
    return query_unit, candidates


def _loads_tool_payload(raw_output: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw_output)
    except json.JSONDecodeError:
        return {"ok": False, "error": raw_output[:500]}
    return payload if isinstance(payload, dict) else {"ok": False, "error": "tool output is not a JSON object"}


def _tool_run_to_dict(tool_run: ToolRun) -> dict[str, Any]:
    return {
        "name": tool_run.name,
        "arguments": tool_run.arguments,
        "raw_output": tool_run.raw_output,
        "parse_error": tool_run.parse_error,
    }
