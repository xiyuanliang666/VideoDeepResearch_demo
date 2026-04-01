"""Agentic pipeline with LLM-driven search planning."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.core.anchors import build_anchors
from src.core.grounding import bind_web_candidates
from src.core.observation import build_low_risk_observations
from src.core.preprocessing import build_observation_candidates
from src.core.reasoning import build_final_answer, build_judge_result
from src.core.retrieval import run_local_retrieval, run_web_retrieval
from src.core.store import build_evidence_store
from src.schemas import Anchor, Sample, ToolCallRecord, utc_now_iso
from src.tools.env_tools import normalize_openai_base_url
from src.tools.model_router import get_model
from src.tools.prompt_router import load_prompt_pair


PROMPT_ROOT = Path(__file__).resolve().parents[2] / "prompts" / "agentic_planner"


def _extract_json_dict(text: str) -> dict[str, Any] | None:
    payload = text.strip()
    if not payload:
        return None
    try:
        data = json.loads(payload)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    left = payload.find("{")
    right = payload.rfind("}")
    if left < 0 or right <= left:
        return None
    try:
        data = json.loads(payload[left : right + 1])
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        return None
    return None


def _summarize_anchors(anchors: list[Anchor], *, limit: int = 8) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for anchor in anchors[:limit]:
        out.append(
            {
                "anchor_id": anchor.anchor_id,
                "time_span": anchor.time_span or [],
                "entities": anchor.entities[:6],
                "actions": anchor.actions[:6],
                "scene_summary": anchor.scene_summary[:160],
                "suggested_query": (anchor.search_queries[0] if anchor.search_queries else ""),
                "confidence": anchor.confidence,
            }
        )
    return out


def _anchor_with_query(anchor: Anchor, query_text: str) -> Anchor:
    payload = anchor.to_dict()
    payload["search_queries"] = [query_text] if query_text else list(anchor.search_queries)
    return Anchor(**payload)


def _call_planner_llm(
    *,
    question: str,
    benchmark_name: str,
    anchors: list[Anchor],
    searched_anchor_ids: list[str],
    previous_queries: list[str],
    evidence_count: int,
    model_profile: dict | None,
) -> tuple[dict[str, Any] | None, str]:
    api_key = os.getenv("LLM_API_KEY", "").strip()
    base_url = normalize_openai_base_url(os.getenv("LLM_BASE_URL", ""))
    if not api_key or not base_url:
        return None, "missing LLM_API_KEY or LLM_BASE_URL"

    try:
        from openai import OpenAI
    except ModuleNotFoundError:
        return None, "openai package is not installed"

    system_prompt, user_template = load_prompt_pair(
        prompt_root=PROMPT_ROOT,
        benchmark_name=benchmark_name,
        fallback_system=(
            "You are a research supervisor. Choose one action each turn: "
            "search or finalize. Return JSON only."
        ),
        fallback_user=(
            "Question: {question}\n"
            "Anchors: {anchors_json}\n"
            "Searched anchors: {searched_anchor_ids_json}\n"
            "Previous queries: {previous_queries_json}\n"
            "Evidence count: {evidence_count}\n\n"
            "Return JSON with action, anchor_id, query, reason."
        ),
    )

    user_prompt = (
        user_template.replace("{question}", question)
        .replace("{anchors_json}", json.dumps(_summarize_anchors(anchors), ensure_ascii=False))
        .replace("{searched_anchor_ids_json}", json.dumps(searched_anchor_ids, ensure_ascii=False))
        .replace("{previous_queries_json}", json.dumps(previous_queries[-8:], ensure_ascii=False))
        .replace("{evidence_count}", str(evidence_count))
    )

    timeout_seconds = float(os.getenv("TIMEOUT_SECONDS", "30") or 30)
    model_name = get_model("reasoning", model_profile)
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=400,
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"planner chat.completions failed: {type(exc).__name__}: {exc}"

    content = ""
    if getattr(response, "choices", None):
        message = response.choices[0].message
        content = getattr(message, "content", "") or ""
    parsed = _extract_json_dict(content)
    if parsed is None:
        return None, "planner response is not valid JSON object"
    return parsed, ""


def run_agentic_pipeline(sample: Sample, task_profile: dict, model_profile: dict) -> dict:
    """Execute agentic pipeline with LLM-guided query planning."""
    topk_local = int(task_profile.get("topk_local", 5))
    topk_web = int(task_profile.get("topk_web", 5))
    max_steps = int(task_profile.get("max_steps", 3))
    max_agent_iterations = int(task_profile.get("max_agent_iterations", max_steps))

    clips, observations, transcript_payload = build_observation_candidates(sample, task_profile)
    observations = build_low_risk_observations(
        observations,
        sample.question,
        benchmark_name=sample.benchmark_name,
        task_profile=task_profile,
        model_profile=model_profile,
    )

    local_query, local_candidates = run_local_retrieval(
        sample.question,
        observations,
        topk=topk_local,
        step_index=0,
    )
    anchors = build_anchors(
        observations=observations,
        local_candidates=local_candidates,
        question=sample.question,
    )

    anchor_by_id = {anchor.anchor_id: anchor for anchor in anchors}
    observation_by_clip = {
        observation.metadata.get("clip_id", observation.observation_id): observation
        for observation in observations
    }

    web_queries = []
    web_candidates = []
    evidences = []
    bindings = []
    tool_trace: list[ToolCallRecord] = []
    agent_trace: list[dict[str, Any]] = []
    searched_anchor_ids: list[str] = []
    previous_queries: list[str] = []
    planner_errors: list[str] = []

    for step_index in range(max_agent_iterations):
        decision, planner_error = _call_planner_llm(
            question=sample.question,
            benchmark_name=sample.benchmark_name,
            anchors=anchors,
            searched_anchor_ids=searched_anchor_ids,
            previous_queries=previous_queries,
            evidence_count=len(evidences),
            model_profile=model_profile,
        )
        if planner_error:
            planner_errors.append(planner_error)

        action = str((decision or {}).get("action", "")).strip().lower()
        anchor_id = str((decision or {}).get("anchor_id", "")).strip()
        query_text = str((decision or {}).get("query", "")).strip()
        reason = str((decision or {}).get("reason", "")).strip()

        if action not in {"search", "finalize"}:
            action = "search"
        if action == "finalize" and not tool_trace and anchors:
            action = "search"
            reason = (reason + " | ").strip(" |") + "forced initial search"

        if action == "finalize" or not anchors:
            agent_trace.append(
                {
                    "step_index": step_index,
                    "action": "finalize",
                    "anchor_id": anchor_id,
                    "query": query_text,
                    "reason": reason,
                    "planner_error": planner_error,
                    "evidence_count": len(evidences),
                }
            )
            break

        if anchor_id and anchor_id in anchor_by_id:
            selected_anchor = anchor_by_id[anchor_id]
        else:
            selected_anchor = anchors[step_index % len(anchors)]
            anchor_id = selected_anchor.anchor_id

        if not query_text:
            query_text = selected_anchor.search_queries[0] if selected_anchor.search_queries else sample.question

        working_anchor = _anchor_with_query(selected_anchor, query_text)
        web_query, step_web_candidates = run_web_retrieval(
            working_anchor,
            sample.question,
            topk=topk_web,
            step_index=step_index,
        )
        web_queries.append(web_query)
        web_candidates.extend(step_web_candidates)
        tool_trace.append(
            ToolCallRecord(
                tool_call_id=f"tool_agent_web_search_{step_index:04d}",
                tool_name="search_web",
                tool_input={
                    "query": web_query.query_text,
                    "anchor_id": anchor_id,
                    "topk": topk_web,
                    "planner_reason": reason,
                },
                tool_output_ref=web_query.query_id,
                start_time=utc_now_iso(),
                end_time=utc_now_iso(),
                status="completed",
            )
        )

        step_evidences, step_bindings = bind_web_candidates(
            selected_anchor,
            step_web_candidates,
            observation_by_clip,
        )
        evidences.extend(step_evidences)
        bindings.extend(step_bindings)
        searched_anchor_ids.append(anchor_id)
        previous_queries.append(web_query.query_text)
        agent_trace.append(
            {
                "step_index": step_index,
                "action": "search",
                "anchor_id": anchor_id,
                "query": web_query.query_text,
                "reason": reason,
                "planner_error": planner_error,
                "results_count": len(step_web_candidates),
                "evidence_count": len(evidences),
            }
        )

    if planner_errors:
        agent_trace.append(
            {
                "step_index": len(agent_trace),
                "action": "planner_diagnostics",
                "errors": planner_errors[-8:],
            }
        )

    evidence_store = build_evidence_store(
        store_id=f"store_{sample.sample_id}",
        observations=[obs.to_dict() for obs in observations],
        anchors=[anchor.to_dict() for anchor in anchors],
        queries=[local_query.to_dict()] + [item.to_dict() for item in web_queries],
        retrieval_candidates=[candidate.to_dict() for candidate in local_candidates]
        + [candidate.to_dict() for candidate in web_candidates],
        evidences=[evidence.to_dict() for evidence in evidences],
        bindings=[binding.to_dict() for binding in bindings],
        open_questions=["Need stronger grounded support." if not bindings else ""],
        step_summaries=[
            f"Built {len(observations)} observations.",
            f"Retrieved {len(local_candidates)} local candidates and {len(anchors)} anchors.",
            f"Agentic loop executed {len([x for x in agent_trace if x.get('action') == 'search'])} search steps.",
            f"Collected {len(web_candidates)} web candidates and {len(bindings)} bindings.",
        ],
    )
    final_answer = build_final_answer(
        sample.question,
        evidence_store,
        benchmark_name=sample.benchmark_name,
        model_profile=model_profile,
    )
    judge_input, judge_result = build_judge_result(
        task_input={
            "sample_id": sample.sample_id,
            "benchmark_name": sample.benchmark_name,
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
        "execution_mode": "agentic",
        "status": "answer_ready",
        "clips": [clip.to_dict() for clip in clips],
        "observations": [obs.to_dict() for obs in observations],
        "transcript": transcript_payload,
        "local_query": local_query.to_dict(),
        "local_candidates": [candidate.to_dict() for candidate in local_candidates],
        "anchors": [anchor.to_dict() for anchor in anchors],
        "web_queries": [item.to_dict() for item in web_queries],
        "web_candidates": [candidate.to_dict() for candidate in web_candidates],
        "evidences": [evidence.to_dict() for evidence in evidences],
        "bindings": [binding.to_dict() for binding in bindings],
        "evidence_store": evidence_store.to_dict(),
        "final_answer": final_answer.to_dict(),
        "judge_input": judge_input.to_dict(),
        "judge_result": judge_result.to_dict() if judge_result else None,
        "tool_trace": [record.to_dict() for record in tool_trace],
        "agent_trace": agent_trace,
        "question": sample.question,
    }

