"""Compact method-first pipeline."""

from __future__ import annotations

from src.core.anchors import build_anchors
from src.core.grounding import bind_web_candidates
from src.core.observation import build_low_risk_observations
from src.core.preprocessing import build_observation_candidates
from src.core.reasoning import build_final_answer, build_judge_result
from src.core.retrieval import run_local_retrieval, run_web_retrieval
from src.core.store import build_evidence_store
from src.schemas import Sample, ToolCallRecord, utc_now_iso


def run_pipeline(sample: Sample, task_profile: dict, model_profile: dict) -> dict:
    """Execute the compact VideoDR-first method core."""
    topk_local = int(task_profile.get("topk_local", 5))
    topk_web = int(task_profile.get("topk_web", 5))
    max_steps = int(task_profile.get("max_steps", 3))

    clips, observations, transcript_payload = build_observation_candidates(sample, task_profile)
    observations = build_low_risk_observations(observations, sample.question)

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

    observation_by_clip = {
        observation.metadata.get("clip_id", observation.observation_id): observation
        for observation in observations
    }
    web_queries = []
    web_candidates = []
    evidences = []
    bindings = []
    tool_trace: list[ToolCallRecord] = []

    for step_index, anchor in enumerate(anchors[:max_steps]):
        web_query, anchor_web_candidates = run_web_retrieval(
            anchor,
            sample.question,
            topk=topk_web,
            step_index=step_index,
        )
        web_queries.append(web_query)
        web_candidates.extend(anchor_web_candidates)
        tool_trace.append(
            ToolCallRecord(
                tool_call_id=f"tool_web_search_{step_index:04d}",
                tool_name="search_web",
                tool_input={
                    "query": web_query.query_text,
                    "anchor_id": anchor.anchor_id,
                    "topk": topk_web,
                },
                tool_output_ref=web_query.query_id,
                start_time=utc_now_iso(),
                end_time=utc_now_iso(),
                status="completed",
            )
        )
        step_evidences, step_bindings = bind_web_candidates(
            anchor,
            anchor_web_candidates,
            observation_by_clip,
        )
        evidences.extend(step_evidences)
        bindings.extend(step_bindings)

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
            f"Collected {len(web_candidates)} web candidates and {len(bindings)} bindings.",
        ],
    )
    final_answer = build_final_answer(sample.question, evidence_store)
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
        "question": sample.question,
    }
