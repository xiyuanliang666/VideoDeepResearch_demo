"""Compact method-first pipeline."""

from __future__ import annotations

from src.core.anchors import build_anchors
from src.core.grounding import bind_web_candidates
from src.core.observation import build_event_observations, build_low_risk_observations
from src.core.preprocessing import build_dense_observations, build_observation_candidates
from src.core.query_planner import slots_from_intents
from src.core.reasoning import build_final_answer, build_judge_result
from src.core.retrieval import run_local_retrieval, run_web_retrieval
from src.core.store import build_evidence_store
from src.schemas import Anchor, QueryUnit, Sample, TemporalState, ToolCallRecord, utc_now_iso
from src.videodeepresearch.ablation import normalize_ablation_mode, uses_video, uses_web


def _question_query(sample: Sample, *, query_id: str = "question_query_0000") -> QueryUnit:
    return QueryUnit(
        query_id=query_id,
        anchor_id="question",
        query_text=sample.question,
        query_type="question_only",
        step_index=0,
        motivation="Use only the question text for this ablation setting.",
        query_source="question",
    )


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


def run_pipeline(sample: Sample, task_profile: dict, model_profile: dict) -> dict:
    """Execute the compact VideoDR-first method core."""
    topk_local = int(task_profile.get("topk_local", 5))
    topk_web = int(task_profile.get("topk_web", 5))
    max_steps = int(task_profile.get("max_steps", 3))
    ablation_mode = normalize_ablation_mode(task_profile)
    prompt_group = "video" if (sample.input_type or "").strip().lower() == "video" else "image"

    clips = []
    observations = []
    transcript_payload = {"audio_path": "", "segments": [], "aligned_by_clip": {}}
    local_query = _question_query(sample, query_id="local_query_text_only_0000")
    local_candidates = []
    anchors: list[Anchor] = []
    state_anchor_payloads: list[dict] = []

    states: list[TemporalState] = []
    raw_frames: list = []

    if uses_video(ablation_mode):
        enable_state_proposal = bool(task_profile.get("enable_state_proposal", False))

        if enable_state_proposal and sample.input_type == "video":
            primary_media = sample.media_paths[0] if sample.media_paths else ""
            raw_frames, observations = build_dense_observations(
                primary_media, sample.sample_id, task_profile
            )
            observations = build_event_observations(
                observations,
                sample.question,
                prompt_group=prompt_group,
                task_profile=task_profile,
                model_profile=model_profile,
            )
            # Propose temporal states from event observations
            from src.core.state_constructor import propose_states
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
                question=sample.question,
                observations=obs_dicts,
                model_profile=model_profile,
                max_states=int(task_profile.get("max_states", 6)),
            )
            # Select supporting frames and extract anchors per state
            if states and raw_frames:
                from src.core.anchor_extractor import extract_anchors_for_states
                from src.core.frame_selector import select_frames_for_states
                state_frames = select_frames_for_states(
                    states=states,
                    all_frames=raw_frames,
                    top_k=int(task_profile.get("max_images_per_multimodal_call", 4)),
                )
                state_anchor_payloads = extract_anchors_for_states(
                    question=sample.question,
                    states=states,
                    state_frames=state_frames,
                    model_profile=model_profile,
                )
                # Propagate search_queries into observation metadata for build_anchors
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
                observations,
                sample.question,
                prompt_group=prompt_group,
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
    elif ablation_mode == "web_only":
        anchors = [_question_anchor(sample)]

    observation_by_clip = {
        observation.metadata.get("clip_id", observation.observation_id): observation
        for observation in observations
    }
    web_queries = []
    web_candidates = []
    evidences = []
    bindings = []
    tool_trace: list[ToolCallRecord] = []

    hypotheses: list = []
    intents: list = []
    retrieval_memory: dict = {}
    coverage_report: dict = {}
    frame_coverage_report: dict = {}

    if uses_web(ablation_mode):
        enable_phase_b = bool(task_profile.get("enable_state_proposal", False)) and states
        if enable_phase_b:
            from src.core.coverage_controller import (
                check_evidence_coverage,
                check_state_frame_coverage,
                merge_coverage_reports,
            )
            from src.core.hypothesis_generator import generate_hypotheses
            from src.core.query_planner import plan_retrieval_intents
            from src.core.retrieval_manager import run_retrieval_for_intents
            from src.core.evidence_binder import bind_evidence_to_states

            frame_coverage_report = check_state_frame_coverage(
                states=states,
                state_frames={
                    state.state_id: [
                        {"frame_path": frame_path}
                        for frame_path in state.supporting_frame_paths
                    ]
                    for state in states
                },
            )
            intents = plan_retrieval_intents(
                question=sample.question,
                states=states,
                anchors=[a.to_dict() for a in anchors] + state_anchor_payloads,
                hypotheses=[],
                model_profile=model_profile,
                max_intents=int(task_profile.get("retrieval_budget", 5)),
            )
            web_queries, web_candidates, retrieval_memory = run_retrieval_for_intents(
                intents,
                topk=topk_web,
                max_rounds=2,
                return_memory=True,
            )
            hypotheses = generate_hypotheses(
                question=sample.question,
                states=states,
                anchors=[a.to_dict() for a in anchors] + state_anchor_payloads,
                web_candidates=web_candidates,
                model_profile=model_profile,
            )
            evidences, bindings = bind_evidence_to_states(
                question=sample.question,
                states=states,
                hypotheses=hypotheses,
                web_candidates=web_candidates,
                model_profile=model_profile,
            )
            # Phase C: score hypotheses
            from src.core.candidate_evaluator import evaluate_candidates
            evaluate_candidates(
                question=sample.question,
                hypotheses=hypotheses,
                evidences=evidences,
                bindings=bindings,
                model_profile=model_profile,
            )
            evidence_coverage_report = check_evidence_coverage(
                states=states,
                intents=intents,
                evidences=evidences,
                bindings=bindings,
                required_support_targets=task_profile.get("required_support_targets") or [],
            )
            coverage_report = merge_coverage_reports(
                frame_coverage_report,
                evidence_coverage_report,
            )
            tool_trace.extend([
                ToolCallRecord(
                    tool_call_id=f"tool_web_search_{i:04d}",
                    tool_name="search_web",
                    tool_input={"query": q.query_text, "intent_id": q.anchor_id},
                    tool_output_ref=q.query_id,
                    start_time=utc_now_iso(),
                    end_time=utc_now_iso(),
                    status="completed",
                )
                for i, q in enumerate(web_queries)
            ])
        else:
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
                            "ablation_mode": ablation_mode,
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

    state_anchor_store_rows = [
        _state_to_anchor_row(state)
        for state in states
    ]
    evidence_store = build_evidence_store(
        store_id=f"store_{sample.sample_id}",
        observations=[obs.to_dict() for obs in observations],
        anchors=[anchor.to_dict() for anchor in anchors] + state_anchor_store_rows,
        queries=[local_query.to_dict()] + [item.to_dict() for item in web_queries],
        retrieval_candidates=[candidate.to_dict() for candidate in local_candidates]
        + [candidate.to_dict() for candidate in web_candidates],
        evidences=[evidence.to_dict() for evidence in evidences],
        bindings=[binding.to_dict() for binding in bindings],
        claims=[_hypothesis_to_claim(hypothesis) for hypothesis in hypotheses],
        open_questions=["Need stronger grounded support." if not bindings and uses_web(ablation_mode) else ""],
        step_summaries=[
            f"Ablation mode: {ablation_mode}.",
            f"Built {len(observations)} observations.",
            f"Retrieved {len(local_candidates)} local candidates and {len(anchors)} anchors.",
            f"Collected {len(web_candidates)} web candidates and {len(bindings)} bindings.",
            f"Coverage status: {coverage_report.get('status', 'not_run')}.",
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
            "ablation_mode": ablation_mode,
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
        "execution_mode": "workflow",
        "ablation_mode": ablation_mode,
        "clips": [clip.to_dict() for clip in clips],
        "states": [s.to_dict() for s in states],
        "hypotheses": [h.to_dict() for h in hypotheses],
        "retrieval_slots": slots_from_intents(intents),
        "intents": [i.to_dict() for i in intents],
        "retrieval_memory": retrieval_memory,
        "retrieval_history": retrieval_memory.get("retrieval_history", []) if isinstance(retrieval_memory, dict) else [],
        "observations": [obs.to_dict() for obs in observations],
        "transcript": transcript_payload,
        "local_query": local_query.to_dict(),
        "local_candidates": [candidate.to_dict() for candidate in local_candidates],
        "anchors": [anchor.to_dict() for anchor in anchors],
        "state_anchors": state_anchor_payloads,
        "web_queries": [item.to_dict() for item in web_queries],
        "web_candidates": [candidate.to_dict() for candidate in web_candidates],
        "evidences": [evidence.to_dict() for evidence in evidences],
        "bindings": [binding.to_dict() for binding in bindings],
        "coverage_report": coverage_report,
        "evidence_store": evidence_store.to_dict(),
        "final_answer": final_answer.to_dict(),
        "judge_input": judge_input.to_dict(),
        "judge_result": judge_result.to_dict() if judge_result else None,
        "tool_trace": [record.to_dict() for record in tool_trace],
        "question": sample.question,
    }


def _state_to_anchor_row(state: TemporalState) -> dict:
    span = state.temporal_span if len(state.temporal_span) >= 2 else []
    return {
        "anchor_id": state.state_id,
        "source_clip_ids": [],
        "source_observation_ids": [],
        "time_span": span,
        "anchor_type": "temporal_state",
        "scene_summary": " ".join(
            part for part in [
                state.sub_question,
                state.sub_answer,
                "Predicates: " + "; ".join(state.required_predicates) if state.required_predicates else "",
            ]
            if part
        ),
        "search_queries": list(state.search_queries),
        "confidence": state.confidence,
        "status": "tentative",
        "priority_score": state.confidence,
        "open_slots": ["web_grounding"],
        "linked_state_ids": [state.state_id],
        "supporting_frame_paths": list(state.supporting_frame_paths),
        "visual_anchors": list(state.visual_anchors),
    }


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
