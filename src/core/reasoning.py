"""Compact reasoning utilities."""

from __future__ import annotations

from src.evaluation.judge import run_judge
from src.schemas import FinalAnswerBundle, JudgeInputBundle
from src.tools.reasoning_tools import get_last_reasoning_error, run_online_reasoning


def build_final_answer(question: str, evidence_store, model_profile: dict | None = None):
    online = run_online_reasoning(
        question=question,
        evidence_store=evidence_store.to_dict() if hasattr(evidence_store, "to_dict") else dict(evidence_store),
        model_profile=model_profile,
        max_tokens=800,
    )
    reasoning_error = get_last_reasoning_error()
    reasoning_trace_refs = [f"online_reasoning_error:{reasoning_error}"] if reasoning_error else []
    if isinstance(online, dict):
        final_answer_text = str(online.get("final_answer", "")).strip()
        if final_answer_text:
            try:
                confidence = float(online.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            video_refs = online.get("supporting_video_evidence", [])
            web_refs = online.get("supporting_web_evidence", [])
            if not isinstance(video_refs, list):
                video_refs = []
            if not isinstance(web_refs, list):
                web_refs = []
            return FinalAnswerBundle(
                task_id=evidence_store.store_id,
                question=question,
                final_answer=final_answer_text,
                confidence=max(0.0, min(1.0, confidence)),
                supporting_video_evidence=[str(x) for x in video_refs],
                supporting_web_evidence=[str(x) for x in web_refs],
                tool_trace_refs=[],
                reasoning_trace_refs=[],
            )

    anchors = evidence_store.anchors or []
    bindings = evidence_store.bindings or []
    evidences = {item["evidence_id"]: item for item in evidence_store.evidences}

    supporting_bindings = [binding for binding in bindings if binding.get("relation") == "supports"]
    selected_bindings = supporting_bindings or bindings[:3]

    video_support: list[str] = []
    web_support: list[str] = []
    rationale_parts: list[str] = []

    for binding in selected_bindings[:3]:
        anchor = next((item for item in anchors if item.get("anchor_id") == binding.get("anchor_id")), None)
        evidence = evidences.get(binding.get("evidence_id"), {})
        if anchor:
            clip_ids = anchor.get("source_clip_ids", [])
            time_span = anchor.get("time_span") or []
            if len(time_span) == 2:
                video_support.append(
                    f"{clip_ids[0] if clip_ids else anchor.get('anchor_id')}@{time_span[0]:.1f}-{time_span[1]:.1f}s"
                )
            else:
                video_support.append(clip_ids[0] if clip_ids else anchor.get("anchor_id", "unknown_anchor"))
            scene_summary = anchor.get("scene_summary", "")
            if scene_summary:
                rationale_parts.append(scene_summary[:160])
        if evidence:
            url = evidence.get("source_url") or evidence.get("source_ref")
            if url:
                web_support.append(url)
            summary = evidence.get("content_summary", "")
            if summary:
                rationale_parts.append(summary[:160])

    rationale_parts = [part for part in rationale_parts if part]
    if supporting_bindings:
        answer_text = (
            "Based on the currently retrieved video and web evidence, the best-supported answer is: "
            + " ".join(rationale_parts[:2])
        ).strip()
        confidence = min(0.85, 0.45 + 0.1 * len(supporting_bindings))
    elif rationale_parts:
        answer_text = (
            "The current evidence is weak but suggests the following answer direction: "
            + " ".join(rationale_parts[:2])
        ).strip()
        confidence = 0.35
    else:
        answer_text = (
            "I do not yet have enough grounded video-web evidence to answer confidently. "
            "The current pipeline reached preprocessing and retrieval but found no strong support."
        )
        confidence = 0.1

    return FinalAnswerBundle(
        task_id=evidence_store.store_id,
        question=question,
        final_answer=answer_text,
        confidence=confidence,
        supporting_video_evidence=video_support,
        supporting_web_evidence=web_support,
        tool_trace_refs=[],
        reasoning_trace_refs=reasoning_trace_refs,
    )


def build_judge_result(
    *,
    task_input: dict,
    final_answer,
    evidence_chain: list[dict],
    tool_trace: list[dict],
    task_profile: dict,
    model_profile: dict,
):
    judge_input = JudgeInputBundle(
        judge_input_id="judge_input_v0",
        task_input=task_input,
        model_output=final_answer.to_dict(),
        reasoning_trace=[],
        evidence_chain=evidence_chain,
        tool_trace=tool_trace,
        rubric={
            "groundedness": "Does the answer reflect retrieved video/web evidence?",
            "trace_completeness": "Does the run preserve analyzable intermediate state?",
        },
        metadata={"task_type": task_profile.get("output_mode", "short_answer")},
    )
    if not bool(task_profile.get("enable_llm_judge", False)):
        return judge_input, None
    judge_result = run_judge(
        judge_input=judge_input,
        model_name=model_profile.get("judge_model", "heuristic-judge"),
        prompt_version=task_profile.get("judge_prompt_version", "v0"),
    )
    return judge_input, judge_result
