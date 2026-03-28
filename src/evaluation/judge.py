"""LLM-as-a-Judge entrypoints."""

from __future__ import annotations

from src.schemas import JudgeResult


def run_judge(judge_input, model_name: str, prompt_version: str):
    """Run the configured judge over a bundled input.

    v0 uses a heuristic judge so the pipeline can produce a stable score without
    external model calls. The interface is intentionally compatible with a future
    LLM-backed judge.
    """
    model_output = judge_input.model_output or {}
    evidence_chain = judge_input.evidence_chain or []
    tool_trace = judge_input.tool_trace or []

    answer_text = str(model_output.get("final_answer", "")).strip()
    evidence_count = len(evidence_chain)
    tool_count = len(tool_trace)
    answer_present = 1.0 if answer_text else 0.0
    evidence_support = min(1.0, evidence_count / 3.0)
    trace_completeness = min(1.0, tool_count / 4.0)
    overall = round(0.4 * answer_present + 0.4 * evidence_support + 0.2 * trace_completeness, 4)

    if overall >= 0.75:
        verdict = "pass"
        explanation = "The run produced an answer with multiple evidence items and a reasonably complete trace."
        failure_tags: list[str] = []
    elif overall >= 0.4:
        verdict = "weak_pass"
        explanation = "The run is analyzable but evidence or trace completeness is still limited."
        failure_tags = ["limited_evidence"] if evidence_support < 0.67 else []
    else:
        verdict = "fail"
        explanation = "The run lacks enough grounded evidence or did not produce a usable answer."
        failure_tags = ["insufficient_answer", "insufficient_evidence"]

    return JudgeResult(
        judge_result_id=judge_input.judge_input_id.replace("judge_input", "judge_result"),
        judge_model=model_name,
        judge_prompt_version=prompt_version,
        task_type=judge_input.metadata.get("task_type", "vqa"),
        overall_score=overall,
        dimension_scores={
            "answer_present": answer_present,
            "evidence_support": evidence_support,
            "trace_completeness": trace_completeness,
        },
        verdict=verdict,
        explanation=explanation,
        failure_tags=failure_tags,
        raw_judge_output=(
            f"heuristic_judge(answer_present={answer_present}, "
            f"evidence_support={evidence_support}, trace_completeness={trace_completeness})"
        ),
    )
