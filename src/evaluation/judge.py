"""LLM-as-a-Judge entrypoints."""

from __future__ import annotations

from src.schemas import JudgeResult
from src.tools.judge_tools import get_last_judge_error, run_online_judge


def _heuristic_judge(judge_input, model_name: str, prompt_version: str):
    """Local fallback judge that is deterministic and network-free."""
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


def run_judge(judge_input, model_name: str, prompt_version: str, benchmark_name: str = ""):
    """Run judge with online LLM first, then fallback to heuristic.

    Set ENABLE_ONLINE_JUDGE=true in .env to enable provider-routed online judge.
    """
    online_payload = run_online_judge(
        judge_input=judge_input.to_dict() if hasattr(judge_input, "to_dict") else dict(judge_input),
        model_name=model_name,
        benchmark_name=benchmark_name,
        max_tokens=800,
    )
    judge_error = get_last_judge_error()
    if isinstance(online_payload, dict):
        try:
            overall = float(online_payload.get("overall_score", 0.0))
        except (TypeError, ValueError):
            overall = 0.0
        dimension_scores = online_payload.get("dimension_scores", {})
        if not isinstance(dimension_scores, dict):
            dimension_scores = {}
        failure_tags = online_payload.get("failure_tags", [])
        if not isinstance(failure_tags, list):
            failure_tags = []
        return JudgeResult(
            judge_result_id=judge_input.judge_input_id.replace("judge_input", "judge_result"),
            judge_model=model_name,
            judge_prompt_version=prompt_version,
            task_type=judge_input.metadata.get("task_type", "vqa"),
            overall_score=max(0.0, min(1.0, overall)),
            dimension_scores={str(k): float(v) for k, v in dimension_scores.items() if isinstance(v, (int, float))},
            verdict=str(online_payload.get("verdict", "") or ""),
            explanation=str(online_payload.get("explanation", "") or ""),
            failure_tags=[str(item) for item in failure_tags],
            raw_judge_output=json_dumps_safe(online_payload),
        )
    fallback = _heuristic_judge(judge_input, model_name, prompt_version)
    if judge_error:
        fallback.raw_judge_output = f"{fallback.raw_judge_output}; online_judge_error={judge_error}"
    return fallback


def json_dumps_safe(payload: dict) -> str:
    try:
        import json

        return json.dumps(payload, ensure_ascii=False)
    except Exception:
        return str(payload)
