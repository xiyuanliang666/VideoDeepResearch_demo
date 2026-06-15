"""Relevance evaluation: heuristic pre-filter + LLM judgment."""

from __future__ import annotations

import json
from typing import Any

from src.rsagent.llm_client import LLMClient
from src.rsagent.types import RelevanceVerdict, SubQuestion, ToolRun

_JUDGE_PROMPT = """Judge whether each tool result below is relevant to answering this sub-question:

"{sq_text}"

Tool results this round:
{results_text}

Output a JSON array with one entry per result:
[{{"index":1,"keep":true,"reason":"brief reason"}}, ...]

Rules:
- keep=true if the result directly supports, directly refutes, provides a discriminating fact, OR establishes a necessary intermediate entity for a later hop in this sub-question
- keep=false for broad background, loosely related candidates, name-only matches without a usable entity link, or facts about a different plausible candidate
- keep=false when the result cannot distinguish between competing candidates
- When in doubt, keep=false and explain what is missing
- Be concise in reasons"""


def evaluate_relevance(
    llm: LLMClient,
    sub_question: SubQuestion,
    tool_runs: list[ToolRun],
) -> list[RelevanceVerdict]:
    """Evaluate relevance of tool results: heuristic pre-filter then LLM judgment."""
    verdicts: list[RelevanceVerdict] = []
    needs_llm: list[tuple[int, ToolRun]] = []

    # Heuristic pre-filter
    for i, tr in enumerate(tool_runs):
        if tr.parse_error:
            verdicts.append(RelevanceVerdict(tool_run_index=i, keep=False, reason="parse error"))
            continue
        # video_frame_extract: always keep (judge cannot see images)
        if tr.name == "video_frame_extract":
            try:
                data = json.loads(tr.raw_output)
                if data.get("ok") and data.get("frames"):
                    verdicts.append(RelevanceVerdict(tool_run_index=i, keep=True, reason="visual evidence kept"))
                else:
                    verdicts.append(RelevanceVerdict(tool_run_index=i, keep=False, reason=data.get("error", "no frames")))
            except (json.JSONDecodeError, ValueError):
                verdicts.append(RelevanceVerdict(tool_run_index=i, keep=False, reason="invalid JSON"))
            continue
        try:
            data = json.loads(tr.raw_output)
        except (json.JSONDecodeError, ValueError):
            verdicts.append(RelevanceVerdict(tool_run_index=i, keep=False, reason="invalid JSON"))
            continue
        if not data.get("ok", False):
            verdicts.append(RelevanceVerdict(tool_run_index=i, keep=False, reason=data.get("error", "failed")))
            continue
        if data.get("retryable_error"):
            verdicts.append(RelevanceVerdict(tool_run_index=i, keep=False, reason=f"TOOL_ERROR_RETRYABLE: {data.get('error', 'web search failed')}"))
            continue
        needs_llm.append((i, tr))

    if not needs_llm:
        return verdicts

    # LLM judgment for remaining results
    results_text = ""
    for idx, (orig_i, tr) in enumerate(needs_llm, 1):
        brief = _brief_description(tr)
        results_text += f"[{idx}] {tr.name}: {brief}\n"

    prompt = _JUDGE_PROMPT.format(sq_text=sub_question.text, results_text=results_text)

    try:
        response = llm.complete([
            {"role": "system", "content": "You are an evidence relevance judge. Output only JSON."},
            {"role": "user", "content": prompt},
        ])
        llm_verdicts = _parse_llm_verdicts(response)
    except Exception:
        for orig_i, _ in needs_llm:
            verdicts.append(RelevanceVerdict(tool_run_index=orig_i, keep=False, reason="judge failed, default discard"))
        return verdicts

    for idx, (orig_i, _) in enumerate(needs_llm):
        if idx < len(llm_verdicts):
            v = llm_verdicts[idx]
            verdicts.append(RelevanceVerdict(tool_run_index=orig_i, keep=v.get("keep", True), reason=v.get("reason", "")))
        else:
            verdicts.append(RelevanceVerdict(tool_run_index=orig_i, keep=False, reason="missing judge verdict"))

    return verdicts


def _brief_description(tr: ToolRun) -> str:
    try:
        data = json.loads(tr.raw_output)
    except (json.JSONDecodeError, ValueError):
        return tr.raw_output[:150]
    if tr.name == "video_frame_extract":
        return f"{data.get('count', 0)} frames from {data.get('time_range', [0,0])}s"
    if tr.name == "deep_research_web_search":
        snippets = []
        for worker in data.get("workers", [])[:2]:
            for result in worker.get("results", [])[:2]:
                title = result.get("title", "")[:80]
                snippet = result.get("snippet", "")[:180]
                snippets.append(f"{title} | {snippet}")
        summary = data.get("summary", "")[:240]
        return f"query: {data.get('query', '')}; results: {' || '.join(snippets)}; summary: {summary}"
    return tr.raw_output[:150]


def _parse_llm_verdicts(response: str) -> list[dict[str, Any]]:
    start = response.find("[")
    end = response.rfind("]") + 1
    if start >= 0 and end > start:
        return json.loads(response[start:end])
    return []
