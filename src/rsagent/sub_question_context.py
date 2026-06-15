"""SubQuestionContext: isolated evidence-gathering context for one sub-question."""

from __future__ import annotations

import json
from typing import Any, List

from src.rsagent.llm_client import image_content_part
from src.rsagent.types import RelevanceVerdict, SubQuestion, ToolRun


class SubQuestionContext:
    """Per-sub-question conversation context with evidence filtering."""

    def __init__(
        self,
        *,
        system: str,
        sub_question: SubQuestion,
        overview_frames: list[dict[str, Any]],
        preprocessing_result: dict[str, Any] | None = None,
    ) -> None:
        self._sub_question = sub_question
        self._messages: List[dict[str, Any]] = [{"role": "system", "content": system}]
        self._resolved = False
        self._summary = ""
        self._round = 0
        self._kept_evidence: list[str] = []

        # Initial user message: overview frames + sub-question + plan
        parts: list[dict[str, Any]] = []
        for f in overview_frames[:6]:
            parts.append(image_content_part(f["file"]))
        text = f"## Sub-question\n{sub_question.text}\n\n## Evidence Plan\nVisual: {sub_question.visual_evidence_plan}\nWeb: {sub_question.web_evidence_plan}"
        if preprocessing_result:
            text += "\n\n## Video Preprocessing Result\n" + _compact_preprocessing_result(preprocessing_result)
            text += "\nUse preprocessing entities, OCR text, or detector outputs only as auxiliary candidates; verify them with frame evidence before answering."
        parts.append({"type": "text", "text": text})
        self._messages.append({"role": "user", "content": parts})

    @property
    def sub_question(self) -> SubQuestion:
        return self._sub_question

    @property
    def resolved(self) -> bool:
        return self._resolved

    def messages(self) -> List[dict[str, Any]]:
        max_messages = 6
        if len(self._messages) <= max_messages:
            return self._messages
        compacted = self._messages[:2]
        dropped = len(self._messages) - max_messages
        compacted.append({"role": "user", "content": f"<compacted_history dropped_messages={dropped}; prior rounds contained non-final tool attempts and discarded or superseded evidence. Use the remaining recent evidence only. />"})
        compacted.extend(self._messages[-(max_messages - 3):])
        return compacted

    def append_assistant(self, content: str) -> None:
        self._messages.append({"role": "assistant", "content": content})

    def append_tool_feedback(self, parts: list[dict[str, Any]]) -> None:
        """Append tool results as a user message. Index stored for potential discard."""
        self._round += 1
        self._messages.append({"role": "user", "content": parts})

    def apply_verdicts(self, verdicts: list[RelevanceVerdict], tool_runs: list[ToolRun] | None = None) -> None:
        """Replace the last user message content based on verdicts and remember kept evidence.

        For tool results marked keep=False, replace their content portions with a
        one-line placeholder. Keeps images from kept results, discards images from
        rejected ones.
        """
        if tool_runs:
            self._record_kept_evidence(verdicts, tool_runs)
        if not verdicts or not self._messages:
            return
        last_msg = self._messages[-1]
        if last_msg["role"] != "user":
            return

        content = last_msg["content"]
        if not isinstance(content, list):
            # Pure text feedback - check if all should be discarded
            if all(not v.keep for v in verdicts):
                reasons = "; ".join(v.reason for v in verdicts if v.reason)
                last_msg["content"] = f"<discarded round={self._round} reason=\"{reasons}\" />"
            return

        # Multimodal content: decide which parts to keep
        all_discard = all(not v.keep for v in verdicts)
        if all_discard:
            reasons = "; ".join(v.reason for v in verdicts if v.reason)
            last_msg["content"] = [{"type": "text", "text": f"<discarded round={self._round} reason=\"{reasons}\" />"}]
        # If mixed (some keep, some discard), keep the message as-is
        # since images and text are interleaved and hard to split per-tool

    def has_kept_evidence(self) -> bool:
        return bool(self._kept_evidence)

    def kept_evidence_text(self, *, max_items: int = 8, max_chars: int = 3000) -> str:
        text = "\n".join(self._kept_evidence[-max_items:])
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...[truncated kept evidence]..."
        return text

    def inject_force_summary_request(self) -> None:
        evidence = self.kept_evidence_text()
        if not evidence:
            evidence = "No kept evidence has been recorded. Output an insufficient-evidence summary."
        self._messages.append({
            "role": "user",
            "content": (
                "## Forced Evidence Summary\n"
                "You have reached the tool-call limit for this sub-question. Do NOT call tools again.\n"
                "Use only the kept evidence below and output the required SUMMARY structure.\n"
                "If the kept evidence supports the sub-question, provide the answer; otherwise output INSUFFICIENT_EVIDENCE.\n\n"
                f"Kept evidence:\n{evidence}"
            ),
        })

    def _record_kept_evidence(self, verdicts: list[RelevanceVerdict], tool_runs: list[ToolRun]) -> None:
        for verdict in verdicts:
            if not verdict.keep:
                continue
            if verdict.tool_run_index < 0 or verdict.tool_run_index >= len(tool_runs):
                continue
            tr = tool_runs[verdict.tool_run_index]
            self._kept_evidence.append(_brief_kept_evidence(tr, verdict.reason))

    def inject_directive(self, directive: str) -> None:
        """Inject coordinator's directive as a user message."""
        self._messages.append({"role": "user", "content": f"[Coordinator] {directive}"})

    def inject_known_facts(self, known_facts: str) -> None:
        """Inject resolved facts from other sub-questions for entity-anchored searches."""
        if not known_facts.strip():
            return
        text = (
            "## Known Facts from other sub-questions\n"
            f"{known_facts.strip()}\n\n"
            "Use these canonical entities as search anchors. Do not replace them with pronouns or vague visual descriptions."
        )
        self._messages.append({"role": "user", "content": text})

    def mark_resolved(self, summary: str, *, sufficient: bool = True) -> None:
        self._resolved = True
        self._summary = summary
        self._sub_question.sufficient = sufficient
        self._sub_question.answer = summary

    def mark_insufficient(self, summary: str) -> None:
        self.mark_resolved(summary, sufficient=False)

    def export_summary(self) -> str:
        return self._summary


def _compact_preprocessing_result(preprocessing_result: dict[str, Any], *, max_chars: int = 2500) -> str:
    text = json.dumps(preprocessing_result, ensure_ascii=False, indent=2)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated preprocessing result]..."
    return text


def _brief_kept_evidence(tool_run: ToolRun, reason: str) -> str:
    try:
        data = json.loads(tool_run.raw_output)
    except (json.JSONDecodeError, ValueError):
        return f"- {tool_run.name}: {reason}; raw={tool_run.raw_output[:300]}"

    if tool_run.name == "video_frame_extract":
        frames = data.get("frames", []) if isinstance(data, dict) else []
        times = ", ".join(str(frame.get("time_sec", "")) for frame in frames[:6])
        return f"- video_frame_extract {data.get('time_range')}: kept visual frames at [{times}]; relevance={reason}"

    if tool_run.name == "deep_research_web_search":
        query = data.get("query", "") if isinstance(data, dict) else ""
        snippets: list[str] = []
        for worker in data.get("workers", [])[:2]:
            for result in worker.get("results", [])[:2]:
                title = (result.get("title") or "")[:100]
                snippet = (result.get("snippet") or "")[:220]
                url = result.get("url") or ""
                snippets.append(f"{title} | {snippet} | {url}")
        summary = (data.get("summary") or "")[:500]
        parts = [f"- web_search query='{query}': relevance={reason}"]
        if snippets:
            parts.append("  results: " + " || ".join(snippets))
        if summary:
            parts.append("  page_summary: " + summary)
        return "\n".join(parts)

    return f"- {tool_run.name}: {reason}; raw={tool_run.raw_output[:300]}"
