"""MainContext: coordinator-level context for the evidence loop."""

from __future__ import annotations

from typing import Any, List

from src.rsagent.types import CoordinatorAction, SubQuestion


class MainContext:
    """High-level coordinator context: manages plan, status, and summaries. No images."""

    def __init__(self, *, system: str, question: str, sub_questions: list[SubQuestion]) -> None:
        self._system = system
        self._question = question
        self._sub_questions = list(sub_questions)
        self._statuses: list[str] = ["pending"] * len(sub_questions)  # pending|active|resolved|insufficient
        self._summaries: list[str] = [""] * len(sub_questions)
        self._history: List[dict[str, Any]] = []  # assistant/user turn pairs

    def build_messages(self) -> List[dict[str, Any]]:
        msgs: List[dict[str, Any]] = [{"role": "system", "content": self._system}]
        user_text = f"## Research Question\n{self._question}\n\n## Sub-questions\n{self._format_plan()}\n\n## Status\n{self._format_status_table()}"
        msgs.append({"role": "user", "content": user_text})
        msgs.extend(self._history)
        return msgs

    def append_coordinator_response(self, response: str) -> None:
        self._history.append({"role": "assistant", "content": response})

    def inject_resolution(self, sq_id: int, summary: str, *, sufficient: bool = True) -> None:
        self._statuses[sq_id] = "resolved" if sufficient else "insufficient"
        self._summaries[sq_id] = summary
        self._sub_questions[sq_id].sufficient = sufficient
        self._sub_questions[sq_id].answer = summary
        label = "RESOLVED" if sufficient else "INSUFFICIENT"
        update = f"## Sub-question {sq_id + 1} {label}\n{summary}\n\n## Updated Status\n{self._format_status_table()}"
        self._history.append({"role": "user", "content": update})

    def update_partial_progress(self, sq_id: int, note: str = "") -> None:
        self._statuses[sq_id] = "active"
        update = f"## Sub-question {sq_id + 1} progress\n{note or 'Needs more evidence.'}\n\n## Updated Status\n{self._format_status_table()}"
        self._history.append({"role": "user", "content": update})

    def inject_error(self, error_msg: str) -> None:
        """Inject an error message so the coordinator can correct its action."""
        self._history.append({"role": "user", "content": f"## ERROR\n{error_msg}"})

    def inject_plan_adjustment(self, action: CoordinatorAction) -> None:
        for sq_dict in action.add_sub_questions:
            new_sq = SubQuestion(
                text=sq_dict.get("text", ""),
                visual_evidence_plan=sq_dict.get("visual_plan", ""),
                web_evidence_plan=sq_dict.get("web_plan", ""),
            )
            self._sub_questions.append(new_sq)
            self._statuses.append("pending")
            self._summaries.append("")
        # Merge: mark merged sub-questions as resolved with redirect
        for mid in action.merge_ids:
            if 0 <= mid < len(self._statuses):
                self._statuses[mid] = "resolved"
                self._summaries[mid] = "(merged into another sub-question)"

    @property
    def sub_questions(self) -> list[SubQuestion]:
        return self._sub_questions

    def all_resolved(self) -> bool:
        return all(s in {"resolved", "insufficient"} for s in self._statuses)

    def known_facts_text(self, *, exclude_sq_id: int | None = None, max_chars: int = 3000) -> str:
        """Return resolved sub-question summaries for downstream workers."""
        chunks = []
        for i, (sq, status, summary) in enumerate(zip(self._sub_questions, self._statuses, self._summaries)):
            if i == exclude_sq_id or status != "resolved" or not summary:
                continue
            chunks.append(f"sq_id={i}: {sq.text}\n{summary.strip()}")
        text = "\n\n".join(chunks)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...[truncated known facts]..."
        return text

    def _format_plan(self) -> str:
        lines = []
        for i, sq in enumerate(self._sub_questions):
            lines.append(f"sq_id={i}: {sq.text}")
            if sq.visual_evidence_plan:
                lines.append(f"   Visual: {sq.visual_evidence_plan}")
            if sq.web_evidence_plan:
                lines.append(f"   Web: {sq.web_evidence_plan}")
        return "\n".join(lines)

    def _format_status_table(self) -> str:
        header = "| Status | ID | Question | Summary |\n|--------|-----|----------|----------|\n"
        rows = []
        for i, (sq, status, summary) in enumerate(zip(self._sub_questions, self._statuses, self._summaries)):
            mark = "✓" if status == "resolved" else ("!" if status == "insufficient" else ("→" if status == "active" else "✗"))
            s = summary[:80] if summary else ""
            rows.append(f"| {mark} | sq_id={i} | {sq.text[:50]} | {s} |")
        return header + "\n".join(rows)
