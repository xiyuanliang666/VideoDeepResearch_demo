"""
Deep-research Critic: FOCUS technical check + **Top-3 frame ranking** (metadata) for injection;
search snippet scoring + Jina Top-3 + LLM synthesis.

Implements CriticAgent like OpenAICriticAgent; complete_round may use multiple stateless LLM calls
and only appends the final feedback to CriticContext. Exposes ``focus_inject_paths`` for the loop.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from video_dr_agent.critic_context import CriticContext
from video_dr_agent.critic_prompts import (
    CRITIC_SYSTEM_PROMPT_DEEP,
    CRITIC_USER_FOCUS_TOP3_TEMPLATE,
    CRITIC_USER_INTEGRATE_TEMPLATE,
    CRITIC_USER_RANK_URLS_TEMPLATE,
)
from video_dr_agent.focus_keyframe_tool import FOCUS_SELECT_KEYFRAMES_TOOL_NAME
from video_dr_agent.jina_reader import jina_reader_fetch, truncate_for_llm
from video_dr_agent.protocols import CriticAgent, CriticPromptBuilder
from video_dr_agent.serper_web_search_tool import DEEP_RESEARCH_WEB_SEARCH_TOOL_NAME
from video_dr_agent.types import ToolRun


CRITIC_ENVELOPE_VERSION = 1


def _parse_json_from_llm(text: str) -> dict[str, Any] | None:
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```\s*$", "", t)
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            obj = json.loads(m.group())
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _focus_expected_count(arguments: dict[str, Any]) -> int | None:
    nk = arguments.get("num_keyframes")
    if nk is None:
        return None
    try:
        return int(nk)
    except (TypeError, ValueError):
        return None


def _evaluate_focus(raw_output: str, arguments: dict[str, Any]) -> tuple[bool, str]:
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        return False, f"JSON parse failed: {exc}"
    if not data.get("ok"):
        return False, str(data.get("error", "ok=false"))
    ts = data.get("times_sec")
    if not isinstance(ts, list) or len(ts) == 0:
        return False, "times_sec missing or empty"
    exp = _focus_expected_count(arguments)
    if exp is not None and len(ts) != exp:
        return (
            False,
            f"times_sec length {len(ts)} != expected num_keyframes={exp}",
        )
    return True, "success criteria met"


def _is_serper_row_error(row: dict[str, Any]) -> bool:
    url = (row.get("url") or "").strip()
    title = (row.get("title") or "").strip()
    snippet = (row.get("snippet") or "").strip()
    low = (title + snippet).lower()
    if not url:
        return True
    if title == "Search failed" or "search failed" in low:
        return True
    if row.get("source") == "error":
        return True
    return False


def _search_error_stats(raw_output: str) -> tuple[int, int, str]:
    """(error_count, total_count, note)"""
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        return 1, 1, f"Search JSON parse failed: {exc}"
    if not data.get("ok"):
        return 1, 1, str(data.get("error", "ok=false"))
    total = 0
    err = 0
    for w in data.get("workers", []) or []:
        for r in w.get("results", []) or []:
            if not isinstance(r, dict):
                continue
            total += 1
            if _is_serper_row_error(r):
                err += 1
    if total == 0:
        return 1, 1, "no search result rows"
    return err, total, ""


def _focus_query_from_arguments(arguments: dict[str, Any]) -> str:
    return (arguments.get("question") or arguments.get("query") or "").strip()


def _parse_focus_frame_rows(raw_output: str) -> tuple[list[dict[str, Any]], str]:
    """Parse FOCUS JSON; return frame dicts and a plain-text table for the critic."""
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError:
        return [], ""
    frames = data.get("frames")
    if not isinstance(frames, list):
        return [], ""
    rows: list[dict[str, Any]] = []
    lines: list[str] = []
    for rec in frames:
        if not isinstance(rec, dict):
            continue
        fp = (rec.get("file") or "").strip()
        if not fp:
            continue
        idx = rec.get("frame_index", "")
        ts = rec.get("time_sec", "")
        rows.append(rec)
        lines.append(f"path={fp}\tframe_index={idx}\ttime_sec={ts}")
    return rows, "\n".join(lines) if lines else ""


def _resolve_path_key(path_str: str) -> str:
    p = (path_str or "").strip()
    if not p:
        return ""
    try:
        return str(Path(p).resolve())
    except OSError:
        return p


def _fallback_top3_paths(rows: list[dict[str, Any]], *, max_n: int = 3) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for rec in rows:
        if len(out) >= max_n:
            break
        fp = (rec.get("file") or "").strip()
        if not fp:
            continue
        p = Path(fp)
        if not p.is_file():
            continue
        key = _resolve_path_key(fp)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(str(p.resolve()))
    return out


def _flatten_search_candidates(search_raw_outputs: list[str]) -> str:
    lines: list[str] = []
    idx = 0
    for raw in search_raw_outputs:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for w in data.get("workers", []) or []:
            wid = w.get("worker_id", "")
            label = w.get("label", "")
            for r in w.get("results", []) or []:
                if not isinstance(r, dict):
                    continue
                idx += 1
                url = (r.get("url") or "").strip()
                title = (r.get("title") or "").replace("\n", " ")[:300]
                snippet = (r.get("snippet") or "").replace("\n", " ")[:400]
                lines.append(
                    f"[{idx}] worker={wid} ({label}) url={url}\n"
                    f"    title={title}\n    snippet={snippet}"
                )
    return "\n".join(lines) if lines else "(no candidates)"


class DeepResearchCriticPromptBuilder(CriticPromptBuilder):
    """Build JSON envelope consumed by DeepResearchCriticAgent."""

    def build(
        self,
        *,
        round_index: int,
        research_assistant_text: str,
        tool_runs: Sequence[ToolRun],
        extra: Any = None,
    ) -> str:
        extra = extra or {}
        rq = (extra.get("research_question") or "").strip()
        envelope: dict[str, Any] = {
            "schema_version": CRITIC_ENVELOPE_VERSION,
            "research_question": rq,
            "round_index": round_index,
            "research_assistant_excerpt": (research_assistant_text or "")[:8000],
            "tool_runs": [
                {
                    "name": tr.name,
                    "arguments": dict(tr.arguments),
                    "raw_output": tr.raw_output,
                    "parse_error": tr.parse_error,
                }
                for tr in tool_runs
            ],
        }
        return json.dumps(envelope, ensure_ascii=False)


@dataclass
class DeepResearchCriticConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.2
    max_tokens_rank: int = 1200
    max_tokens_integrate: int = 4096
    search_failure_ratio_threshold: float = 0.5
    jina_max_chars_per_page: int = 24000
    max_tokens_focus_top3: int = 900


class DeepResearchCriticAgent(CriticAgent):
    def __init__(
        self,
        cfg: DeepResearchCriticConfig,
        critic_ctx: CriticContext,
        *,
        jina_api_key: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.ctx = critic_ctx
        self.jina_api_key = jina_api_key
        self.focus_inject_paths: list[str] = []
        from openai import OpenAI

        self._client = OpenAI(base_url=cfg.base_url.rstrip("/"), api_key=cfg.api_key)

    def _chat(self, system: str, user: str, *, max_tokens: int) -> str:
        r = self._client.chat.completions.create(
            model=self.cfg.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=self.cfg.temperature,
        )
        return (r.choices[0].message.content or "").strip()

    def _rank_focus_top3(
        self,
        research_question: str,
        focus_query: str,
        rows: list[dict[str, Any]],
        frames_table: str,
    ) -> tuple[list[tuple[str, str]], str]:
        """
        Ask the critic LLM to pick up to 3 frame paths. Returns
        ``(ordered list of (absolute_path, reason), raw_model_text)``.
        """
        if not rows or not frames_table.strip():
            return [], ""
        allowed: dict[str, str] = {}
        for rec in rows:
            fp = (rec.get("file") or "").strip()
            if not fp:
                continue
            p = Path(fp)
            if not p.is_file():
                continue
            abs_path = str(p.resolve())
            for key in {_resolve_path_key(fp), abs_path, fp}:
                if key:
                    allowed[key] = abs_path
        fq = focus_query or "(no query in tool arguments; infer from research question)"
        user = CRITIC_USER_FOCUS_TOP3_TEMPLATE.format(
            research_question=research_question,
            focus_query=fq,
            frames_table=frames_table,
        )
        raw_out = self._chat(
            CRITIC_SYSTEM_PROMPT_DEEP,
            user,
            max_tokens=self.cfg.max_tokens_focus_top3,
        )
        obj = _parse_json_from_llm(raw_out) or {}
        picked: list[tuple[str, str]] = []
        seen_abs: set[str] = set()
        top_list = obj.get("top3")
        if isinstance(top_list, list):
            for item in top_list:
                if len(picked) >= 3:
                    break
                if not isinstance(item, dict):
                    continue
                file_guess = (item.get("file") or "").strip()
                reason = (item.get("one_line_reason") or "").strip() or "selected"
                abs_path = allowed.get(file_guess) or allowed.get(
                    _resolve_path_key(file_guess)
                )
                if not abs_path:
                    try:
                        p2 = Path(file_guess)
                        if p2.is_file():
                            abs_path = str(p2.resolve())
                    except OSError:
                        abs_path = ""
                if not abs_path or abs_path in seen_abs:
                    continue
                seen_abs.add(abs_path)
                picked.append((abs_path, reason))
        if not picked:
            for fp in _fallback_top3_paths(rows, max_n=3):
                picked.append((fp, "fallback: first valid frames from tool output order"))
        return picked, raw_out

    def complete_round(self, user_payload: str) -> str:
        try:
            payload = json.loads(user_payload)
        except json.JSONDecodeError:
            fb = f"[critic_error] failed to parse critic input JSON: {user_payload[:500]}"
            self.ctx.append_assistant(fb)
            return fb

        self.focus_inject_paths = []

        rq = (payload.get("research_question") or "").strip() or "(research_question not provided)"
        runs: list[dict[str, Any]] = payload.get("tool_runs") or []
        rnd = int(payload.get("round_index") or 0)

        sections: list[str] = [f"## Critic report (round {rnd})"]

        focus_blocks: list[str] = []
        search_raw_list: list[str] = []

        for tr in runs:
            name = tr.get("name") or ""
            args = tr.get("arguments") or {}
            raw = tr.get("raw_output") or ""
            perr = tr.get("parse_error")

            if perr:
                sections.append(
                    f"### Tool `{name}`\n**Status**: failed (parse error)\n```\n{perr}\n```"
                )
                continue

            if name == FOCUS_SELECT_KEYFRAMES_TOOL_NAME:
                ok, reason = _evaluate_focus(raw, args)
                if not ok:
                    excerpt = raw[:4000] + ("..." if len(raw) > 4000 else "")
                    focus_blocks.append(
                        f"### FOCUS keyframes (`{name}`)\n"
                        f"**Verdict**: FAILED — {reason}\n"
                        f"**Raw excerpt**:\n```json\n{excerpt}\n```"
                    )
                    continue
                rows, table = _parse_focus_frame_rows(raw)
                if not rows or not table.strip():
                    focus_blocks.append(
                        f"### FOCUS keyframes (`{name}`)\n"
                        f"**Verdict**: SUCCESS — {reason}, but no frame rows to rank.\n"
                    )
                    continue
                pairs, rank_dbg = self._rank_focus_top3(
                    rq,
                    _focus_query_from_arguments(args),
                    rows,
                    table,
                )
                for abs_path, _r in pairs:
                    if abs_path not in self.focus_inject_paths:
                        self.focus_inject_paths.append(abs_path)
                lines = [
                    f"### FOCUS keyframes (`{name}`)\n"
                    f"**Verdict**: SUCCESS — {reason}\n"
                    "**Top frames for Research (≤3, also injected as images when enabled)**:\n"
                ]
                for i, (ap, rs) in enumerate(pairs, 1):
                    lines.append(f"{i}. `{ap}` — {rs}")
                if not pairs:
                    lines.append(
                        "\nNo on-disk frame files were available to rank or inject."
                    )
                    if rank_dbg:
                        lines.append(
                            "\n**Rank stage output (excerpt)**:\n```\n"
                            + rank_dbg[:1500]
                            + "\n```"
                        )
                focus_blocks.append("\n".join(lines))
            elif name == DEEP_RESEARCH_WEB_SEARCH_TOOL_NAME:
                search_raw_list.append(raw)
            else:
                sections.append(
                    f"### Tool `{name}`\n**Raw output**:\n```\n{raw[:4000]}\n```"
                )

        sections.extend(focus_blocks)

        if search_raw_list:
            sections.append(self._process_search(rq, search_raw_list))

        feedback = "\n\n".join(sections)
        self.ctx.append_assistant(feedback)
        return feedback

    def _process_search(self, research_question: str, raw_list: list[str]) -> str:
        agg_err = 0
        agg_tot = 0
        notes: list[str] = []
        for raw in raw_list:
            e, t, note = _search_error_stats(raw)
            agg_err += e
            agg_tot += t
            if note:
                notes.append(note)

        ratio = (agg_err / agg_tot) if agg_tot else 1.0
        thr = self.cfg.search_failure_ratio_threshold
        failed = ratio >= thr

        head = (
            f"### Web search (`{DEEP_RESEARCH_WEB_SEARCH_TOOL_NAME}`)\n"
            f"**Heuristic**: error-row ratio {ratio:.2%} (threshold {thr:.0%}) → "
            f"{'FAILED — skipping Jina and synthesis' if failed else 'SUCCESS — running Jina + synthesis'}\n"
        )
        if notes:
            head += "**Notes**: " + "; ".join(notes) + "\n"

        if failed:
            head += "**Raw JSON excerpt**:\n```json\n"
            head += raw_list[0][:6000] if raw_list else ""
            head += "\n```"
            return head

        candidates = _flatten_search_candidates(raw_list)
        rank_user = CRITIC_USER_RANK_URLS_TEMPLATE.format(
            research_question=research_question,
            candidates_block=candidates,
        )
        rank_out = self._chat(
            CRITIC_SYSTEM_PROMPT_DEEP,
            rank_user,
            max_tokens=self.cfg.max_tokens_rank,
        )
        rank_obj = _parse_json_from_llm(rank_out) or {}
        top3 = rank_obj.get("top3") or []
        urls: list[str] = []
        if isinstance(top3, list):
            for item in top3[:3]:
                if isinstance(item, dict):
                    u = (item.get("url") or "").strip()
                    if u.startswith("http"):
                        urls.append(u)

        if not urls:
            return (
                head
                + "**Rank stage returned no valid URLs**. Rank model output excerpt:\n```\n"
                + rank_out[:2000]
                + "\n```"
            )

        jina_blocks: list[str] = []
        for i, u in enumerate(urls, 1):
            got = jina_reader_fetch(u, api_key=self.jina_api_key)
            if got["ok"]:
                body = truncate_for_llm(
                    got["content"], max_chars=self.cfg.jina_max_chars_per_page
                )
                jina_blocks.append(
                    f"### Source {i}: {got['url']}\n{body}"
                )
            else:
                jina_blocks.append(
                    f"### Source {i}: {got['url']}\n**Fetch failed**: {got['error']}"
                )

        integ_user = CRITIC_USER_INTEGRATE_TEMPLATE.format(
            research_question=research_question,
            jina_blocks="\n\n".join(jina_blocks),
        )
        answer = self._chat(
            CRITIC_SYSTEM_PROMPT_DEEP,
            integ_user,
            max_tokens=self.cfg.max_tokens_integrate,
        )

        return (
            head
            + f"**Selected URLs (Top {len(urls)})**:\n"
            + "\n".join(f"- {u}" for u in urls)
            + "\n\n**Synthesized answer (from Jina bodies)**:\n"
            + answer
        )
