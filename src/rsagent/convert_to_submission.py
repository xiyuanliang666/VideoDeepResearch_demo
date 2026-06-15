#!/usr/bin/env python3
"""Convert rsagent-v4 result.json traces → VDR pilot submission JSON.

Supports both old traces (raw_output empty) and new traces (raw_output populated,
worker summaries with [src:tool_N] tags, binding with [src:R{}_sq{}_t{}] refs).

Usage:
  python convert_to_submission.py \
    --runs-dir latest_13 \
    --manifest ../../vdr_investigation/benchmark/pilot_v1/pilot_manifest_v0.json \
    --out /tmp/rsagent_v4_submission.json
"""

from __future__ import annotations

import argparse
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json(raw: str) -> dict[str, Any] | None:
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def _find_verdict(verdicts: list[dict[str, Any]], tool_index: int) -> dict[str, Any] | None:
    for v in verdicts:
        if isinstance(v, dict) and v.get("tool_run_index") == tool_index:
            return v
    return None


def _get_rounds(result: dict[str, Any]) -> list[dict[str, Any]]:
    trace = result.get("trace") or {}
    if not isinstance(trace, dict):
        return []
    stages = trace.get("stages") or {}
    if not isinstance(stages, dict):
        return []
    eloop = stages.get("evidence_loop") or {}
    if not isinstance(eloop, dict):
        return []
    rounds = eloop.get("rounds") or []
    return rounds if isinstance(rounds, list) else []


def _extract_urls(text: str) -> list[str]:
    return re.findall(r"https?://[^\s)>\]\"',;]+", str(text or ""))


# ---------------------------------------------------------------------------
# P0 helpers: parse web results from raw_output
# ---------------------------------------------------------------------------

def _parse_web_results(raw_output: str) -> list[dict[str, str]]:
    """Parse {url, title, snippet} from a Serper raw_output JSON string."""
    results: list[dict[str, str]] = []
    data = _parse_json(raw_output)
    if data is None or not data.get("ok"):
        return results
    for worker in data.get("workers") or []:
        if not isinstance(worker, dict):
            continue
        for wr in worker.get("results") or []:
            if not isinstance(wr, dict):
                continue
            url = str(wr.get("url") or "").strip()
            if url:
                results.append({
                    "url": url,
                    "title": str(wr.get("title") or "")[:300],
                    "snippet": str(wr.get("snippet") or "")[:500],
                })
    return results


def _parse_web_summary(raw_output: str) -> str:
    """Extract LLM summary field from raw_output."""
    data = _parse_json(raw_output)
    if data is None:
        return ""
    return str(data.get("summary") or "")[:600]


# ---------------------------------------------------------------------------
# P2 helpers: parse structured evidence tags from worker SUMMARY
# ---------------------------------------------------------------------------

_SRC_TAG = re.compile(r"\[src:tool_(\d+)\]\s*\[target:(\S+)\]\s*\[relation:(\S+)\]")
_BINDING_SRC_TAG = re.compile(r"\[src:R(\d+)_sq(\d+)_t(\d+)\]")


def _parse_worker_evidence_tags(summary_text: str) -> list[dict[str, Any]]:
    """Extract evidence links from worker SUMMARY with [src:tool_N][target:...][relation:...] tags.

    Returns list of {tool_index, target, relation, description}.
    """
    links: list[dict[str, Any]] = []
    for m in _SRC_TAG.finditer(summary_text):
        links.append({
            "tool_index": int(m.group(1)),
            "target": m.group(2),
            "relation": m.group(3),
        })
    return links


def _parse_binding_source_tags(binding_text: str) -> list[dict[str, Any]]:
    """Extract source references from binding text with [src:R{r}_sq{s}_t{t}] tags.

    Returns list of {round, sq_id, tool_idx}.
    """
    refs: list[dict[str, Any]] = []
    for m in _BINDING_SRC_TAG.finditer(binding_text):
        refs.append({
            "round": int(m.group(1)),
            "sq_id": int(m.group(2)),
            "tool_idx": int(m.group(3)),
        })
    return refs


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

def _extract_video_evidence(result: dict[str, Any]) -> list[dict[str, Any]]:
    """From kept video_frame_extract tool calls in evidence_rounds."""
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()

    for rd_idx, rd in enumerate(_get_rounds(result)):
        for task in rd.get("tasks") or []:
            if not isinstance(task, dict):
                continue
            sq_id = task.get("sq_id", -1)
            for ir in task.get("inner_rounds") or []:
                if not isinstance(ir, dict):
                    continue
                tools = ir.get("tools") or []
                verdicts = ir.get("verdicts") or []
                for ti, tool in enumerate(tools):
                    if not isinstance(tool, dict):
                        continue
                    if tool.get("name") != "video_frame_extract":
                        continue
                    v = _find_verdict(verdicts, ti)
                    if v is None or not v.get("keep"):
                        continue

                    args = tool.get("arguments") or {}
                    if not isinstance(args, dict):
                        continue
                    t_start = float(args.get("time_start_sec") or 0)
                    t_end = float(args.get("time_end_sec") or 0)
                    if t_start >= t_end:
                        continue

                    key = (int(t_start), int(t_end))
                    if key in seen:
                        continue
                    seen.add(key)

                    # Use raw_output for frame count if available (P0)
                    raw = str(tool.get("raw_output") or "")
                    data = _parse_json(raw)
                    n_frames = (data or {}).get("count", 0) if data else 0

                    rows.append({
                        "evidence_id": f"vfe_{uuid.uuid4().hex[:8]}",
                        "start_sec": t_start,
                        "end_sec": t_end,
                        "description": (
                            f"Agent video_frame_extract [{t_start:.1f}s, {t_end:.1f}s], "
                            f"{n_frames} frames. sq_id={sq_id}, round={rd_idx+1}, tool_{ti}."
                        ),
                        "linked_state_ids": [f"sq_{sq_id}"] if sq_id >= 0 else [],
                        "confidence": 0.8,
                        "metadata": {"sq_id": sq_id, "round": rd_idx + 1, "tool_index": ti,
                                      "frame_count": n_frames},
                    })

    # Fallback: planning-stage visual evidence plans
    for sq_idx, sq in enumerate(result.get("sub_questions") or []):
        if not isinstance(sq, dict):
            continue
        vplan = sq.get("visual_evidence_plan")
        if not isinstance(vplan, dict):
            continue
        times = vplan.get("time_ranges") or []
        for i in range(0, len(times) - 1, 2):
            try:
                t_start, t_end = float(times[i]), float(times[i + 1])
            except (TypeError, ValueError):
                continue
            if t_start >= t_end:
                continue
            rows.append({
                "evidence_id": f"plan_{uuid.uuid4().hex[:8]}",
                "start_sec": t_start,
                "end_sec": t_end,
                "description": f"Planned: {sq.get('text', '')[:200]}",
                "linked_state_ids": [f"sq_{sq_idx}"],
                "confidence": 0.3,
                "metadata": {"sq_id": sq_idx, "source": "planning"},
            })
    return rows


def _extract_web_retrieved(result: dict[str, Any]) -> list[dict[str, Any]]:
    """All web search results. Uses raw_output when available (P0), else fallback."""
    rows: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    rank = 0

    for rd_idx, rd in enumerate(_get_rounds(result)):
        for task in rd.get("tasks") or []:
            if not isinstance(task, dict):
                continue
            sq_id = task.get("sq_id", -1)
            for ir in task.get("inner_rounds") or []:
                if not isinstance(ir, dict):
                    continue
                for ti, tool in enumerate(ir.get("tools") or []):
                    if not isinstance(tool, dict):
                        continue
                    if tool.get("name") != "deep_research_web_search":
                        continue
                    args = tool.get("arguments") or {}
                    query = str((args if isinstance(args, dict) else {}).get("query", ""))
                    if not query:
                        continue

                    raw = str(tool.get("raw_output") or "")
                    parsed = _parse_web_results(raw) if raw else []

                    if parsed:
                        for wr in parsed:
                            if wr["url"] in seen_urls:
                                continue
                            seen_urls.add(wr["url"])
                            rank += 1
                            rows.append({
                                "rank": rank, "query": query,
                                "hop_id": f"sq_{sq_id}" if sq_id >= 0 else "",
                                "url": wr["url"], "title": wr["title"], "snippet": wr["snippet"],
                                "linked_state_ids": [f"sq_{sq_id}"] if sq_id >= 0 else [],
                                "metadata": {"sq_id": sq_id, "round": rd_idx + 1, "tool_index": ti},
                            })
                    else:
                        # Old trace: raw_output empty, use query as placeholder
                        rank += 1
                        rows.append({
                            "rank": rank, "query": query,
                            "hop_id": f"sq_{sq_id}" if sq_id >= 0 else "",
                            "url": f"query://{query[:80]}",
                            "title": query[:200],
                            "snippet": f"Web query: {query}",
                            "linked_state_ids": [f"sq_{sq_id}"] if sq_id >= 0 else [],
                            "metadata": {"sq_id": sq_id, "note": "raw_output_empty"},
                        })

    # Augment from binding URLs
    for url in _extract_urls(_get_binding_text(result)):
        if url not in seen_urls:
            seen_urls.add(url)
            rank += 1
            rows.append({
                "rank": rank, "query": "", "hop_id": "binding",
                "url": url, "title": url,
                "snippet": "URL in binding stage.",
                "linked_state_ids": [],
                "metadata": {"source": "binding"},
            })
    return rows


def _extract_web_used(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Only KEPT web search calls. Uses raw_output when available (P0)."""
    rows: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for rd_idx, rd in enumerate(_get_rounds(result)):
        for task in rd.get("tasks") or []:
            if not isinstance(task, dict):
                continue
            sq_id = task.get("sq_id", -1)
            for ir in task.get("inner_rounds") or []:
                if not isinstance(ir, dict):
                    continue
                tools = ir.get("tools") or []
                verdicts = ir.get("verdicts") or []
                for ti, tool in enumerate(tools):
                    if not isinstance(tool, dict):
                        continue
                    if tool.get("name") != "deep_research_web_search":
                        continue
                    v = _find_verdict(verdicts, ti)
                    if v is None or not v.get("keep"):
                        continue

                    args = tool.get("arguments") or {}
                    query = str((args if isinstance(args, dict) else {}).get("query", ""))
                    if not query:
                        continue

                    raw = str(tool.get("raw_output") or "")
                    parsed = _parse_web_results(raw) if raw else []
                    summary = _parse_web_summary(raw)

                    if parsed:
                        for wr in parsed:
                            if wr["url"] in seen_urls:
                                continue
                            seen_urls.add(wr["url"])
                            snippet = wr["snippet"]
                            if summary:
                                snippet += f"\n[LLM Summary] {summary}"
                            rows.append({
                                "evidence_id": f"web_{uuid.uuid4().hex[:8]}",
                                "url": wr["url"], "title": wr["title"],
                                "evidence_snippet": snippet[:1000],
                                "support_claim": summary[:500] if summary else wr["snippet"][:500],
                                "support_target": _infer_support_target(query, sq_id, result),
                                "linked_state_ids": [f"sq_{sq_id}"] if sq_id >= 0 else [],
                                "confidence": 0.6,
                                "metadata": {"sq_id": sq_id, "query": query, "round": rd_idx + 1,
                                              "tool_index": ti, "relevance": v.get("reason", "")},
                            })
                    else:
                        # Old trace fallback
                        rows.append({
                            "evidence_id": f"web_{uuid.uuid4().hex[:8]}",
                            "url": f"query://{query[:80]}",
                            "title": query[:200],
                            "evidence_snippet": f"Kept web search: {query}. Relevance: {v.get('reason', '')[:200]}",
                            "support_claim": query[:500],
                            "support_target": _infer_support_target(query, sq_id, result),
                            "linked_state_ids": [f"sq_{sq_id}"] if sq_id >= 0 else [],
                            "confidence": 0.6,
                            "metadata": {"sq_id": sq_id, "query": query, "round": rd_idx + 1,
                                          "tool_index": ti, "relevance": v.get("reason", ""),
                                          "note": "raw_output_empty"},
                        })

    # Augment from binding URLs
    for url in _extract_urls(_get_binding_text(result)):
        if url not in seen_urls:
            seen_urls.add(url)
            rows.append({
                "evidence_id": f"web_{uuid.uuid4().hex[:8]}",
                "url": url, "title": url,
                "evidence_snippet": "URL cited in binding stage.",
                "support_claim": "",
                "support_target": "final_answer",
                "linked_state_ids": [],
                "confidence": 0.5,
                "metadata": {"source": "binding"},
            })
    return rows


def _get_binding_text(result: dict[str, Any]) -> str:
    trace = result.get("trace") or {}
    stages = (trace if isinstance(trace, dict) else {}).get("stages") or {}
    binding = (stages if isinstance(stages, dict) else {}).get("binding") or {}
    return str((binding if isinstance(binding, dict) else {}).get("response") or "")


def _extract_evidence_links(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Build evidence_links from:
    1. Worker SUMMARY [src:tool_N][target:...][relation:...] tags (P2)
    2. Binding [src:R{round}_sq{sq}_t{tool}] references (P3)
    3. Fallback: sub_question planning data
    """
    links: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()

    def _add(src_type: str, src_id: str, tgt_type: str, tgt_id: str, rel: str) -> None:
        key = (src_type, src_id, tgt_type, tgt_id)
        if key not in seen:
            seen.add(key)
            links.append({
                "source_type": src_type, "source_id": src_id,
                "target_type": tgt_type, "target_id": tgt_id,
                "relation": rel,
            })

    # P2: Parse worker SUMMARY tags
    for rd_idx, rd in enumerate(_get_rounds(result)):
        for task in rd.get("tasks") or []:
            if not isinstance(task, dict):
                continue
            sq_id = task.get("sq_id", -1)
            for ir in task.get("inner_rounds") or []:
                if not isinstance(ir, dict):
                    continue
                # Check resolved summary for structured tags
                if ir.get("resolved") and ir.get("sufficient"):
                    # The worker SUMMARY is stored in the sub_question answer field
                    pass  # P2 tags are in summary text, parsed below

    # P2+P3: Parse from sub_question answers (worker summaries) and binding text
    for sq_idx, sq in enumerate(result.get("sub_questions") or []):
        if not isinstance(sq, dict):
            continue
        sid = f"sq_{sq_idx}"
        answer = str(sq.get("answer") or "")

        # P2 tags in worker summary
        for tag in _parse_worker_evidence_tags(answer):
            tool_idx = tag["tool_index"]
            src_id = f"web_sq{sq_idx}_tool{tool_idx}" if "video" not in tag.get("target", "") else f"vfe_sq{sq_idx}_tool{tool_idx}"
            src_type = "web_evidence"
            tgt_type = "candidate" if "candidate" in tag.get("target", "") else "state"
            _add(src_type, src_id, tgt_type, tag["target"], tag["relation"])

        # P3 tags in binding
        for ref in _parse_binding_source_tags(answer):
            _add("web_evidence",
                 f"web_R{ref['round']}_sq{ref['sq_id']}_t{ref['tool_idx']}",
                 "state", sid, "supports")

        # Fallback: planning-based links
        vplan = sq.get("visual_evidence_plan")
        if isinstance(vplan, dict) and vplan.get("time_ranges"):
            _add("video_evidence", f"plan_sq_{sq_idx}", "state", sid, "supports_temporal_condition")

        wplan = sq.get("web_evidence_plan")
        if isinstance(wplan, list) and wplan:
            _add("web_evidence", f"web_plan_sq_{sq_idx}", "state", sid, "supports_record")

        if sq.get("sufficient"):
            _add("state", sid, "final_answer", "final", "supports")

    # P3: Parse binding source references
    binding_text = _get_binding_text(result)
    for ref in _parse_binding_source_tags(binding_text):
        _add("web_evidence",
             f"web_R{ref['round']}_sq{ref['sq_id']}_t{ref['tool_idx']}",
             "final_answer", "final", "supports")

    return links


def _extract_dependency_graph(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Infer weak_order edges from coordinator dispatch order across rounds."""
    edges: list[dict[str, Any]] = []
    prev_ids: set[int] = set()

    for rd in _get_rounds(result):
        cur_ids: set[int] = set()
        for task in rd.get("tasks") or []:
            if isinstance(task, dict) and task.get("sq_id", -1) >= 0:
                cur_ids.add(task["sq_id"])
        new_ids = cur_ids - prev_ids
        if new_ids and prev_ids:
            for ni in sorted(new_ids):
                for pi in sorted(prev_ids):
                    edges.append({"from": f"sq_{pi}", "to": f"sq_{ni}", "relation": "weak_order"})
        prev_ids |= cur_ids

    return edges


def _infer_support_target(query: str, sq_id: int, result: dict[str, Any]) -> str:
    sub_qs = result.get("sub_questions") or []
    sq_text = ""
    if isinstance(sub_qs, list) and 0 <= sq_id < len(sub_qs):
        sq = sub_qs[sq_id]
        sq_text = str((sq or {}).get("text", "") if isinstance(sq, dict) else "").lower()

    if any(w in sq_text for w in ["who", "name of", "person", "actor", "player", "record holder"]):
        return "candidate_identity"
    if any(w in sq_text for w in ["what is", "how many", "how long", "duration", "height", "distance"]):
        return "candidate_attribute"
    if any(w in sq_text for w in ["match", "game", "event", "happened", "occurred", "year"]):
        return "event_record"
    return "final_answer"


def _build_trace_summary(result: dict[str, Any]) -> str:
    sub_qs = result.get("sub_questions") or []
    if not isinstance(sub_qs, list):
        sub_qs = []
    n_sq = len(sub_qs)
    n_suff = sum(1 for sq in sub_qs if isinstance(sq, dict) and sq.get("sufficient"))
    n_rounds = 0
    eloop = result.get("trace", {}).get("stages", {}).get("evidence_loop", {})
    if isinstance(eloop, dict):
        n_rounds = eloop.get("total_rounds", 0)
    n_has_raw = sum(1 for rd in _get_rounds(result)
                    for t in rd.get("tasks", []) or []
                    for ir in (t if isinstance(t, dict) else {}).get("inner_rounds", []) or []
                    for tool in (ir if isinstance(ir, dict) else {}).get("tools", []) or []
                    if isinstance(tool, dict) and tool.get("raw_output", ""))
    return (f"rsagent-v4: {n_sq} sub-questions, {n_suff} sufficient, "
            f"{n_rounds} evidence rounds, {n_has_raw} tools with raw_output.")


def _final_answer_text(result: dict[str, Any]) -> str:
    fa = result.get("final_answer", "")
    if isinstance(fa, dict):
        return str(fa.get("answer_text") or fa.get("answer") or "")
    return str(fa or "")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def convert_one(sample_id: str, result: dict[str, Any], mode: str = "full_vdr") -> dict[str, Any]:
    video_ev = _extract_video_evidence(result)
    web_ret = _extract_web_retrieved(result)
    web_used = _extract_web_used(result)
    links = _extract_evidence_links(result)
    dep = _extract_dependency_graph(result)
    fa_text = _final_answer_text(result)

    return {
        "sample_id": sample_id,
        "ablation_setting": mode,
        "final_answer": {
            "answer_text": fa_text,
            "answer_type": "",
            "confidence": 0.5,
            "supporting_video_evidence": [e["evidence_id"] for e in video_ev[:10]],
            "supporting_web_evidence": [e["evidence_id"] for e in web_used[:10]],
        },
        "predicted_candidate_id": None,
        "used_video_evidence": video_ev,
        "retrieved_web_results": web_ret,
        "used_web_evidence": web_used,
        "evidence_links": links,
        "dependency_graph": dep,
        "reasoning_trace_summary": _build_trace_summary(result),
        "errors": [],
        "metadata": {"source": "rsagent-v4"},
    }


def auto_discover(runs_dir: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for video_dir in sorted(runs_dir.iterdir()):
        if not video_dir.is_dir():
            continue
        for q_dir in sorted(video_dir.iterdir()):
            if not q_dir.is_dir():
                continue
            if not (q_dir / "result.json").is_file():
                continue
            sample_id = f"{video_dir.name}_{q_dir.name}"
            mapping[sample_id] = f"{video_dir.name}/{q_dir.name}"
    return mapping


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="rsagent-v4 trace → VDR submission JSON")
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", default="full_vdr")
    p.add_argument("--model-name", default="Qwen3-VL-32B-Instruct")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir).expanduser().resolve()
    if not runs_dir.is_dir():
        raise SystemExit(f"Not a directory: {runs_dir}")

    mapping = auto_discover(runs_dir)
    print(f"Found {len(mapping)} runs in {runs_dir}")

    if args.manifest:
        manifest_path = Path(args.manifest).expanduser().resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_ids = {
            str(s["sample_id"]) for s in (manifest.get("samples") or [])
            if isinstance(s, dict) and s.get("sample_id")
        }
        matched: dict[str, str] = {}
        for auto_id, rel in mapping.items():
            video_name, q_name = rel.split("/")
            expected = f"vdr_pilot_{video_name}_{q_name}"
            if expected in manifest_ids:
                matched[expected] = rel
        if matched:
            print(f"Matched {len(matched)}/{len(manifest_ids)} manifest IDs")
            mapping = matched
        else:
            print(f"WARNING: 0 matched. Auto IDs sample: {list(mapping.keys())[:3]}")

    samples: list[dict[str, Any]] = []
    video_total = web_ret_total = web_used_total = 0
    for sample_id, rel in sorted(mapping.items()):
        path = runs_dir / rel / "result.json"
        if not path.is_file():
            print(f"  SKIP {sample_id}: not found")
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        sample = convert_one(sample_id, result, args.mode)
        samples.append(sample)
        video_total += len(sample["used_video_evidence"])
        web_ret_total += len(sample["retrieved_web_results"])
        web_used_total += len(sample["used_web_evidence"])
        print(f"  OK  {sample_id}")

    if not samples:
        raise SystemExit("No samples converted")

    payload = {
        "submission_id": f"rsagent_v4_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "benchmark_version": "",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "system": {
            "system_name": "rsagent-v4",
            "system_version": "latest_13",
            "run_mode": args.mode,
            "model_name": args.model_name,
            "tooling_notes": (
                "Coordinator+Worker dual-context agent. "
                "Tools: video_frame_extract, deep_research_web_search (Serper+Jina). "
                "VLM: Qwen3-VL-32B-Instruct via vLLM."
            ),
        },
        "samples": samples,
    }

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n→ {out_path}")
    print(f"  samples={len(samples)}")
    print(f"  video_evidence={video_total}  web_retrieved={web_ret_total}  web_used={web_used_total}")


if __name__ == "__main__":
    main()
