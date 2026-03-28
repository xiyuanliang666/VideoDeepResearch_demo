"""Compact retrieval utilities."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

from src.schemas import Anchor, ObservationUnit, QueryUnit, RetrievalCandidate


def _tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", text.lower()) if token]


def _overlap_score(query_tokens: set[str], candidate_tokens: list[str]) -> float:
    candidate_token_set = set(candidate_tokens)
    if not query_tokens or not candidate_token_set:
        return 0.0
    overlap = query_tokens & candidate_token_set
    return len(overlap) / max(len(query_tokens), 1)


def _fallback_candidates(query_unit: QueryUnit, query: str, topk: int) -> list[RetrievalCandidate]:
    normalized_query = " ".join(query.split())[:120]
    return [
        RetrievalCandidate(
            candidate_id=f"web_fallback_{index:04d}",
            candidate_type="web_result",
            source_query_id=query_unit.query_id,
            source_ref=f"fallback://web/{index}",
            score=max(0.05 - index * 0.01, 0.01),
            rank=index + 1,
            metadata={
                "title": f"Fallback web result for: {normalized_query}",
                "snippet": (
                    "Web search is unavailable in the current environment, so this is a "
                    "placeholder candidate preserved for pipeline debugging."
                ),
                "url": "",
            },
            retriever_name="fallback_web_retriever",
            normalized_score=max(0.05 - index * 0.01, 0.01),
            keep_label="weak_keep",
            keep_reason="Fallback candidate generated because no live web search is available.",
        )
        for index in range(max(topk, 1))
    ]


def _search_tavily(query: str, api_key: str, topk: int) -> list[dict]:
    payload = {
        "query": query,
        "max_results": topk,
        "search_depth": "basic",
        "include_answer": False,
        "include_raw_content": False,
    }
    request = urllib.request.Request(
        url="https://api.tavily.com/search",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data.get("results", [])


def run_local_retrieval(
    question: str,
    observations: list[ObservationUnit],
    *,
    topk: int,
    step_index: int = 0,
) -> tuple[QueryUnit, list[RetrievalCandidate]]:
    query_unit = QueryUnit(
        query_id=f"local_query_{step_index:04d}",
        anchor_id="bootstrap",
        query_text=question,
        query_type="bootstrap",
        step_index=step_index,
        motivation="Use the question itself to bootstrap local retrieval.",
        query_source="question",
    )
    query_tokens = set(_tokenize(question))
    candidates: list[RetrievalCandidate] = []
    for rank_hint, observation in enumerate(observations):
        transcript = observation.transcript_text or ""
        ocr_text = observation.ocr_text or ""
        candidate_text = " ".join([transcript, ocr_text, " ".join(observation.scene_clues)]).strip()
        candidate_tokens = _tokenize(candidate_text)
        score = _overlap_score(query_tokens, candidate_tokens)
        if not candidate_text:
            score = max(score, 0.01)
        candidates.append(
            RetrievalCandidate(
                candidate_id=f"cand_{rank_hint:04d}",
                candidate_type="clip_candidate",
                source_query_id=query_unit.query_id,
                source_ref=observation.metadata.get("clip_id", observation.observation_id),
                score=score,
                rank=rank_hint,
                metadata={
                    "observation_id": observation.observation_id,
                    "timestamp_start": observation.timestamp_start,
                    "timestamp_end": observation.timestamp_end,
                    "frame_count": len(observation.frame_paths),
                    "transcript_preview": transcript[:160],
                },
                retriever_name="token_overlap_local_retriever",
            )
        )
    sorted_candidates = sorted(candidates, key=lambda item: item.score, reverse=True)
    top_candidates: list[RetrievalCandidate] = []
    for final_rank, candidate in enumerate(sorted_candidates[:topk], start=1):
        candidate.rank = final_rank
        candidate.normalized_score = candidate.score
        candidate.keep_label = "keep" if candidate.score > 0 else "weak_keep"
        candidate.keep_reason = (
            "Matched transcript/OCR/scene clues."
            if candidate.score > 0
            else "Kept as fallback due to sparse local clues."
        )
        top_candidates.append(candidate)
    return query_unit, top_candidates


def run_web_retrieval(
    anchor: Anchor,
    question: str,
    *,
    topk: int,
    step_index: int,
) -> tuple[QueryUnit, list[RetrievalCandidate]]:
    query_text = anchor.search_queries[0] if anchor.search_queries else question
    query_unit = QueryUnit(
        query_id=f"web_query_{step_index:04d}_{anchor.anchor_id or 'global'}",
        anchor_id=anchor.anchor_id,
        query_text=query_text,
        query_type="web_search",
        step_index=step_index,
        motivation="Search the open web for external evidence grounded by the current anchor.",
        query_source="anchor_search_query",
    )

    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        return query_unit, _fallback_candidates(query_unit, query_text, topk)

    try:
        raw_results = _search_tavily(query_text, api_key=api_key, topk=topk)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return query_unit, _fallback_candidates(query_unit, query_text, topk)

    query_tokens = set(_tokenize(query_text))
    candidates: list[RetrievalCandidate] = []
    for rank, item in enumerate(raw_results[:topk], start=1):
        title = (item.get("title") or "").strip()
        snippet = (item.get("content") or item.get("snippet") or "").strip()
        url = (item.get("url") or "").strip()
        overlap = len(query_tokens & set(_tokenize(" ".join([title, snippet]))))
        normalized_score = min(1.0, 0.2 * overlap + max(0.0, 1.0 - 0.1 * (rank - 1)))
        candidates.append(
            RetrievalCandidate(
                candidate_id=f"web_{step_index:04d}_{rank:04d}",
                candidate_type="web_result",
                source_query_id=query_unit.query_id,
                source_ref=url or f"web_result_{rank}",
                score=float(item.get("score", 0.0) or 0.0),
                rank=rank,
                metadata={
                    "title": title,
                    "snippet": snippet[:400],
                    "url": url,
                    "raw_result": item,
                },
                retriever_name="tavily_web_retriever",
                normalized_score=normalized_score,
                keep_label="keep" if overlap > 0 else "weak_keep",
                keep_reason=(
                    "Title/snippet overlaps with the query terms."
                    if overlap > 0
                    else "Retained as a low-confidence web candidate."
                ),
            )
        )
    if not candidates:
        return query_unit, _fallback_candidates(query_unit, query_text, topk)
    return query_unit, candidates
