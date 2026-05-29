"""Serper-backed multi-worker web search tool."""

from __future__ import annotations

import http.client
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Any

from .base import ToolDispatcher
from .evidence_policy import DOMAIN_WORKER_SITE_OVERRIDES, normalize_evidence_domain

DEEP_RESEARCH_WEB_SEARCH_TOOL_NAME = "deep_research_web_search"

DEEP_RESEARCH_WORKER_CONFIGS: list[dict[str, Any]] = [
    {
        "id": "FACT_CHECKER",
        "gl": "us",
        "hl": "en",
        "sites": [],
        "label": "Real-time fact and background verification",
    },
    {
        "id": "ACADEMIC_TECH",
        "gl": "us",
        "hl": "en",
        "sites": ["arxiv.org", "scholar.google.com", "github.com", "paperswithcode.com", "huggingface.co"],
        "label": "Academic papers and technical provenance",
    },
    {
        "id": "COMMUNITY_CONTEXT",
        "gl": "us",
        "hl": "en",
        "sites": ["reddit.com", "stackoverflow.com", "news.ycombinator.com", "quora.com"],
        "label": "Community and discourse context",
    },
    {
        "id": "VIDEO_METADATA",
        "gl": "us",
        "hl": "en",
        "sites": ["youtube.com", "wikipedia.org"],
        "label": "Video metadata and encyclopedia context",
    },
]


def apply_evidence_domain_to_worker_configs(configs: list[dict[str, Any]], domain: str | None) -> str:
    """Apply domain-specific site overrides to a worker config list."""
    normalized = normalize_evidence_domain(domain)
    overrides = DOMAIN_WORKER_SITE_OVERRIDES.get(normalized, {})
    for config in configs:
        worker_id = config.get("id")
        if worker_id in overrides:
            config["sites"] = list(overrides[worker_id])
    return normalized


def serper_search(
    query: str,
    *,
    gl: str = "us",
    hl: str = "en",
    sites: list[str] | None = None,
    num: int = 2,
    api_key: str,
) -> list[dict[str, Any]]:
    """Call Google Serper and return structured organic-style rows."""
    submitted_query = query
    if sites:
        site_filter = " OR ".join(f"site:{site}" for site in sites)
        submitted_query = f"{query} ({site_filter})"

    conn = http.client.HTTPSConnection("google.serper.dev")
    try:
        payload = json.dumps({"q": submitted_query, "gl": gl, "hl": hl, "num": num})
        headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
        conn.request("POST", "/search", payload, headers)
        response = conn.getresponse()
        raw = response.read().decode()
        if response.status >= 400:
            raise RuntimeError(f"Serper HTTP {response.status}: {raw[:500]}")
        data = json.loads(raw)
    finally:
        conn.close()

    results: list[dict[str, Any]] = []
    if kg := data.get("knowledgeGraph"):
        results.append(
            {
                "title": kg.get("title", ""),
                "snippet": kg.get("description", ""),
                "url": kg.get("website", ""),
                "source": "knowledgeGraph",
            }
        )
    for item in data.get("organic", []):
        results.append(
            {
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "url": item.get("link", ""),
                "source": "organic",
            }
        )
    return results[:num]


def run_deep_research_parallel_serper(
    query: str,
    *,
    api_key: str,
    worker_configs: list[dict[str, Any]] | None = None,
    results_per_worker: int = 2,
    max_workers: int | None = None,
) -> list[dict[str, Any]]:
    """Run one Serper request per worker profile in parallel."""
    configs = worker_configs or DEEP_RESEARCH_WORKER_CONFIGS
    n_workers = max_workers or len(configs)

    def one(config: dict[str, Any]) -> dict[str, Any]:
        worker_id = config["id"]
        gl = config.get("gl", "us")
        hl = config.get("hl", "en")
        sites = config.get("sites") or []
        try:
            rows = serper_search(
                query,
                gl=gl,
                hl=hl,
                sites=sites if sites else None,
                num=results_per_worker,
                api_key=api_key,
            )
        except Exception as exc:  # noqa: BLE001
            rows = [
                {
                    "title": "Search failed",
                    "snippet": f"{type(exc).__name__}: {exc}",
                    "url": "",
                    "source": "error",
                }
            ]
        return {
            "worker_id": worker_id,
            "label": config.get("label", worker_id),
            "gl": gl,
            "hl": hl,
            "sites": list(sites),
            "query_submitted": query,
            "results": rows[:results_per_worker],
        }

    out: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(n_workers, len(configs))) as pool:
        futures = [pool.submit(one, config) for config in configs]
        for future in as_completed(futures):
            out.append(future.result())
    return out


class DeepResearchWebSearchToolDispatcher:
    """Tool dispatcher for `deep_research_web_search`; delegates unknown tools."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        fallback: ToolDispatcher,
        results_per_worker: int = 2,
        worker_configs: list[dict[str, Any]] | None = None,
        default_evidence_domain: str | None = None,
    ) -> None:
        self.api_key = (api_key or os.environ.get("SERPER_API_KEY") or "").strip()
        self.fallback = fallback
        self.results_per_worker = results_per_worker
        self.worker_configs = worker_configs
        self.default_evidence_domain = (default_evidence_domain or "").strip() or None

    def dispatch(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        round_index: int | None = None,
    ) -> str:
        if name != DEEP_RESEARCH_WEB_SEARCH_TOOL_NAME:
            return self.fallback.dispatch(name, arguments, round_index=round_index)
        return self._run_search(arguments)

    def _run_search(self, arguments: dict[str, Any]) -> str:
        query = (arguments.get("query") or arguments.get("question") or "").strip()
        if not query:
            return json.dumps(
                {"ok": False, "tool": DEEP_RESEARCH_WEB_SEARCH_TOOL_NAME, "error": "query_or_question_required"},
                ensure_ascii=False,
            )
        if not self.api_key:
            return json.dumps(
                {"ok": False, "tool": DEEP_RESEARCH_WEB_SEARCH_TOOL_NAME, "error": "SERPER_API_KEY is not set"},
                ensure_ascii=False,
            )

        profile_raw = arguments.get("search_profile") or arguments.get("evidence_domain") or self.default_evidence_domain
        worker_configs = deepcopy(self.worker_configs if self.worker_configs is not None else DEEP_RESEARCH_WORKER_CONFIGS)
        resolved = apply_evidence_domain_to_worker_configs(worker_configs, profile_raw)
        workers = run_deep_research_parallel_serper(
            query,
            api_key=self.api_key,
            worker_configs=worker_configs,
            results_per_worker=self.results_per_worker,
        )
        return json.dumps(
            {
                "ok": True,
                "tool": DEEP_RESEARCH_WEB_SEARCH_TOOL_NAME,
                "query": query,
                "results_per_worker": self.results_per_worker,
                "search_profile_resolved": resolved,
                "workers": workers,
            },
            ensure_ascii=False,
        )


def maybe_wrap_serper_search(
    inner: ToolDispatcher,
    *,
    api_key: str | None = None,
    results_per_worker: int = 2,
    default_evidence_domain: str | None = None,
) -> ToolDispatcher:
    """Wrap a dispatcher with Serper search when a key is available."""
    key = (api_key or os.environ.get("SERPER_API_KEY") or "").strip()
    if not key:
        return inner
    return DeepResearchWebSearchToolDispatcher(
        api_key=key,
        fallback=inner,
        results_per_worker=results_per_worker,
        default_evidence_domain=default_evidence_domain,
    )
