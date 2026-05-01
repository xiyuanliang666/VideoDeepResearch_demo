"""
多源网页搜索工具：并行调用 Serper（与 search_agent.serper_search 同协议），无 LLM 总结。

工具名: deep_research_web_search

<tool_call> arguments 示例:
  {"query": "要检索的问题"}   # 或使用 "question"
"""

from __future__ import annotations

import http.client
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from video_dr_agent.protocols import ToolDispatcher

DEEP_RESEARCH_WEB_SEARCH_TOOL_NAME = "deep_research_web_search"

# 针对视频深度调研优化的 Worker 配置（与用户需求一致）
DEEP_RESEARCH_WORKER_CONFIGS: list[dict[str, Any]] = [
    {
        "id": "FACT_CHECKER",
        "gl": "us",
        "hl": "en",
        "sites": [],
        "label": "Real-time fact and background verification",
        "description": "Verify people, companies, news events, or general facts mentioned in the video. Best for time-sensitive claims.",
    },
    {
        "id": "ACADEMIC_TECH",
        "gl": "us",
        "hl": "en",
        "sites": [
            "arxiv.org",
            "scholar.google.com",
            "github.com",
            "paperswithcode.com",
            "huggingface.co",
        ],
        "label": "Academic papers and deep technical provenance",
        "description": "Use when the video covers AI models, algorithms, formulas, or research. Surfaces more rigorous underlying logic than the video alone.",
    },
    {
        "id": "COMMUNITY_CONTEXT",
        "gl": "us",
        "hl": "en",
        "sites": [
            "reddit.com",
            "stackoverflow.com",
            "news.ycombinator.com",
            "quora.com",
        ],
        "label": "Developer communities and discourse analysis",
        "description": "Multi-angle takes on the video's claims, practical fixes for code errors, or industry debates and controversies.",
    },
    {
        "id": "VIDEO_METADATA",
        "gl": "us",
        "hl": "en",
        "sites": ["youtube.com", "wikipedia.org"],
        "label": "Video metadata and encyclopedia context",
        "description": "Comment summaries, timestamp guides, or encyclopedic background for related episodes or lectures tied to the video.",
    },
]


def serper_search(
    query: str,
    *,
    gl: str = "us",
    hl: str = "en",
    sites: list[str] | None = None,
    num: int = 2,
    api_key: str,
) -> list[dict[str, Any]]:
    """
    调用 Google Serper API，返回结构化列表（与 search_agent.serper_search 行为一致）。
    不包含任何 LLM 后处理。
    """
    q = query
    if sites:
        site_filter = " OR ".join(f"site:{s}" for s in sites)
        q = f"{query} ({site_filter})"

    conn = http.client.HTTPSConnection("google.serper.dev")
    try:
        payload = json.dumps({"q": q, "gl": gl, "hl": hl, "num": num})
        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
        }
        conn.request("POST", "/search", payload, headers)
        resp = conn.getresponse()
        raw = resp.read().decode()
        if resp.status >= 400:
            raise RuntimeError(f"Serper HTTP {resp.status}: {raw[:500]}")
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
    """
    对每个 Worker 配置并发执行一次 serper_search，每个 Worker 最多 results_per_worker 条。
    返回列表顺序不保证；每条含 worker 元数据 + results。
    """
    configs = worker_configs or DEEP_RESEARCH_WORKER_CONFIGS
    n_workers = max_workers or len(configs)

    def one(cfg: dict[str, Any]) -> dict[str, Any]:
        wid = cfg["id"]
        gl = cfg.get("gl", "us")
        hl = cfg.get("hl", "en")
        sites = cfg.get("sites") or []
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
            "worker_id": wid,
            "label": cfg.get("label", wid),
            "description": cfg.get("description", ""),
            "gl": gl,
            "hl": hl,
            "sites": list(sites),
            "query_submitted": query,
            "results": rows[:results_per_worker],
        }

    out: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(n_workers, len(configs))) as pool:
        futures = [pool.submit(one, c) for c in configs]
        for fut in as_completed(futures):
            out.append(fut.result())
    return out


class DeepResearchWebSearchToolDispatcher(ToolDispatcher):
    """
    仅处理 ``deep_research_web_search``；其余 ``name`` 交给 ``fallback``。

    与 FOCUS 等工具 **互不依赖**：每一轮 ``dispatch`` 只对应 research 输出的 **一个**
    ``name``；命中本类则只跑 Serper，否则原样转发。**不存在**「先网页后 FOCUS」的固定流水线；
    嵌套 ``fallback`` 只是实现上的 **责任链路由**（按层判断 name），执行哪个工具完全由
    research agent 的 ``<tool_call>`` 决定。
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        fallback: ToolDispatcher,
        results_per_worker: int = 2,
        worker_configs: list[dict[str, Any]] | None = None,
    ) -> None:
        self.api_key = (api_key or os.environ.get("SERPER_API_KEY") or "").strip()
        self.fallback = fallback
        self.results_per_worker = results_per_worker
        self.worker_configs = worker_configs

    def dispatch(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        round_index: int | None = None,
    ) -> str:
        _ = round_index
        if name != DEEP_RESEARCH_WEB_SEARCH_TOOL_NAME:
            return self.fallback.dispatch(
                name, arguments, round_index=round_index
            )
        return self._run_search(arguments)

    def _run_search(self, arguments: dict[str, Any]) -> str:
        q = (arguments.get("query") or arguments.get("question") or "").strip()
        if not q:
            return json.dumps(
                {
                    "ok": False,
                    "tool": DEEP_RESEARCH_WEB_SEARCH_TOOL_NAME,
                    "error": "arguments.query 或 arguments.question 不能为空",
                },
                ensure_ascii=False,
            )
        if not self.api_key:
            return json.dumps(
                {
                    "ok": False,
                    "tool": DEEP_RESEARCH_WEB_SEARCH_TOOL_NAME,
                    "error": "未配置 Serper API Key（构造参数 api_key 或环境变量 SERPER_API_KEY）",
                },
                ensure_ascii=False,
            )

        workers_out = run_deep_research_parallel_serper(
            q,
            api_key=self.api_key,
            worker_configs=self.worker_configs,
            results_per_worker=self.results_per_worker,
        )
        payload = {
            "ok": True,
            "tool": DEEP_RESEARCH_WEB_SEARCH_TOOL_NAME,
            "query": q,
            "results_per_worker": self.results_per_worker,
            "workers": workers_out,
        }
        return json.dumps(payload, ensure_ascii=False)


def maybe_wrap_serper_search(
    inner: ToolDispatcher,
    *,
    api_key: str | None = None,
    results_per_worker: int = 2,
) -> ToolDispatcher:
    """
    若 api_key 或环境变量 SERPER_API_KEY 存在，则在最外层增加 **按 name 分流** 的搜索层。

    ``inner``（如 FOCUS + stub）与搜索层 **平级互斥执行**：单次 ``dispatch`` 只会进入
    其中一个分支。包装顺序不代表「先执行网页搜索再执行 FOCUS」，只表示 **先匹配**
    ``deep_research_web_search``，未匹配再交给内层匹配其它工具名。
    """
    key = (api_key or os.environ.get("SERPER_API_KEY") or "").strip()
    if not key:
        return inner
    return DeepResearchWebSearchToolDispatcher(
        api_key=key,
        fallback=inner,
        results_per_worker=results_per_worker,
    )
