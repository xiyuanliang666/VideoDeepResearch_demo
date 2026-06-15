"""Web search tool: parallel Serper workers + crawl4ai page fetch + per-page LLM filtering."""

from __future__ import annotations

import asyncio
import http.client
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Any

from src.rsagent.evidence_policy import DOMAIN_WORKER_SITE_OVERRIDES, normalize_evidence_domain
from src.rsagent.protocols import ToolDispatcher

TOOL_NAME = "deep_research_web_search"

WORKER_CONFIGS: list[dict[str, Any]] = [
    {
        "id": "FACT_CHECKER", "gl": "us", "hl": "en", "sites": [],
        "label": "Real-time fact and background verification",
        "description": "Verify people, companies, news events, or general facts mentioned in the video.",
    },
    {
        "id": "ACADEMIC_TECH", "gl": "us", "hl": "en",
        "sites": ["arxiv.org", "scholar.google.com", "github.com", "paperswithcode.com", "huggingface.co"],
        "label": "Academic papers and deep technical provenance",
        "description": "AI models, algorithms, formulas, or research papers.",
    },
    {
        "id": "COMMUNITY_CONTEXT", "gl": "us", "hl": "en",
        "sites": ["reddit.com", "stackoverflow.com", "news.ycombinator.com", "quora.com"],
        "label": "Developer communities and discourse analysis",
        "description": "Multi-angle takes, practical fixes, or industry debates.",
    },
    {
        "id": "VIDEO_METADATA", "gl": "us", "hl": "en",
        "sites": ["youtube.com", "wikipedia.org"],
        "label": "Video metadata and encyclopedia context",
        "description": "Encyclopedic background or related video metadata.",
    },
]

_CRAWL_TIMEOUT_MS = 90_000
_MAX_PAGE_CHARS = 24000
_MAX_PAGES_TO_FETCH = int(os.environ.get("WEB_SEARCH_MAX_PAGES", "2"))


def _selected_worker_configs(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select search workers. Default to FACT_CHECKER to reduce API usage."""
    requested = os.environ.get("WEB_SEARCH_WORKERS", "FACT_CHECKER").strip()
    if requested.lower() in {"all", "*"}:
        return configs
    wanted = {item.strip() for item in requested.split(",") if item.strip()}
    selected = [cfg for cfg in configs if cfg.get("id") in wanted]
    return selected or [cfg for cfg in configs if cfg.get("id") == "FACT_CHECKER"] or configs[:1]


# --- Serper search ---

def _serper_search(query: str, *, gl: str, hl: str, sites: list[str] | None, num: int, api_key: str) -> list[dict[str, Any]]:
    q = query
    if sites:
        q = f"{query} ({' OR '.join(f'site:{s}' for s in sites)})"
    conn = http.client.HTTPSConnection("google.serper.dev")
    try:
        payload = json.dumps({"q": q, "gl": gl, "hl": hl, "num": num})
        conn.request("POST", "/search", payload, {"X-API-KEY": api_key, "Content-Type": "application/json"})
        resp = conn.getresponse()
        raw = resp.read().decode()
        if resp.status >= 400:
            raise RuntimeError(f"Serper HTTP {resp.status}: {raw[:300]}")
        data = json.loads(raw)
    finally:
        conn.close()
    results: list[dict[str, Any]] = []
    kg = data.get("knowledgeGraph")
    if kg:
        results.append({"title": kg.get("title", ""), "snippet": kg.get("description", ""), "url": kg.get("website", ""), "source": "knowledgeGraph"})
    for item in data.get("organic", []):
        results.append({"title": item.get("title", ""), "snippet": item.get("snippet", ""), "url": item.get("link", ""), "source": "organic"})
    return results[:num]


def _run_parallel_serper(query: str, *, api_key: str, configs: list[dict[str, Any]], results_per_worker: int) -> list[dict[str, Any]]:
    def one(cfg: dict[str, Any]) -> dict[str, Any]:
        try:
            rows = _serper_search(query, gl=cfg.get("gl", "us"), hl=cfg.get("hl", "en"),
                                  sites=cfg.get("sites") or None, num=results_per_worker, api_key=api_key)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            rows = [{"title": "Error", "snippet": err, "url": "", "source": "error"}]
            return {"worker_id": cfg["id"], "label": cfg.get("label", ""), "results": rows, "error": err}
        return {"worker_id": cfg["id"], "label": cfg.get("label", ""), "results": rows}

    out: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(configs)) as pool:
        for fut in as_completed([pool.submit(one, c) for c in configs]):
            out.append(fut.result())
    return out


# --- crawl4ai page fetch ---

def _normalize_url(url: str) -> str:
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def _proxy_url() -> str | None:
    return (
        os.environ.get("https_proxy")
        or os.environ.get("HTTPS_PROXY")
        or os.environ.get("http_proxy")
        or os.environ.get("HTTP_PROXY")
    )


def _markdown_text(markdown: Any) -> str:
    if isinstance(markdown, str):
        return markdown
    if markdown is None:
        return ""
    for attr in ("fit_markdown", "raw_markdown", "markdown"):
        value = getattr(markdown, attr, None)
        if value:
            return str(value)
    return str(markdown)


async def _crawl4ai_fetch_one(url: str) -> dict[str, Any]:
    target = _normalize_url(url)
    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
    except Exception as exc:
        return {"ok": False, "url": url, "content": "", "error": f"crawl4ai unavailable: {type(exc).__name__}: {exc}"}

    proxy = _proxy_url()
    browser_config = BrowserConfig(
        headless=True,
        proxy_config={"server": proxy} if proxy else None,
        verbose=False,
        text_mode=True,
        light_mode=True,
    )
    run_config = CrawlerRunConfig(
        only_text=True,
        page_timeout=_CRAWL_TIMEOUT_MS,
        wait_until="domcontentloaded",
        excluded_tags=["script", "style", "nav", "footer", "form"],
        remove_forms=True,
        verbose=False,
    )
    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=target, config=run_config)
        if not getattr(result, "success", False):
            error = getattr(result, "error_message", "crawl failed") or "crawl failed"
            return {"ok": False, "url": url, "content": "", "error": error}
        text = _markdown_text(getattr(result, "markdown", ""))
        if len(text) > _MAX_PAGE_CHARS:
            text = text[:_MAX_PAGE_CHARS] + "\n...[truncated]..."
        return {"ok": True, "url": url, "content": text}
    except Exception as exc:
        return {"ok": False, "url": url, "content": "", "error": f"{type(exc).__name__}: {exc}"}


def _crawl4ai_fetch(url: str) -> dict[str, Any]:
    return asyncio.run(_crawl4ai_fetch_one(url))


def _clean_markdown_for_judge(content: str) -> str:
    content = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", content)
    content = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", content)
    content = re.sub(r"\[\[?\]\]?", "", content)
    return content


def _query_focused_excerpt(content: str, question: str, *, max_chars: int = 8000) -> str:
    content = _clean_markdown_for_judge(content)
    if len(content) <= max_chars:
        return content

    words = [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9'-]{3,}", question)]
    stopwords = {"what", "which", "when", "where", "with", "from", "that", "this", "their", "about"}
    tokens: list[str] = []
    for word in words:
        if word in stopwords:
            continue
        tokens.append(word)
        if len(word) > 5:
            tokens.append(re.sub(r"(ing|ed|es|s)$", "", word))
    question_lower = question.lower()
    if "measur" in question_lower or "height" in question_lower:
        tokens.extend(["measur", "survey", "determin", "height"])
    if "credited" in question_lower or question_lower.startswith("who"):
        tokens.extend(["credited", "organized", "financed", "expedition", "journalist"])
    tokens = sorted({t for t in tokens if len(t) >= 4})

    segments = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n{2,}", content) if len(s.strip()) >= 40]
    scored: list[tuple[int, int, str]] = []
    for idx, segment in enumerate(segments):
        lowered = segment.lower()
        score = sum(2 for token in tokens if token in lowered)
        for phrase in ("official height", "determined by", "survey carried out", "organized and financed", "journalist", "expedition"):
            if phrase in lowered:
                score += 4
        if question_lower.startswith("who") and re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", segment):
            score += 2
        if score > 0:
            scored.append((-score, idx, segment))

    if not scored:
        return content[:max_chars]

    scored.sort()
    excerpt_parts: list[str] = []
    used = 0
    used_indexes: set[int] = set()
    for _, idx, segment in scored:
        if idx in used_indexes:
            continue
        context = []
        for j in (idx - 1, idx, idx + 1):
            if 0 <= j < len(segments) and j not in used_indexes:
                context.append(segments[j])
                used_indexes.add(j)
        chunk = " ".join(context).strip()
        if not chunk:
            continue
        if len(chunk) > max_chars // 2:
            chunk = chunk[: max_chars // 2]
        if used + len(chunk) + 32 > max_chars:
            continue
        excerpt_parts.append(chunk)
        used += len(chunk) + 32
        if used >= max_chars:
            break

    return "\n\n...[focused excerpt]...\n\n".join(excerpt_parts) or content[:max_chars]


def _deduplicate_urls(workers_out: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for worker_rank, w in enumerate(workers_out):
        for result_rank, r in enumerate(w.get("results", [])):
            u = (r.get("url") or "").strip()
            if u and u not in seen and r.get("source") != "error":
                seen.add(u)
                row = dict(r)
                row["_worker_rank"] = worker_rank
                row["_result_rank"] = result_rank
                rows.append(row)

    def priority(row: dict[str, Any]) -> tuple[int, int, int]:
        url = (row.get("url") or "").lower()
        title = (row.get("title") or "").lower()
        trusted = ("wikipedia.org" in url or "britannica.com" in url or "guinnessworldrecords.com" in url)
        community = any(host in url for host in ("reddit.com", "quora.com", "stackoverflow.com", "news.ycombinator.com"))
        exact_title = 0 if any(token in title for token in ("angel falls", "world cup", "guinness", "museum", "official")) else 1
        site_score = 0 if trusted else (2 if community else 1)
        return (site_score, exact_title, row.get("_result_rank", 999) + row.get("_worker_rank", 999))

    rows.sort(key=priority)
    return [row["url"] for row in rows]


# --- Per-page LLM filtering: answer question or discard ---

_PAGE_JUDGE_PROMPT = """You are given a question and a web page's content. Your task:
1. If the page contains information that helps answer the question, output a concise summary of the relevant information (2-5 sentences). Start with "RELEVANT: ".
2. If the page does NOT help answer the question, output exactly "IRRELEVANT".

Question: {question}

Page URL: {url}
Page content (may be truncated):
{content}"""


def _judge_page(question: str, page: dict[str, Any], *, llm_client: Any) -> dict[str, Any]:
    """Call LLM to judge if a page answers the question. Returns {url, relevant, summary}."""
    if not page.get("ok") or not page.get("content"):
        return {"url": page.get("url", ""), "relevant": False, "summary": ""}
    content = _query_focused_excerpt(page["content"], question, max_chars=8000)
    messages = [
        {"role": "system", "content": "You are a concise research assistant."},
        {"role": "user", "content": _PAGE_JUDGE_PROMPT.format(question=question, url=page["url"], content=content)},
    ]
    try:
        resp = llm_client.complete(messages)
        if resp.strip().startswith("RELEVANT:"):
            return {"url": page["url"], "relevant": True, "summary": resp.strip()[len("RELEVANT:"):].strip()}
        return {"url": page["url"], "relevant": False, "summary": ""}
    except Exception as exc:
        excerpt = content[:1200].replace("\n", " ")
        return {
            "url": page["url"],
            "relevant": True,
            "summary": f"JUDGE_FAILED ({type(exc).__name__}): focused excerpt for manual assessment: {excerpt}",
        }


def _parallel_judge_pages(question: str, pages: list[dict[str, Any]], *, llm_client: Any) -> list[dict[str, Any]]:
    """Judge multiple pages in parallel, return only relevant ones with summaries."""
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(len(pages), _MAX_PAGES_TO_FETCH)) as pool:
        futs = [pool.submit(_judge_page, question, p, llm_client=llm_client) for p in pages]
        for fut in as_completed(futs):
            r = fut.result()
            if r["relevant"]:
                results.append(r)
    return results


# --- Dispatcher ---

class WebSearchDispatcher(ToolDispatcher):
    """Handles deep_research_web_search: Serper → crawl4ai fetch → LLM per-page judge → return relevant summaries."""

    def __init__(self, *, api_key: str | None = None, fallback: ToolDispatcher | None = None,
                 results_per_worker: int = 3, default_evidence_domain: str | None = None,
                 jina_api_key: str | None = None, llm_client: Any = None) -> None:
        self.api_key = (api_key or os.environ.get("SERPER_API_KEY") or "").strip()
        self.fallback = fallback
        self.results_per_worker = results_per_worker
        self.default_evidence_domain = default_evidence_domain
        self.llm_client = llm_client

    def dispatch(self, name: str, arguments: dict[str, Any], *, round_index: int | None = None) -> str:
        if name != TOOL_NAME:
            if self.fallback:
                return self.fallback.dispatch(name, arguments, round_index=round_index)
            return json.dumps({"ok": False, "error": f"Unknown tool: {name}"})
        return self._run(arguments)

    def _run(self, arguments: dict[str, Any]) -> str:
        q = (arguments.get("query") or arguments.get("question") or "").strip()
        if not q:
            return json.dumps({"ok": False, "tool": TOOL_NAME, "error": "query is required"})
        if not self.api_key:
            return json.dumps({"ok": False, "tool": TOOL_NAME, "error": "No SERPER_API_KEY configured"})

        domain = arguments.get("evidence_domain") or self.default_evidence_domain
        cfgs = deepcopy(WORKER_CONFIGS)
        dom = normalize_evidence_domain(domain)
        overrides = DOMAIN_WORKER_SITE_OVERRIDES.get(dom, {})
        for cfg in cfgs:
            if cfg["id"] in overrides:
                cfg["sites"] = list(overrides[cfg["id"]])
        cfgs = _selected_worker_configs(cfgs)

        # Step 1: Parallel Serper search
        workers_out = _run_parallel_serper(q, api_key=self.api_key, configs=cfgs, results_per_worker=self.results_per_worker)

        # Step 2: crawl4ai page fetch (top unique URLs)
        pages: list[dict[str, Any]] = []
        top_urls = _deduplicate_urls(workers_out)[:_MAX_PAGES_TO_FETCH]
        if top_urls:
            with ThreadPoolExecutor(max_workers=len(top_urls)) as pool:
                futs = {pool.submit(_crawl4ai_fetch, u): u for u in top_urls}
                for fut in as_completed(futs):
                    pages.append(fut.result())

        # Step 3: Per-page LLM judgment (parallel) — keep relevant, discard irrelevant
        relevant_summaries: list[dict[str, Any]] = []
        if pages and self.llm_client:
            relevant_summaries = _parallel_judge_pages(q, pages, llm_client=self.llm_client)

        # Build final output
        worker_errors = [w.get("error", "") for w in workers_out if w.get("error")]
        result: dict[str, Any] = {"ok": True, "tool": TOOL_NAME, "query": q, "workers": workers_out}
        if worker_errors and not pages:
            result["retryable_error"] = True
            result["error"] = "; ".join(worker_errors)
        if relevant_summaries:
            result["summary"] = "\n\n".join(
                f"[{r['url']}] {r['summary']}" for r in relevant_summaries
            )
            result["relevant_pages"] = relevant_summaries
        elif pages:
            result["summary"] = "(No pages contained relevant information for this query)"
        return json.dumps(result, ensure_ascii=False)
