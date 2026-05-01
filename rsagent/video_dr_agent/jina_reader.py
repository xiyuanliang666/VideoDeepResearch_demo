"""
Jina Reader: synchronous fetch of page text (same idea as the Jina branch in visit_tool).

Default: official Reader GET `https://r.jina.ai/{url}` with header `Authorization: Bearer <JINA_API_KEY>`.
Env: `JINA_API_KEY`.
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse


def normalize_http_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return u
    p = urlparse(u)
    if not p.scheme:
        return f"https://{u}"
    return u


def jina_reader_fetch(
    url: str,
    api_key: str | None = None,
    *,
    reader_base: str | None = None,
    timeout_sec: float = 90.0,
) -> dict[str, Any]:
    """
    Returns {"ok": bool, "url": str, "content": str, "error": str}.
    """
    key = (api_key or os.environ.get("JINA_API_KEY") or "").strip()
    if not key:
        return {"ok": False, "url": url, "content": "", "error": "JINA_API_KEY is not set"}

    try:
        import requests
    except ImportError as exc:
        return {
            "ok": False,
            "url": url,
            "content": "",
            "error": f"requests package required: {exc}",
        }

    target = normalize_http_url(url)
    base = (reader_base or os.environ.get("JINA_READER_BASE", "https://r.jina.ai")).rstrip("/")
    jina_url = f"{base}/{target}"
    headers = {"Authorization": f"Bearer {key}"}
    try:
        resp = requests.get(jina_url, headers=headers, timeout=timeout_sec)
        if resp.status_code >= 400:
            return {
                "ok": False,
                "url": target,
                "content": "",
                "error": f"HTTP {resp.status_code}: {resp.text[:500]}",
            }
        text = (resp.text or "").strip()
        if not text:
            return {
                "ok": False,
                "url": target,
                "content": "",
                "error": "empty response body",
            }
        return {"ok": True, "url": target, "content": text, "error": ""}
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "url": target,
            "content": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def truncate_for_llm(text: str, max_chars: int = 24000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n...[truncated for LLM]..."
