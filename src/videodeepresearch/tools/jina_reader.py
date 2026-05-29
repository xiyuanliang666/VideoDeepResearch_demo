"""Jina Reader helper for fetching page bodies."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse


def normalize_http_url(url: str) -> str:
    """Ensure a URL has an HTTP scheme."""
    text = (url or "").strip()
    if not text:
        return text
    parsed = urlparse(text)
    if not parsed.scheme:
        return f"https://{text}"
    return text


def jina_reader_fetch(
    url: str,
    api_key: str | None = None,
    *,
    reader_base: str | None = None,
    timeout_sec: float = 90.0,
) -> dict[str, Any]:
    """Fetch URL body through Jina Reader.

    Returns `{"ok": bool, "url": str, "content": str, "error": str}`.
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
        response = requests.get(jina_url, headers=headers, timeout=timeout_sec)
        if response.status_code >= 400:
            return {
                "ok": False,
                "url": target,
                "content": "",
                "error": f"HTTP {response.status_code}: {response.text[:500]}",
            }
        text = (response.text or "").strip()
        if not text:
            return {"ok": False, "url": target, "content": "", "error": "empty response body"}
        return {"ok": True, "url": target, "content": text, "error": ""}
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "url": target,
            "content": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def truncate_for_llm(text: str, max_chars: int = 24000) -> str:
    """Trim long page text before LLM integration."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n...[truncated for LLM]..."
