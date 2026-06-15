"""Simplified message buffer for rsagent-v4 pipeline."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, List


class ResearchContext:
    def __init__(self, *, system: str | None = None) -> None:
        self._messages: List[dict[str, Any]] = []
        if system:
            self._messages.append({"role": "system", "content": system})

    def messages(self) -> List[dict[str, Any]]:
        return self._messages

    def append_user(self, content: str | list[dict[str, Any]]) -> None:
        self._messages.append({"role": "user", "content": content})

    def append_assistant(self, content: str) -> None:
        self._messages.append({"role": "assistant", "content": content})

    def snapshot_redacted(self) -> List[dict[str, Any]]:
        """Return messages with large base64 payloads truncated."""
        out: List[dict[str, Any]] = []
        for m in self._messages:
            mc = deepcopy(m)
            c = mc.get("content")
            if isinstance(c, list):
                mc["content"] = [_redact_part(p) if isinstance(p, dict) else p for p in c]
            out.append(mc)
        return out


def _redact_part(p: dict[str, Any]) -> dict[str, Any]:
    p = dict(p)
    if p.get("type") == "image_url":
        url = (p.get("image_url") or {}).get("url", "")
        if isinstance(url, str) and "base64," in url and len(url) > 200:
            p["image_url"] = {"url": url[:60] + "...[base64 redacted]..."}
    return p
