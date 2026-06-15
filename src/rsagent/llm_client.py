"""vLLM OpenAI-compatible client for Qwen3-VL (adapted from rsagent-v3)."""

from __future__ import annotations

import base64
import mimetypes
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, List


@dataclass
class VLLMConfig:
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "EMPTY"
    model: str = "Qwen3-VL-8B-Instruct"
    max_tokens: int = 4096
    temperature: float = 0.7


def image_content_part(image_path: str, *, max_side: int | None = None) -> dict[str, Any]:
    """Local path -> base64 data URL image_url part."""
    v = image_path.strip()
    if v.startswith(("http://", "https://")):
        return {"type": "image_url", "image_url": {"url": v}}

    path = Path(v).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    side = max_side if max_side is not None else _image_max_side_from_env()
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "image/jpeg"

    raw = path.read_bytes()
    if side > 0:
        raw, mime = _downscale(raw, max_side=side, fallback_mime=mime)

    b64 = base64.standard_b64encode(raw).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def image_content_part_from_bytes(data: bytes, mime: str = "image/jpeg") -> dict[str, Any]:
    """Raw bytes -> base64 data URL image_url part."""
    b64 = base64.standard_b64encode(data).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def chat_complete(cfg: VLLMConfig, messages: List[dict[str, Any]]) -> str:
    """Call vLLM chat completions (OpenAI SDK with httpx fallback)."""
    try:
        from openai import OpenAI
        client = OpenAI(base_url=cfg.base_url.rstrip("/"), api_key=cfg.api_key, timeout=600.0)
        r = client.chat.completions.create(
            model=cfg.model, messages=messages,
            max_tokens=cfg.max_tokens, temperature=cfg.temperature,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        msg = f"{type(e).__name__}: {e}".lower()
        if not any(t in msg for t in ("validation", "video_url", "image_url", "extra_forbidden")):
            raise
        import httpx
        url = cfg.base_url.rstrip("/") + "/chat/completions"
        payload = {"model": cfg.model, "messages": messages, "max_tokens": cfg.max_tokens, "temperature": cfg.temperature}
        headers = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}
        with httpx.Client(timeout=600.0) as h:
            resp = h.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            return (resp.json()["choices"][0]["message"].get("content") or "").strip()


class LLMClient:
    def __init__(self, cfg: VLLMConfig) -> None:
        self.cfg = cfg

    def complete(self, messages: List[dict[str, Any]]) -> str:
        return chat_complete(self.cfg, messages)


def _image_max_side_from_env() -> int:
    raw = (os.environ.get("RESEARCH_IMAGE_MAX_SIDE") or "").strip()
    try:
        return max(0, int(raw)) if raw else 0
    except ValueError:
        return 0


def _downscale(raw: bytes, *, max_side: int, fallback_mime: str) -> tuple[bytes, str]:
    if max_side <= 0:
        return raw, fallback_mime
    try:
        from PIL import Image
        im = Image.open(BytesIO(raw)).convert("RGB")
        im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=88)
        return buf.getvalue(), "image/jpeg"
    except Exception:
        return raw, fallback_mime
