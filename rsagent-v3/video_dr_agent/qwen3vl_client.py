"""
vLLM OpenAI-compatible chat for Qwen3-VL (see chat_qwen3vl_video.py on Vision-DeepResearch).
"""

from __future__ import annotations

import base64
import mimetypes
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, List

from video_dr_agent.evidence_policy import normalize_evidence_domain


def _research_image_max_side_from_env() -> int:
    raw = (os.environ.get("RESEARCH_IMAGE_MAX_SIDE") or "").strip().strip("\"'")
    if not raw:
        return 0
    try:
        return max(0, int(raw, 10))
    except ValueError:
        return 0


def _maybe_downscale_image_bytes(
    raw: bytes,
    *,
    max_side: int,
) -> tuple[bytes, str]:
    """将图像缩放到最长边不超过 ``max_side``，返回 JPEG bytes 与 mime。"""
    if max_side <= 0:
        return raw, ""
    try:
        from PIL import Image
    except ImportError:
        return raw, ""
    try:
        im = Image.open(BytesIO(raw))
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=88, optimize=True)
        return buf.getvalue(), "image/jpeg"
    except Exception:
        return raw, ""


@dataclass
class VLLMResearchConfig:
    base_url: str  # e.g. http://127.0.0.1:8000/v1
    api_key: str = "EMPTY"
    model: str = "Qwen3-VL-8B-Instruct"
    max_tokens: int = 2048
    temperature: float = 0.7


def image_content_part(image: str, *, max_side: int | None = None) -> dict[str, Any]:
    """
    本地路径会读入并编码为 data URL。若 ``max_side`` 未传，则读环境变量
    ``RESEARCH_IMAGE_MAX_SIDE``（>0 时按最长边缩放，减轻 vLLM 多模态 token 与 ``max_model_len`` 压力）。
    """
    v = image.strip()
    if v.startswith(("http://", "https://")):
        return {"type": "image_url", "image_url": {"url": v}}

    path = Path(v).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image path not found: {path}")

    side = max_side if max_side is not None else _research_image_max_side_from_env()

    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "image/jpeg"
    if not mime.startswith("image/"):
        mime = "image/jpeg"

    raw = path.read_bytes()
    if side > 0:
        new_bytes, new_mime = _maybe_downscale_image_bytes(raw, max_side=side)
        if new_mime:
            raw = new_bytes
            mime = new_mime

    b64 = base64.standard_b64encode(raw).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{b64}"},
    }


def video_content_part(video: str) -> dict[str, Any]:
    v = video.strip()
    if v.startswith(("http://", "https://")):
        return {"type": "video_url", "video_url": {"url": v}}

    path = Path(v).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Video path not found: {path}")

    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "video/mp4"
    suffix = path.suffix.lower().lstrip(".") or "mp4"
    if not mime.startswith("video/"):
        mime = f"video/{suffix if suffix != 'jpg' else 'mp4'}"

    raw = path.read_bytes()
    b64 = base64.standard_b64encode(raw).decode("ascii")
    return {
        "type": "video_url",
        "video_url": {"url": f"data:{mime};base64,{b64}"},
    }


def build_initial_user_content(
    *,
    question: str,
    frame_paths: List[str] | None = None,
    video_path: str | None = None,
    include_video: bool = True,
    evidence_domain: str | None = None,
) -> List[dict[str, Any]]:
    parts: List[dict[str, Any]] = []
    if video_path and include_video:
        parts.append(video_content_part(video_path))
    if frame_paths:
        for p in frame_paths:
            parts.append(image_content_part(p))
    dom = normalize_evidence_domain(evidence_domain)
    q = question.strip()
    if dom != "general":
        meta = (
            "[Host — Task metadata]\n"
            f"evidence_domain: {dom}\n"
            "The tool host applies this profile to `deep_research_web_search` worker site filters "
            "unless you pass `search_profile` or `evidence_domain` in that tool call to override.\n\n"
        )
        q = meta + q
    parts.append({"type": "text", "text": q})
    return parts


def chat_create(
    cfg: VLLMResearchConfig,
    messages: List[dict[str, Any]],
) -> str:
    """OpenAI SDK first; httpx fallback on multimodal validation errors."""
    try:
        from openai import OpenAI

        client = OpenAI(base_url=cfg.base_url.rstrip("/"), api_key=cfg.api_key)
        r = client.chat.completions.create(
            model=cfg.model,
            messages=messages,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        msg = f"{type(e).__name__}: {e}".lower()
        if not any(
            t in msg
            for t in (
                "validation",
                "unexpected keyword",
                "video_url",
                "image_url",
                "extra_forbidden",
                "not a valid",
            )
        ):
            raise
        try:
            import httpx
        except ImportError as ie:
            raise RuntimeError(
                "openai SDK rejected multimodal payload; install httpx for fallback: pip install httpx"
            ) from ie

        url = cfg.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": cfg.model,
            "messages": messages,
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
        }
        headers = {
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=600.0) as h:
            resp = h.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        return (data["choices"][0]["message"].get("content") or "").strip()


class Qwen3VLClient:
    def __init__(self, cfg: VLLMResearchConfig) -> None:
        self.cfg = cfg

    def complete(self, messages: List[dict[str, Any]]) -> str:
        return chat_create(self.cfg, messages)
