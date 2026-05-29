"""Event-oriented VLM observation for scene segments."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any


def _load_image_b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")


def _pick_representative_frames(
    frame_paths: list[str],
    max_frames: int = 4,
) -> list[str]:
    """Pick evenly-spaced frames from a segment."""
    if not frame_paths:
        return []
    if len(frame_paths) <= max_frames:
        return frame_paths
    step = len(frame_paths) / max_frames
    return [frame_paths[int(i * step)] for i in range(max_frames)]


def observe_scene_segment(
    *,
    question: str,
    frame_paths: list[str],
    start_sec: float,
    end_sec: float,
    model_profile: dict | None = None,
    max_frames: int = 4,
    max_tokens: int = 400,
) -> dict[str, Any]:
    """Call a lightweight VLM to produce an event-oriented observation for one scene segment.

    Returns a dict with: event_description, entities, actions, temporal_markers, confidence.
    Falls back to an empty observation on any failure.
    """
    from src.tools.env_tools import resolve_openai_config
    from src.tools.model_router import get_model

    config = resolve_openai_config("event_observer")
    fallback = {
        "event_description": "",
        "entities": [],
        "actions": [],
        "temporal_markers": [],
        "confidence": 0.0,
        "error": "",
    }

    if not config.available:
        fallback["error"] = config.missing_message
        return fallback

    try:
        from openai import OpenAI
    except ImportError:
        fallback["error"] = "openai package not installed"
        return fallback

    selected = _pick_representative_frames(frame_paths, max_frames)
    if not selected:
        fallback["error"] = "no frames available"
        return fallback

    system_prompt = (
        "You are a precise video analyst. "
        "Return JSON only with keys: event_description, entities, actions, temporal_markers, confidence."
    )
    user_text = (
        f"Question context: {question}\n\n"
        f"Video segment: {start_sec:.1f}s – {end_sec:.1f}s\n\n"
        "Describe the key event(s) in this segment. "
        "Focus on: what happens, who is involved, what actions occur, "
        "and any temporal markers (before/after, transitions, changes). "
        "Return JSON: {event_description, entities: [], actions: [], temporal_markers: [], confidence: 0-1}"
    )

    content_parts: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for path in selected:
        try:
            b64 = _load_image_b64(path)
            suffix = Path(path).suffix.lower().lstrip(".")
            mime = f"image/{suffix}" if suffix in {"jpg", "jpeg", "png", "webp"} else "image/jpeg"
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })
        except OSError:
            continue

    model_name = get_model("event_observer", model_profile)
    timeout = float(os.getenv("TIMEOUT_SECONDS", "30") or 30)
    client = OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=timeout)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_parts},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        raw = (response.choices[0].message.content or "").strip() if response.choices else ""
    except Exception as exc:
        fallback["error"] = f"{type(exc).__name__}: {exc}"
        return fallback

    import json
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        left, right = raw.find("{"), raw.rfind("}")
        if left >= 0 and right > left:
            try:
                data = json.loads(raw[left:right + 1])
            except json.JSONDecodeError:
                fallback["error"] = "model response not valid JSON"
                return fallback
        else:
            fallback["error"] = "model response not valid JSON"
            return fallback

    if not isinstance(data, dict):
        fallback["error"] = "model response not a JSON object"
        return fallback

    return {
        "event_description": str(data.get("event_description") or ""),
        "entities": [str(e) for e in (data.get("entities") or []) if e],
        "actions": [str(a) for a in (data.get("actions") or []) if a],
        "temporal_markers": [str(m) for m in (data.get("temporal_markers") or []) if m],
        "confidence": max(0.0, min(1.0, float(data.get("confidence") or 0.0))),
        "error": "",
    }


def observe_all_segments(
    *,
    question: str,
    segments: list[dict[str, Any]],
    all_frames: list[dict[str, Any]],
    model_profile: dict | None = None,
    max_frames_per_segment: int = 4,
    max_tokens: int = 400,
) -> list[dict[str, Any]]:
    """Run event observation on every scene segment.

    segments: output of local_cv.detect_scene_changes
    all_frames: output of local_cv.extract_dense_frames
    Returns list of observation dicts, one per segment.
    """
    results: list[dict[str, Any]] = []
    for seg in segments:
        indices = seg.get("frame_indices") or []
        frame_paths = [all_frames[i]["frame_path"] for i in indices if i < len(all_frames)]
        obs = observe_scene_segment(
            question=question,
            frame_paths=frame_paths,
            start_sec=seg.get("start_sec", 0.0),
            end_sec=seg.get("end_sec", 0.0),
            model_profile=model_profile,
            max_frames=max_frames_per_segment,
            max_tokens=max_tokens,
        )
        obs["start_sec"] = seg.get("start_sec", 0.0)
        obs["end_sec"] = seg.get("end_sec", 0.0)
        obs["frame_paths"] = frame_paths
        results.append(obs)
    return results
