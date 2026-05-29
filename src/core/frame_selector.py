"""Evidence-oriented frame selection using CLIP cosine similarity."""

from __future__ import annotations

from typing import Any


def select_frames_for_states(
    *,
    states: list[Any],
    all_frames: list[dict[str, Any]],
    top_k: int = 5,
    min_evidence_count: int = 1,
) -> dict[str, list[dict[str, Any]]]:
    """Select supporting frames for each TemporalState using CLIP similarity.

    Falls back to temporal proximity when CLIP is unavailable.
    Returns {state_id: [{frame_path, time_sec, confidence}]}.
    """
    from src.tools.local_cv import encode_frames_clip, encode_text_clip

    frame_paths = [f["frame_path"] for f in all_frames]
    frame_embeddings = encode_frames_clip(frame_paths) if frame_paths else None

    result: dict[str, list[dict[str, Any]]] = {}
    for state in states:
        state_id = state.state_id
        span = state.temporal_span  # [start_sec, end_sec]
        query = state.sub_question or " ".join(state.required_predicates)

        if frame_embeddings and query:
            text_embs = encode_text_clip([query])
            if text_embs:
                selected = _clip_select(
                    all_frames=all_frames,
                    frame_embeddings=frame_embeddings,
                    text_embedding=text_embs[0],
                    top_k=max(top_k, min_evidence_count),
                )
                result[state_id] = selected
                continue

        # Fallback: pick frames within temporal span, or nearest frames
        result[state_id] = _temporal_select(all_frames, span, max(top_k, min_evidence_count))

    return result


def _clip_select(
    *,
    all_frames: list[dict[str, Any]],
    frame_embeddings: list[list[float]],
    text_embedding: list[float],
    top_k: int,
) -> list[dict[str, Any]]:
    import math

    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb + 1e-9)

    scored = [
        (cosine(emb, text_embedding), frame)
        for emb, frame in zip(frame_embeddings, all_frames)
        if emb
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {"frame_path": f["frame_path"], "time_sec": f["time_sec"], "confidence": round(score, 4)}
        for score, f in scored[:top_k]
    ]


def _temporal_select(
    all_frames: list[dict[str, Any]],
    span: list[float],
    max_frames: int,
) -> list[dict[str, Any]]:
    if len(span) >= 2:
        in_span = [f for f in all_frames if span[0] <= f["time_sec"] <= span[1]]
        if in_span:
            step = max(1, len(in_span) // max_frames)
            selected = in_span[::step][:max_frames]
            return [{"frame_path": f["frame_path"], "time_sec": f["time_sec"], "confidence": 0.3}
                    for f in selected]

    # No span: evenly sample
    step = max(1, len(all_frames) // max_frames)
    return [{"frame_path": f["frame_path"], "time_sec": f["time_sec"], "confidence": 0.1}
            for f in all_frames[::step][:max_frames]]
