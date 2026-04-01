"""Low-risk observation utilities."""

from __future__ import annotations

import re

from src.schemas import ObservationUnit
from src.tools.multimodal_tools import analyze_observation_multimodal, get_last_multimodal_error


def _sentence_chunks(text: str) -> list[str]:
    return [item.strip() for item in re.split(r"[。！？!?;\n]+", text) if item.strip()]


def _extract_entities(question: str, text: str) -> list[str]:
    candidates = re.findall(r"\b[A-Z][a-zA-Z0-9\-]{2,}\b", question + " " + text)
    cjk_candidates = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    seen = set()
    entities: list[str] = []
    for token in candidates + cjk_candidates:
        lowered = token.lower()
        if lowered not in seen:
            seen.add(lowered)
            entities.append(token)
    return entities[:8]


def _dedupe(items: list[str], limit: int) -> list[str]:
    seen = set()
    out: list[str] = []
    for item in items:
        value = str(item).strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
        if len(out) >= limit:
            break
    return out


def build_low_risk_observations(
    observations: list[ObservationUnit],
    question: str,
    *,
    task_profile: dict | None = None,
    model_profile: dict | None = None,
) -> list[ObservationUnit]:
    """Populate observations with low-risk, question-conditioned clues.

    The function intentionally avoids strong semantic commitments and instead
    stores candidate entities, short clue spans, and explicit uncertainty notes.
    """
    for observation in observations:
        transcript = observation.transcript_text.strip()
        ocr_text = observation.ocr_text.strip()
        combined_text = " ".join([transcript, ocr_text]).strip()

        observation.scene_clues = _sentence_chunks(combined_text)[:2]
        observation.speech_clues = _sentence_chunks(transcript)[:3]
        observation.candidate_entities = _extract_entities(question, combined_text)
        observation.candidate_actions = []
        observation.confidence = 0.2 if combined_text else 0.05
        observation.uncertainty_notes = [
            "low_risk_observation",
            "candidate_layer_only",
        ]
        if not combined_text and not observation.frame_paths:
            observation.uncertainty_notes.append("very_sparse_signal")
        elif not combined_text:
            observation.uncertainty_notes.append("visual_only_sparse_text")

    task_profile = task_profile or {}
    model_profile = model_profile or {}
    if not bool(task_profile.get("enable_online_multimodal", False)):
        return observations

    max_calls = int(task_profile.get("max_multimodal_observation_calls", 2))
    max_images = int(task_profile.get("max_images_per_multimodal_call", 3))

    used_calls = 0
    for observation in observations:
        if used_calls >= max_calls:
            break
        media_candidates = observation.frame_paths[:max_images]
        if not media_candidates and observation.source_type == "image" and observation.source_path:
            media_candidates = [observation.source_path]
        hints = analyze_observation_multimodal(
            question=question,
            media_paths=media_candidates,
            model_profile=model_profile,
            max_tokens=500,
        )
        used_calls += 1
        if not hints:
            observation.uncertainty_notes.append("online_multimodal_failed_or_unavailable")
            error_message = get_last_multimodal_error()
            if error_message:
                observation.uncertainty_notes.append(f"online_multimodal_error:{error_message}")
            continue

        observation.scene_clues = _dedupe(observation.scene_clues + list(hints.get("scene_clues", [])), 6)
        observation.speech_clues = _dedupe(observation.speech_clues + list(hints.get("speech_clues", [])), 6)
        observation.candidate_entities = _dedupe(
            observation.candidate_entities + list(hints.get("candidate_entities", [])),
            10,
        )
        observation.candidate_actions = _dedupe(
            observation.candidate_actions + list(hints.get("candidate_actions", [])),
            8,
        )
        try:
            llm_confidence = float(hints.get("confidence", 0.0))
        except (TypeError, ValueError):
            llm_confidence = 0.0
        observation.confidence = max(observation.confidence, min(max(llm_confidence, 0.0), 1.0))
        observation.uncertainty_notes.append("online_multimodal_enhanced")
    return observations
