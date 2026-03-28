"""Low-risk observation utilities."""

from __future__ import annotations

import re

from src.schemas import ObservationUnit


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


def build_low_risk_observations(observations: list[ObservationUnit], question: str) -> list[ObservationUnit]:
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
    return observations
