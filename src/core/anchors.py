"""Anchor building utilities."""

from __future__ import annotations

import re

from src.schemas import Anchor, ObservationUnit, RetrievalCandidate


def _sentence_chunks(text: str) -> list[str]:
    parts = re.split(r"[。！？!?;\n]+", text)
    return [part.strip() for part in parts if part.strip()]


def _extract_entities(question: str, transcript_text: str) -> list[str]:
    candidates = re.findall(r"\b[A-Z][a-zA-Z0-9\-]{2,}\b", question + " " + transcript_text)
    cjk_candidates = re.findall(r"[\u4e00-\u9fff]{2,}", transcript_text)
    entities = []
    seen = set()
    for token in candidates + cjk_candidates:
        lowered = token.lower()
        if lowered not in seen:
            seen.add(lowered)
            entities.append(token)
    return entities[:8]


def build_anchors(
    observations: list[ObservationUnit],
    local_candidates: list[RetrievalCandidate],
    question: str,
) -> list[Anchor]:
    """Build anchors from low-risk observations and local candidates."""
    observation_by_clip = {
        obs.metadata.get("clip_id", obs.observation_id): obs
        for obs in observations
    }
    anchors: list[Anchor] = []

    for index, candidate in enumerate(local_candidates):
        observation = observation_by_clip.get(candidate.source_ref)
        if observation is None:
            observation = next(
                (item for item in observations if item.observation_id == candidate.source_ref),
                None,
            )
        if observation is None:
            continue

        transcript_text = observation.transcript_text.strip()
        summary = transcript_text or " ".join(observation.scene_clues).strip() or (
            f"Observation from {observation.timestamp_start} to {observation.timestamp_end}."
        )
        # Use state-derived search queries when available (vdr_v1 path)
        state_queries = list(observation.metadata.get("search_queries") or [])
        if state_queries:
            search_queries = state_queries[:3]
        else:
            search_queries = [question]
            if transcript_text:
                search_queries.extend(_sentence_chunks(transcript_text)[:2])

        anchors.append(
            Anchor(
                anchor_id=f"anchor_{index:04d}",
                source_clip_ids=[candidate.source_ref],
                source_observation_ids=[observation.observation_id],
                time_span=[observation.timestamp_start or 0.0, observation.timestamp_end or 0.0],
                anchor_type="temporal_event",
                entities=observation.candidate_entities or _extract_entities(question, transcript_text),
                actions=observation.candidate_actions,
                scene_summary=summary[:300],
                ocr_clues=[observation.ocr_text] if observation.ocr_text else [],
                subtitle_clues=[],
                speech_clues=observation.speech_clues or (_sentence_chunks(transcript_text)[:3] if transcript_text else []),
                temporal_clues=[f"{observation.timestamp_start}-{observation.timestamp_end}"],
                search_queries=[query for query in search_queries if query][:3],
                confidence=max(candidate.score, observation.confidence, 0.1),
                status="tentative",
                priority_score=max(candidate.score, observation.confidence),
                evidence_ids=[],
                open_slots=["web_grounding", "claim_validation"],
            )
        )

    return anchors
