"""Compact grounding utilities."""

from __future__ import annotations

import re

from src.schemas import Anchor, Binding, Evidence, ObservationUnit, RetrievalCandidate


def _tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", text.lower()) if token]


def bind_web_candidates(
    anchor: Anchor,
    web_candidates: list[RetrievalCandidate],
    observation_by_clip: dict[str, ObservationUnit],
) -> tuple[list[Evidence], list[Binding]]:
    evidences: list[Evidence] = []
    bindings: list[Binding] = []
    local_context = {}
    if anchor.source_clip_ids:
        observation = observation_by_clip.get(anchor.source_clip_ids[0])
        if observation is not None:
            local_context = {
                "clip_id": anchor.source_clip_ids[0],
                "timestamp_start": observation.timestamp_start,
                "timestamp_end": observation.timestamp_end,
            }
    for candidate in web_candidates:
        candidate_title = candidate.metadata.get("title", "")
        candidate_snippet = candidate.metadata.get("snippet", "")
        candidate_text = " ".join([candidate_title, candidate_snippet]).strip()
        anchor_terms = list(anchor.entities) + list(anchor.actions) + list(anchor.speech_clues) + [anchor.scene_summary]
        overlap = sorted(set(_tokenize(" ".join(anchor_terms))) & set(_tokenize(candidate_text)))
        base_confidence = float(candidate.normalized_score or candidate.score or 0.0)
        overlap_bonus = min(0.5, 0.08 * len(overlap))
        confidence = max(0.05, min(1.0, base_confidence + overlap_bonus))
        relation = "supports" if overlap else "uncertain"
        evidence = Evidence(
            evidence_id=f"evidence_{anchor.anchor_id}_{candidate.candidate_id}",
            evidence_type="web",
            source_ref=candidate.source_ref,
            content_summary=(candidate_title + " | " + candidate_snippet).strip(" |")[:400],
            span=list(anchor.time_span) if anchor.time_span else None,
            confidence=confidence,
            source_url=candidate.metadata.get("url", ""),
            source_clip_id=local_context.get("clip_id", ""),
            source_timestamp=local_context.get("timestamp_start"),
            raw_excerpt=(candidate_snippet[:240] if candidate_snippet else candidate_title[:240]),
        )
        binding = Binding(
            binding_id=f"binding_{anchor.anchor_id}_{candidate.candidate_id}",
            anchor_id=anchor.anchor_id,
            evidence_id=evidence.evidence_id,
            relation=relation,
            matched_entities=[entity for entity in anchor.entities if entity.lower() in overlap],
            matched_actions=[action for action in anchor.actions if action.lower() in overlap],
            matched_temporal_clues=list(anchor.temporal_clues),
            reason=(
                f"Matched lexical clues between anchor and web result: {', '.join(overlap[:8])}"
                if overlap
                else "No explicit lexical overlap found; kept as weak contextual evidence."
            ),
            confidence=confidence,
            judgeable_claim=(
                f"Web result {'supports' if overlap else 'may relate to'} anchor {anchor.anchor_id} "
                f"for clip(s) {', '.join(anchor.source_clip_ids)}."
            ),
        )
        evidences.append(evidence)
        bindings.append(binding)
    return evidences, bindings
