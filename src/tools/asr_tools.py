"""ASR utilities for the v0 scaffold."""

from __future__ import annotations

from pathlib import Path


def transcribe_audio(audio_path: str, backend: str = "none") -> list[dict]:
    """Return timestamped transcript segments.

    v0 fallback behavior:
    - if no audio is available, return an empty transcript
    - if backend is not implemented yet, return a placeholder empty transcript
    """
    if not audio_path or not Path(audio_path).exists():
        return []

    if backend in {"none", "", "placeholder", "whisper"}:
        # Whisper integration is intentionally deferred in v0. We keep the shape
        # of the output stable so later stages can already consume transcript data.
        return []

    return []


def align_transcript_to_clips(transcript: list[dict], clips) -> dict[str, list[dict]]:
    """Align transcript segments to generated clips by timestamp overlap."""
    aligned: dict[str, list[dict]] = {}
    for clip in clips:
        clip_segments: list[dict] = []
        for segment in transcript:
            seg_start = float(segment.get("start", 0.0))
            seg_end = float(segment.get("end", seg_start))
            if seg_end >= clip.start_time and seg_start <= clip.end_time:
                clip_segments.append(segment)
        aligned[clip.clip_id] = clip_segments
    return aligned
