"""VideoDR adapter."""

from __future__ import annotations

from src.schemas import Sample


def adapt_sample(raw_sample: dict, index: int = 0) -> Sample:
    sample_id = str(
        raw_sample.get("sample_id") or raw_sample.get("id") or raw_sample.get("qid") or f"videodr_{index:05d}"
    )
    video_path = raw_sample.get("video_path") or raw_sample.get("video") or raw_sample.get("media_path") or ""
    return Sample(
        sample_id=sample_id,
        benchmark_name="VideoDR",
        input_type="video",
        media_paths=[str(video_path)] if video_path else [],
        question=str(raw_sample.get("question") or raw_sample.get("query") or raw_sample.get("prompt") or ""),
        reference_answer=str(raw_sample.get("reference_answer", raw_sample.get("answer", ""))),
        output_mode="short_answer",
        metadata=raw_sample.get("metadata", {}),
    )

