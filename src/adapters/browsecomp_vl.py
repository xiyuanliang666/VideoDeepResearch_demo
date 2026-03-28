"""BrowseComp-VL adapter."""

from __future__ import annotations

from src.schemas import Sample


def adapt_sample(raw_sample: dict, index: int = 0) -> Sample:
    sample_id = str(raw_sample.get("sample_id") or raw_sample.get("id") or f"browsecomp_vl_{index:05d}")
    media_paths = raw_sample.get("media_paths") or []
    if not media_paths:
        primary = raw_sample.get("image_path") or raw_sample.get("media_path") or raw_sample.get("video_path") or ""
        media_paths = [primary] if primary else []
    input_type = "image_set" if len(media_paths) > 1 else "image"
    return Sample(
        sample_id=sample_id,
        benchmark_name="BrowseComp-VL",
        input_type=input_type,
        media_paths=[str(item) for item in media_paths],
        question=str(raw_sample.get("question") or raw_sample.get("query") or raw_sample.get("prompt") or ""),
        reference_answer=str(raw_sample.get("reference_answer", raw_sample.get("answer", ""))),
        output_mode="short_answer",
        metadata=raw_sample.get("metadata", {}),
    )
