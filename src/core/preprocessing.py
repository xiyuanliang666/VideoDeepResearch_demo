"""Compact preprocessing entrypoints."""

from __future__ import annotations

from pathlib import Path

from src.schemas import ObservationUnit, Sample
from src.tools.asr_tools import align_transcript_to_clips, transcribe_audio
from src.tools.audio_tools import extract_audio
from src.tools.cache_tools import save_json
from src.tools.video_tools import cut_video, extract_frames


def build_clips(video_path: str, clip_duration: int, clips_dir: str = "data/clips") -> list:
    return cut_video(
        video_path=video_path,
        clip_duration=clip_duration,
        output_dir=clips_dir,
    )


def build_frames(clip_path: str, fps: float, max_frames: int, frame_dir: str) -> list[str]:
    return extract_frames(
        clip_path=clip_path,
        fps=fps,
        max_frames=max_frames,
        output_dir=frame_dir,
    )


def build_transcript(video_path: str, clips: list, *, asr_backend: str = "none") -> dict:
    audio_path = extract_audio(video_path, output_dir="data/audio")
    transcript_segments = transcribe_audio(audio_path, backend=asr_backend)
    aligned_by_clip = align_transcript_to_clips(transcript_segments, clips)
    payload = {
        "audio_path": audio_path,
        "segments": transcript_segments,
        "aligned_by_clip": aligned_by_clip,
    }
    save_json(payload, Path("data/transcripts") / f"{Path(video_path).stem}_transcript.json")
    return payload


def build_dense_observations(
    video_path: str,
    sample_id: str,
    task_profile: dict,
) -> tuple[list, list[ObservationUnit]]:
    """Extract 1fps dense frames and group into scene-segment observations.

    Used by vdr_v1 pipeline. Returns (raw_frames, observations).
    raw_frames is the list[dict] from local_cv.extract_dense_frames.
    """
    from src.tools.local_cv import (
        consolidate_temporal_semantic_segments,
        detect_scene_changes_for_video,
        extract_dense_frames,
        extract_frame_artifacts,
    )

    dense_fps = float(task_profile.get("dense_frame_fps", 1.0))
    scene_threshold = float(task_profile.get("scene_change_threshold", 30.0))
    min_scene_segments = int(task_profile.get("min_scene_segments", 2))
    scene_fallback_window_sec = float(task_profile.get("scene_fallback_window_sec", 30.0))
    enable_semantic_consolidation = bool(task_profile.get("enable_temporal_semantic_consolidation", True))
    semantic_similarity_threshold = float(task_profile.get("semantic_merge_similarity_threshold", 0.82))
    semantic_adjacent_window_sec = float(task_profile.get("semantic_merge_adjacent_window_sec", 3.0))
    semantic_max_duration_sec = float(task_profile.get("semantic_merge_max_duration_sec", 90.0))
    max_semantic_segments = int(task_profile.get("max_semantic_segments", 24))
    semantic_clip_enabled = bool(task_profile.get("enable_semantic_clip_merge", True))
    semantic_clip_model = str(task_profile.get("semantic_clip_model", "MobileCLIP-S2"))
    semantic_clip_pretrained = str(task_profile.get("semantic_clip_pretrained", "datacompdr"))
    semantic_merge_frames = int(task_profile.get("max_semantic_merge_frames_per_segment", 8))
    artifact_frames = int(task_profile.get("max_artifact_frames_per_segment", 4))
    enable_ocr = bool(task_profile.get("enable_local_ocr", True))
    enable_object_detection = bool(task_profile.get("enable_local_object_detection", True))
    frame_dir = str(Path("data/dense_frames") / Path(video_path).stem)

    raw_frames = extract_dense_frames(video_path, fps=dense_fps, output_dir=frame_dir)
    segments = detect_scene_changes_for_video(
        video_path,
        raw_frames,
        threshold=scene_threshold,
        min_segments=min_scene_segments,
        fallback_window_sec=scene_fallback_window_sec,
    )
    for proposal_index, segment in enumerate(segments):
        segment["proposal_index"] = proposal_index
    if enable_semantic_consolidation:
        segments = consolidate_temporal_semantic_segments(
            segments,
            raw_frames,
            enable_clip=semantic_clip_enabled,
            clip_model_name=semantic_clip_model,
            clip_pretrained=semantic_clip_pretrained,
            similarity_threshold=semantic_similarity_threshold,
            adjacent_window_sec=semantic_adjacent_window_sec,
            max_merge_duration_sec=semantic_max_duration_sec,
            max_segments=max_semantic_segments,
            max_frames_per_segment=semantic_merge_frames,
            max_representative_frames=artifact_frames,
            enable_ocr=enable_ocr,
            enable_object_detection=enable_object_detection,
        )

    observations: list[ObservationUnit] = []
    for index, seg in enumerate(segments):
        indices = seg.get("frame_indices") or []
        seg_frames = [raw_frames[i]["frame_path"] for i in indices if i < len(raw_frames)]
        artifacts = seg.get("artifacts") or extract_frame_artifacts(
            seg_frames,
            max_frames=artifact_frames,
            enable_ocr=enable_ocr,
            enable_object_detection=enable_object_detection,
        )
        scene_clues = []
        if artifacts.get("scene_tags"):
            scene_clues.append("Scene tags: " + ", ".join(artifacts["scene_tags"]))
        if artifacts.get("object_tags"):
            scene_clues.append("Objects: " + ", ".join(artifacts["object_tags"][:12]))
        if artifacts.get("ocr_text"):
            scene_clues.append("OCR: " + str(artifacts["ocr_text"])[:240])
        observations.append(
            ObservationUnit(
                observation_id=f"obs_seg_{index:04d}",
                source_type="video",
                source_path=video_path,
                order_index=index,
                timestamp_start=seg.get("start_sec", 0.0),
                timestamp_end=seg.get("end_sec", 0.0),
                frame_paths=seg_frames,
                ocr_text=str(artifacts.get("ocr_text") or ""),
                scene_clues=scene_clues,
                candidate_entities=[
                    str(item) for item in artifacts.get("object_tags", [])
                ],
                candidate_actions=[
                    str(item) for item in artifacts.get("scene_tags", [])
                ],
                confidence=0.2 if scene_clues else 0.05,
                uncertainty_notes=[
                    "local_preprocess_artifacts",
                    *[f"artifact_warning:{error}" for error in artifacts.get("errors", [])],
                ],
                metadata={
                    "sample_id": sample_id,
                    "segment_index": index,
                    "frame_indices": indices,
                    "source_frame_indices": seg.get("source_frame_indices", indices),
                    "representative_frame_indices": seg.get("representative_frame_indices", indices),
                    "scene_detector": seg.get("detector", "histogram_fallback"),
                    "proposal_detector": seg.get("proposal_detector", ""),
                    "proposal_segment_indices": seg.get("proposal_segment_indices", []),
                    "proposal_segment_count": seg.get("proposal_segment_count", 1),
                    "semantic_merge_count": seg.get("semantic_merge_count", 0),
                    "semantic_merge_scores": seg.get("semantic_merge_scores", []),
                    "semantic_merge_signals": seg.get("semantic_merge_signals", []),
                    "artifact_frame_paths": artifacts.get("frame_paths", []),
                    "object_tags": artifacts.get("object_tags", []),
                    "scene_tags": artifacts.get("scene_tags", []),
                    "visual_tags": artifacts.get("visual_tags", []),
                    "ocr_texts": artifacts.get("ocr_texts", []),
                },
            )
        )
    return raw_frames, observations


def build_observation_candidates(sample: Sample, task_profile: dict) -> tuple[list, list[ObservationUnit], dict]:
    """Create low-level observations from raw media.

    Current v0 focuses on video input while keeping the interface open for image
    and image-set adapters later.
    """
    clip_duration = int(task_profile.get("clip_duration", 30))
    fps = float(task_profile.get("frame_sampling_fps", 2))
    max_frames = int(task_profile.get("max_frames_per_clip", 16))
    enable_asr = bool(task_profile.get("enable_asr", False))
    asr_backend = task_profile.get("asr_backend", "none") if enable_asr else "none"

    primary_media = sample.media_paths[0] if sample.media_paths else ""

    if sample.input_type == "video":
        clips = build_clips(video_path=primary_media, clip_duration=clip_duration)
        observations: list[ObservationUnit] = []
        for index, clip in enumerate(clips):
            clip_frame_dir = str(Path("data/dense_frames") / clip.clip_id)
            frame_paths = build_frames(
                clip_path=clip.clip_path,
                fps=fps,
                max_frames=max_frames,
                frame_dir=clip_frame_dir,
            )
            clip.frame_dir = clip_frame_dir
            clip.frame_count = len(frame_paths)
            observations.append(
                ObservationUnit(
                    observation_id=f"obs_{clip.clip_id}",
                    source_type="video",
                    source_path=clip.clip_path,
                    order_index=index,
                    timestamp_start=clip.start_time,
                    timestamp_end=clip.end_time,
                    frame_paths=frame_paths,
                    metadata={
                        "clip_id": clip.clip_id,
                        "video_name": clip.video_name,
                        "sample_id": sample.sample_id,
                    },
                    quality_flags=list(clip.quality_flags),
                )
            )

        transcript_payload = build_transcript(
            video_path=primary_media,
            clips=clips,
            asr_backend=asr_backend,
        )

        for observation in observations:
            clip_id = observation.metadata.get("clip_id", "")
            aligned_segments = transcript_payload["aligned_by_clip"].get(clip_id, [])
            observation.transcript_text = " ".join(
                segment.get("text", "").strip() for segment in aligned_segments if segment.get("text")
            ).strip()
            observation.audio_path = transcript_payload.get("audio_path", "")
        return clips, observations, transcript_payload

    if sample.input_type in {"image", "image_set"}:
        observations = []
        for index, media_path in enumerate(sample.media_paths):
            observations.append(
                ObservationUnit(
                    observation_id=f"obs_{sample.sample_id}_{index:03d}",
                    source_type="image",
                    source_path=media_path,
                    order_index=index,
                    frame_paths=[media_path],
                    metadata={"sample_id": sample.sample_id},
                    quality_flags=[],
                )
            )
        return [], observations, {"audio_path": "", "segments": [], "aligned_by_clip": {}}

    raise ValueError(f"Unsupported sample input_type: {sample.input_type}")


__all__ = [
    "build_clips",
    "build_frames",
    "build_transcript",
    "build_observation_candidates",
    "build_dense_observations",
]
