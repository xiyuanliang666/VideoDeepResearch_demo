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
]
