"""Local CV utilities: dense frame extraction and scene change detection.

Requires: opencv-python, Pillow
Optional: torch + open_clip_torch for CLIP embeddings (MPS-accelerated on Apple Silicon)
"""

from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path
from typing import Any


def extract_dense_frames(
    video_path: str,
    fps: float = 1.0,
    output_dir: str = "data/dense_frames",
) -> list[dict[str, Any]]:
    """Extract frames at uniform fps from the full video.

    Returns list of {frame_path, time_sec}.
    Falls back to empty list when ffmpeg is unavailable.
    """
    from src.tools.io_tools import ensure_dir

    ensure_dir(output_dir)
    out_dir = Path(output_dir)
    pattern = str(out_dir / "frame_%06d.jpg")

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",
        pattern,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return []

    frames = sorted(out_dir.glob("frame_*.jpg"))
    interval = 1.0 / fps
    return [
        {"frame_path": str(f), "time_sec": round(i * interval, 3)}
        for i, f in enumerate(frames)
    ]


def detect_scene_changes(
    frames: list[dict[str, Any]],
    threshold: float = 30.0,
) -> list[dict[str, Any]]:
    """Group frames into scene segments using histogram difference.

    Returns list of {start_sec, end_sec, frame_indices}.
    Falls back to a single segment when opencv is unavailable.
    """
    if not frames:
        return []

    try:
        import cv2
        import numpy as np
    except ImportError:
        # No opencv: treat entire video as one segment
        return [{
            "start_sec": frames[0]["time_sec"],
            "end_sec": frames[-1]["time_sec"],
            "frame_indices": list(range(len(frames))),
        }]

    segments: list[dict[str, Any]] = []
    seg_start_idx = 0
    prev_hist = None

    for i, frame_info in enumerate(frames):
        img = cv2.imread(frame_info["frame_path"])
        if img is None:
            continue
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        if prev_hist is not None:
            diff = float(np.sum(np.abs(hist - prev_hist)))
            if diff > threshold:
                segments.append({
                    "start_sec": frames[seg_start_idx]["time_sec"],
                    "end_sec": frames[i - 1]["time_sec"],
                    "frame_indices": list(range(seg_start_idx, i)),
                })
                seg_start_idx = i
        prev_hist = hist

    # Final segment
    segments.append({
        "start_sec": frames[seg_start_idx]["time_sec"],
        "end_sec": frames[-1]["time_sec"],
        "frame_indices": list(range(seg_start_idx, len(frames))),
    })
    return segments


def detect_scene_changes_for_video(
    video_path: str,
    frames: list[dict[str, Any]],
    threshold: float = 27.0,
    adaptive_threshold: float = 3.0,
    min_segments: int = 2,
    fallback_window_sec: float = 15.0,
) -> list[dict[str, Any]]:
    """Detect scene segments with PySceneDetect when available.

    Falls back to the lightweight histogram detector so the pipeline remains
    usable in minimal environments.
    """
    if not frames:
        return []
    try:
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import AdaptiveDetector, ContentDetector
    except ImportError:
        return _ensure_scene_diversity(
            detect_scene_changes(frames, threshold=threshold),
            frames,
            min_segments=min_segments,
            fallback_window_sec=fallback_window_sec,
            fallback_reason="pyscenedetect_unavailable",
        )

    try:
        video = open_video(video_path)
        manager = SceneManager()
        # ContentDetector: frame-to-frame content change (hard cuts)
        manager.add_detector(ContentDetector(threshold=threshold))
        # AdaptiveDetector: rolling-average luminance (fades / gradual transitions)
        manager.add_detector(AdaptiveDetector(adaptive_threshold=adaptive_threshold))
        manager.detect_scenes(video)
        scene_list = manager.get_scene_list()
    except Exception:
        return _ensure_scene_diversity(
            detect_scene_changes(frames, threshold=threshold),
            frames,
            min_segments=min_segments,
            fallback_window_sec=fallback_window_sec,
            fallback_reason="pyscenedetect_failed",
        )

    if not scene_list:
        return _ensure_scene_diversity(
            detect_scene_changes(frames, threshold=threshold),
            frames,
            min_segments=min_segments,
            fallback_window_sec=fallback_window_sec,
            fallback_reason="pyscenedetect_no_scenes",
        )

    # Merge very short adjacent scenes (< 2 sec) into neighbours
    merged = _merge_short_scenes(scene_list, min_duration_sec=2.0)

    segments: list[dict[str, Any]] = []
    for start_sec, end_sec in merged:
        indices = [
            index for index, frame in enumerate(frames)
            if start_sec <= float(frame.get("time_sec", 0.0)) <= end_sec
        ]
        if not indices:
            continue
        segments.append({
            "start_sec": start_sec,
            "end_sec": end_sec,
            "frame_indices": indices,
            "detector": "pyscenedetect_hybrid",
        })
    return _ensure_scene_diversity(
        segments or detect_scene_changes(frames, threshold=threshold),
        frames,
        min_segments=min_segments,
        fallback_window_sec=fallback_window_sec,
        fallback_reason="single_scene_detected",
    )


def _merge_short_scenes(
    scene_list: list,
    min_duration_sec: float = 2.0,
) -> list[tuple[float, float]]:
    """Merge adjacent scenes shorter than min_duration_sec into neighbours."""
    if not scene_list:
        return []
    intervals = [
        (float(s.get_seconds()), float(e.get_seconds()))
        for s, e in scene_list
    ]
    merged: list[tuple[float, float]] = []
    i = 0
    while i < len(intervals):
        start, end = intervals[i]
        j = i + 1
        while j < len(intervals):
            dur = intervals[j][1] - intervals[j][0]
            if dur < min_duration_sec:
                end = intervals[j][1]
                j += 1
            else:
                break
        merged.append((start, end))
        i = j
    return merged


def _ensure_scene_diversity(
    segments: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    *,
    min_segments: int,
    fallback_window_sec: float,
    fallback_reason: str,
) -> list[dict[str, Any]]:
    if len(segments) >= max(1, min_segments):
        return segments
    windowed = _split_frames_by_time_window(frames, window_sec=fallback_window_sec)
    if len(windowed) > len(segments):
        for segment in windowed:
            segment["detector"] = "time_window_fallback"
            segment["fallback_reason"] = fallback_reason
        return windowed
    return segments


def _split_frames_by_time_window(
    frames: list[dict[str, Any]],
    *,
    window_sec: float,
) -> list[dict[str, Any]]:
    if not frames:
        return []
    window_sec = max(1.0, float(window_sec or 30.0))
    segments: list[dict[str, Any]] = []
    seg_start_idx = 0
    seg_start_time = float(frames[0].get("time_sec", 0.0))
    for index, frame in enumerate(frames):
        current_time = float(frame.get("time_sec", 0.0))
        if index > seg_start_idx and current_time - seg_start_time >= window_sec:
            segments.append({
                "start_sec": float(frames[seg_start_idx].get("time_sec", 0.0)),
                "end_sec": float(frames[index - 1].get("time_sec", current_time)),
                "frame_indices": list(range(seg_start_idx, index)),
            })
            seg_start_idx = index
            seg_start_time = current_time
    segments.append({
        "start_sec": float(frames[seg_start_idx].get("time_sec", 0.0)),
        "end_sec": float(frames[-1].get("time_sec", 0.0)),
        "frame_indices": list(range(seg_start_idx, len(frames))),
    })
    return [segment for segment in segments if segment["frame_indices"]]


def extract_frame_artifacts(
    frame_paths: list[str],
    *,
    max_frames: int = 4,
    enable_ocr: bool = True,
    enable_object_detection: bool = True,
) -> dict[str, Any]:
    """Extract low-cost local artifacts from representative frames.

    Optional dependencies are used when installed:
    - PaddleOCR for OCR.
    - ultralytics YOLO for object tags.

    The function always returns a stable dict and never raises for missing
    dependencies, so it is safe in benchmark dry runs.
    """
    selected = _pick_evenly(frame_paths, max_frames=max_frames)
    visual_tags = _extract_visual_tags(selected)
    # YOLO (PyTorch) must run before PaddleOCR to avoid a native-library SIGSEGV
    # caused by PaddlePaddle corrupting GPU state that PyTorch depends on.
    object_tags, object_error = _extract_object_tags(selected) if enable_object_detection else ([], "")
    ocr_texts, ocr_error = _extract_ocr_texts(selected) if enable_ocr else ([], "")

    scene_tags = _dedupe(visual_tags + _scene_tags_from_objects(object_tags), limit=12)
    errors = [error for error in [ocr_error, object_error] if error]
    return {
        "frame_paths": selected,
        "ocr_text": " ".join(ocr_texts).strip(),
        "ocr_texts": ocr_texts,
        "object_tags": object_tags,
        "scene_tags": scene_tags,
        "visual_tags": visual_tags,
        "errors": errors,
    }


def consolidate_temporal_semantic_segments(
    segments: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    *,
    enable_clip: bool = True,
    clip_model_name: str = "MobileCLIP-S2",
    clip_pretrained: str = "datacompdr",
    similarity_threshold: float = 0.82,
    adjacent_window_sec: float = 3.0,
    max_merge_duration_sec: float = 90.0,
    max_segments: int = 24,
    max_frames_per_segment: int = 8,
    max_representative_frames: int = 4,
    enable_ocr: bool = True,
    enable_object_detection: bool = True,
) -> list[dict[str, Any]]:
    """Merge adjacent cut proposals into semantic-state segments.

    The merge is intentionally local in time. Editing cuts are allowed to merge
    only with neighboring proposals when CLIP and/or OCR/object cues indicate
    they are likely the same semantic state.
    """
    if not segments or not frames:
        return segments

    proposals = [
        _build_segment_signature(
            segment,
            frames,
            max_frames=max_frames_per_segment,
            enable_ocr=enable_ocr,
            enable_object_detection=enable_object_detection,
        )
        for segment in segments
        if segment.get("frame_indices")
    ]
    if not proposals:
        return segments

    if enable_clip:
        selected_paths: list[str] = []
        selected_refs: list[tuple[int, int]] = []
        for proposal_index, proposal in enumerate(proposals):
            for selected_index, path in enumerate(proposal["selected_frame_paths"]):
                selected_paths.append(path)
                selected_refs.append((proposal_index, selected_index))
        clip_embeddings = encode_frames_clip(
            selected_paths,
            model_name=clip_model_name,
            pretrained=clip_pretrained,
        ) if selected_paths else None
        if clip_embeddings:
            for (proposal_index, _), embedding in zip(selected_refs, clip_embeddings):
                if embedding:
                    proposals[proposal_index]["clip_embeddings"].append(embedding)
            for proposal in proposals:
                proposal["clip_centroid"] = _centroid(proposal["clip_embeddings"])

    merged: list[dict[str, Any]] = []
    current = proposals[0]
    for proposal in proposals[1:]:
        score, signals = _adjacent_semantic_similarity(current, proposal)
        gap = float(proposal["start_sec"]) - float(current["end_sec"])
        merged_duration = float(proposal["end_sec"]) - float(current["start_sec"])
        should_merge = (
            gap <= adjacent_window_sec
            and merged_duration <= max_merge_duration_sec
            and score >= similarity_threshold
        )
        if should_merge:
            current = _merge_segment_signatures(current, proposal, score=score, signals=signals)
        else:
            merged.append(current)
            current = proposal
    merged.append(current)
    if max_segments > 0 and len(merged) > max_segments:
        merged = _agglomerative_reduce_adjacent_segments(
            merged,
            max_segments=max_segments,
            max_duration_sec=max_merge_duration_sec,
        )

    consolidated: list[dict[str, Any]] = []
    for index, item in enumerate(merged):
        all_indices = sorted(set(item["frame_indices"]))
        representative_indices = _pick_evenly(all_indices, max_frames=max_representative_frames)
        proposal_count = len(item["proposal_indices"])
        consolidated.append({
            "start_sec": float(item["start_sec"]),
            "end_sec": float(item["end_sec"]),
            "frame_indices": representative_indices,
            "source_frame_indices": all_indices,
            "representative_frame_indices": representative_indices,
            "detector": "temporal_semantic_consolidation",
            "proposal_detector": item.get("proposal_detector", ""),
            "proposal_segment_indices": item["proposal_indices"],
            "proposal_segment_count": proposal_count,
            "semantic_merge_count": max(0, proposal_count - 1),
            "semantic_merge_scores": item.get("merge_scores", []),
            "semantic_merge_signals": item.get("merge_signals", []),
            "artifacts": item["artifacts"],
        })
    return consolidated


def _agglomerative_reduce_adjacent_segments(
    segments: list[dict[str, Any]],
    *,
    max_segments: int,
    max_duration_sec: float,
) -> list[dict[str, Any]]:
    """Reduce over-fragmented adjacent segments by best local semantic match."""
    out = list(segments)
    while len(out) > max_segments:
        best_index = -1
        best_score = -1.0
        best_signals: dict[str, Any] = {}
        for index in range(len(out) - 1):
            left, right = out[index], out[index + 1]
            duration = float(right["end_sec"]) - float(left["start_sec"])
            if duration > max_duration_sec:
                continue
            score, signals = _adjacent_semantic_similarity(left, right)
            if score > best_score:
                best_index = index
                best_score = score
                best_signals = signals
        if best_index < 0:
            break
        best_signals = dict(best_signals)
        best_signals["forced_by_max_segments"] = True
        merged = _merge_segment_signatures(
            out[best_index],
            out[best_index + 1],
            score=max(best_score, 0.0),
            signals=best_signals,
        )
        out = out[:best_index] + [merged] + out[best_index + 2:]
    return out


def _build_segment_signature(
    segment: dict[str, Any],
    frames: list[dict[str, Any]],
    *,
    max_frames: int,
    enable_ocr: bool,
    enable_object_detection: bool,
) -> dict[str, Any]:
    indices = [int(i) for i in segment.get("frame_indices", []) if int(i) < len(frames)]
    selected_indices = _pick_evenly(indices, max_frames=max_frames)
    selected_paths = [str(frames[i]["frame_path"]) for i in selected_indices]
    artifacts = extract_frame_artifacts(
        selected_paths,
        max_frames=max_frames,
        enable_ocr=enable_ocr,
        enable_object_detection=enable_object_detection,
    )
    return {
        "start_sec": float(segment.get("start_sec", 0.0)),
        "end_sec": float(segment.get("end_sec", 0.0)),
        "frame_indices": indices,
        "selected_frame_indices": selected_indices,
        "selected_frame_paths": selected_paths,
        "proposal_indices": [int(segment.get("proposal_index", len(indices)))],
        "proposal_detector": str(segment.get("detector", "")),
        "clip_embeddings": [],
        "clip_centroid": [],
        "artifacts": artifacts,
        "merge_scores": [],
        "merge_signals": [],
    }


def _adjacent_semantic_similarity(left: dict[str, Any], right: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    clip_score = _cosine(left.get("clip_centroid") or [], right.get("clip_centroid") or [])
    left_artifacts = left.get("artifacts") or {}
    right_artifacts = right.get("artifacts") or {}
    object_score = _jaccard(left_artifacts.get("object_tags", []), right_artifacts.get("object_tags", []))
    scene_score = _jaccard(left_artifacts.get("scene_tags", []), right_artifacts.get("scene_tags", []))
    visual_score = _jaccard(left_artifacts.get("visual_tags", []), right_artifacts.get("visual_tags", []))
    ocr_score = _token_jaccard(
        str(left_artifacts.get("ocr_text") or ""),
        str(right_artifacts.get("ocr_text") or ""),
    )
    overlap_score = max(object_score, scene_score, visual_score, ocr_score)

    if clip_score > 0:
        score = (0.72 * clip_score) + (0.18 * overlap_score) + (0.10 * scene_score)
    else:
        score = (0.45 * object_score) + (0.35 * scene_score) + (0.15 * visual_score) + (0.05 * ocr_score)
    return float(score), {
        "clip": round(float(clip_score), 4),
        "object_overlap": round(float(object_score), 4),
        "scene_overlap": round(float(scene_score), 4),
        "visual_overlap": round(float(visual_score), 4),
        "ocr_overlap": round(float(ocr_score), 4),
        "combined": round(float(score), 4),
    }


def _merge_segment_signatures(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    score: float,
    signals: dict[str, Any],
) -> dict[str, Any]:
    merged_embeddings = (left.get("clip_embeddings") or []) + (right.get("clip_embeddings") or [])
    return {
        "start_sec": min(float(left["start_sec"]), float(right["start_sec"])),
        "end_sec": max(float(left["end_sec"]), float(right["end_sec"])),
        "frame_indices": sorted(set((left.get("frame_indices") or []) + (right.get("frame_indices") or []))),
        "selected_frame_indices": (left.get("selected_frame_indices") or []) + (right.get("selected_frame_indices") or []),
        "selected_frame_paths": (left.get("selected_frame_paths") or []) + (right.get("selected_frame_paths") or []),
        "proposal_indices": (left.get("proposal_indices") or []) + (right.get("proposal_indices") or []),
        "proposal_detector": left.get("proposal_detector") or right.get("proposal_detector") or "",
        "clip_embeddings": merged_embeddings,
        "clip_centroid": _centroid(merged_embeddings),
        "artifacts": _merge_artifacts(left.get("artifacts") or {}, right.get("artifacts") or {}),
        "merge_scores": (left.get("merge_scores") or []) + [round(float(score), 4)] + (right.get("merge_scores") or []),
        "merge_signals": (left.get("merge_signals") or []) + [signals] + (right.get("merge_signals") or []),
    }


def _merge_artifacts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {
        "frame_paths": _dedupe((left.get("frame_paths") or []) + (right.get("frame_paths") or []), limit=32),
        "ocr_text": " ".join([str(left.get("ocr_text") or ""), str(right.get("ocr_text") or "")]).strip(),
        "ocr_texts": _dedupe((left.get("ocr_texts") or []) + (right.get("ocr_texts") or []), limit=40),
        "object_tags": _dedupe((left.get("object_tags") or []) + (right.get("object_tags") or []), limit=30),
        "scene_tags": _dedupe((left.get("scene_tags") or []) + (right.get("scene_tags") or []), limit=30),
        "visual_tags": _dedupe((left.get("visual_tags") or []) + (right.get("visual_tags") or []), limit=30),
        "errors": _dedupe((left.get("errors") or []) + (right.get("errors") or []), limit=20),
    }


def _centroid(vectors: list[list[float]]) -> list[float]:
    usable = [vector for vector in vectors if vector]
    if not usable:
        return []
    size = len(usable[0])
    values = [0.0] * size
    count = 0
    for vector in usable:
        if len(vector) != size:
            continue
        count += 1
        for index, value in enumerate(vector):
            values[index] += float(value)
    if count <= 0:
        return []
    values = [value / count for value in values]
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values] if norm else []


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return max(0.0, min(1.0, sum(float(a) * float(b) for a, b in zip(left, right))))


def _jaccard(left: list[str], right: list[str]) -> float:
    left_set = {str(item).strip().lower() for item in left if str(item).strip()}
    right_set = {str(item).strip().lower() for item in right if str(item).strip()}
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = {token for token in left.lower().split() if len(token) > 2}
    right_tokens = {token for token in right.lower().split() if len(token) > 2}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _pick_evenly(items: list[str], *, max_frames: int) -> list[str]:
    if not items or max_frames <= 0:
        return []
    if len(items) <= max_frames:
        return list(items)
    step = len(items) / max_frames
    return [items[int(i * step)] for i in range(max_frames)]


def _extract_visual_tags(frame_paths: list[str]) -> list[str]:
    if not frame_paths:
        return []
    try:
        import cv2
        import numpy as np
    except ImportError:
        return _extract_visual_tags_pillow(frame_paths)

    tags: list[str] = []
    brightness_values: list[float] = []
    saturation_values: list[float] = []
    green_ratios: list[float] = []
    edge_values: list[float] = []
    for path in frame_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        brightness_values.append(float(np.mean(hsv[:, :, 2])))
        saturation_values.append(float(np.mean(hsv[:, :, 1])))
        green_mask = (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85) & (hsv[:, :, 1] > 40)
        green_ratios.append(float(np.mean(green_mask)))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        edge_values.append(float(np.mean(edges > 0)))

    if not brightness_values:
        return ["visual_frames_available"]

    brightness = sum(brightness_values) / len(brightness_values)
    saturation = sum(saturation_values) / len(saturation_values)
    green_ratio = sum(green_ratios) / len(green_ratios)
    edge_density = sum(edge_values) / len(edge_values)

    tags.append("bright_scene" if brightness >= 120 else "dark_scene")
    if saturation >= 75:
        tags.append("high_color_saturation")
    if green_ratio >= 0.25:
        tags.append("green_field_or_outdoor_surface")
    if edge_density >= 0.08:
        tags.append("high_detail_scene")
    return tags


def _extract_visual_tags_pillow(frame_paths: list[str]) -> list[str]:
    try:
        from PIL import Image, ImageStat
    except ImportError:
        return ["visual_frames_available"]

    brightness_values: list[float] = []
    saturation_values: list[float] = []
    green_ratios: list[float] = []
    for path in frame_paths:
        try:
            image = Image.open(path).convert("RGB").resize((96, 96))
        except Exception:
            continue
        stat = ImageStat.Stat(image)
        r_mean, g_mean, b_mean = stat.mean
        brightness_values.append((r_mean + g_mean + b_mean) / 3.0)
        saturation_values.append(max(stat.stddev))
        pixels = list(image.getdata())
        green_pixels = [
            1 for r, g, b in pixels
            if g > r * 1.15 and g > b * 1.15 and g > 45
        ]
        green_ratios.append(len(green_pixels) / max(len(pixels), 1))

    if not brightness_values:
        return ["visual_frames_available"]

    brightness = sum(brightness_values) / len(brightness_values)
    saturation = sum(saturation_values) / len(saturation_values)
    green_ratio = sum(green_ratios) / len(green_ratios)
    tags = ["bright_scene" if brightness >= 120 else "dark_scene"]
    if saturation >= 45:
        tags.append("high_color_variation")
    if green_ratio >= 0.25:
        tags.append("green_field_or_outdoor_surface")
    return tags


def _extract_ocr_texts(frame_paths: list[str]) -> tuple[list[str], str]:
    if not frame_paths:
        return [], ""
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        return [], "paddleocr_not_installed"

    ocr = None
    init_errors: list[str] = []
    init_attempts = [
        {"use_angle_cls": True, "lang": "en", "show_log": False},
        {"use_textline_orientation": True, "lang": "en"},
        {"lang": "en"},
    ]
    for kwargs in init_attempts:
        try:
            ocr = PaddleOCR(**kwargs)
            break
        except Exception as exc:
            init_errors.append(f"{type(exc).__name__}:{str(exc)[:160]}")
    if ocr is None:
        return [], "paddleocr_init_failed:" + " | ".join(init_errors[:3])

    texts: list[str] = []
    for path in frame_paths:
        try:
            result = ocr.ocr(path, cls=True)
        except Exception:
            continue
        for page in result or []:
            for row in page or []:
                if not isinstance(row, (list, tuple)) or len(row) < 2:
                    continue
                payload = row[1]
                if isinstance(payload, (list, tuple)) and payload:
                    text = str(payload[0]).strip()
                    if text:
                        texts.append(text)
    return _dedupe(texts, limit=20), ""


def _extract_object_tags(frame_paths: list[str]) -> tuple[list[str], str]:
    if not frame_paths:
        return [], ""
    try:
        from ultralytics import YOLO
    except ImportError:
        return [], "ultralytics_not_installed"

    model_path = os.getenv("YOLO_MODEL_PATH", "").strip() or "yolo11n.pt"
    try:
        model = YOLO(model_path)
    except Exception as exc:
        return [], f"yolo_init_failed:{type(exc).__name__}:{str(exc)[:160]}"

    tags: list[str] = []
    for path in frame_paths:
        try:
            results = model(path, verbose=False)
        except Exception:
            continue
        for result in results or []:
            names = getattr(result, "names", {}) or {}
            boxes = getattr(result, "boxes", None)
            classes = getattr(boxes, "cls", []) if boxes is not None else []
            for cls in classes:
                try:
                    tags.append(str(names.get(int(cls), int(cls))))
                except Exception:
                    continue
    return _dedupe(tags, limit=20), ""


def _scene_tags_from_objects(object_tags: list[str]) -> list[str]:
    lowered = {tag.lower() for tag in object_tags}
    tags: list[str] = []
    if {"person", "sports ball"} & lowered:
        tags.append("people_and_ball_activity")
    if "sports ball" in lowered:
        tags.append("sports_scene")
    if "car" in lowered or "bus" in lowered or "truck" in lowered:
        tags.append("road_or_vehicle_scene")
    if "chair" in lowered or "dining table" in lowered:
        tags.append("indoor_room_scene")
    return tags


def _dedupe(items: list[str], *, limit: int) -> list[str]:
    seen: set[str] = set()
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


def encode_frames_clip(
    frame_paths: list[str],
    model_name: str = "MobileCLIP-S2",
    pretrained: str = "datacompdr",
) -> list[list[float]] | None:
    """Encode frames with CLIP. Returns list of embedding vectors, or None on failure."""
    try:
        import open_clip
        import torch
        from PIL import Image
    except ImportError:
        return None

    try:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        model = model.to(device).eval()
    except Exception:
        return None

    embeddings: list[list[float]] = []
    with torch.no_grad():
        for path in frame_paths:
            try:
                img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
                feat = model.encode_image(img)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                embeddings.append(feat.squeeze(0).cpu().tolist())
            except Exception:
                embeddings.append([])
    return embeddings


def encode_text_clip(
    texts: list[str],
    model_name: str = "MobileCLIP-S2",
    pretrained: str = "datacompdr",
) -> list[list[float]] | None:
    """Encode text queries with CLIP. Returns list of embedding vectors, or None on failure."""
    try:
        import open_clip
        import torch
    except ImportError:
        return None

    try:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        tokenizer = open_clip.get_tokenizer(model_name)
        model = model.to(device).eval()
    except Exception:
        return None

    with torch.no_grad():
        tokens = tokenizer(texts).to(device)
        feat = model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().tolist()
