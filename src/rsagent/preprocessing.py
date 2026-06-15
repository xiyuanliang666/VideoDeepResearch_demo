"""Optional visual preprocessing: scene segmentation, OCR, and object detection."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def uniform_sample_frames(video_path: str, n_frames: int = 20) -> list[dict[str, Any]]:
    """Sample n_frames uniformly from video, return [{file, time_sec, frame_index}]."""
    from decord import VideoReader, cpu
    from PIL import Image

    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    fps = float(vr.get_avg_fps())
    indices = np.linspace(0, total - 1, n_frames, dtype=int).tolist()

    max_side = int(os.environ.get("RESEARCH_IMAGE_MAX_SIDE", "720") or 720)
    out_dir = Path(os.environ.get("VDR_FRAME_CACHE_DIR", "/tmp/vdr_frames")) / "overview"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for idx in indices:
        raw = vr[idx]
        arr = raw.asnumpy() if hasattr(raw, "asnumpy") else np.array(raw)
        im = Image.fromarray(arr)
        if max_side > 0:
            im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        time_sec = round(idx / fps, 3)
        fname = f"overview_{time_sec:.3f}s.jpg"
        fpath = out_dir / fname
        im.save(fpath, quality=88)
        results.append({"file": str(fpath), "time_sec": time_sec, "frame_index": int(idx)})
    return results


PREPROCESSING_PROMPT = """You are a video analysis assistant. Given uniformly sampled overview frames from a video, analyze and output a structured JSON describing the video content.

Output a JSON object with:
{
  "scenes": [
    {
      "scene_id": "scene_1",
      "time_range": [start_sec, end_sec],
      "description": "Brief scene description",
      "key_entities": ["entity1", "entity2"],
      "representative_frame_times": [t1, t2]
    }
  ],
  "video_summary": "One paragraph summary of the entire video"
}

Rules:
- Segment into logical scenes (3-10 scenes typically)
- Identify key entities: people, objects, locations, text shown
- Each scene should have at least one representative frame time from the overview
- Output ONLY valid JSON, no other text"""


def run_preprocessing(
    video_path: str,
    overview_frames: list[dict[str, Any]],
    llm_client: Any,
    system_prompt: str | None = None,
    *,
    enable_ocr: bool = True,
    enable_yolo: bool = True,
) -> dict[str, Any]:
    """Run visual preprocessing over overview frames."""
    from src.rsagent.llm_client import image_content_part
    from src.rsagent.research_context import ResearchContext

    sys_prompt = system_prompt or PREPROCESSING_PROMPT
    ctx = ResearchContext(system=sys_prompt)

    parts: list[dict[str, Any]] = []
    for frame in overview_frames:
        parts.append(image_content_part(frame["file"]))
    time_info = ", ".join(f"{f['time_sec']:.1f}s" for f in overview_frames)
    parts.append({"type": "text", "text": f"Overview frame timestamps: [{time_info}]\nAnalyze these frames and output the scene segmentation JSON."})

    ctx.append_user(parts)
    response = llm_client.complete(ctx.messages())

    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(response[start:end])
        else:
            result = {"scenes": [], "video_summary": response, "parse_error": "No JSON found"}
    except json.JSONDecodeError:
        result = {"scenes": [], "video_summary": response, "parse_error": "Invalid JSON"}

    result["video_path"] = video_path
    return enrich_preprocessing_result(
        result,
        overview_frames,
        enable_ocr=enable_ocr,
        enable_yolo=enable_yolo,
    )


def enrich_preprocessing_result(
    result: dict[str, Any],
    overview_frames: list[dict[str, Any]],
    *,
    enable_ocr: bool = True,
    enable_yolo: bool = True,
) -> dict[str, Any]:
    """Add optional OCR and detector metadata to an existing preprocessing result."""
    enriched = dict(result)
    warnings: list[str] = list(enriched.get("preprocessing_warnings") or [])

    ocr = run_ocr_on_frames(overview_frames, enabled=enable_ocr)
    enriched["ocr"] = ocr
    _extend_warnings(warnings, ocr)

    detections = run_object_detection_on_frames(overview_frames, enabled=enable_yolo)
    enriched["detections"] = detections
    _extend_warnings(warnings, detections)

    enriched["preprocessing_warnings"] = warnings
    return enriched


def run_ocr_on_frames(
    frames: list[dict[str, Any]],
    *,
    enabled: bool = True,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Run optional OCR over sampled frames with graceful degradation."""
    if not enabled or not _env_bool("VDR_ENABLE_OCR", True):
        return {"enabled": False, "engine": "rapidocr_onnxruntime", "reason": "disabled"}

    try:
        from rapidocr_onnxruntime import RapidOCR
    except Exception as exc:
        return {
            "enabled": False,
            "engine": "rapidocr_onnxruntime",
            "error": "dependency_missing",
            "message": f"Install rapidocr-onnxruntime to enable OCR ({type(exc).__name__}: {exc}).",
        }

    limit = max_frames if max_frames is not None else _env_int("VDR_OCR_MAX_FRAMES", 20)
    min_conf = _env_float("VDR_OCR_MIN_CONF", 0.35)
    max_texts = _env_int("VDR_OCR_MAX_TEXTS_PER_FRAME", 20)

    try:
        ocr = RapidOCR()
    except Exception as exc:
        return {
            "enabled": False,
            "engine": "rapidocr_onnxruntime",
            "error": "init_failed",
            "message": f"OCR initialization failed: {type(exc).__name__}: {exc}",
        }

    frame_rows: list[dict[str, Any]] = []
    all_text: list[str] = []
    seen_texts: set[str] = set()
    warnings: list[str] = []

    for frame in frames[: max(0, limit)]:
        texts: list[dict[str, Any]] = []
        try:
            raw_result = ocr(frame["file"])
            for item in _iter_ocr_items(raw_result):
                text = str(item.get("text", "")).strip()
                conf = float(item.get("confidence", 0.0) or 0.0)
                if not text or conf < min_conf:
                    continue
                texts.append({
                    "text": text,
                    "confidence": round(conf, 4),
                    "bbox": item.get("bbox"),
                })
                key = " ".join(text.lower().split())
                if key and key not in seen_texts:
                    seen_texts.add(key)
                    all_text.append(text)
                if len(texts) >= max_texts:
                    break
        except Exception as exc:
            warnings.append(f"OCR failed at {frame.get('time_sec')}s: {type(exc).__name__}: {exc}")

        if texts:
            frame_rows.append({
                "time_sec": frame.get("time_sec"),
                "frame_index": frame.get("frame_index"),
                "file": frame.get("file"),
                "texts": texts,
            })

    result: dict[str, Any] = {
        "enabled": True,
        "engine": "rapidocr_onnxruntime",
        "frames": frame_rows,
        "all_text": all_text[: _env_int("VDR_OCR_MAX_ALL_TEXT", 80)],
    }
    if warnings:
        result["warnings"] = warnings[:20]
    return result


def run_object_detection_on_frames(
    frames: list[dict[str, Any]],
    *,
    enabled: bool = True,
    model_name: str | None = None,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Run optional YOLO object detection over sampled frames with graceful degradation."""
    if not enabled or not _env_bool("VDR_ENABLE_YOLO", True):
        return {"enabled": False, "engine": "ultralytics_yolo", "reason": "disabled"}

    try:
        from ultralytics import YOLO
    except Exception as exc:
        return {
            "enabled": False,
            "engine": "ultralytics_yolo",
            "error": "dependency_missing",
            "message": f"Install ultralytics to enable YOLO detection ({type(exc).__name__}: {exc}).",
        }

    model_id = model_name or os.environ.get("VDR_YOLO_MODEL") or "yolov8n.pt"
    limit = max_frames if max_frames is not None else _env_int("VDR_YOLO_MAX_FRAMES", 20)
    conf = _env_float("VDR_YOLO_CONF", 0.25)
    max_objects = _env_int("VDR_YOLO_MAX_OBJECTS_PER_FRAME", 30)

    try:
        model = YOLO(model_id)
    except Exception as exc:
        return {
            "enabled": False,
            "engine": "ultralytics_yolo",
            "model": model_id,
            "error": "init_failed",
            "message": f"YOLO initialization failed: {type(exc).__name__}: {exc}",
        }

    frame_rows: list[dict[str, Any]] = []
    labels = Counter()
    warnings: list[str] = []

    for frame in frames[: max(0, limit)]:
        objects: list[dict[str, Any]] = []
        try:
            predictions = model(frame["file"], conf=conf, verbose=False)
            for pred in predictions:
                names = getattr(pred, "names", {}) or {}
                boxes = getattr(pred, "boxes", None)
                if boxes is None:
                    continue
                for box in boxes[:max_objects]:
                    cls_id = int(box.cls[0].item()) if hasattr(box.cls[0], "item") else int(box.cls[0])
                    label = str(names.get(cls_id, cls_id))
                    score = float(box.conf[0].item()) if hasattr(box.conf[0], "item") else float(box.conf[0])
                    xyxy = box.xyxy[0].tolist() if hasattr(box.xyxy[0], "tolist") else list(box.xyxy[0])
                    objects.append({
                        "label": label,
                        "confidence": round(score, 4),
                        "bbox_xyxy": [round(float(x), 2) for x in xyxy],
                    })
                    labels[label] += 1
                    if len(objects) >= max_objects:
                        break
                if len(objects) >= max_objects:
                    break
        except Exception as exc:
            warnings.append(f"YOLO failed at {frame.get('time_sec')}s: {type(exc).__name__}: {exc}")

        if objects:
            frame_rows.append({
                "time_sec": frame.get("time_sec"),
                "frame_index": frame.get("frame_index"),
                "file": frame.get("file"),
                "objects": objects,
            })

    result: dict[str, Any] = {
        "enabled": True,
        "engine": "ultralytics_yolo",
        "model": model_id,
        "frames": frame_rows,
        "top_labels": [label for label, _ in labels.most_common(_env_int("VDR_YOLO_MAX_TOP_LABELS", 30))],
    }
    if warnings:
        result["warnings"] = warnings[:20]
    return result


def _iter_ocr_items(raw_result: Any) -> list[dict[str, Any]]:
    """Normalize common RapidOCR output variants into {bbox,text,confidence}."""
    if raw_result is None:
        return []

    if isinstance(raw_result, tuple):
        raw_result = raw_result[0] if raw_result else []

    if hasattr(raw_result, "to_json"):
        try:
            raw_result = json.loads(raw_result.to_json())
        except Exception:
            pass

    if isinstance(raw_result, dict):
        for key in ("result", "results", "data", "rec_res"):
            value = raw_result.get(key)
            if isinstance(value, list):
                raw_result = value
                break

    items: list[dict[str, Any]] = []
    if not isinstance(raw_result, list):
        return items

    for row in raw_result:
        if isinstance(row, dict):
            text = row.get("text") or row.get("transcription") or row.get("rec_text") or ""
            confidence = row.get("confidence") or row.get("score") or row.get("rec_score") or 0.0
            bbox = row.get("bbox") or row.get("box") or row.get("dt_box")
            items.append({"text": text, "confidence": confidence, "bbox": _jsonify_bbox(bbox)})
            continue

        if isinstance(row, (list, tuple)) and len(row) >= 3:
            bbox, text, confidence = row[0], row[1], row[2]
            items.append({"text": text, "confidence": confidence, "bbox": _jsonify_bbox(bbox)})

    return items


def _jsonify_bbox(bbox: Any) -> Any:
    if bbox is None:
        return None
    if hasattr(bbox, "tolist"):
        bbox = bbox.tolist()
    if isinstance(bbox, tuple):
        bbox = list(bbox)
    if isinstance(bbox, list):
        return [_jsonify_bbox(item) for item in bbox]
    try:
        return round(float(bbox), 2)
    except (TypeError, ValueError):
        return bbox


def _extend_warnings(warnings: list[str], section: dict[str, Any]) -> None:
    if not section.get("enabled", False):
        message = section.get("message") or section.get("reason")
        if message:
            warnings.append(str(message))
    for warning in section.get("warnings", []) or []:
        warnings.append(str(warning))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default
