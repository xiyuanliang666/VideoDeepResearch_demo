#!/usr/bin/env python3
"""Run rsagent-v4 visual preprocessing without the full research agent."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from video_research.env_loader import load_dotenv
from video_research.llm_client import LLMClient, VLLMConfig
from video_research.preprocessing import enrich_preprocessing_result, run_preprocessing, uniform_sample_frames


def main() -> None:
    load_dotenv(_ROOT)

    parser = argparse.ArgumentParser(description="Run rsagent-v4 visual preprocessing and save JSON output")
    parser.add_argument("--video", required=True, help="Local video file path")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--overview-frames", type=int, default=20, help="Number of uniform overview frames")
    parser.add_argument("--skip-scene-llm", action="store_true", help="Skip VLM scene segmentation and run only OCR/YOLO enrichment")
    parser.add_argument("--disable-ocr", action="store_true", help="Disable OCR preprocessing")
    parser.add_argument("--disable-yolo", action="store_true", help="Disable YOLO object detection preprocessing")
    parser.add_argument("--vllm-base-url", default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--vllm-model", default=os.environ.get("VLLM_MODEL", "Qwen/Qwen3-VL-32B-Instruct"))
    parser.add_argument("--vllm-api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    video_path = str(Path(args.video).expanduser().resolve())
    if not Path(video_path).is_file():
        parser.error(f"Video not found: {video_path}")

    print(f"Sampling {args.overview_frames} overview frames...")
    overview_frames = uniform_sample_frames(video_path, n_frames=args.overview_frames)
    print(f"  -> {len(overview_frames)} frames extracted")

    enable_ocr = not args.disable_ocr
    enable_yolo = not args.disable_yolo

    if args.skip_scene_llm:
        print("Skipping scene LLM; running OCR/YOLO enrichment only...")
        result: dict[str, Any] = {
            "video_path": video_path,
            "scenes": [],
            "video_summary": "",
            "scene_segmentation_skipped": True,
        }
        result = enrich_preprocessing_result(
            result,
            overview_frames,
            enable_ocr=enable_ocr,
            enable_yolo=enable_yolo,
        )
    else:
        print("Running scene LLM + OCR/YOLO enrichment...")
        cfg = VLLMConfig(
            base_url=args.vllm_base_url,
            api_key=args.vllm_api_key,
            model=args.vllm_model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        llm = LLMClient(cfg)
        result = run_preprocessing(
            video_path,
            overview_frames,
            llm,
            enable_ocr=enable_ocr,
            enable_yolo=enable_yolo,
        )

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Preprocessing JSON saved to: {out_path}")
    _print_summary(result)


def _print_summary(result: dict[str, Any]) -> None:
    scenes = result.get("scenes", []) if isinstance(result.get("scenes"), list) else []
    ocr = result.get("ocr", {}) if isinstance(result.get("ocr"), dict) else {}
    detections = result.get("detections", {}) if isinstance(result.get("detections"), dict) else {}
    warnings = result.get("preprocessing_warnings", []) or []

    ocr_texts = ocr.get("all_text", []) if isinstance(ocr.get("all_text"), list) else []
    det_labels = detections.get("top_labels", []) if isinstance(detections.get("top_labels"), list) else []

    print("Summary:")
    print(f"  scenes: {len(scenes)}")
    print(f"  ocr_enabled: {ocr.get('enabled', False)}")
    print(f"  ocr_text_count: {len(ocr_texts)}")
    if ocr_texts:
        print(f"  ocr_text_preview: {ocr_texts[:10]}")
    print(f"  yolo_enabled: {detections.get('enabled', False)}")
    print(f"  yolo_top_labels: {det_labels[:15]}")
    if warnings:
        print("  warnings:")
        for warning in warnings[:10]:
            print(f"    - {warning}")


if __name__ == "__main__":
    main()
