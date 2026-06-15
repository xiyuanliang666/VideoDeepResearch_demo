#!/usr/bin/env python3
"""rsagent-v4 baseline entrypoint: single video + single question research agent.

Calls src.rsagent.pipeline.run_pipeline with the Coordinator+Worker dual-context
architecture. Produces a result.json trace suitable for convert_to_submission.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_PROJ = Path(__file__).resolve().parents[1]
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from src.rsagent.env_loader import load_dotenv
from src.rsagent.llm_client import LLMClient, VLLMConfig
from src.rsagent.pipeline import run_pipeline
from src.rsagent.preprocessing import run_preprocessing, uniform_sample_frames
from src.rsagent.tools.dispatcher import build_dispatcher


def _load_prompt(name: str) -> str:
    path = _PROJ / "prompts" / "rsagent" / name
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    return ""


def main() -> None:
    load_dotenv(_PROJ)

    p = argparse.ArgumentParser(description="rsagent-v4: structured video research agent")
    p.add_argument("--video", required=True, help="Local video file path")
    p.add_argument("--question", required=True, help="Single question to answer")
    p.add_argument("--out-dir", default=None, help="Output directory for trace files")
    p.add_argument("--skip-preprocessing", action="store_true", help="Skip visual preprocessing stage")
    p.add_argument("--disable-ocr", action="store_true", help="Disable OCR preprocessing")
    p.add_argument("--disable-yolo", action="store_true", help="Disable YOLO object detection preprocessing")
    p.add_argument("--overview-frames", type=int, default=20, help="Number of uniform overview frames")
    p.add_argument("--max-rounds", type=int, default=12, help="Max evidence collection rounds")
    p.add_argument("--vllm-base-url", default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    p.add_argument("--vllm-model", default=os.environ.get("VLLM_MODEL", "Qwen/Qwen3-VL-32B-Instruct"))
    p.add_argument("--vllm-api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--serper-api-key", default=os.environ.get("SERPER_API_KEY"))
    p.add_argument("--jina-api-key", default=os.environ.get("JINA_API_KEY"))
    p.add_argument("--serper-results-per-worker", type=int, default=3)
    p.add_argument("--evidence-domain", default=None)
    args = p.parse_args()

    video_path = str(Path(args.video).expanduser().resolve())
    if not Path(video_path).is_file():
        p.error(f"Video not found: {video_path}")

    # Build LLM client
    cfg = VLLMConfig(
        base_url=args.vllm_base_url, api_key=args.vllm_api_key,
        model=args.vllm_model, max_tokens=args.max_tokens, temperature=args.temperature,
    )
    llm = LLMClient(cfg)

    # Build tool dispatcher
    dispatcher = build_dispatcher(
        video_path=video_path, serper_api_key=args.serper_api_key,
        jina_api_key=args.jina_api_key, llm_client=llm,
        results_per_worker=args.serper_results_per_worker, evidence_domain=args.evidence_domain,
    )

    # Stage 1: Overview frames + optional preprocessing
    print(f"Sampling {args.overview_frames} overview frames...")
    overview_frames = uniform_sample_frames(video_path, n_frames=args.overview_frames)
    print(f"  -> {len(overview_frames)} frames extracted")

    preprocessing_result = None
    if not args.skip_preprocessing:
        print("Running visual preprocessing...")
        preprocessing_result = run_preprocessing(
            video_path,
            overview_frames,
            llm,
            enable_ocr=not args.disable_ocr,
            enable_yolo=not args.disable_yolo,
        )
        n_scenes = len(preprocessing_result.get("scenes", []))
        ocr = preprocessing_result.get("ocr", {}) if isinstance(preprocessing_result, dict) else {}
        detections = preprocessing_result.get("detections", {}) if isinstance(preprocessing_result, dict) else {}
        print(f"  -> {n_scenes} scenes identified")
        print(f"  -> OCR enabled: {ocr.get('enabled', False)}; YOLO enabled: {detections.get('enabled', False)}")

    # Load prompts from shared prompts/rsagent/
    system_prompt = _load_prompt("system.txt")
    planning_prompt = _load_prompt("planning.txt")
    evidence_loop_prompt = _load_prompt("evidence_loop.txt")
    binding_prompt = _load_prompt("binding.txt")
    final_answer_prompt = _load_prompt("final_answer.txt")
    coordinator_prompt = _load_prompt("coordinator.txt")
    worker_prompt = _load_prompt("worker.txt")

    # Run pipeline
    out_dir = Path(args.out_dir) if args.out_dir else None
    print(f"Running pipeline (max_rounds={args.max_rounds})...")

    result = run_pipeline(
        video_path=video_path, question=args.question, llm=llm, dispatcher=dispatcher,
        system_prompt=system_prompt, planning_prompt=planning_prompt,
        evidence_loop_prompt=evidence_loop_prompt, binding_prompt=binding_prompt,
        final_answer_prompt=final_answer_prompt, overview_frames=overview_frames,
        preprocessing_result=preprocessing_result, max_rounds=args.max_rounds, out_dir=out_dir,
        coordinator_prompt=coordinator_prompt, worker_prompt=worker_prompt,
    )

    # Output
    print("\n" + "=" * 72)
    print("FINAL ANSWER")
    print("=" * 72)
    print(result.get("final_answer", "(empty)"))

    if out_dir:
        print(f"\nTrace saved to: {out_dir}/")


if __name__ == "__main__":
    main()
