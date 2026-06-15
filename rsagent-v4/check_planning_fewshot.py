#!/usr/bin/env python3
"""Quick planning-only validation for annotation few-shot prompt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from video_research.env_loader import load_dotenv
from video_research.llm_client import LLMClient, VLLMConfig
from video_research.pipeline import _stage_planning
from video_research.preprocessing import run_preprocessing, uniform_sample_frames

CASES = {
    "up": {
        "video": "/tmp/video_cache/EHoHhdbEq1Y.mp4",
        "question": "In the video, the same unrealized destination appears in many ways. That destination corresponds to a real-world place. Who measured the height of that real-world place?",
    },
    "greenbook": {
        "video": "/tmp/video_cache/QxGWpfO9qQw_h264.mp4",
        "question": "In the video, two travelers are released from a police-station situation after a phone-call chain involving a key helper. External sources can identify the real historical figure who was pressured in that chain. What was that person's real name?",
    },
    "vangogh": {
        "video": "/tmp/video_cache/VeMNKQdQekQ_h264.mp4",
        "question": "In the video, the owner of the location where the conversation takes place is visually based on a particular portrait. Which person donated the initial collection of the museum that holds that portrait?",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(CASES), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--overview-frames", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--vllm-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--vllm-model", default="Qwen/Qwen3-VL-32B-Instruct")
    parser.add_argument("--vllm-api-key", default="EMPTY")
    args = parser.parse_args()

    load_dotenv(ROOT)
    case = CASES[args.case]
    system_prompt = (ROOT / "prompts" / "system.txt").read_text(encoding="utf-8").strip()
    planning_prompt = (ROOT / "prompts" / "planning.txt").read_text(encoding="utf-8").strip()
    llm = LLMClient(VLLMConfig(
        base_url=args.vllm_base_url,
        api_key=args.vllm_api_key,
        model=args.vllm_model,
        temperature=args.temperature,
        max_tokens=4096,
    ))

    frames = uniform_sample_frames(case["video"], n_frames=args.overview_frames)
    pre = run_preprocessing(case["video"], frames, llm, enable_ocr=True, enable_yolo=True)
    trace = {"stages": {}}
    sqs = _stage_planning(
        llm=llm,
        question=case["question"],
        system_prompt=system_prompt,
        planning_prompt=planning_prompt,
        overview_frames=frames,
        preprocessing_result=pre,
        trace=trace,
        out_dir=None,
    )

    result = {
        "case": args.case,
        "question": case["question"],
        "ocr_text": pre.get("ocr", {}).get("all_text", []),
        "detector_labels": pre.get("detections", {}).get("top_labels", []),
        "sub_questions": [
            {
                "text": sq.text,
                "visual_evidence_plan": sq.visual_evidence_plan,
                "web_evidence_plan": sq.web_evidence_plan,
            }
            for sq in sqs
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
