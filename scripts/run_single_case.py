"""Run a single sample through the compact method core."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.schemas import Sample
from src.tools.cache_tools import save_json
from src.tools.config_tools import load_config
from src.tools.env_tools import load_env_file
from src.tools.io_tools import build_run_dir, build_run_id, write_text, write_trace
from videodeepresearch import run as run_videodeepresearch


def parse_args():
    parser = argparse.ArgumentParser(description="Run a single sample with the compact VideoDR-first core.")
    parser.add_argument("--video-path", required=True, help="Path to the input video.")
    parser.add_argument("--question", required=True, help="Question to answer.")
    parser.add_argument(
        "--task-profile",
        default="configs/task_profiles/videodr_v0.yaml",
        help="Path to the task profile yaml.",
    )
    parser.add_argument(
        "--model-profile",
        default="configs/model_profiles/default.yaml",
        help="Path to the model profile yaml.",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional run id. If omitted, one will be generated.",
    )
    parser.add_argument(
        "--mode",
        default="",
        help="Execution mode: workflow or agentic. Empty means follow task profile.",
    )
    return parser.parse_args()


def _resolve_execution_mode(mode_label: str, task_profile: dict) -> str:
    cli_mode = str(mode_label or "").strip().lower()
    if cli_mode in {"workflow", "agentic"}:
        return cli_mode
    configured = str(task_profile.get("mode", "")).strip().lower()
    if configured in {"workflow", "agentic"}:
        return configured
    return "workflow"


def main():
    load_env_file(".env")
    args = parse_args()
    task_profile = load_config(args.task_profile)
    model_profile = load_config(args.model_profile)
    run_id = args.run_id or build_run_id("single_case")
    run_dir = build_run_dir("outputs/traces", run_id)

    sample = Sample(
        sample_id=run_id,
        benchmark_name="single_case",
        input_type="video",
        media_paths=[str(Path(args.video_path))],
        question=args.question,
        output_mode=str(task_profile.get("output_mode", "short_answer")),
        metadata={},
    )

    result = run_videodeepresearch(
        sample=sample,
        task_profile=task_profile,
        model_profile=model_profile,
        mode=args.mode,
        include_raw_trace=True,
    )
    execution_mode = str(result.get("mode") or "")

    write_trace(run_dir, "trace", result)
    save_json(result, Path("outputs/answers") / f"{run_id}.json")
    write_text(Path(run_dir) / "question.txt", args.question)

    print(f"run_id={run_id}")
    print(f"trace_dir={run_dir}")
    print(f"answer_file=outputs/answers/{run_id}.json")
    print(f"execution_mode={execution_mode}")
    print(f"status={result['status']}")
    diagnostics = result.get("diagnostics") or {}
    print(f"clips={diagnostics.get('clip_count', 0)}")
    print(f"observations={diagnostics.get('observation_count', 0)}")


if __name__ == "__main__":
    main()
