"""Stable single-sample entrypoint for benchmark adapters.

This wrapper intentionally calls the method-first `src/` pipeline rather than
the historical `rsagent*` snapshots. It gives external benchmark projects a
small, stable CLI surface while preserving the raw trace for later conversion.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.schemas import Sample
from src.tools.cache_tools import save_json
from src.tools.config_tools import load_config
from src.tools.env_tools import load_env_file
from src.tools.io_tools import build_run_dir, build_run_id, write_trace
from videodeepresearch import run as run_videodeepresearch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one VDR sample through the stable method-core entrypoint.")
    parser.add_argument("--sample-id", default="", help="Stable sample id from the benchmark manifest.")
    parser.add_argument("--video-path", required=True, help="Path to the input video.")
    parser.add_argument("--question", required=True, help="Question to answer.")
    parser.add_argument("--reference-answer", default="", help="Optional reference answer for trace metadata.")
    parser.add_argument("--benchmark-name", default="vdr_pilot", help="Benchmark name stored in the trace.")
    parser.add_argument("--run-id", default="", help="Optional run id. If omitted, one is generated.")
    parser.add_argument(
        "--task-profile",
        default="configs/task_profiles/videodr_v0.yaml",
        help="Path to the task profile yaml, relative to the demo repo unless absolute.",
    )
    parser.add_argument(
        "--model-profile",
        default="configs/model_profiles/default.yaml",
        help="Path to the model profile yaml, relative to the demo repo unless absolute.",
    )
    parser.add_argument(
        "--mode",
        default="",
        help="Execution mode: workflow or agentic. Empty means follow task profile.",
    )
    parser.add_argument(
        "--ablation-mode",
        default="",
        choices=["", "full_vdr", "video_only", "web_only", "text_only"],
        help="Optional ablation setting injected into task_profile.",
    )
    parser.add_argument(
        "--out-json",
        required=True,
        help="Where to write the stable entrypoint output JSON.",
    )
    return parser.parse_args()


def _resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    load_env_file(str(PROJECT_ROOT / ".env"), override=True)

    task_profile = load_config(_resolve_repo_path(args.task_profile))
    model_profile = load_config(_resolve_repo_path(args.model_profile))
    if args.ablation_mode:
        task_profile["ablation_mode"] = args.ablation_mode
    sample_id = args.sample_id or build_run_id("vdr_sample")
    run_id = args.run_id or sample_id
    out_path = _resolve_repo_path(args.out_json).expanduser()
    output_root = out_path.parent
    run_dir = Path(build_run_dir(output_root / "traces", run_id))

    sample = Sample(
        sample_id=sample_id,
        benchmark_name=args.benchmark_name,
        input_type="video",
        media_paths=[str(Path(args.video_path).expanduser())],
        question=args.question,
        reference_answer=args.reference_answer,
        output_mode=str(task_profile.get("output_mode", "short_answer")),
        metadata={
            "entrypoint": "baseline_entrypoints/run_vdr_single.py",
            "run_id": run_id,
        },
    )

    os.chdir(output_root)

    result = run_videodeepresearch(
        sample=sample,
        task_profile=task_profile,
        model_profile=model_profile,
        mode=args.mode,
        include_raw_trace=True,
    )
    execution_mode = str(result.get("mode") or "")

    trace_path = run_dir / "trace.json"
    answer_path = output_root / "answers" / f"{run_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    answer_path.parent.mkdir(parents=True, exist_ok=True)

    write_trace(run_dir, "trace", result)
    save_json(result, answer_path)

    stable_output = {
        "entrypoint": "baseline_entrypoints/run_vdr_single.py",
        "sample_id": sample_id,
        "run_id": run_id,
        "execution_mode": execution_mode,
        "trace_path": str(trace_path),
        "answer_path": str(answer_path),
        "result": result,
    }
    save_json(stable_output, out_path)

    print(f"sample_id={sample_id}")
    print(f"run_id={run_id}")
    print(f"execution_mode={execution_mode}")
    print(f"out_json={out_path}")
    print(f"trace_path={trace_path}")
    print(f"answer_path={answer_path}")
    print(f"status={result.get('status', '')}")


if __name__ == "__main__":
    main()
