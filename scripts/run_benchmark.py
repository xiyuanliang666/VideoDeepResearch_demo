"""Run a benchmark split through the compact method core and adapters."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.adapters import get_adapter, list_adapters
from src.evaluation.aggregate import build_comparison_row, summarize_results, write_comparison_csv
from src.tools.cache_tools import load_json, load_jsonl, save_json, save_jsonl
from src.tools.config_tools import load_config
from src.tools.env_tools import load_env_file
from src.tools.io_tools import build_run_dir, build_run_id, write_trace
from videodeepresearch import run as run_videodeepresearch


def parse_args():
    parser = argparse.ArgumentParser(description="Run a benchmark split with the compact core and benchmark adapters.")
    parser.add_argument("--input-file", default="", help="Path to a benchmark .json or .jsonl file.")
    parser.add_argument(
        "--benchmark-config",
        default="",
        help="Optional benchmark yaml config under configs/benchmarks/.",
    )
    parser.add_argument("--task-profile", default="configs/task_profiles/videodr_v0.yaml")
    parser.add_argument("--model-profile", default="configs/model_profiles/default.yaml")
    parser.add_argument("--benchmark-name", default="videodr")
    parser.add_argument(
        "--adapter",
        default="",
        help=f"Adapter name. Available: {', '.join(list_adapters())}",
    )
    parser.add_argument(
        "--mode",
        default="framework",
        help="Run label. If set to 'workflow' or 'agentic', it also controls execution mode.",
    )
    parser.add_argument(
        "--experiment-config",
        default="",
        help="Optional experiment yaml config, e.g. configs/experiment.yaml",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional sample limit. 0 means no limit.")
    parser.add_argument("--run-id", default="", help="Optional benchmark run id.")
    return parser.parse_args()


def load_samples(path: str | Path) -> list[dict]:
    target = Path(path)
    if target.suffix.lower() == ".jsonl":
        rows = load_jsonl(target)
        return [row for row in rows if isinstance(row, dict)]
    payload = load_json(target)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("samples"), list):
            return [row for row in payload["samples"] if isinstance(row, dict)]
        if isinstance(payload.get("data"), list):
            return [row for row in payload["data"] if isinstance(row, dict)]
    raise ValueError(f"Unsupported benchmark input format: {target}")


def build_result_row(
    result: dict,
    sample: dict,
    sample_id: str,
    run_id: str,
    benchmark_name: str,
    mode: str,
    execution_mode: str,
) -> dict:
    final_answer = result.get("final_answer") or {}
    judge_result = result.get("judge_result") or {}
    diagnostics = result.get("diagnostics") or {}
    return {
        "run_id": run_id,
        "benchmark_name": benchmark_name,
        "mode": mode,
        "execution_mode": execution_mode,
        "sample_id": sample_id,
        "video_path": sample.get("video_path") or sample.get("video") or sample.get("media_path") or "",
        "question": sample.get("question") or sample.get("query") or sample.get("prompt") or "",
        "reference_answer": sample.get("reference_answer", sample.get("answer", "")),
        "status": result.get("status", ""),
        "final_answer": final_answer.get("answer_text") or final_answer.get("final_answer", ""),
        "answer_confidence": final_answer.get("confidence"),
        "judge_score": (judge_result or {}).get("overall_score") if isinstance(judge_result, dict) else None,
        "judge_verdict": (judge_result or {}).get("verdict", "") if isinstance(judge_result, dict) else "",
        "anchor_count": diagnostics.get("anchor_count", 0),
        "evidence_count": diagnostics.get("web_evidence_count", 0),
        "binding_count": diagnostics.get("binding_count", 0),
        "supporting_video_evidence": final_answer.get("supporting_video_evidence", []),
        "supporting_web_evidence": final_answer.get("supporting_web_evidence", []),
        "error_message": "",
    }


def _model_tag(name: str) -> str:
    tag = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip())
    tag = re.sub(r"_+", "_", tag).strip("_")
    return tag[:48] if tag else "model"


def _resolve_execution_mode(mode_label: str, task_profile: dict) -> str:
    cli_mode = str(mode_label or "").strip().lower()
    if cli_mode in {"workflow", "agentic"}:
        return cli_mode
    configured = str(task_profile.get("mode", "")).strip().lower()
    if configured in {"workflow", "agentic"}:
        return configured
    return "workflow"


def _run_once(
    *,
    samples: list[dict],
    adapter,
    task_profile: dict,
    model_profile: dict,
    benchmark_name: str,
    mode: str,
    run_id: str,
) -> dict:
    execution_mode = _resolve_execution_mode(mode, task_profile)
    trace_root = Path(build_run_dir("outputs/traces", run_id))
    eval_dir = Path(build_run_dir("outputs/eval", run_id))
    answers_path = Path("outputs/answers") / f"{run_id}.jsonl"
    results_path = eval_dir / "results.jsonl"
    summary_path = eval_dir / "summary.json"
    comparison_path = eval_dir / "compare.csv"

    result_rows: list[dict] = []
    answer_rows: list[dict] = []

    for index, sample in enumerate(samples):
        sample_id = str(sample.get("sample_id") or sample.get("id") or sample.get("qid") or f"sample_{index:05d}")
        try:
            adapted_sample = adapter(sample, index)
            sample_id = adapted_sample.sample_id
            result = run_videodeepresearch(
                sample=adapted_sample,
                task_profile=task_profile,
                model_profile=model_profile,
                mode=execution_mode,
                include_raw_trace=True,
            )
            execution_mode = str(result.get("mode") or execution_mode)
            sample_trace_dir = trace_root / sample_id
            sample_trace_dir.mkdir(parents=True, exist_ok=True)
            write_trace(sample_trace_dir, "trace", result)

            result_row = build_result_row(
                result=result,
                sample=sample,
                sample_id=sample_id,
                run_id=run_id,
                benchmark_name=benchmark_name,
                mode=mode,
                execution_mode=execution_mode,
            )
        except Exception as exc:  # noqa: BLE001
            sample_trace_dir = trace_root / sample_id
            sample_trace_dir.mkdir(parents=True, exist_ok=True)
            write_trace(
                sample_trace_dir,
                "trace",
                {
                    "sample_id": sample_id,
                    "status": "error",
                    "error_message": str(exc),
                    "task_input": sample,
                },
            )
            result_row = {
                "run_id": run_id,
                "benchmark_name": benchmark_name,
                "mode": mode,
                "execution_mode": execution_mode,
                "sample_id": sample_id,
                "video_path": sample.get("video_path") or sample.get("video") or sample.get("media_path") or "",
                "question": sample.get("question") or sample.get("query") or sample.get("prompt") or "",
                "reference_answer": sample.get("reference_answer", sample.get("answer", "")),
                "status": "error",
                "final_answer": "",
                "answer_confidence": 0.0,
                "judge_score": 0.0,
                "judge_verdict": "error",
                "anchor_count": 0,
                "evidence_count": 0,
                "binding_count": 0,
                "supporting_video_evidence": [],
                "supporting_web_evidence": [],
                "error_message": str(exc),
            }

        result_rows.append(result_row)
        answer_rows.append(
            {
                "run_id": run_id,
                "execution_mode": execution_mode,
                "sample_id": sample_id,
                "question": result_row["question"],
                "final_answer": result_row["final_answer"],
                "answer_confidence": result_row["answer_confidence"],
                "supporting_video_evidence": result_row["supporting_video_evidence"],
                "supporting_web_evidence": result_row["supporting_web_evidence"],
                "error_message": result_row.get("error_message", ""),
            }
        )

    save_jsonl(answer_rows, answers_path)
    save_jsonl(result_rows, results_path)
    summary = summarize_results(result_rows)
    save_json(summary, summary_path)
    write_comparison_csv([build_comparison_row(summary)], comparison_path)
    return {
        "run_id": run_id,
        "execution_mode": execution_mode,
        "answers_path": answers_path,
        "results_path": results_path,
        "summary_path": summary_path,
        "comparison_path": comparison_path,
        "samples": len(result_rows),
    }


def main():
    load_env_file(".env")
    args = parse_args()
    benchmark_config = load_config(args.benchmark_config) if args.benchmark_config else {}

    task_profile_path = benchmark_config.get("task_profile", args.task_profile)
    model_profile_path = benchmark_config.get("model_profile", args.model_profile)
    input_file = args.input_file or benchmark_config.get("input_file") or ""
    if not input_file:
        raise ValueError("An input file is required. Pass --input-file or provide it in --benchmark-config.")

    benchmark_name = benchmark_config.get("benchmark_name", args.benchmark_name)
    adapter_name = args.adapter or benchmark_config.get("adapter") or benchmark_name

    task_profile = load_config(task_profile_path)
    model_profile = load_config(model_profile_path)
    samples = load_samples(input_file)
    adapter = get_adapter(adapter_name)
    if args.limit > 0:
        samples = samples[: args.limit]

    exp_cfg = load_config(args.experiment_config) if args.experiment_config else {}
    reasoning_models = exp_cfg.get("reasoning_models", []) if isinstance(exp_cfg, dict) else []
    if not isinstance(reasoning_models, list):
        reasoning_models = []
    reasoning_models = [str(x).strip() for x in reasoning_models if str(x).strip()]

    if not reasoning_models:
        run_id = args.run_id or build_run_id(benchmark_name)
        result = _run_once(
            samples=samples,
            adapter=adapter,
            task_profile=task_profile,
            model_profile=model_profile,
            benchmark_name=benchmark_name,
            mode=args.mode,
            run_id=run_id,
        )
        print(f"run_id={result['run_id']}")
        print(f"benchmark_name={benchmark_name}")
        print(f"adapter={adapter_name}")
        print(f"input_file={input_file}")
        print(f"execution_mode={result['execution_mode']}")
        print(f"samples={result['samples']}")
        print(f"answers_file={result['answers_path']}")
        print(f"results_file={result['results_path']}")
        print(f"summary_file={result['summary_path']}")
        print(f"comparison_file={result['comparison_path']}")
        return

    for model_name in reasoning_models:
        run_model_profile = dict(model_profile)
        run_model_profile["reasoning_model"] = model_name
        tag = _model_tag(model_name)
        run_id = f"{args.run_id}_{tag}" if args.run_id else build_run_id(f"{benchmark_name}_{tag}")
        result = _run_once(
            samples=samples,
            adapter=adapter,
            task_profile=task_profile,
            model_profile=run_model_profile,
            benchmark_name=benchmark_name,
            mode=f"{args.mode}:{tag}",
            run_id=run_id,
        )
        print(f"run_id={result['run_id']}")
        print(f"benchmark_name={benchmark_name}")
        print(f"adapter={adapter_name}")
        print(f"reasoning_model={model_name}")
        print(f"input_file={input_file}")
        print(f"execution_mode={result['execution_mode']}")
        print(f"samples={result['samples']}")
        print(f"answers_file={result['answers_path']}")
        print(f"results_file={result['results_path']}")
        print(f"summary_file={result['summary_path']}")
        print(f"comparison_file={result['comparison_path']}")


if __name__ == "__main__":
    main()
