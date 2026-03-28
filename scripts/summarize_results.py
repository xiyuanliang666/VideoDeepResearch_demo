"""Summarize benchmark outputs for the v0 scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.aggregate import build_comparison_row, summarize_results, write_comparison_csv
from src.tools.cache_tools import load_jsonl, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize benchmark result rows.")
    parser.add_argument("--results-file", required=True, help="Path to benchmark results .jsonl.")
    parser.add_argument(
        "--summary-file",
        default="",
        help="Optional output path for summary.json. Defaults next to results file.",
    )
    parser.add_argument(
        "--comparison-csv",
        default="",
        help="Optional output path for compare.csv. Defaults next to results file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_path = Path(args.results_file)
    rows = load_jsonl(results_path)
    summary = summarize_results(rows)

    summary_path = Path(args.summary_file) if args.summary_file else results_path.with_name("summary.json")
    comparison_path = (
        Path(args.comparison_csv) if args.comparison_csv else results_path.with_name("compare.csv")
    )

    save_json(summary, summary_path)
    write_comparison_csv([build_comparison_row(summary)], comparison_path)

    print(f"results_file={results_path}")
    print(f"summary_file={summary_path}")
    print(f"comparison_file={comparison_path}")
    print(f"sample_count={summary['sample_count']}")
    print(f"avg_judge_score={summary['avg_judge_score']}")


if __name__ == "__main__":
    main()
