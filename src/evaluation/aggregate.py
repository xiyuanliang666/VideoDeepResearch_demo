"""Evaluation aggregation helpers."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


def summarize_results(result_rows):
    """Aggregate per-sample results into benchmark summaries."""
    rows = list(result_rows)
    sample_count = len(rows)
    status_counts = Counter(row.get("status", "unknown") for row in rows)
    verdict_counts = Counter(row.get("judge_verdict", "missing") for row in rows)

    judge_scores = [
        float(row["judge_score"])
        for row in rows
        if row.get("judge_score") is not None
    ]
    answer_confidences = [
        float(row["answer_confidence"])
        for row in rows
        if row.get("answer_confidence") is not None
    ]
    anchor_counts = [int(row.get("anchor_count", 0)) for row in rows]
    evidence_counts = [int(row.get("evidence_count", 0)) for row in rows]
    binding_counts = [int(row.get("binding_count", 0)) for row in rows]

    answered_count = sum(1 for row in rows if row.get("final_answer"))

    def _avg(values):
        return round(sum(values) / len(values), 4) if values else 0.0

    benchmark_name = rows[0].get("benchmark_name", "") if rows else ""
    mode = rows[0].get("mode", "") if rows else ""
    run_id = rows[0].get("run_id", "") if rows else ""

    return {
        "run_id": run_id,
        "benchmark_name": benchmark_name,
        "mode": mode,
        "sample_count": sample_count,
        "answered_count": answered_count,
        "answer_rate": round(answered_count / sample_count, 4) if sample_count else 0.0,
        "avg_judge_score": _avg(judge_scores),
        "avg_answer_confidence": _avg(answer_confidences),
        "avg_anchor_count": _avg(anchor_counts),
        "avg_evidence_count": _avg(evidence_counts),
        "avg_binding_count": _avg(binding_counts),
        "status_counts": dict(status_counts),
        "judge_verdict_counts": dict(verdict_counts),
    }


def build_comparison_row(summary: dict) -> dict:
    """Flatten a summary dict into a CSV-friendly comparison row."""
    return {
        "run_id": summary.get("run_id", ""),
        "benchmark_name": summary.get("benchmark_name", ""),
        "mode": summary.get("mode", ""),
        "sample_count": summary.get("sample_count", 0),
        "answer_rate": summary.get("answer_rate", 0.0),
        "avg_judge_score": summary.get("avg_judge_score", 0.0),
        "avg_answer_confidence": summary.get("avg_answer_confidence", 0.0),
        "avg_anchor_count": summary.get("avg_anchor_count", 0.0),
        "avg_evidence_count": summary.get("avg_evidence_count", 0.0),
        "avg_binding_count": summary.get("avg_binding_count", 0.0),
    }


def write_comparison_csv(rows: list[dict], path: str | Path) -> str:
    """Write flattened summary rows to CSV."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "benchmark_name",
        "mode",
        "sample_count",
        "answer_rate",
        "avg_judge_score",
        "avg_answer_confidence",
        "avg_anchor_count",
        "avg_evidence_count",
        "avg_binding_count",
    ]
    with open(target, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return str(target)
