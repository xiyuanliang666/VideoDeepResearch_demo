"""General IO helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .cache_tools import save_json


def ensure_dir(path: str | Path) -> str:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return str(target)


def build_run_id(prefix: str = "run") -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}_{uuid4().hex[:8]}"


def build_run_dir(base_dir: str | Path, run_id: str | None = None) -> str:
    resolved_run_id = run_id or build_run_id()
    run_dir = Path(base_dir) / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return str(run_dir)


def write_text(path: str | Path, content: str) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return str(target)


def write_json(path: str | Path, payload: Any) -> str:
    save_json(payload, path)
    return str(path)


def write_trace(run_dir: str | Path, trace_name: str, payload: Any) -> str:
    trace_path = Path(run_dir) / f"{trace_name}.json"
    save_json(payload, trace_path)
    return str(trace_path)
