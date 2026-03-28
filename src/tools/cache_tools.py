"""Cache read/write helpers."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable

from pydantic import BaseModel


def _ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {key: _to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(value) for value in obj]
    return obj


def save_json(obj: Any, path: str | Path, *, indent: int = 2) -> None:
    target = _ensure_parent(path)
    target.write_text(
        json.dumps(_to_jsonable(obj), ensure_ascii=False, indent=indent),
        encoding="utf-8",
    )


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_jsonl(rows: Iterable[Any], path: str | Path) -> None:
    target = _ensure_parent(path)
    with open(target, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_to_jsonable(row), ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[Any]:
    rows: list[Any] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_pickle(obj: Any, path: str | Path) -> None:
    target = _ensure_parent(path)
    with open(target, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def atomic_write_json(obj: Any, path: str | Path, *, indent: int = 2) -> None:
    target = _ensure_parent(path)
    with NamedTemporaryFile("w", delete=False, dir=target.parent, encoding="utf-8") as tmp:
        json.dump(_to_jsonable(obj), tmp, ensure_ascii=False, indent=indent)
        tmp_path = Path(tmp.name)
    tmp_path.replace(target)
