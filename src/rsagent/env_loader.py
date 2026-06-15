"""Simple .env loader for rsagent-v4."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(root: Path | str | None = None) -> None:
    """Load .env from project root directory.

    Searches upward from `root` (or this file's location) until a .env is found
    or the filesystem root is reached.
    """
    if root is None:
        root = Path(__file__).resolve().parent  # src/rsagent/
    start = Path(root).resolve()
    # Walk upward until we find .env or hit /
    candidate = start
    while candidate != candidate.parent:
        env_file = candidate / ".env"
        if env_file.is_file():
            break
        candidate = candidate.parent
    else:
        return  # No .env found

    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = val
