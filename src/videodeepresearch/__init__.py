"""Public framework entrypoint for VideoDeepResearch.

The framework exposes one stable API with two execution modes:

- ``workflow``: fixed stage pipeline.
- ``agentic``: planner/search loop pipeline.

Both modes return the same normalized run structure so downstream benchmark
evaluators do not need to know which internal strategy was used.
"""

from .runner import (
    SUPPORTED_MODES,
    normalize_mode,
    run,
    run_raw,
)

__all__ = [
    "SUPPORTED_MODES",
    "normalize_mode",
    "run",
    "run_raw",
]
