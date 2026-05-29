"""Check hybrid pipeline model routing without making network calls.

All online LLM/VLM stages share LLM_API_KEY and LLM_BASE_URL. Per-stage
variables only select model names.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.env_tools import load_env_file, resolve_openai_config
from src.tools.model_router import get_model


ROLES = [
    "event_observer",
    "anchor_extractor",
    "state_constructor",
    "query_planner",
    "hypothesis_generator",
    "evidence_binder",
    "candidate_evaluator",
    "coverage_checker",
    "frame_reranker",
    "reasoning",
    "judge",
]


def _mask(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def main() -> int:
    load_env_file(PROJECT_ROOT / ".env", override=True)
    ok = True
    for role in ROLES:
        config = resolve_openai_config(role)
        model = get_model(role, {})
        status = "OK" if config.available and model else "MISSING"
        ok = ok and status == "OK"
        print(
            f"{status:7} {role:22} model={model or '-'} "
            f"api={config.api_key_var}({_mask(config.api_key) or 'empty'}) "
            f"base={config.base_url_var}({config.base_url or 'empty'})"
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
