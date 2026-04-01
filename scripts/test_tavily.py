"""Small Tavily SDK smoke test script."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.env_tools import load_env_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Tavily SDK connectivity with qna_search.")
    parser.add_argument(
        "--query",
        default="Who is Leo Messi?",
        help="Question for Tavily qna_search.",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Optional Tavily API key override. If empty, uses TAVILY_API_KEY from .env.",
    )
    return parser.parse_args()


def main() -> int:
    load_env_file(".env")
    args = parse_args()

    api_key = args.api_key.strip()
    if not api_key:
        import os

        api_key = os.getenv("TAVILY_API_KEY", "").strip()

    if not api_key:
        print("Tavily test failed: missing API key. Set TAVILY_API_KEY in .env or pass --api-key.")
        return 2

    try:
        from tavily import TavilyClient
    except ModuleNotFoundError:
        print("Tavily test failed: tavily-python is not installed. Run `pip install -r requirements.txt`.")
        return 3

    client = TavilyClient(api_key=api_key)
    try:
        answer = client.qna_search(query=args.query)
    except Exception as exc:  # noqa: BLE001
        print(f"Tavily test failed: {type(exc).__name__}: {exc}")
        return 1

    print("Tavily test succeeded.")
    print(f"query: {args.query}")
    print("answer:")
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
