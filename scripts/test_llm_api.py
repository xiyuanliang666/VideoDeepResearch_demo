"""Quick LLM API connectivity test (OpenAI-compatible), reading config from .env."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from openai import APIConnectionError, APIStatusError, OpenAI, PermissionDeniedError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.env_tools import load_env_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test OpenAI-compatible chat completion API.")
    parser.add_argument("--prompt", default="Hello, this is a connectivity test.", help="Prompt to send.")
    parser.add_argument("--model", default="", help="Optional model override.")
    parser.add_argument("--base-url", default="", help="Optional base_url override.")
    return parser.parse_args()


def normalize_base_url(raw: str) -> str:
    value = raw.strip().rstrip("/")
    if not value:
        return ""
    if value.endswith("/chat/completions"):
        value = value[: -len("/chat/completions")]
    if not value.endswith("/v1"):
        value = f"{value}/v1"
    return value


def main() -> int:
    load_env_file(".env", override=True)
    args = parse_args()

    api_key = (
        os.getenv("API_GPT_GE_API_KEY", "")
        or os.getenv("LLM_API_KEY", "")
        or os.getenv("OPENAI_API_KEY", "")
    ).strip()

    base_url = normalize_base_url(
        args.base_url
        or os.getenv("API_GPT_GE_BASE_URL", "")
        or os.getenv("LLM_BASE_URL", "")
        or os.getenv("OPENAI_BASE_URL", "")
        or "https://api.gpt.ge"
    )

    model = (
        args.model
        or os.getenv("API_GPT_GE_MODEL", "")
        or os.getenv("REASONING_MODEL", "")
        or os.getenv("LLM_MODEL", "")
        or "gpt-4o-mini"
    ).strip()

    if not api_key:
        print("LLM API test failed: missing API key (API_GPT_GE_API_KEY / LLM_API_KEY / OPENAI_API_KEY).")
        return 2
    if not base_url:
        print("LLM API test failed: missing base_url.")
        return 2

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=45)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": args.prompt}],
            temperature=0.0,
            max_tokens=120,
        )
    except PermissionDeniedError as exc:
        print("LLM API test failed: provider/model blocked (403).")
        print(str(exc))
        return 1
    except APIConnectionError as exc:
        print("LLM API test failed: network/DNS connection error.")
        print(str(exc))
        return 1
    except APIStatusError as exc:
        print(f"LLM API test failed: HTTP {exc.status_code}.")
        print(str(exc))
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"LLM API test failed: {type(exc).__name__}: {exc}")
        return 1

    text = ""
    if getattr(response, "choices", None):
        text = response.choices[0].message.content or ""

    print("LLM API test succeeded.")
    print(f"base_url={base_url}")
    print(f"model={model}")
    print("response:")
    print(text.strip() or "<empty content>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
