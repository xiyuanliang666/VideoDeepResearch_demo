import os
import sys
from pathlib import Path

from openai import OpenAI
from openai import APIConnectionError, APIStatusError, PermissionDeniedError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.env_tools import load_env_file

load_env_file('.env', override=True)

api_key = (os.getenv('OPENROUTER_API_KEY', '') or os.getenv('LLM_API_KEY', '')).strip()
model_name = (
    os.getenv('OPENROUTER_MODEL', '')
    or os.getenv('REASONING_MODEL', '')
    or os.getenv('LLM_MODEL', '')
).strip()
base_url = (
    os.getenv('OPENROUTER_BASE_URL', '')
    or os.getenv('LLM_BASE_URL', '')
    or 'https://openrouter.ai/api/v1'
).strip()
http_referer = (
    os.getenv('OPENROUTER_HTTP_REFERER', '')
    or os.getenv('OPENROUTER_SITE_URL', '')
).strip()
x_openrouter_title = (
    os.getenv('OPENROUTER_X_TITLE', '')
    or os.getenv('OPENROUTER_APP_TITLE', '')
    or os.getenv('OPENROUTER_SITE_NAME', '')
).strip()

if not api_key:
    raise SystemExit('Missing OPENROUTER_API_KEY or LLM_API_KEY in .env')
if not model_name:
    raise SystemExit('Missing OPENROUTER_MODEL/REASONING_MODEL/LLM_MODEL in .env')
if not base_url:
    raise SystemExit('Missing OPENROUTER_BASE_URL/LLM_BASE_URL in .env')

default_headers = {}
if http_referer:
    default_headers['HTTP-Referer'] = http_referer
if x_openrouter_title:
    default_headers['X-OpenRouter-Title'] = x_openrouter_title

client = OpenAI(
    base_url=base_url,
    api_key=api_key,
    default_headers=default_headers or None,
)

def create_with_reasoning(messages):
    return client.chat.completions.create(
        model=model_name,
        messages=messages,
        extra_body={'reasoning': {'enabled': True}},
    )


try:
    # First API call with reasoning
    response = create_with_reasoning(
        [
            {
                'role': 'user',
                'content': "How many r's are in the word 'strawberry'?",
            }
        ]
    )
except PermissionDeniedError as exc:
    print('OpenRouter test failed: provider/model is blocked by policy (403).')
    print(f'model={model_name}')
    print(str(exc))
    raise SystemExit(1)
except APIConnectionError as exc:
    print('OpenRouter test failed: network/DNS connection error.')
    print(str(exc))
    raise SystemExit(1)
except APIStatusError as exc:
    print(f'OpenRouter test failed: HTTP {exc.status_code}')
    print(str(exc))
    raise SystemExit(1)
except Exception as exc:  # noqa: BLE001
    print(f'OpenRouter test failed: {type(exc).__name__}: {exc}')
    raise SystemExit(1)

# Extract the assistant message with reasoning_details
response = response.choices[0].message

# Preserve the assistant message with reasoning_details
messages = [
    {'role': 'user', 'content': "How many r's are in the word 'strawberry'?"},
    {
        'role': 'assistant',
        'content': response.content,
        'reasoning_details': response.reasoning_details,  # Pass back unmodified
    },
    {'role': 'user', 'content': 'Are you sure? Think carefully.'},
]

# Second API call - model continues reasoning from where it left off
try:
    response2 = create_with_reasoning(messages)
except PermissionDeniedError as exc:
    print('OpenRouter second call failed: provider/model is blocked by policy (403).')
    print(str(exc))
    raise SystemExit(1)
except APIConnectionError as exc:
    print('OpenRouter second call failed: network/DNS connection error.')
    print(str(exc))
    raise SystemExit(1)
except APIStatusError as exc:
    print(f'OpenRouter second call failed: HTTP {exc.status_code}')
    print(str(exc))
    raise SystemExit(1)
except Exception as exc:  # noqa: BLE001
    print(f'OpenRouter second call failed: {type(exc).__name__}: {exc}')
    raise SystemExit(1)

print('Model:', model_name)
print('Base URL:', base_url)
if http_referer:
    print('HTTP-Referer:', http_referer)
if x_openrouter_title:
    print('X-OpenRouter-Title:', x_openrouter_title)
print('First response:', response.content)
print('Second response:', response2.choices[0].message.content)
