"""Critic via OpenAI-compatible chat.completions (same shape as rsagent/test_prompt.py)."""

from __future__ import annotations

from dataclasses import dataclass

from video_dr_agent.critic_context import CriticContext
from video_dr_agent.protocols import CriticAgent


@dataclass
class OpenAICriticConfig:
    base_url: str
    api_key: str
    model: str
    max_tokens: int = 1024
    temperature: float = 0.3


class OpenAICriticAgent(CriticAgent):
    def __init__(self, cfg: OpenAICriticConfig, critic_ctx: CriticContext) -> None:
        self.cfg = cfg
        self.ctx = critic_ctx
        from openai import OpenAI

        self._client = OpenAI(base_url=cfg.base_url.rstrip("/"), api_key=cfg.api_key)

    def complete_round(self, user_payload: str) -> str:
        messages = self.ctx.messages_for_api(user_payload)
        r = self._client.chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
        )
        text = (r.choices[0].message.content or "").strip()
        self.ctx.append_assistant(text)
        return text
