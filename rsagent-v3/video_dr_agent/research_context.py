"""Research-side message buffer: prompt, research assistants, critic user feedback only."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, List


def _count_image_url_parts(content: str | list[dict[str, Any]]) -> int:
    if isinstance(content, str):
        return 0
    if not isinstance(content, list):
        return 0
    return sum(
        1
        for p in content
        if isinstance(p, dict) and p.get("type") == "image_url"
    )


def extract_research_question_from_context(research: "ResearchContext") -> str:
    """取首条 user 中的纯文本（多模态时拼接所有 text part）。"""
    for m in research.messages():
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str):
            return c.strip()
        if isinstance(c, list):
            parts: list[str] = []
            for p in c:
                if isinstance(p, dict) and p.get("type") == "text":
                    parts.append(str(p.get("text", "")))
            return "\n".join(parts).strip()
    return ""


class ResearchContext:
    def __init__(
        self,
        *,
        system_content: str | None = None,
        initial_user_content: str | list[dict[str, Any]] = "",
    ) -> None:
        self._messages: List[dict[str, Any]] = []
        if system_content is not None and system_content.strip():
            self._messages.append({"role": "system", "content": system_content})
        self._messages.append({"role": "user", "content": initial_user_content})
        # 仅指向循环内 `append_focus_keyframe_injection` 追加的多模态消息；首轮 user 永不写入此处。
        self._focus_injection_message_index: int | None = None
        self._last_focus_injection_round: int | None = None

    def messages(self) -> List[dict[str, Any]]:
        return self._messages

    def append_assistant(self, content: str) -> None:
        self._messages.append({"role": "assistant", "content": content})

    def append_critic_feedback(self, feedback: str) -> None:
        self._messages.append({"role": "user", "content": feedback})

    def append_user_content(self, content: str | list[dict[str, Any]]) -> None:
        """追加一条 user 消息（纯文本或多模态 parts 列表）。"""
        self._messages.append({"role": "user", "content": content})

    def append_focus_keyframe_injection(
        self,
        parts: list[dict[str, Any]],
        *,
        round_index: int,
    ) -> None:
        """
        追加本轮 FOCUS 关键帧多模态 user。若存在「上一轮」由本方法写入的注入消息，则先将其整段
        图像 parts 删除，替换为一对占位标签包裹的短文本，以控制上下文体积；**不修改**首轮 user
        （初始视频/概览帧等）。
        """
        if self._focus_injection_message_index is not None:
            idx = self._focus_injection_message_index
            if 0 <= idx < len(self._messages):
                prev = self._messages[idx]
                if prev.get("role") == "user":
                    n_img = _count_image_url_parts(prev.get("content"))
                    if n_img > 0:
                        prev_r = (
                            self._last_focus_injection_round
                            if self._last_focus_injection_round is not None
                            else "?"
                        )
                        self._messages[idx] = {
                            "role": "user",
                            "content": (
                                f'<focus_keyframes_omitted round="{prev_r}" image_count="{n_img}">'
                                "已移除该轮注入的 FOCUS 关键帧图像，避免上下文膨胀；"
                                "视觉信息应已由当时 assistant 推理或后续工具结果吸收。"
                                "</focus_keyframes_omitted>"
                            ),
                        }
        self._messages.append({"role": "user", "content": parts})
        self._focus_injection_message_index = len(self._messages) - 1
        self._last_focus_injection_round = round_index

    def snapshot_redacted(self) -> List[dict[str, Any]]:
        return redact_messages_for_log(self._messages)


def redact_messages_for_log(messages: List[dict[str, Any]]) -> List[dict[str, Any]]:
    """Replace long base64 data URLs in content parts for JSON traces."""

    def redact_part(p: dict[str, Any]) -> dict[str, Any]:
        p = dict(p)
        if p.get("type") == "image_url":
            url = (p.get("image_url") or {}).get("url", "")
            if isinstance(url, str) and "base64," in url and len(url) > 200:
                p["image_url"] = {"url": url[:80] + "...[redacted base64]..."}
        if p.get("type") == "video_url":
            url = (p.get("video_url") or {}).get("url", "")
            if isinstance(url, str) and "base64," in url and len(url) > 200:
                p["video_url"] = {"url": url[:80] + "...[redacted base64]..."}
        return p

    out: List[dict[str, Any]] = []
    for m in messages:
        mc = deepcopy(m)
        c = mc.get("content")
        if isinstance(c, list):
            mc["content"] = [redact_part(x) if isinstance(x, dict) else x for x in c]
        out.append(mc)
    return out
