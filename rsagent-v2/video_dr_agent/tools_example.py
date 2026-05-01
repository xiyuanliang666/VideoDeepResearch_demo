"""
示例工具：`ToolDispatcher` 的最小实现模板。

在 system / user 里约定模型输出：
  <tool_call>
  {"name": "example_noop", "arguments": {}}
  </tool_call>

复制本文件为 `tools.py` 后，在 `dispatch` 里按 name 增加分支即可。
"""

from __future__ import annotations

import json
from typing import Any

from video_dr_agent.protocols import ToolDispatcher

# 与模型侧约定的空工具名称（可按需改名，并与 prompt 一致）
EXAMPLE_NOOP_TOOL_NAME = "example_noop"


class ExampleToolDispatcher(ToolDispatcher):
    """含一个无副作用空工具；未知工具名返回说明字符串（便于调试）。"""

    def dispatch(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        round_index: int | None = None,
    ) -> str:
        _ = round_index
        if name == EXAMPLE_NOOP_TOOL_NAME:
            return self._example_noop(arguments)
        return self._unknown_tool(name, arguments)

    def _example_noop(self, arguments: dict[str, Any]) -> str:
        """空工具：不接网络、不读盘；在此替换为你的真实逻辑。"""
        _ = arguments
        return f"[{EXAMPLE_NOOP_TOOL_NAME}] 已调用（无操作示例）。"

    def _unknown_tool(self, name: str, arguments: dict[str, Any]) -> str:
        return (
            f"[unknown_tool] name={name!r} "
            f"arguments={json.dumps(arguments, ensure_ascii=False)[:400]}"
        )
