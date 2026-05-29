# VideoDeepResearch 框架统一方案

日期：2026-05-22

## 目标

最终只保留一个对外框架：`videodeepresearch`。框架内部允许两种执行范式：

- `workflow`：固定流程 baseline。
- `agentic`：planner / tool-call / critic loop baseline。

两种模式输出同一套结构，因此 benchmark evaluator 不需要知道内部到底运行的是 fixed pipeline 还是 agent loop。

## 当前主入口

```python
from videodeepresearch import run

result = run(
    sample=sample,
    task_profile=task_profile,
    model_profile=model_profile,
    mode="workflow",  # or "agentic"
)
```

`run` 返回统一结构：

- `schema_version`
- `mode`
- `status`
- `sample`
- `question`
- `final_answer`
- `used_video_evidence`
- `retrieved_web_results`
- `used_web_evidence`
- `evidence_links`
- `dependency_graph`
- `tool_trace`
- `judge_result`
- `diagnostics`
- `raw_trace`

其中 `raw_trace` 只用于调试和迁移，不应作为 evaluator 的主要输入。

## 当前目录

```text
baseline_entrypoints/
configs/
data/
docs/
prompts/
scripts/
src/
videodeepresearch/
```

当前 active runtime 入口是 `videodeepresearch.run(...)` 与 `baseline_entrypoints/run_vdr_single.py`。历史 agent 快照（rsagent/、rsagent-v2/、rsagent-v3/）、恢复材料、旧设计草稿和一次性脚本已全部删除。

## 已完成归并

- `src/videodeepresearch/runner.py`：统一 `workflow` / `agentic` mode 切换。
- `src/videodeepresearch/agentic/`：统一 agentic runner、tool-call parsing、tool round execution。
- `src/videodeepresearch/tools/`：统一工具层，包括 web search、Jina reader、evidence policy、可选 FOCUS wrapper。
- `baseline_entrypoints/run_vdr_single.py`：外部 benchmark adapter 的稳定单样本入口。

## 当前边界

- `src/core/` 仍承载 workflow pipeline 的内部实现。
- `agentic` runner 已切到 `src/videodeepresearch/agentic/runner.py`，但目前仍是 deterministic host plan，不是完整 LLM research planning。
- structured critic verdict 尚未接入。
- FOCUS 不再随仓库保留；如需启用，设置 `FOCUS_PACKAGE_DIR` 指向外部 FOCUS checkout。
- 无搜索 API key 时会生成可追踪 fallback evidence，用于 dry run，不代表真实检索能力。

## 下一步

1. 接入 critic feedback / structured verdict。
3. 将 `src/core/` 内仍稳定可复用的模块逐步迁移到 `src/videodeepresearch/` 下更清晰的子包。
