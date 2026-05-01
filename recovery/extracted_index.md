### 072e0a10cc6eef2b169a291920f3e47575ec97dd.json
- mime: application/json
- lines: 32
- head:
    {
      "tool": "focus_select_keyframes",
      "video_path": "/mnt/sda/Datasets/VideoDR/video/16.mp4",
      "video_path_in_arguments": null,
      "used_default_video_fallback": false,
      "question": "relative position of spherical tower to classical building",

### 0ad0faf8049b312cb58cbbd812a5365e4d73fc45.txt
- mime: text/plain
- lines: 93
- head:
    """Video DeepResearch agent framework (vLLM Qwen3-VL research + OpenAI-compatible critic)."""
    
    from video_dr_agent.critic_context import CriticContext
    from video_dr_agent.critic_deep_research import (
        DeepResearchCriticAgent,
        DeepResearchCriticConfig,

### 132e50b47e3615e59194264f2d57816229ce1058.txt
- mime: text/plain
- lines: 149
- head:
    ---
    name: Video DeepResearch 框架
    overview: 在 `rsagent` 下新增轻量、解耦的 Video DeepResearch 主循环包：**Research 走本地 vLLM 部署的 Qwen3-VL**（OpenAI 兼容 `/v1/chat/completions` + vLLM 多模态 `messages` 格式）、从模型输出解析 `<tool_call>`（对齐 VDR/dryrun 协议）、**双通道上下文防污染**（research 仅见 prompt/自身输出/critic 反馈；critic 独立、调用方式对齐 test_prompt）、每轮落盘；具体 prompt 与工具实现仅保留接口与占位。
    todos:
      - id: scaffold-pkg
        content: 在 rsagent/video_dr_agent/ 创建包：protocols、parsing、research_context 与 critic_context（双缓冲）、qwen3vl_client、openai_critic 骨架

### 17d586053181aac3dc3d1ddc0e2616795b0ce6da.json
- mime: application/json
- lines: 32
- head:
    {
      "tool": "focus_select_keyframes",
      "video_path": "/mnt/sda/Datasets/VideoDR/video/16.mp4",
      "video_path_in_arguments": null,
      "used_default_video_fallback": false,
      "question": "tall spherical tower",

### 19d51e76c491483f89a94e4f690dbcc6be59889a.json
- mime: application/json
- lines: 32
- head:
    {
      "tool": "focus_select_keyframes",
      "video_path": "/mnt/sda/Datasets/VideoDR/video/56.mp4",
      "video_path_in_arguments": null,
      "used_default_video_fallback": false,
      "question": "Identify company names on screen",

### 1c1328bf47f6df6b4cf45b2dbdd45827749c22ed.json
- mime: application/json
- lines: 32
- head:
    {
      "tool": "focus_select_keyframes",
      "video_path": "/mnt/sda/Datasets/VideoDR/video/56.mp4",
      "video_path_in_arguments": null,
      "used_default_video_fallback": false,
      "question": "Identify company names on screen",

### 1c91da55c4da9620f3f7f5c73bd1b4b7cc3b36d6.json
- mime: application/json
- lines: 19
- head:
    {
      "mode": "custom_single_video",
      "video_path": "/mnt/sda/Datasets/VideoDR/video/2.mp4",
      "query_preview": "The video shows the making process of a tool. According to the official description, in which year of Japan’s Meiji era was this tool invented?",
      "budget_usage": {
        "total_budget_used": 52,

### 1d5e185a812d70b40a5614828c9c76ee06b1b893.txt
- mime: text/plain
- lines: 84
- head:
    """Video DeepResearch agent framework (vLLM Qwen3-VL research + OpenAI-compatible critic)."""
    
    from video_dr_agent.critic_context import CriticContext
    from video_dr_agent.critic_deep_research import (
        DeepResearchCriticAgent,
        DeepResearchCriticConfig,

### 227b236d49964bc9735be77f0d90a4b1107bd6aa.json
- mime: application/json
- lines: 32
- head:
    {
      "tool": "focus_select_keyframes",
      "video_path": "/mnt/sda/Datasets/VideoDR/video/56.mp4",
      "video_path_in_arguments": null,
      "used_default_video_fallback": false,
      "question": "Panelists and company names on stage at Six Little Dragons Wuzhen Dialogue",

### 2894419425eeeae5830ec37a115f8186c1fa347e.txt
- mime: text/plain
- lines: 109
- head:
    """
    Critic system prompt and staged user-prompt templates (deep research: FOCUS + web search + Jina).
    """
    
    from __future__ import annotations
    

### 2f1d12674b6e156518cb2d98c6881139ad8d61bf.txt
- mime: text/plain
- lines: 60
- head:
    """Placeholder tool / critic implementations."""
    
    from __future__ import annotations
    
    import json
    from typing import Any, Sequence

### 30b31cae990cdfdbf117e73ca9adff7bafbf34da.txt
- mime: text/plain
- lines: 47
- head:
    """
    示例工具：`ToolDispatcher` 的最小实现模板。
    
    在 system / user 里约定模型输出：
      <tool_call>
      {"name": "example_noop", "arguments": {}}

### 370ee47fff92c931ea018bf726c89236b4cec527.py
- mime: text/x-script.python
- lines: 519
- head:
    #!/usr/bin/env python3
    """
    CLI entry for video_dr_agent.
    
    Recommended environment (add only missing deps, avoid upgrading unrelated packages):
      conda activate AKS

### 3b28b545a724129687c9cb09d54230d25bd3eba6.txt
- mime: text/plain
- lines: 72
- head:
    You are the **Research** model in a video deep-research loop. Solve the user’s task with **video evidence** and **web search**, following a **written Research Plan** that you already produced in an earlier turn (Host “Planning phase”) and the subsequent “Execution phase” instructions.
    
    ## What you see in the transcript
    
    - **User task / question** (may be any natural language; answer in the same language unless the user asks otherwise).
    - **Uniformly sampled overview frames** (~20 typical): a **sparse** timeline, not a full watch.

### 45140ab63240c0e7f1f9eb59a661d26fb01badea.json
- mime: application/json
- lines: 32
- head:
    {
      "tool": "focus_select_keyframes",
      "video_path": "/mnt/sda/Datasets/VideoDR/video/56.mp4",
      "video_path_in_arguments": null,
      "used_default_video_fallback": false,
      "question": "Identify company names on screen",

### 4d8587f3f17190fdd8859d5078737db382af8928.txt
- mime: text/plain
- lines: 162
- head:
    # 帧筛选评估
    
    本文档合并两部分内容：
    
    1. **自拟 benchmark**：状态池子状态、关键帧 GT、时间阈值 + SSIM 最优匹配 + 冗余惩罚等形式化定义（基于前文讨论总结）。
    2. **论文 / 官方实现参考**：`/mnt/sda/TStar/LVHaystackBench/val_tstar_results.py` 中与 LV-Haystack / T* 相关工作一致的 **Temporal PRF** 与 **SSIM Precision/Recall/F1** 的计算方式及特点。

### 4d9cba22c30e724e30d3f620e26f671636dd151a.txt
- mime: text/plain
- lines: 125
- head:
    """
    Run ``test_data_frame/uniform_sample_frames.py`` before the research loop when configured.
    """
    
    from __future__ import annotations
    

### 5a05b307c4d337f80e89030cc0f0ea1d95af2cdd.json
- mime: application/json
- lines: 32
- head:
    {
      "tool": "focus_select_keyframes",
      "video_path": "/mnt/sda/Datasets/VideoDR/video/16.mp4",
      "video_path_in_arguments": null,
      "used_default_video_fallback": false,
      "question": "close-up of the classical building with a large clock face (Bank of China building) in Shanghai",

### 6441dafe2bcc5b982291f89f2f36348172705a06.sh
- mime: text/x-shellscript
- lines: 95
- head:
    #!/bin/bash
    
    # FOCUS Keyframe Extraction Example Script
    # This script demonstrates how to use FOCUS for keyframe extraction
    
    echo "FOCUS: Frame-Optimistic Confidence Upper-bound Selection"

### 6600c5bf84c98f677b79ebbbfcb227bfa7f6b5cb.txt
- mime: text/plain
- lines: 125
- head:
    ---
    name: rsagent-v2 流程与 Critic 改造
    overview: 在 [rsagent-v2](/mnt/sda/Projects/video-deep-research/rsagent-v2) 中：以**强制「详细 Research Plan → 再按计划在循环中调用工具」**为主架构，缓解当前 prompt 下 research 跳过 `focus_select_keyframes`、证据不足就结束的问题；Deep Critic 在 FOCUS 成功时选 Top3 帧驱动注入；工具顺序由**计划内显式约定**而非单一全局硬规则。**凡框架写入、会进入模型上下文的提示文案（system / Host user / Critic 模板 / 循环 nudge / 规划 handoff / 多模态注入说明等）一律用英文**；用户任务与问题正文可保留原语言。
    todos:
      - id: prompt-plan-first
        content: 重写 research_system_prompt.txt（英文）：强制两阶段（先书面详细计划、再执行）；计划模板含视觉/FOCUS/网页步骤与完成标准；执行轮次声明当前计划步骤；抑制过早无 tool_call 结束

### 67e2d71a4074d894147ff96da92539dd921ad65d.txt
- mime: text/plain
- lines: 82
- head:
    You are the **Research** model in a video deep-research loop (**rsagent-v3**). Solve the user’s task with **video evidence** and **web search**, following a **written Research Plan** that you already produced in an earlier turn (Host “Planning phase”) and the subsequent “Execution phase” instructions.
    
    ## What you see in the transcript
    
    - **User task / question** (may be any natural language; answer in the same language unless the user asks otherwise).
    - **Uniformly sampled overview frames** (~20 typical): a **sparse** timeline, not a full watch.

### 680e1119ad4dacd26f0fb93b7eca5e07c0e52402.sh
- mime: text/x-shellscript
- lines: 17
- head:
    #!/usr/bin/env bash
    # 在已配置好的 AKS conda 环境中跑 1 条 LongVideoBench 样本（FOCUS 目录下已含 datasets/longvideobench_smoke）
    set -euo pipefail
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate AKS
    cd "$(dirname "$0")"

### 6c0d2f4d857600ca275e20a779a48f27bb72b1bc.txt
- mime: text/plain
- lines: 47
- head:
    """
    Run ``test_data_frame/uniform_sample_frames.py`` before the research loop when configured.
    """
    
    from __future__ import annotations
    

### 6c2ea8f61f38105973d87027cb73da849996877c.txt
- mime: text/plain
- lines: 128
- head:
    """Research-side message buffer: prompt, research assistants, critic user feedback only."""
    
    from __future__ import annotations
    
    from copy import deepcopy
    from typing import Any, List

### 6c7394bc7f90dea47fa741925a781063ad8281ed.json
- mime: application/json
- lines: 32
- head:
    {
      "tool": "focus_select_keyframes",
      "video_path": "/mnt/sda/Datasets/VideoDR/video/16.mp4",
      "video_path_in_arguments": null,
      "used_default_video_fallback": false,
      "question": "close-up of the tall spherical tower (Oriental Pearl Tower) in Shanghai",

### 6c84e98946981f0609ba926a4dcff0f07fb24ed5.txt
- mime: text/plain
- lines: 0
- head:
    /mnt/sda/Projects/video-deep-research/rsagent copy/FOCUS/datasets/longvideobench_smoke/videos/86CxyhFV9MI.mp4
### 75a47a3006a3fc3481716f8200b83ef2b0c0ef83.txt
- mime: text/plain
- lines: 56
- head:
    """Parse <tool_call> blocks (VDR-style)."""
    
    from __future__ import annotations
    
    import re
    from typing import List

### 76b1896256eac0ccbe5ff132ba27e30469b2682f.json
- mime: application/json
- lines: 47
- head:
    {
      "video_id": "86CxyhFV9MI",
      "question": "In the video, which subtitles appear at the same time as the man with black hair, dressed in grey clothes with black sleeves, on stage?",
      "question_wo_referring_query": "Which subtitles appear at the same time?",
      "candidates": [
        "promisc has come to an end, in and run away countless times, i was just scared, i still",

### 780368ff3779a8b053626aac799c4144b5603893.txt
- mime: text/plain
- lines: 171
- head:
    # FOCUS: Efficient Keyframe Selection for Long Video Understanding
    
    > 🎉 **NEWS**: Our paper has been accepted by ICLR 2026! 
    > 
    > 📄 [Read the paper on OpenReview](https://openreview.net/forum?id=1OQKqLFcbB)
    

### 7ab89f46c69f6b2196a66c07037b9f563cea3a03.txt
- mime: text/plain
- lines: 5
- head:
    > **Deprecated — do not use for new runs.**  
    > The repository default is `prompts/research_system_prompt.txt` (**plan-first**, no global mandatory tool order).  
    > Earlier versions of this file required calling `focus_select_keyframes` before web search; that **global** requirement has been **removed**. Configure tools only through your Research Plan and the host (FOCUS enabled or not).
    
    Copy `prompts/research_system_prompt.txt` (or set `RESEARCH_SYSTEM_PROMPT_PATH`) and adjust wording there if you need stricter per-run behavior.

### 7c07d5cdabd2c1680e91910446f50b381f90f554.py
- mime: text/x-script.python
- lines: 91
- head:
    #!/usr/bin/env python3
    """
    Offline smoke: core ``run_loop`` + stub tools/critic + trace JSON (no vLLM).
    
    Run from ``rsagent/``:
      python scripts/smoke_offline_loop.py

### 7d5238f866e7096d526beb3aebb111991aede881.json
- mime: application/json
- lines: 32
- head:
    {
      "tool": "focus_select_keyframes",
      "video_path": "/mnt/sda/Datasets/VideoDR/video/9.mp4",
      "video_path_in_arguments": null,
      "used_default_video_fallback": false,
      "question": "Identify visual landmarks in the video that confirm the location is Cincinnati, Ohio.",

### 7e40b113a8d08d540fab10de6a07a2c6c9768e1a.json
- mime: application/json
- lines: 32
- head:
    {
      "tool": "focus_select_keyframes",
      "video_path": "/mnt/sda/Datasets/VideoDR/video/56.mp4",
      "video_path_in_arguments": null,
      "used_default_video_fallback": false,
      "question": "Identify company names on screen",

### 7ed8bd197792149d7e258884fb5b409015ef0126.py
- mime: text/x-script.python
- lines: 339
- head:
    #!/usr/bin/env python3
    """
    将 FOCUS 筛选结果整理到独立子目录：
      output_dir/frames/     选中的帧（图片）
      output_dir/question/   问题文本 + 元数据 JSON
      output_dir/video/      原视频（默认符号链接，可用 --copy-video 复制）

### 7ff8553e66f2df80e721604d4316c6661f47f869.txt
- mime: text/plain
- lines: 144
- head:
    """
    Critic system prompt and staged user-prompt templates (deep research: FOCUS + web search + Jina).
    
    rsagent-v3: evidence tiers + structured verdict fields for Research.
    """
    

### 823ea7a0bd55199143571bb1c1be156d0897d9e2.py
- mime: text/x-script.python
- lines: 57
- head:
    #!/usr/bin/env python3
    """
    Lightweight checks that the Video DeepResearch stack imports and wires correctly:
    
    - Default research prompt file resolves
    - Deep critic prompts import

### 82e37a41735c3441622547bec1574fed51dec00f.txt
- mime: text/plain
- lines: 387
- head:
    """
    Main research loop (Video DeepResearch plan).
    
    Flow (**rsagent-v3**): optional **mandatory planning** (no tools) → **vLLM research** → parse ``<tool_call>`` →
    **ToolDispatcher** (raw tool JSON does not go directly to research) → **CriticPromptBuilder** +
    **CriticAgent** → append **critic text**; if the round includes successful ``focus_select_keyframes``,

### 87a37a7c03ad6f8f9e714681735cb6a719fd3238.json
- mime: application/json
- lines: 32
- head:
    {
      "tool": "focus_select_keyframes",
      "video_path": "/mnt/sda/Datasets/VideoDR/video/56.mp4",
      "video_path_in_arguments": null,
      "used_default_video_fallback": false,
      "question": "Identify the names and titles of the six speakers on the panel at the 'Six Little Dragons Wuzhen Dialogue' event.",

### 902b13ccfa02c550fe05ddac03e791884ce74b84.txt
- mime: text/plain
- lines: 47
- head:
    """
    Protocols for Video DeepResearch (see repo plan: research/critic separation).
    
    - ``ToolDispatcher``: execute tools; outputs go to critic payload only, not research messages.
    - ``CriticPromptBuilder``: assemble this round's user payload (may embed tool runs).
    - ``CriticAgent``: ``complete_round(user_payload) -> str`` feedback for research's next user turn.

### 91088d8981ad943901256c6b6620471e77b8c6df.txt
- mime: text/plain
- lines: 18
- head:
    .env
    __pycache__/
    *.pyc
    outputs/
    data/*
    !data/benchmarks/

### 93007a68977eea4336b86bf68e928b83ae52b26d.txt
- mime: text/plain
- lines: 128
- head:
    """Research-side message buffer: prompt, research assistants, critic user feedback only."""
    
    from __future__ import annotations
    
    from copy import deepcopy
    from typing import Any, List

### 9655d974e3329ef0d5e0d1f80be4b8e5181d7c90.json
- mime: application/json
- lines: 32
- head:
    {
      "tool": "focus_select_keyframes",
      "video_path": "/mnt/sda/Datasets/VideoDR/video/1.mp4",
      "video_path_in_arguments": null,
      "used_default_video_fallback": false,
      "question": "temple complex with tiered roofs and lotus ponds",

### 9a7b01fe3dfba90eeff6641b8159404c60fbfbc0.py
- mime: text/x-script.python
- lines: 464
- head:
    #!/usr/bin/env python3
    """
    CLI entry for video_dr_agent.
    
    Recommended environment (add only missing deps, avoid upgrading unrelated packages):
      conda activate AKS

### 9fd9ba0d4ac7d2e8444bf5d786d20c02fc68162b.json
- mime: application/json
- lines: 32
- head:
    {
      "tool": "focus_select_keyframes",
      "video_path": "/mnt/sda/Datasets/VideoDR/video/56.mp4",
      "video_path_in_arguments": null,
      "used_default_video_fallback": false,
      "question": "Man in front of ‘群核科技’ backdrop",

### a87cfe6bd70aa842984412fc07acf9b9d8372a26.json
- mime: application/json
- lines: 32
- head:
    {
      "tool": "focus_select_keyframes",
      "video_path": "/mnt/sda/Datasets/VideoDR/video/16.mp4",
      "video_path_in_arguments": null,
      "used_default_video_fallback": false,
      "question": "classical building with large clock face",

### ab785cd6c7b952e6549f04e68876d5dc0dfc840a.txt
- mime: text/plain
- lines: 519
- head:
    """
    Deep-research Critic: FOCUS technical check + **Top-3 frame ranking** (metadata) for injection;
    search snippet scoring + Jina Top-3 + LLM synthesis.
    
    Implements CriticAgent like OpenAICriticAgent; complete_round may use multiple stateless LLM calls
    and only appends the final feedback to CriticContext. Exposes ``focus_inject_paths`` for the loop.

### ac2f4788f13c43a8e2eee6e7d5b08c725bed0f59.txt
- mime: text/plain
- lines: 277
- head:
    """
    多源网页搜索工具：并行调用 Serper（与 search_agent.serper_search 同协议），无 LLM 总结。
    
    工具名: deep_research_web_search
    
    <tool_call> arguments 示例:

### ac929f452328d8b7a7dcaaa6acc8314a9d04b2db.txt
- mime: text/plain
- lines: 350
- head:
    """
    Deep-research Critic: FOCUS pass/fail (no prose summary) + search snippet scoring + Jina Top-3 + LLM synthesis.
    
    Implements CriticAgent like OpenAICriticAgent; complete_round may use multiple stateless LLM calls
    and only appends the final feedback to CriticContext.
    """

### ad325e41537b04f91b3abfa1fe17ce8c8b6cdd8a.txt
- mime: text/plain
- lines: 849
- head:
    """
    FOCUS: Frame-Optimistic Confidence Upper-bound Selection
    
    Core algorithm implementation for keyframe extraction using confidence upper-bound bandit approach.
    This module contains only the algorithm logic, without data processing or I/O operations.
    """

### b38aaee12c4a010f8cc35d37ab99368fa041692e.txt
- mime: text/plain
- lines: 173
- head:
    You are the **Research** model in a video deep-research loop. Your job is to solve the user’s task using **video evidence** plus **web search**, in a disciplined multi-step way.
    
    ---
    
    ## [TEST MODE — MANDATORY `focus_select_keyframes`] Read this block first
    

### b4298592ca2df1058a07366e31e3dded09c9268c.txt
- mime: text/plain
- lines: 82
- head:
    """
    Critic system prompt and staged user-prompt templates (deep research: FOCUS + web search + Jina).
    """
    
    from __future__ import annotations
    

### b517b29f0c173d01b9914f319eee0487032e5572.txt
- mime: text/plain
- lines: 229
- head:
    """
    Load `video_dr_agent/.env` from the repository root and apply name aliases.
    
    Supported aliases (after dotenv load):
      DEEP_RESEARCH_BASE_URL -> VLLM_BASE_URL (if unset)
      DEEP_RESEARCH_MODEL     -> VLLM_MODEL (if unset)

### b57e246e8908d004382397f40a25d09b51cbb665.txt
- mime: text/plain
- lines: 208
- head:
    """
    Load `video_dr_agent/.env` from the repository root and apply name aliases.
    
    Supported aliases (after dotenv load):
      DEEP_RESEARCH_BASE_URL -> VLLM_BASE_URL (if unset)
      DEEP_RESEARCH_MODEL     -> VLLM_MODEL (if unset)

### b685285ede4ba853debac9475052a369d5f91372.txt
- mime: text/plain
- lines: 247
- head:
    """
    Load `video_dr_agent/.env` from the repository root and apply name aliases.
    
    Supported aliases (after dotenv load):
      DEEP_RESEARCH_BASE_URL -> VLLM_BASE_URL (if unset)
      DEEP_RESEARCH_MODEL     -> VLLM_MODEL (if unset)

### b8be559d38716c3eb80c8a08fcdbb868e73e5b35.txt
- mime: text/plain
- lines: 426
- head:
    """
    FOCUS 关键帧筛选工具：调用 rsagent/FOCUS/select_keyframe.py 中的逻辑。
    
    工具名: focus_select_keyframes
    
    <tool_call> 建议 arguments:

### bf1ff3870f0c23502fa01e95609f98d6ad76b350.txt
- mime: text/plain
- lines: 158
- head:
    """
    Evidence-tier policy (rsagent-v3): constants referenced by critic and research prompts.
    
    Registry-tier facts require primary / official sources; travel and UGC sites must not be
    upgraded to “verified registry” in synthesis.
    

### c0f3ce978482879a1656ede0a5ab69b6165cf669.txt
- mime: text/plain
- lines: 906
- head:
    """
    FOCUS Data Processing and I/O Module
    
    This module handles data processing, video loading, and result output
    for the FOCUS keyframe extraction algorithm.
    """

### c230852d662fa43780421c58becfbd6ec99a3df5.json
- mime: application/json
- lines: 32
- head:
    {
      "tool": "focus_select_keyframes",
      "video_path": "/mnt/sda/Datasets/VideoDR/video/56.mp4",
      "video_path_in_arguments": null,
      "used_default_video_fallback": false,
      "question": "App store listing for ‘千问-阿里最强大模型官方AI助手’",

### c51215db4ff7f0e379b3044c8cd1fc1ed8cbec05.txt
- mime: text/plain
- lines: 318
- head:
    """
    多源网页搜索工具：并行调用 Serper（与 search_agent.serper_search 同协议），无 LLM 总结。
    
    工具名: deep_research_web_search
    
    <tool_call> arguments 示例:

### cde7dcb760aabb3668686331d882caff93cc0cdd.txt
- mime: text/plain
- lines: 565
- head:
    """
    Deep-research Critic: FOCUS technical check + **Top-3 frame ranking** (metadata) for injection;
    search snippet scoring + Jina Top-3 + LLM synthesis.
    
    Implements CriticAgent like OpenAICriticAgent; complete_round may use multiple stateless LLM calls
    and only appends the final feedback to CriticContext. Exposes ``focus_inject_paths`` for the loop.

### d08321adbcb8d6d7db4ad565b73de15f4cee84b2.txt
- mime: text/plain
- lines: 156
- head:
    You are the **Research** model in a video deep-research loop. Your job is to solve the user’s task using **video evidence** plus **web search**, in a disciplined multi-step way.
    
    ## What you receive at initialization
    
    - A **natural-language task / question** (the main objective).
    - **Uniformly sampled frames** from the video (typical workflow: **~20** frames for a **coarse global view**). These are **not** an exhaustive watch of the video.

### d1f0cb0dc87c90ebaa3cba09edc5b33a2644a6f0.txt
- mime: text/plain
- lines: 10
- head:
    # Local secrets (video_dr_agent pack-in config)
    video_dr_agent/.env
    
    __pycache__/
    *.py[cod]
    *.egg-info/

### d30a0c5d9214ccd46ded3a92719059a7e305290c.py
- mime: text/x-script.python
- lines: 100
- head:
    #!/usr/bin/env python3
    """
    Offline smoke: core ``run_loop`` + stub tools/critic + trace JSON (no vLLM).
    
    Run from ``rsagent/``:
      python scripts/smoke_offline_loop.py

### db399b7284fd5b51f116988fceeaf892bdca7eaa.json
- mime: application/json
- lines: 32
- head:
    {
      "tool": "focus_select_keyframes",
      "video_path": "/mnt/sda/Datasets/VideoDR/video/56.mp4",
      "video_path_in_arguments": null,
      "used_default_video_fallback": false,
      "question": "Identify company names on screen",

### db42a9249637562fce429007250a1e58e0e365e2.json
- mime: application/json
- lines: 32
- head:
    {
      "tool": "focus_select_keyframes",
      "video_path": "/mnt/sda/Datasets/VideoDR/video/1.mp4",
      "video_path_in_arguments": null,
      "used_default_video_fallback": false,
      "question": "aerial view of a waterfall surrounded by lush greenery",

### e023b92238473869a7ef349bbafbf9e0e09670fe.py
- mime: text/x-script.python
- lines: 552
- head:
    #!/usr/bin/env python3
    """
    CLI entry for video_dr_agent (**rsagent-v3**).
    
    v3 adds **evidence-tier** rules and **structured critic verdicts** (see ``critic_prompts``,
    ``evidence_policy``, and ``prompts/research_system_prompt.txt``) so Research does not treat

### e0584d8e1c50e7f90a954f094f5f6d5f9b16cc87.txt
- mime: text/plain
- lines: 32
- head:
    """Critic-side history: system + critic assistants only (no persistent tool-output user turns)."""
    
    from __future__ import annotations
    
    from copy import deepcopy
    from typing import Any, List

### e198d0ac4e4398e46da2b1458e9abaee8648472c.txt
- mime: text/plain
- lines: 46
- head:
    """
    Default research (Qwen3-VL) system prompt resolution — aligns with Video DeepResearch plan:
    system + first multimodal user; tool protocol lives in the system prompt file.
    """
    
    from __future__ import annotations

### e451b8b24a2acb65e151701630e72ab230909556.json
- mime: application/json
- lines: 26
- head:
    {
      "tool": "focus_select_keyframes",
      "video_path": "/mnt/sda/Datasets/VideoDR/video/1.mp4",
      "video_path_in_arguments": null,
      "used_default_video_fallback": false,
      "question": "temple on water with tiered roof",

### e45593e473669e02f183a735a0a15ad443f52391.json
- mime: application/json
- lines: 32
- head:
    {
      "tool": "focus_select_keyframes",
      "video_path": "/mnt/sda/Datasets/VideoDR/video/9.mp4",
      "video_path_in_arguments": null,
      "used_default_video_fallback": false,
      "question": "Identify the aircraft associated with the team 'Polse i Vaffel' that achieved the longest glide distance.",

### e7dee6557cea283726b32685f0bd71fe1fbaf6f3.txt
- mime: text/plain
- lines: 105
- head:
    """Video DeepResearch agent framework (vLLM Qwen3-VL research + OpenAI-compatible critic)."""
    
    from video_dr_agent.critic_context import CriticContext
    from video_dr_agent.critic_deep_research import (
        DeepResearchCriticAgent,
        DeepResearchCriticConfig,

### eb24c6bb5266f0a7267673959ea01a1899c3822e.txt
- mime: text/plain
- lines: 376
- head:
    """
    Main research loop (Video DeepResearch plan).
    
    Flow: optional **mandatory planning** (no tools) → **vLLM research** → parse ``<tool_call>`` →
    **ToolDispatcher** (raw tool JSON does not go directly to research) → **CriticPromptBuilder** +
    **CriticAgent** → append **critic text**; if the round includes successful ``focus_select_keyframes``,

### ee1615e97d67c0587dc863ba578645c8d5d7597d.txt
- mime: text/plain
- lines: 16
- head:
    # Additional dependencies for video_dr_agent (keep existing conda packages stable).
    #
    #   conda activate qwen3-30B
    #   pip install -r requirements-video-dr-agent.txt --upgrade-strategy only-if-needed
    #
    # only-if-needed: upgrade a package only when a dependency requires a newer version.

### f59c6f67b1d5fade0a304d1260ef5b7da893a4b2.txt
- mime: text/plain
- lines: 281
- head:
    """
    Main research loop (Video DeepResearch plan).
    
    Flow: **vLLM research** → parse ``<tool_call>`` → **ToolDispatcher** (raw tool JSON 不直接进 research) →
    **CriticPromptBuilder** + **CriticAgent** → append **critic 文本**；若本轮含成功的 ``focus_select_keyframes``，
    可再追加一条 **多模态 user**（关键帧 ``image_url``），让模型直接看到画面。

### f641f6f18b3d1b61285d5d79a16879d98fea6c1b.txt
- mime: text/plain
- lines: 38
- head:
    """Critic via OpenAI-compatible chat.completions (same shape as rsagent/test_prompt.py)."""
    
    from __future__ import annotations
    
    from dataclasses import dataclass
    

### f93903c7a650c0c2c4961e031734dd599fb918c6.txt
- mime: text/plain
- lines: 426
- head:
    """
    FOCUS 关键帧筛选工具：调用 rsagent/FOCUS/select_keyframe.py 中的逻辑。
    
    工具名: focus_select_keyframes
    
    <tool_call> 建议 arguments:

### fda19036841a5c262f2b07952163cfb0ef8469ba.txt
- mime: text/plain
- lines: 207
- head:
    """
    vLLM OpenAI-compatible chat for Qwen3-VL (see chat_qwen3vl_video.py on Vision-DeepResearch).
    """
    
    from __future__ import annotations
    

