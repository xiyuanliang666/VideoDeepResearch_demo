# Video-Grounded Deep Research Framework

## 1. 设计目标

当前框架的首要目标不是做一个“大而全”的多模态平台，而是尽快支持下面两件事：

1. 在 `VideoDR` 上找到真正有效的方法创新点。
2. 在不放大工程复杂度的前提下，兼容以下五个 benchmark 的测试：
   - `MMSearch-Plus`
   - `VDR-Bench`
   - `VideoDR`
   - `BrowseComp-VL`
   - `MMDeepResearch-Bench`

因此，新的框架采用：

- `method-first core`
- `thin benchmark adapters`
- `low-risk observation layer`

核心原则是：

- 方法主干要小
- observation 不能变成新的 drift 源头
- benchmark 差异放到 adapter 层解决
- 输出差异放到 compiler / evaluator 层解决

## 2. 核心设计判断

### 2.1 先收缩，再扩展

当前阶段不应优先建设：

- 多 agent
- 复杂 planner / supervisor
- 重型 memory 系统
- 完整训练模块
- 深度 / 热红外主链
- 大型 prompt 平台

当前阶段只保留和方法创新直接相关的主干：

- `preprocessing`
- `observation`
- `anchors`
- `retrieval`
- `grounding`
- `reasoning`
- `evaluation`

### 2.2 observation 必须是“低风险候选层”

observation 不是事实结论，不是最终 memory，也不应成为单点锁死的摘要。

新的 observation 设计目标是：

- `low-commitment`
- `question-conditioned`
- `multi-candidate`
- `refreshable`
- `traceable`

也就是说，observation 只保存“值得后续验证的局部观察”，而不是提前替 agent 下结论。

## 3. Low-Risk Observation 设计

## 3.1 Observation 的角色

在新框架里，observation 的定位是：

- 输入模态经过预处理后的局部观察单元
- 是 anchor 的候选来源
- 不是最终研究单元
- 不是不可修改的全局摘要

主链关系应为：

`raw input -> observation candidates -> anchors -> retrieval -> binding -> answer`

而不是：

`raw input -> one-shot observation summary -> whole pipeline`

## 3.2 Low-Risk Observation 原则

### 原则 1：弱语义，不强结论

observation 只记录：

- 看到什么
- 听到什么
- 出现在哪个时间范围
- 可能有哪些待验证线索

避免一开始就写：

- “这是某事件”
- “这是某人物”
- “这是某因果关系”

应优先写成：

- `scene clues`
- `speech clues`
- `ocr clues`
- `candidate entities`
- `candidate actions`
- `uncertain slots`

### 原则 2：多候选并存

不要只保留一个 observation。

应保留：

- `top-k observation candidates`
- `不同来源 observation`
  - frame-based
  - transcript-based
  - OCR-based

后续由 anchor selection 和 evidence binding 决定谁更可信。

### 原则 3：问题条件化

observation 不应该只是“视频概览”。

它应该至少部分受问题约束，避免把无关但显眼的内容放大。

因此 observation 生成要支持两种模式：

- `generic observation`
- `question-conditioned observation`

第一版优先保留后者。

### 原则 4：可刷新、可回退

如果后续 evidence 表明当前 observation 带偏了流程，应支持：

- 降权
- 替换
- 回到上游重新选 observation

所以 observation 必须是可更新状态，而不是一次性静态产物。

### 原则 5：和 anchor 分离

`observation != anchor`

两者关系应为：

- observation：局部候选观察
- anchor：当前被选中、驱动检索与绑定的工作单元

这样 observation 的错误不会直接等价于主推理链错误。

## 3.3 Observation 的最小字段

建议保留以下字段：

- `observation_id`
- `source_type`
- `source_path`
- `timestamp_start`
- `timestamp_end`
- `frame_paths`
- `transcript_text`
- `ocr_text`
- `scene_clues`
- `speech_clues`
- `candidate_entities`
- `candidate_actions`
- `confidence`
- `uncertainty_notes`
- `quality_flags`

其中最关键的是：

- `candidate_entities`
- `candidate_actions`
- `uncertainty_notes`

这些字段会直接帮助后续做更稳的 anchor selection。

## 4. 新框架总览

新框架收缩为四层：

1. `Benchmark Adapter Layer`
2. `Method Core Layer`
3. `Output & Evaluation Layer`
4. `Future Training Hooks`

这样拆分的目的：

- 主干只服务方法创新
- benchmark 兼容通过 adapter 解决
- 输出差异不污染核心推理链
- 训练接口只预留，不进入第一阶段主战场

## 5. Benchmark Adapter Layer

这一层只负责把不同 benchmark 的原始样本格式转成统一 `Sample`。

不要把 benchmark 的特殊逻辑写进主 pipeline。

建议每个 benchmark 一个 adapter：

- `mmsearch_plus.py`
- `vdr_bench.py`
- `videodr.py`
- `browsecomp_vl.py`
- `mmdeepresearch_bench.py`

每个 adapter 负责：

- 读取样本
- 规范输入路径
- 规范问题字段
- 规范参考答案字段
- 指定 output mode

这一层不做研究推理。

## 6. Method Core Layer

这是整个项目真正应该投入的方法主干。

建议只保留 6 个模块。

### 6.1 Preprocessing

职责：

- 视频切片
- 抽帧
- 音频抽取
- ASR
- OCR
- 单图 / 多图 / 视频统一转 observation candidates

输入：

- `image`
- `image_set`
- `video`

输出：

- `Observation[]`

### 6.2 Observation Manager

职责：

- 生成 low-risk observations
- observation 过滤与降权
- question-conditioned observation refresh

这是新的关键中间层。

建议支持：

- `build_observations`
- `score_observations`
- `refresh_observations`
- `drop_stale_observations`

### 6.3 Anchor Builder

职责：

- 从 observation candidates 中构建 anchor
- 选择 active anchors
- 为每个 anchor 生成 search query

anchor 是真正进入 research loop 的工作单元。

建议支持：

- `extract_anchors`
- `select_active_anchors`
- `rewrite_anchor_queries`

### 6.4 Retrieval

职责：

- 本地检索
- 开放网页检索
- 基础 rerank / filter

显式分成两类：

- `local retrieval`
  - 面向 frame / clip / OCR / ASR transcript
- `web retrieval`
  - 面向开放网页证据

### 6.5 Grounding

职责：

- 将网页候选与视频 anchor 对齐
- 生成 evidence
- 生成 binding

这是区分“普通多模态拼接”和“video-grounded deep research”的核心模块。

建议支持：

- `bind_web_to_anchor`
- `verify_temporal_alignment`
- `build_claims`

### 6.6 Reasoning

职责：

- 基于 evidence chain 输出答案或报告
- 短答案与长报告走不同 compiler

第一版只需：

- `short_answer_compiler`
- `report_compiler`

但 report 模式不是当前主战场。

## 7. Output & Evaluation Layer

这一层只负责：

- 保存 trace
- 保存 answers / reports
- 聚合 benchmark 结果
- 执行 judge
- 输出 failure taxonomy

建议模块：

- `judge.py`
- `aggregate.py`
- `case_study.py`

## 8. Future Training Hooks

训练不是当前阶段重点，但需要留接口。

只预留导出能力：

- `export_sft_examples`
- `export_preference_pairs`
- `export_trajectory_records`

对应未来可能的：

- `SFT`
- `DPO`
- `DRPO`

第一阶段不应投入训练实现。

## 9. 统一数据对象

当前阶段不需要过多 schema，只保留最小一组。

### 9.1 Sample

统一 benchmark 输入。

字段建议：

- `sample_id`
- `benchmark_name`
- `input_type`
- `media_paths`
- `question`
- `reference_answer`
- `metadata`

### 9.2 Observation

低风险候选观察单元。

字段建议：

- `observation_id`
- `source_type`
- `source_path`
- `timestamp_start`
- `timestamp_end`
- `frame_paths`
- `transcript_text`
- `ocr_text`
- `scene_clues`
- `speech_clues`
- `candidate_entities`
- `candidate_actions`
- `confidence`
- `uncertainty_notes`

### 9.3 Anchor

检索与推理的当前工作单元。

字段建议：

- `anchor_id`
- `source_observation_ids`
- `time_span`
- `entities`
- `actions`
- `scene_summary`
- `speech_clues`
- `search_queries`
- `priority_score`
- `status`

### 9.4 Evidence

正式进入 grounded state 的证据对象。

字段建议：

- `evidence_id`
- `evidence_type`
- `source_ref`
- `content_summary`
- `source_url`
- `source_timestamp`
- `confidence`

### 9.5 Binding

anchor 和 evidence 的显式绑定。

字段建议：

- `binding_id`
- `anchor_id`
- `evidence_id`
- `relation`
- `reason`
- `confidence`

### 9.6 Prediction

统一输出对象。

字段建议：

- `sample_id`
- `output_mode`
- `answer`
- `report`
- `evidence_chain`
- `judge_result`

## 10. 关键创新空位

当前框架必须为你最可能发力的方法点留空位。

## 10.1 关键帧 / observation 生成

建议明确预留：

- `frame_selector`
- `event_selector`
- `query_conditioned_frame_selector`
- `observation_refiner`

这是最值得做创新的地方之一。

## 10.2 视觉锚点构建

建议明确预留：

- `visual_anchor_extractor`
- `cross_frame_anchor_builder`
- `anchor_refresh`

这是最可能直接打到 `VideoDR` 痛点的位置。

## 10.3 视频-网页绑定

建议明确预留：

- `temporal_binding_verifier`
- `web_evidence_grounder`
- `claim_consistency_checker`

这是最可能缓解 `goal drift` 的位置。

## 11. 端到端最小数据流

新的第一版主链应为：

1. 输入 `video / image / image_set`
2. 预处理得到 `Observation[]`
3. 从 observation 中生成 `Anchor[]`
4. 执行 `local retrieval`
5. 围绕 active anchors 执行 `web retrieval`
6. 执行 `evidence binding`
7. 生成短答案或长报告
8. 执行 judge 与结果汇总

对于 `VideoDR`：

`video -> clip/frame/asr -> observations -> anchors -> local retrieval -> web retrieval -> binding -> answer`

对于 `MMSearch-Plus / VDR-Bench / BrowseComp-VL`：

`image or image_set -> observations -> anchors -> web retrieval -> binding -> answer`

对于 `MMDeepResearch-Bench`：

同样沿用上述 evidence 流，只是切换 `report` 输出模式。

## 12. Benchmark 适配策略

新的兼容方式不是把所有 benchmark 特性做进主干，而是：

- 每个 benchmark 一个 adapter
- 共用同一个 method core
- 通过 profile 切换 output mode 和 evaluation mode

推荐映射如下：

### `MMSearch-Plus`

角色：

- 机制热启动

重点验证：

- observation 是否保留关键信号
- anchor 是否帮助后续检索
- evidence binding 是否优于 naive concat

### `VDR-Bench`

角色：

- 机制验证延伸

重点验证：

- 局部视觉线索
- 多轮视觉检索
- visual anchor 是否值得继续深入

### `VideoDR`

角色：

- 主目标 benchmark

重点验证：

- 是否缓解 `goal drift`
- 是否提升视频线索利用率
- 是否减轻 long-horizon degradation

### `BrowseComp-VL`

角色：

- 泛化补充

重点验证：

- 方法是否不仅对 VideoDR 特化

### `MMDeepResearch-Bench`

角色：

- 上限展示

重点验证：

- 同一 evidence 流是否能支撑长报告输出

## 13. 推荐目录结构

新的目录应明显比旧版更小。

```text
VideoGroundedResearch/
├── README.md
├── proposal.md
├── framework.md
├── training-free_plan.md
├── configs/
│   ├── methods/
│   └── benchmarks/
├── src/
│   ├── adapters/
│   │   ├── mmsearch_plus.py
│   │   ├── vdr_bench.py
│   │   ├── videodr.py
│   │   ├── browsecomp_vl.py
│   │   └── mmdeepresearch_bench.py
│   ├── core/
│   │   ├── preprocessing.py
│   │   ├── observation.py
│   │   ├── anchors.py
│   │   ├── retrieval.py
│   │   ├── grounding.py
│   │   ├── reasoning.py
│   │   ├── judge.py
│   │   └── pipeline.py
│   ├── schemas/
│   │   ├── sample.py
│   │   ├── observation.py
│   │   ├── anchor.py
│   │   ├── evidence.py
│   │   └── prediction.py
│   └── tools/
│       ├── video_tools.py
│       ├── audio_tools.py
│       ├── asr_tools.py
│       ├── cache_tools.py
│       └── io_tools.py
├── scripts/
│   ├── run_single_case.py
│   ├── run_benchmark.py
│   └── summarize_results.py
├── data/
│   ├── raw_videos/
│   ├── clips/
│   ├── dense_frames/
│   ├── audio/
│   ├── transcripts/
│   └── benchmarks/
└── outputs/
    ├── traces/
    ├── answers/
    └── eval/
```

## 14. 第一阶段最小实现建议

第一阶段只做：

- 单样本运行
- 小规模 benchmark 子集运行
- 4 个 baseline
- 关键消融
- trace 保存
- judge 汇总

不要一开始就做：

- 全量 benchmark 自动化
- 长报告主实验
- 训练
- 多 agent
- 重 memory

## 15. 框架总结

新的 framework 有三个核心变化：

1. `observation` 被降级为低风险候选层，而不是高承诺摘要层。
2. 项目主干收缩成 `observation -> anchor -> retrieval -> binding -> answer`。
3. 五个 benchmark 的兼容性通过 `thin adapters` 保留，而不是靠扩大主工程来换。

这版框架的真正中心不再是“做多大”，而是：

- 哪种 observation 更稳
- 哪种 visual anchor 更有效
- 哪种 video-web binding 更能抑制 drift

这也更符合当前阶段最重要的目标：尽快在 `VideoDR` 上长出可验证的方法创新点。
