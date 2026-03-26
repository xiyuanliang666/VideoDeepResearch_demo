# Video-Grounded Deep Research Framework 研究方案

## 1. 项目定位

本项目面向一个新兴但重要的问题设定：`video-grounded deep research`。在这一设定中，agent 需要从视频中提取线索，在开放网页中进行检索，完成跨模态证据对齐，并最终输出可验证答案。项目的首要目标不是搭建一个泛化的多模态平台，而是提出一个能够提升 `VideoDR` 类任务表现的框架。

与此同时，由于该方向仍处于早期阶段，直接面向视频 deep research 的 benchmark 仍然较少，因此框架需要天然兼容 `单图`、`多图` 与 `长报告` 等设定。这些附加设定的作用是围绕 VideoDR 主目标进行机制验证、泛化补充与能力展示，而不是取代主线任务。

## 2. 研究目标

我们希望构建一个 `video-grounded deep research framework`，重点解决 VideoDR 类任务中的五个核心瓶颈：

- 长链网页交互中的 `goal drift`
- 检索过程中对视频 anchor 的保持能力不足
- 视频证据与网页证据之间缺乏显式绑定
- 视频内部定位能力弱，导致外部检索前的视频线索利用不足
- 视频中的语音信息未被有效利用

同时，该框架还应保持对图像类多模态 deep research 场景的兼容性，从而支持统一框架下的方法验证、泛化实验与上限能力展示。

## 3. 核心假设

本工作的核心假设是：相比于“先做视频总结，再进行自由网页搜索”的朴素流程，采用一种 `anchor-centric + dual-retrieval` 的 research loop 会更适合 VideoDR。

该假设具体包括：

- 先将视频转换为结构化的 temporal anchors
- 检索前先利用本地视频检索与 ASR 文本检索缩小相关 clip 范围
- 每一步外部网页检索都由当前 anchor 状态驱动
- 网页证据必须显式绑定回视频时间段与 anchor 字段
- 音频优先通过 `ASR -> 文本化 -> 时间对齐` 的方式进入 research loop
- memory 不再是长篇自由文本 history，而是轻量级 anchor/evidence store

如果这一假设成立，框架应当能够减少 drift，提高证据质量，并在多步检索场景下得到更可靠的最终推理结果。

## 4. 方法概述

本方法包含三个核心创新点和一个关键工程支撑。

### 4.1 Temporal Anchor Abstraction

系统不再反复将整段视频暴露给 agent，而是先将视频片段转换为结构化的 `Anchor` 对象。每个 anchor 建议包含：

- 时间范围
- 关键帧
- 实体
- 动作
- 场景/事件描述
- OCR、字幕或 transcript 线索
- speech clues
- 面向检索的 query 候选

这一层承担从原始视频到 deep research 下游流程之间的压缩与抽象作用。

### 4.2 Anchor-Centric Dual Retrieval Loop

Agent 在一个固定步数的循环中运行，第一版可设置为 `K=2~3`。每一步中，系统先基于本地 embedding 检索和 ASR 文本检索在视频内部定位高价值 clip，再基于 active anchors 生成网页 query，执行 open-web 检索，对结果进行过滤与重排序，并仅用 grounded evidence 更新 memory。

这一点与通用 deep research agent 的关键区别在于：

- 检索轨迹不再主要由历史文本自然扩张
- 视频内部检索、语音文本检索与外部网页检索被纳入同一研究循环
- anchor 状态成为整个控制流程的中心

### 4.3 Cross-Modal Evidence Binding

检索得到的网页结果不能直接拼接回上下文，而必须和一个或多个 anchors 建立显式对齐关系，包括：

- 它支持或反驳哪个 anchor
- 它匹配了哪个实体、动作、语音线索或时间线索
- 它具体绑定到视频中的哪个时间戳
- 它允许系统更新什么 claim

这一模块预计会成为 VideoDR 中最关键的部分，因为该 benchmark 明确要求 video-only 与 web-only 单独都不可解。

### 4.4 工程支撑：工具库与缓存式表示层

为了让方法真正可跑、可评估、可扩展，框架需要引入独立的工具库与表示层，负责：

- 视频切片
- 帧采样与提取
- 音频抽取与 ASR 转写
- 字幕抽取与对齐
- embedding 编码与缓存
- clip/path 索引管理
- 质量检查与结果复现

这一部分不是论文主创新，但它是让 `video -> clip -> frame/audio -> embedding/transcript -> retrieval -> research loop` 真正成立的基础设施。

## 5. 框架层原则

尽管方法本身是为 VideoDR 优化的，具体实现仍应被设计成一个可复用的 `multimodal grounded research runtime`。参考现有 `VideoDeepResearch` 一类项目结构时，应采用“**工程分层部分对齐，方法主干不照搬**”的原则。

框架在工程层面应吸收以下优点：

- 编码器与表征层独立组织
- `eval/` 独立成层
- `train/` 提前预留
- 视频工具能力单独封装
- 音频与 ASR 工具能力单独封装

但方法主干必须明确保留以下模块：

- `orchestration`
- `anchors`
- `grounding`
- `memory`
- `compilers`

也就是说，本项目不能被组织成“检索器主导的工程”，而应被组织成“以 grounded research loop 为中心、检索与编码为支撑”的系统。

## 6. 系统实现假设

为了支撑后续实验，本项目的实现将围绕六层结构展开：

1. `Task & Orchestration Layer`
2. `Tool Library Layer`
3. `Preprocessing & Representation Layer`
4. `Research Core Layer`
5. `Evidence / Output / Evaluation Layer`
6. `Training Interface Layer`

其中最关键的工程假设包括：

- 引入独立的 `tools/` 封装视频切片、帧提取、音频抽取、ASR、字幕、缓存、校验等操作
- 引入音频抽取与 ASR 转写链路，并与 clip 时间戳对齐
- 引入独立的 `prompts/` 管理 anchor extraction、query planning、evidence binding 等模板
- 引入本地 embedding 检索与开放网页检索并存的“双检索”架构
- 引入单独的评估与结果汇总模块
- 为 `SFT / DPO / DRPO` 预留训练数据导出接口

## 7. Benchmark 分层策略

本项目中的 benchmark 按三层组织。

### 7.1 第一层：机制验证

用于在大规模 VideoDR 实验前先验证框架核心组件是否有效。

- `MMSearch-Plus`
- `VDR-Bench`
- `BrowseComp-VL` 可作为图像侧补充

核心问题：

- `anchor -> retrieval -> evidence binding` 这条机制链条是否真的能提升 grounded multimodal search 与推理能力？
- 在引入本地检索、ASR 文本检索和结构化 memory 后，系统是否更稳定？

### 7.2 第二层：目标任务验证

这一层是项目的主战场。

- `VideoDR`

核心问题：

- 所提出的框架是否能够提升 video-conditioned open-web reasoning？
- 是否能够缓解 `goal drift`？
- 是否能够在长链交互中更充分地利用视频线索？
- 本地视频检索、ASR 文本检索与外部网页检索结合后，是否优于 naive 拼接式 multimodal baseline？

评估方式上，除了常规 benchmark 指标外，建议加入 `LLM-as-a-Judge` 作为补充，用于判断：

- 最终答案是否真正被视频证据与网页证据共同支持
- 中间证据链是否可信
- 输出是否存在“看起来合理但证据不足”的情况

### 7.3 第三层：能力扩展

这一层主要用于展示迁移性与上限能力，而不是定义主结论。

- `BrowseComp-VL`
- `MMDeepResearch-Bench`

核心问题：

- 该框架是否对更一般的多模态 agent 设定也有帮助？
- 相同的 evidence 表示是否能够支撑 grounded long-form report generation？

在 `MMDeepResearch-Bench` 等长报告设定中，建议重点采用 `LLM-as-a-Judge`，将以下信息打包后进行综合评分：

- 输入任务
- 模型报告
- 参考材料
- 推理链
- 证据链
- 工具调用链

## 8. 实验设计

### 8.1 主要 Baseline

建议至少包含以下几组 baseline：

- `vision-only`：仅基于图像或视频作答
- `vision+asr-only`：仅基于图像/视频与语音转写文本作答，不使用网页检索
- `web-only`：仅基于网页检索作答
- `vision+web naive concat`：将视觉总结与网页结果直接拼接，不做结构化控制
- `general deep research agent + visual pre-summary`：用通用 web research agent，视觉仅作为一次性预处理
- `local retrieval + naive reasoning`：先做本地视频检索与 ASR 文本检索，但不使用 anchor-centric 控制

这些 baseline 有助于区分“只是有多模态输入能力”与“真正来自框架设计的提升”。

评估时建议同时保留两类视角：

- 传统 benchmark 指标
- `LLM-as-a-Judge` 的维度化评分

### 8.2 消融实验

关键消融建议包括：

- 去掉 `anchor extraction`
- 去掉 `local retrieval`
- 去掉 `ASR / speech clues`
- 去掉 `anchor refresh / memory reinjection`
- 去掉 `evidence binding`
- 用单跳检索替代多跳检索
- 限制搜索轮数
- 用全局视频摘要替代 segment-level anchors

这些消融都直接对应论文的机制性主张。

### 8.3 失败案例分析

失败分析应重点关注 VideoDR 特有的问题：

- 多轮网页检索后的 anchor drift
- 本地视频检索召回不到关键 clip
- ASR 错误导致 speech clue 错绑
- 跨帧实体绑定错误
- 虽然检索到了网页证据，但没有绑定到正确的视频时间戳
- 过度依赖参数知识
- 随着交互轮数增加，后续步骤反而降低了回答质量
- Judge 模型指出“答案表面正确但证据链不足”

## 9. 预期贡献

本工作预计产生以下贡献：

- 提出一个以 temporal anchors、双检索控制、speech-aware clues 与 evidence binding 为核心的 `video-grounded deep research framework`
- 构建一套同时兼容 image、multi-image 与 video 的统一 grounded research runtime
- 通过实验表明，相比朴素的 multimodal deep research pipeline，anchor-centric 控制在 VideoDR 类任务上更有效
- 分析 anchor maintenance、local retrieval、speech clues 与 evidence binding 对 drift 和长链推理退化的影响
- 引入 `LLM-as-a-Judge` 作为 VQA 与长报告的补充评估机制
- 给出一条从短答案 grounded reasoning 过渡到长报告 grounded generation 的可行路径

## 10. 第一阶段范围

第一阶段实现需要有意保持简单，但要把主干链路设计正确：

- 固定步数检索循环，`K=2~3`
- 单一 search API
- 单一 MLLM 负责 anchor extraction、reranking、alignment 与 reasoning
- 单一 embedding 方案
- 引入本地 embedding 检索
- 引入 `ASR -> transcript -> 时间对齐` 链路
- 引入独立工具库、prompt 管理与评估汇总模块
- 引入 `LLM-as-a-Judge` 评估模块
- 优先支持短答案输出
- 报告生成与更广泛的泛化实验留到后续阶段

第一阶段的核心目标是先闭环跑通：

- `video -> clip -> frame/audio -> embedding cache + asr transcript -> local retrieval -> anchor -> web retrieval -> evidence -> answer`

## 11. 阶段性里程碑

### Phase A: 机制热启动

- 实现视频处理工具链与 embedding 缓存链
- 实现音频抽取、ASR 与 transcript 对齐链
- 实现 image 与 video 的 anchor extraction
- 在 `VDR-Bench` 与 `MMSearch-Plus` 上完成初步实验
- 验证本地检索、ASR 文本检索与 evidence binding 的基础效果
- 实现 VQA 场景下的 `LLM-as-a-Judge` 补充评估

### Phase B: VideoDR 主实验

- 将同一框架迁移到 `VideoDR`
- 运行 baseline 与 ablation 实验
- 对 drift、local retrieval failure、ASR error 与 evidence grounding 进行失败案例分析
- 引入基于证据链和工具链的 Judge 评估

### Phase C: 扩展与上限展示

- 在 `BrowseComp-VL` 上测试迁移性
- 增加 report compiler，对接 `MMDeepResearch-Bench`
- 验证同一 evidence store 是否能支持 grounded long-form synthesis
- 在长报告设定中使用 `LLM-as-a-Judge` 做多维评分
- 为后续 `SFT / DPO / DRPO` 导出训练数据

## 12. 一句话总结

本项目提出一个 `video-grounded deep research framework`，其核心思想是通过 temporal anchors 保持视频线索，以本地视频检索、ASR 文本检索和开放网页检索构成双检索式研究循环，并将网页证据显式绑定回视频时间段；图像类 benchmark 用于机制验证，VideoDR 用于目标任务证明，其他多模态 benchmark 用于迁移性与上限能力展示。
