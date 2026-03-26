# Multimodal Deep Research Benchmark Summary

## 总览

这份文档用于梳理与 `Video-grounded Deep Research` 相关的 benchmark，并回答三个核心问题：

1. 哪些 benchmark 适合做方法机制验证？
2. 哪个 benchmark 真正对应目标任务？
3. 哪些 benchmark 适合做泛化能力与长报告能力补充？

当前建议的主线是：

- `VideoDR`：主任务 benchmark
- `MMDeepResearch-Bench`：长报告输出能力评估，可选且非第一阶段重点
- 其余 benchmark：拆分验证不同能力，用于 ablation 与热启动

---

## Benchmark 列表

### 1. Vision-DeepResearch / VDR-Bench

论文：

- Vision-DeepResearch: Incentivizing Deep Research Capability in MLLMs
- https://arxiv.org/pdf/2601.22060
- https://github.com/Osilly/Vision-DeepResearch

Benchmark：

- Vision-DeepResearch Benchmark: Rethinking Visual and Textual Search for Multimodal Large Language Models
- https://arxiv.org/pdf/2602.02185
- VDR-Bench: https://huggingface.co/datasets/Osilly/VDR-Bench

任务形态：

- 单张图片
- 单个问题
- 网页搜索
- 短答案

适合验证：

- 是否真正使用了视觉信息
- 是否抑制了 `lazy search`
- 方法是否比 `image + web` 的 naive 拼接更有效

一句话定位：

- 受控的单图多模态 deep research 验证集

---

### 2. MMSearch-Plus

- MMSearch-Plus: Benchmarking Provenance-Aware Search for Multimodal Browsing Agents
- https://arxiv.org/pdf/2508.21475
- https://github.com/mmsearch-plus/MMSearch-Plus
- https://huggingface.co/datasets/Cie1/MMSearch-Plus

任务形态：

- 一张或多张真实截图 / 关键帧
- 单个问题
- 短答案
- 多轮工具调用

典型工具：

- 网页搜索
- 文本搜索
- 图像搜索
- 子图搜索
- zoom-in

适合验证：

- 局部细粒度视觉理解
- 子图检索与局部放大
- 图文互证
- provenance-aware evidence binding
- 多轮工具调用稳定性
- 时空外推

一句话定位：

- 细粒度视觉证据与 provenance 约束最强的 benchmark 之一

---

### 3. BrowseComp-VL

- WebWatcher: Breaking New Frontiers of Vision-Language Deep Research Agent
- https://arxiv.org/pdf/2508.05748
- https://github.com/Alibaba-NLP/DeepResearch/blob/main/WebAgent/WebWatcher/browsecomp-vl/

任务形态：

- 单图
- 单个模糊化问题
- 网络搜索

适合验证：

- 多跳深度检索
- 长链网页浏览
- 模糊实体处理
- agent loop 与 search policy

一句话定位：

- 更偏 agent / browsing / planning 能力的多模态 deep research benchmark

---

### 4. VideoDR

- Watching, Reasoning, and Searching: A Video Deep Research Benchmark on Open Web
- https://arxiv.org/pdf/2601.06943
- https://github.com/QuantaAlpha/VideoDR-Benchmark

任务形态：

- 单视频
- 单个问题
- 看一次视频后不回看
- 开放网页多轮搜索
- 多跳推理
- 短答案

核心流程：

- 感知 -> 查询 -> 检索 -> 推理 -> 输出

适合验证：

- 多帧与跨帧关联
- video grounding
- open-web 多轮检索
- long-horizon reasoning
- `goal drift`
- `video-only` 不可解 / `web-only` 不可解 的双依赖设定

一句话定位：

- 当前最直接对应 video deep research 主任务的 benchmark

---

### 5. MMDeepResearch-Bench

- MMDeepResearch-Bench: A Benchmark for Multimodal Deep Research Agents
- https://arxiv.org/pdf/2601.12346

任务形态：

- 文本问题
- 一组图片或图文捆绑材料
- 输出一篇完整、带引用、带图文对齐的深度研究报告

核心指标：

- `FLAE`：报告质量与长文本生成能力
- `TRACE`：引用与证据忠实度
- `MOSAIC`：图文一致性

适合验证：

- 长报告生成
- 引用与证据绑定
- 图文一致性
- 多模态深度研究输出上限

一句话定位：

- 长报告输出能力的主要 benchmark，但不是第一阶段主战场

---

### 6. LIVEVQA

- LIVEVQA: Live Visual Knowledge Seeking
- http://arxiv.org/pdf/2504.05288v1
- https://huggingface.co/datasets/ONE-Lab/LiveVQA-new

任务形态：

- 单张新闻配图
- 单个问题
- 可选新闻文本上下文
- 可选网页搜索
- 短答案

适合验证：

- 时效性知识获取
- 图文知识融合
- knowledge-seeking 能力
- 模型是否必须依赖 open web

补充说明：

- 这里的 “live” 强调新闻内容新、变化快，因此通常更依赖开放网络搜索

一句话定位：

- knowledge-seeking 的能力下界，不是标准意义上的 deep research benchmark

---

## 实验定位建议

建议将各 benchmark 的角色分成三层：

### 第一层：机制验证

- `VDR-Bench`
- `MMSearch-Plus`
- `LIVEVQA` 可作为知识检索下界补充

作用：

- 验证 `anchor / retrieval / evidence binding` 这条机制链是否有效
- 验证系统是否真的使用视觉与外部知识，而不是靠参数记忆

### 第二层：目标任务验证

- `VideoDR`

作用：

- 验证方法是否真正适用于 `video + web + reasoning` 的目标任务
- 验证是否缓解 `goal drift`、`long-horizon` 退化与 `video grounding` 失败

### 第三层：泛化与输出能力补充

- `BrowseComp-VL`
- `MMDeepResearch-Bench`

作用：

- 验证 agent loop / planning 能力是否具有迁移性
- 验证是否能够输出长报告，并保持引用与证据一致

---

## Benchmark 能力对比

| Benchmark | 最小视觉输入 | 多图/多帧 | 子图/Zoom | 强制多跳搜索 | 去捷径（lazy search） | 复杂网页结构 | 长链浏览 | 模糊实体 | Provenance | 噪声/干扰 | 搜索策略 | 最新知识 | 跨模态融合 | 跨模态对齐 | 时序推理 |
|----------|-------------|----------|-----------|--------------|----------------------|--------------|----------|----------|------------|-----------|-----------|-----------|-------------|-------------|------------|
| LIVEVQA | ✅ | ❌ | ❌ | ⚠️ 弱 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ⚠️ | ❌ |
| VDR-Bench | ✅ | ❌ | ⚠️ 隐式 crop | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ⚠️ | ❌ | ❌ | ⚠️ | ⚠️ | ❌ |
| MMSearch-Plus | ✅ | ✅ | ✅ | ✅ | ⚠️ | ⚠️ | ⚠️ | ❌ | ✅ | ✅ | ⚠️ | ❌ | ✅ | ✅ | ❌ |
| BrowseComp-VL | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ⚠️ | ✅ | ✅ | ❌ | ⚠️ | ⚠️ | ❌ |
| VideoDR | ❌ | ✅ | ⚠️ 隐式 | ✅ | ✅ | ✅ | ✅ | ❌ | ⚠️ | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ |

说明：

- `⚠️` 表示该能力存在，但不是 benchmark 设计重点，或支持较弱
- `VideoDR` 虽然不是“最小视觉输入”，但它是最接近目标任务的完整设定

---

## 能力递进主线

建议按下面的顺序理解这些 benchmark：

`LIVEVQA -> VDR-Bench -> MMSearch-Plus -> BrowseComp-VL -> VideoDR`

这条主线不是按“难度绝对值”排序，而是按“能力成分的逐步叠加”来理解：

- `LIVEVQA`：知识获取起点
- `VDR-Bench`：受控验证视觉参与与强制搜索
- `MMSearch-Plus`：细粒度局部视觉与 evidence-aware 检索
- `BrowseComp-VL`：agent loop 与 search policy
- `VideoDR`：最终完整 deep research 任务

---

## 分 benchmark 解释

### 1. LIVEVQA：knowledge-seeking 起点

它并不是标准意义上的 deep research benchmark，更接近知识获取能力的下界。

它强调的核心是：

- 最新知识
- 图像 + 新闻 + web 的跨模态知识融合

因此它适合回答的问题是：

- 系统是否真的需要 open web？
- 如果不用网络搜索，性能是否明显退化？
- 模型是否存在明显的 knowledge gap？

本质定位：

- `knowledge-seeking setting`

---

### 2. VDR-Bench：受控验证方法是否有效

相较于 LIVEVQA，VDR-Bench 进一步强化了：

- 视觉必须参与
- 搜索必须参与
- 需要抑制 `lazy search`

因此它特别适合验证：

- 方法是否真的“用到了视觉”
- 是否优于 `image/video + web` 的 naive 拼接方案
- 是否能够缓解 `lazy search`

本质定位：

- `controlled multimodal reasoning setting`

---

### 3. MMSearch-Plus：细粒度视觉与证据约束

MMSearch-Plus 在复杂性上又往前推进了一步，重点体现在：

- 多图 / 子图 / zoom-in
- provenance 约束
- 干扰信息与噪声

因此它特别适合验证：

- 是否真的用到了局部视觉信息
- evidence binding 是否准确，而不是“猜对答案”
- 多轮工具调用是否稳定，是否存在越搜越偏的问题

本质定位：

- `fine-grained + evidence-aware setting`

---

### 4. BrowseComp-VL：agent 与 planning 能力

BrowseComp-VL 的重点不再只是“看懂图片并找证据”，而是进一步强调：

- 模糊问题
- 长链浏览
- 搜索策略

因此它适合验证：

- agent loop / strategy 是否有效
- search policy 是否优于 naive search
- 是否容易出现早期错误导致的路径崩塌

本质定位：

- `agent / planning setting`

---

### 5. VideoDR：完整的 Video Deep Research 任务

VideoDR 将前面各 benchmark 的关键能力更强地耦合在一起：

- 视频输入
- 多帧与时序依赖
- open-web 检索
- multi-hop reasoning
- agent loop
- 双依赖约束

因此它最适合回答的问题是：

- 你的方法在真实目标任务上是否成立？
- 是否缓解：
  - `goal drift`
  - `long-horizon` 退化
  - `video grounding` 失败

本质定位：

- `full deep research setting`

---

## 当前建议的实验结论

如果目标是做 `video-grounded deep research framework`，建议这样使用这些 benchmark：

- `VideoDR`：主实验与核心结论
- `VDR-Bench / MMSearch-Plus`：机制验证与热启动
- `BrowseComp-VL`：agent 泛化能力补充
- `MMDeepResearch-Bench`：长报告输出能力补充
- `LIVEVQA`：知识获取能力下界补充，可选

换句话说：

- `VideoDR` 决定“方法在目标任务上是否成立”
- `VDR-Bench / MMSearch-Plus` 决定“核心机制是否有效”
- `BrowseComp-VL / MMDeepResearch-Bench` 决定“是否具备泛化与输出扩展潜力”
