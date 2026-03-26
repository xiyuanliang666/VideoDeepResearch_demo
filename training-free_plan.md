# Training-Free Experiment Plan

## 1. 目标

这份计划回答一个核心问题：

在**不进行任何训练**的前提下，如何基于当前 `video-grounded deep research framework` 在相关 benchmark 上开展实验、评估效果、发现问题，并进一步提炼可发表的创新点。

这里的重点不是训练模型参数，而是验证：

- 推理时框架设计本身是否有效
- 哪些模块真正带来收益
- 系统最常见的失败模式是什么
- 后续训练或方法改进应该瞄准哪里

---

## 2. 核心思路

第一阶段把当前工作定义为一个 **training-free / inference-time framework study**：

- 底层 backbone 固定
- 不做 SFT / DPO / DRPO
- 只比较不同 `controller / retrieval / memory / binding` 设计
- 在相同预算、相同模型、相同输入条件下做公平对比

因此，当前实验的结论不是“模型学到了什么”，而是：

- 这套框架设计本身是否优于 naive pipeline
- 这些设计是否缓解了 VideoDR 的典型问题

---

## 3. 当前阶段能做什么

即使不训练，以下模块也可以直接运行：

- 视频切片
- 帧提取
- 音频抽取与 ASR
- embedding 编码与本地检索
- anchor extraction
- web retrieval
- evidence binding
- final reasoning
- LLM-as-a-Judge

因此，第一阶段完全可以做端到端 benchmark 测评。

---

## 4. 实验目标拆分

建议将 training-free 实验拆成三个层次。

### 4.1 端到端效果验证

回答：

- 当前框架是否比 naive baseline 更强？
- 在目标 benchmark 上是否成立？

### 4.2 机制有效性验证

回答：

- `anchor`
- `local retrieval`
- `ASR`
- `memory refresh`
- `evidence binding`

这些模块分别有没有用？

### 4.3 失败模式归纳

回答：

- 框架最常见失败发生在哪一层？
- 后续创新点应该从哪里长出来？

---

## 5. Benchmark 使用策略

### 5.1 第一层：机制热启动

优先使用：

- `VDR-Bench`
- `MMSearch-Plus`

可选补充：

- `LIVEVQA`

目标：

- 验证框架是否真正使用视觉与检索
- 验证 `anchor / retrieval / evidence binding` 是否有效
- 检查 naive 拼接与结构化 research loop 的差异

### 5.2 第二层：目标任务主实验

主 benchmark：

- `VideoDR`

目标：

- 验证框架是否适用于 `video + web + reasoning`
- 观察是否缓解：
  - `goal drift`
  - `long-horizon` 退化
  - `video grounding` 失败

### 5.3 第三层：泛化与输出能力补充

使用：

- `BrowseComp-VL`
- `MMDeepResearch-Bench`

目标：

- 验证 agent / planning 能力是否可迁移
- 验证 evidence store 是否能支撑长报告输出

说明：

- 这一层不是第一阶段必须完成的主战场

---

## 6. 第一阶段建议先跑哪些数据

不建议一开始就跑全量 benchmark。更稳的做法是先建立小规模 dev 实验包。

### 6.1 小规模热启动集合

建议起步样本量：

- `VDR-Bench`: 50 题
- `MMSearch-Plus`: 50 题
- `VideoDR`: 30 题

用途：

- 检查 pipeline 是否能稳定跑通
- 快速收集 trace
- 观察最典型失败模式

### 6.2 第一轮扩展集合

当小规模实验稳定后，再扩大到：

- `VDR-Bench`: 200 题或官方 test split
- `MMSearch-Plus`: 100~200 题
- `VideoDR`: 完整可用 split

---

## 7. Baseline 设计

第一阶段至少保留以下 baseline。

### 7.1 基础 baseline

- `vision-only`
  - 仅基于图像或视频作答
- `vision+asr-only`
  - 基于视觉与 ASR 文本作答，不使用网页搜索
- `web-only`
  - 仅使用网页检索作答

### 7.2 弱多模态 baseline

- `vision+web naive concat`
  - 视觉总结 + 网页结果直接拼接
- `general deep research agent + visual pre-summary`
  - 通用 web research agent，视觉仅做一次性预处理

### 7.3 局部改进 baseline

- `local retrieval + naive reasoning`
  - 先做本地视频检索与 ASR 文本检索
  - 但不使用 anchor-centric 控制

### 7.4 主方法

- `your framework`
  - local retrieval
  - ASR
  - anchor extraction
  - structured memory
  - evidence binding
  - final reasoning

---

## 8. 控制变量原则

为了让结论可信，第一阶段应尽量固定以下变量：

- 同一个主 MLLM
- 同一个搜索 API
- 同样的网页检索预算
- 同样的最大步数
- 同样的输入可见范围

这样最终结果才能归因于：

- 框架设计差异

而不是归因于：

- 换了更强模型
- 检索次数更多
- 提供了更多上下文

---

## 9. 消融实验设计

建议至少做以下消融。

### 9.1 模块级消融

- 去掉 `anchor extraction`
- 去掉 `local retrieval`
- 去掉 `ASR / speech clues`
- 去掉 `anchor refresh / memory reinjection`
- 去掉 `evidence binding`

### 9.2 策略级消融

- 多步检索改成单步检索
- 限制最大检索轮数
- 用全局视频摘要替代 segment-level anchors
- 关闭 query rewrite

### 9.3 输出级消融

- 去掉 evidence packaging
- 不输出 citations / evidence chain

这些消融可以帮助回答：

- 到底是哪个模块带来提升？
- 提升是否来自“真 grounding”，还是只是“搜得更多”？

---

## 10. 过程指标

除了 benchmark 最终分数，建议额外记录一组过程指标。

### 10.1 本地检索相关

- `Local Retrieval Recall@K`
  - 关键 clip 是否被召回
- `ASR Retrieval Recall@K`
  - 关键 transcript 段是否被召回

### 10.2 Anchor 相关

- `Anchor Coverage`
  - gold 相关实体 / 动作 / 时间线索是否被覆盖
- `Anchor Precision`
  - 被选中的 anchor 是否真的与问题相关

### 10.3 检索与漂移相关

- `Drift Rate`
  - query 是否逐渐偏离原始视频线索
- `Useful Web Retrieval Rate`
  - 检索到的网页结果中有多少真正进入 evidence

### 10.4 对齐与证据相关

- `Binding Precision`
  - 网页证据是否正确绑定到视频时间戳
- `Claim Support Rate`
  - 最终 claim 是否有足够 evidence 支撑

### 10.5 模态利用增益

- `Video Utilization Gain`
  - 去掉视频线索后的性能跌幅
- `ASR Utilization Gain`
  - 去掉 ASR 线索后的性能跌幅
- `Web Utilization Gain`
  - 去掉网页检索后的性能跌幅

---

## 11. 需要保存哪些中间结果

为了后续做 failure analysis 和提炼创新点，每次运行都建议保存完整 trace。

至少保存：

- 输入任务
- 视频 / 图像 / clip 信息
- transcript
- active anchors
- 每一步 query
- 每一步 local retrieval 结果
- 每一步 web retrieval 结果
- rerank 结果
- evidence binding 结果
- claim 更新记录
- final answer / final report
- tool trace
- reasoning trace
- judge input
- judge output

建议输出格式：

- `json`
- `jsonl`
- `csv` 汇总表
- 关键 case 的 `md` 报告

---

## 12. LLM-as-a-Judge 的使用方式

Judge 不是替代 benchmark 指标，而是作为补充评估。

### 12.1 VQA / 短答案

Judge 重点判断：

- 最终答案是否语义正确
- 是否真正使用了视频证据
- 是否真正使用了网页证据
- evidence chain 是否支持答案

### 12.2 长报告

Judge 重点判断：

- 报告是否事实一致
- 引用是否可信
- 图文或图证据是否对齐
- 推理链是否自洽
- 工具调用链是否支持最终结论

### 12.3 Judge 输入打包

建议统一打包：

- 输入任务
- 模型输出
- 参考答案或参考材料
- 推理链
- 证据链
- 工具调用链
- rubric

这样后续可以做：

- 自动聚合
- case study
- 人工抽样复核

---

## 13. 第一轮实验应该怎么跑

建议按下面顺序推进。

### 第一步：确认 pipeline 可运行

只跑极小样本：

- `VDR-Bench`: 10
- `MMSearch-Plus`: 10
- `VideoDR`: 10

目标：

- 检查数据读取是否正常
- 检查视频处理与 ASR 是否稳定
- 检查 search loop 是否能闭环
- 检查 trace 是否完整保存

### 第二步：跑小规模开发集

- `VDR-Bench`: 50
- `MMSearch-Plus`: 50
- `VideoDR`: 30

目标：

- 对比 baseline
- 跑消融
- 收集失败案例

### 第三步：定位主要失败模式

对错误样本分桶，例如：

- `missed_anchor`
- `bad_local_retrieval`
- `bad_asr`
- `bad_web_retrieval`
- `bad_binding`
- `goal_drift`
- `unsupported_final_answer`

### 第四步：形成第一版结论

回答以下问题：

- 框架是否优于 naive baseline？
- 哪个模块最关键？
- 最常见的失败发生在哪一层？
- 哪些问题必须靠训练解决，哪些问题可以继续靠推理时设计解决？

---

## 14. 如何从实验中提炼创新点

创新点最好不是预设出来的，而是从 trace 与失败模式中“长出来”。

### 14.1 如果主要失败是 drift

可能的创新方向：

- `anchor refresh`
- `memory reinjection`
- `anti-drift controller`

### 14.2 如果主要失败是本地召回差

可能的创新方向：

- `clip + ASR fusion retrieval`
- `better local retrieval routing`

### 14.3 如果主要失败是网页证据绑不准

可能的创新方向：

- `cross-modal evidence verifier`
- `timestamp-aware binding`

### 14.4 如果主要失败是最终答案看似合理但证据不足

可能的创新方向：

- `judgeable evidence packaging`
- `claim support validation`

也就是说：

- 第一阶段实验的目标不是直接拿 SOTA
- 而是找到“哪一个模块最值得变成论文主创新”

---

## 15. 第一阶段最值得回答的四个问题

如果现在就开始跑实验，最重要的不是“最终分数最高多少”，而是先回答：

1. 这套 framework 能不能稳定跑完？
2. 它比 naive baseline 有没有稳定收益？
3. 收益主要来自哪个模块？
4. 最常见的失败模式是什么？

这四个问题回答清楚以后，后面无论是继续做 training-free framework paper，还是进入 SFT / DPO / DRPO，都有明确方向。

---

## 16. 当前阶段的预期产出

第一阶段建议形成以下产物：

- benchmark 小规模结果表
- baseline 对比表
- ablation 对比表
- 过程指标统计表
- failure case taxonomy
- judge 评估样例
- 下一阶段创新点候选列表

这些内容本身就足以支撑：

- proposal 修订
- 方法迭代
- paper 里的实验设计章节

---

## 17. 一句话总结

在不训练的前提下，当前框架依然可以通过 **固定 backbone + 对比不同 inference-time pipeline** 的方式在 benchmark 上开展系统实验；第一阶段重点不是追求最终最强结果，而是通过端到端结果、消融、过程指标和失败分析，验证框架是否有效，并从中提炼真正值得进一步训练或强化的方法创新点。
