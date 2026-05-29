# AAAI 2027 提交计划

**Deadline**: 2026-07-28 | **剩余**: 64 天 (9 周)
**目标**: Benchmark + Method 双贡献, 100+ 样本

---

## 团队角色

| 代号 | 成员 | 职责 |
|------|------|------|
| **P0** | 主力 (你) | 框架冻结 + 全部实验 + 论文 Method/Experiments 章节 |
| **P1** | 主力队员 | Benchmark 扩展总负责: 视频审核 + semi-auto pipeline + 评估脚本 + 论文 Benchmark 章节 |
| **P2** | 轻量队员 | 视频海选下载 + contact sheet 生成 + 标注复核 + 图表绘制 + 参考文献整理 |

---

## 资源盘点

| 资源 | 数量 | 缺口 |
|------|------|------|
| 视频 (candidate) | 5 | — |
| 视频 (abandoned, 可救回) | 4 | — |
| 已有 blueprint | 8 | — |
| 已有 pilot 样本 | 13 | — |
| **目标样本** | **100** | **需 ~16-20 个新视频** |

按 4 样本/视频计算，需要 25 个视频。现有 9 个，需新找 16-20 个。若部分视频可做 5 样本，视频需求降至 20 个。

---

## 论文叙事

**Title pattern**: *VideoDeepResearch: A Structured Pipeline for Video-Grounded Open-Web Question Answering*

**Contribution 三件套**:
1. **Benchmark**: 100-sample video QA benchmark with structured annotations (state pool, evidence graph, hard negatives)
2. **Method**: 12-stage pipeline — PySceneDetect → CLIP clustering → VLM state extraction → retrieval → evidence binding → reasoning
3. **Findings**: Video-only ≠ web-only ≠ full; structured state decomposition improves answer quality over end-to-end VLM

---

## Week 1 (5.25 — 6.1): 框架冻结 + 视频海选

| 任务 | 负责人 | 产出 |
|------|--------|------|
| CLIP temporal semantic clustering 实现 + 测试 | P0 | 设计中，见 docs/hybrid_framework_v1.md |
| `.env` 修复 (REASONING_MODEL 切可用模型) | P0 | 全链路 12 阶段跑通 |
| PySceneDetect threshold sweep (确定最优参数) | P0 | 待执行 |
| 13 pilot annotations Review → Finalized | P0 + P2 | 进行中，S1Q3/S4Q1 已重写，其余待审 |
| 视频海选: YouTube/Bilibili 搜索 50+ 候选 URL | P2 | 候选视频列表 spreadsheet |
| 新视频初步筛选 (时长 1-6min, 信息密度, 多场景) | P1 | 初筛 30 个视频 |
| 下载初筛通过的视频 | P2 | 30 个 mp4 文件 |
| Semi-auto pipeline 在 4 个 abandoned 视频上试跑 | P1 | 4 个新 blueprint → 12-16 samples |

**里程碑**: 13 pilot finalized + 30 个新视频待审 + framework 冻结

---

## Week 2 (6.2 — 6.8): Dry Run + 批量标注

| 任务 | 负责人 | 产出 |
|------|--------|------|
| `text_only / web_only / video_only / full_vdr` 在 13 pilot 上跑完 | P0 | ablation 结果表 |
| 评估脚本 (schema validator + metrics calculator) | P1 | `evaluate.py` |
| 30 个新视频 contact sheet 生成 | P2 | 每个视频 20 帧 contact sheet |
| 新视频质量审核 (信息密度、场景多样性、反作弊) | P1 + P2 | 通过 20 个，进入 blueprint |
| Semi-auto blueprint generation 批量跑 20 个通过视频 | P1 | 20 个 blueprint JSON |
| Blueprint feasibility review (semantic reviewer) | P1 + P2 | 每个 blueprint 审过 |
| 已审核 blueprint → sample generation | P1 | 60-80 个 sample candidates |
| 建立 Overleaf project (AAAI LaTeX 模板) | P2 | 论文 repo 就绪 |

**里程碑**: Dry run 结果表 + 60-80 raw samples 生成

---

## Week 3 (6.9 — 6.15): Sample Review + 实验平台

| 任务 | 负责人 | 产出 |
|------|--------|------|
| 60-80 samples 人工复核 (anti-cheat, answer uniqueness) | P1 + P2 | 50+ verified samples |
| Hard negative 绑定路径验证 | P1 | 每个 sample 1-5 hard negatives |
| 合并 13 pilot + 50 new → manifest v1 | P1 | `pilot_manifest_v1.json` (65 samples) |
| 剩余 16 个视频继续标注 | P1 + P2 | 目标 35 samples 在 pipeline 中 |
| 评估脚本完成 + 试跑 (在 13 pilot 上) | P1 | evaluation metrics 输出正常 |
| Framework 集成测试 (全 ablation 模式) 在 5 个新 sample 上试跑 | P0 | 确认 framework 可批量跑 |
| **论文 Section 3 (Method) 框架图 + outline** | P0 | Method section structure + pipeline figure draft |

**里程碑**: ~65 finalized samples + 评估脚本 ready + Method outline

---

## Week 4 (6.16 — 6.22): 批量实验 — 主力模型对比

| 任务 | 负责人 | 产出 |
|------|--------|------|
| **主实验**: 4 模态 ablation × 3 底座模型 on 65 samples | P0 | 12 组实验数据 |
| 模型池: GPT-4o-mini / Claude-Haiku / Qwen-VL | P0 | leaderboard table |
| Web retrieval 引擎对比 (Serper vs Tavily) | P0 | 2 组补充实验 |
| 继续 sample 标注达到 100 | P1 + P2 | manifest v1 达到 100 |
| 100 samples 全部 human review 通过 | P1 + P2 | `pilot_manifest_v1_final.json` |
| **论文 Section 4 (Benchmark) 初稿** | P1 | Benchmark stats, construction pipeline, annotation protocol |
| 收集所有 related work paper 的 bibtex | P2 | `.bib` 文件 |

**里程碑**: 100 samples finalized + 12 组主实验跑完 + Benchmark section draft

---

## Week 5 (6.23 — 6.29): Ablation 实验 + 论文初稿

| 任务 | 负责人 | 产出 |
|------|--------|------|
| **Framework ablation**: 去掉 PySceneDetect / 去掉 CLIP / 去掉 VLM observer / 去掉 web retrieval / 去掉 hypothesis generator | P0 | ablation table (5-6 rows) |
| 100 samples 全量跑最优配置 | P0 | 最终 leaderboard |
| Error analysis: 按 module 归因错误类型 | P0 | error taxonomy table |
| **论文 Section 1 (Intro) 初稿** | P0 | 1.5 page |
| **论文 Section 2 (Related Work) 初稿** | P1 | 2 pages (已有 survey 可复用) |
| **论文 Section 5 (Results + Analysis) 初稿** | P0 | tables + figures + analysis |
| Pipeline architecture figure (full diagram) | P2 | SVG/PDF figure |
| Benchmark construction pipeline figure | P2 | flowchart figure |

**里程碑**: 全部实验完成 + 论文 60% 初稿

---

## Week 6 (6.30 — 7.6): 论文完整初稿

| 任务 | 负责人 | 产出 |
|------|--------|------|
| **论文 Section 3 (Method) 完整初稿** | P0 | 3 pages |
| **论文 Section 6 (Conclusion + Limitations)** | P0 | 0.5 page |
| **Abstract** | P0 + P1 | 第一版 |
| 全论文整合 + 第一轮内部 review | P0 + P1 | 完整 8-page 草稿 |
| Supplementary material (附录实验、prompt templates) | P1 | appendix PDF |
| 补充实验 (reviewer 视角预判弱点,补数据) | P0 | 2-3 组额外实验 |

**里程碑**: 完整论文初稿 (v0.1)

---

## Week 7 (7.7 — 7.13): 论文打磨 + 内部审稿

| 任务 | 负责人 | 产出 |
|------|--------|------|
| 论文 v0.1 → 内部 review (找 2-3 人给 feedback) | P0 + P1 | review comments |
| 根据 feedback 修改 | P0 | v0.2 |
| Figures 精修 (consistent color scheme, font sizes) | P2 | final figures |
| 表格格式化 (AAAI style) | P2 | final tables |
| 英文润色 (Grammarly + GPT 辅助) | P0 | v0.3 |
| Related Work 补充最新 arXiv (2026.06-07) | P1 | updated citations |
| Demo video / website 准备 (optional, 视时间) | P2 | 截图 + 视频 |

**里程碑**: 论文 v0.3 接近提交质量

---

## Week 8 (7.14 — 7.20): 终稿 + 查漏补缺

| 任务 | 负责人 | 产出 |
|------|--------|------|
| 全文最终润色 | P0 | v1.0 |
| AAAI 格式检查 (页数、margin、字体、引用格式) | P2 | checklist |
| 补充实验 final run (确保可复现) | P0 | 冻结的实验数据 |
| 代码仓库整理 + README + 可复现脚本 | P1 | public repo ready |
| 检查所有引用是否完整 | P2 | verified .bib |
| 匿名化处理 (如果需要 double-blind) | P1 | anonymized |

**里程碑**: v1.0 ready

---

## Week 9 (7.21 — 7.28): 提交窗口

| 任务 | 负责人 | 产出 |
|------|--------|------|
| 最终检查 + 提交 | P0 | submitted |
| Supplementary material 打包 | P1 | zip |
| **7.25 前提交** (避免最后一天系统崩溃) | — | — |

---

## 视频瓶颈

```
100 samples / 4 samples per video = 25 videos needed
已有: 5 candidate + 4 abandoned = 9
缺口: 16-20 new videos

Week 1-2: P2 海选 50+ → P1 审核通过 30 → 最终入 16-20
Week 3-4: 持续补视频 + 标注
```

### P2 视频搜索指南

**YouTube 关键词**:
- "documentary clip 5min", "sports highlight analysis", "news report explainer"
- "how it's made segment", "travel vlog highlight", "historical footage analysis"
- "science experiment explained", "cooking technique comparison"

**Bilibili 关键词**:
- 纪录片片段、科普解说、新闻调查、体育集锦

**硬约束**:
- 时长 1-6 分钟
- 英文优先 (或英文硬字幕)
- 包含多个场景/事件变化
- 有可验证的事实性内容 (人物、地点、事件、数字)
- 画面有足够视觉信息 (不只是 talking head)

**避开**:
- 纯音乐 MV、纯游戏实况、纯聊天 vlog
- 画面完全静态的 podcast/访谈
- AI 生成视频、屏幕录制教程

---

## 风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| 找不到 16 个合格视频 | 中 | sample/video 密度提高到 5，减少视频需求到 20 |
| P0/P1 实验跑不完 | 中 | Week 4-5 并行跑, P0 主力模型 + P1 ablation |
| API 费用超预算 | 低 | gpt-4o-mini 为主力, Claude Haiku 便宜 |
| Annotation 质量不够 | 中 | Week 3 设 hard gate, 低于标准直接 discard |
| AAAI 拒稿 | 高 | 同时准备 NeurIPS 2027 workshop (10月) 或 CVPR 2028 |

---

## 关键文件路径

- Framework: `VideoDeepResearch_demo/src/core/pipeline.py`
- 当前预处理: `VideoDeepResearch_demo/src/core/preprocessing.py`
- CV 工具: `VideoDeepResearch_demo/src/tools/local_cv.py`
- Pilot manifest: `vdr_investigation/benchmark/pilot_v1/pilot_manifest_v0.json`
- Submission schema: `vdr_investigation/benchmark/pilot_v1/submission_schema_v1.json`
- Semi-auto pipeline: `vdr_investigation/benchmark/semi-auto_annotation_pipeline/`
- Pilot annotations: `vdr_investigation/benchmark/pilot_annotations/`
- 论文调研: `vdr_investigation/other/`
