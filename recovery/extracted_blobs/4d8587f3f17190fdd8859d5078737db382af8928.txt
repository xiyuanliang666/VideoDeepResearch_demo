# 帧筛选评估

本文档合并两部分内容：

1. **自拟 benchmark**：状态池子状态、关键帧 GT、时间阈值 + SSIM 最优匹配 + 冗余惩罚等形式化定义（基于前文讨论总结）。
2. **论文 / 官方实现参考**：`/mnt/sda/TStar/LVHaystackBench/val_tstar_results.py` 中与 LV-Haystack / T* 相关工作一致的 **Temporal PRF** 与 **SSIM Precision/Recall/F1** 的计算方式及特点。

---

## 一、自拟方案：状态池与关键帧子状态评分

### 1.1 设计前提（协议层）

- **状态池**：每个 case 包含若干 **子状态**；每个子状态由 **子问题** 与 **参考答案 / 参考证据** 构成（含「关键帧提取」类状态）。
- **GT 关键帧**：人工标注为 **对当前子问题最有信息价值** 的帧（可为多张），记为集合 \(G = \{g_1,\ldots,g_m\}\)（每帧有帧号或时间 \(t(\cdot)\)，并可解码为图像 \(I(\cdot)\)）。
- **模型输出**：在预算 \(K\) 下返回候选关键帧集合 \(P = \{p_1,\ldots,p_k\}\)，\(k \le K\)。
- **路由与执行分离**：「模型子问题 → 语义匹配到子状态」与「该状态下的帧筛选质量」**独立计分**，避免路由误差污染帧指标。

### 1.2 单 GT 帧与单预测集合的匹配分（时间门控 × SSIM）

对第 \(i\) 个 GT 帧 \(g_i\) 与任意预测帧 \(p_j\)，定义：

\[
\text{score}(g_i, p_j) =
\mathbb{1}\big(|t(g_i) - t(p_j)| \le \tau\big) \cdot \mathrm{SSIM}\big(I(g_i), I(p_j)\big)
\]

- \(\tau\)：**时间容忍**（与数据一致：秒或帧，全库统一）。
- \(\mathbb{1}(\cdot)\)：超出时间窗则该项为 0，避免「时间很近但高速剪辑导致内容不同」仅靠时间给高分；内容靠 SSIM 区分。

**多预测取最优**（每个 GT 至少可被 \(P\) 中最佳一帧解释）：

\[
\text{match}(g_i) = \max_{p_j \in P} \ \text{score}(g_i, p_j)
\]

### 1.3 子状态覆盖分（连续）

\[
\text{CoverageScore} = \frac{1}{m} \sum_{i=1}^{m} \text{match}(g_i)
\]

- \(m\) 为该子状态下 GT 关键帧数量；若 \(m=0\) 则该状态跳过或按协议处理。

### 1.4 冗余惩罚

对预测集合 \(P\) 定义冗余率 \(\text{Redundancy}(P)\)（示例）：时间差小于 \(\delta\) 的帧对占比、或嵌入相似度过高的比例等，取值 \([0,1]\)。

\[
\text{RedPenalty}(P) = 1 - \lambda \cdot \text{Redundancy}(P),\quad \lambda \in [0,1]
\]

### 1.5 关键帧子状态最终分

\[
\text{KeyframeStateScore} = \text{CoverageScore} \cdot \text{RedPenalty}(P)
\]

### 1.6 可选硬命中率（离散、易解释）

给定 SSIM 阈值 \(\gamma\)，称 \(g_i\) **被命中**当且仅当存在 \(p_j\) 满足：

\[
|t(g_i)-t(p_j)|\le \tau \quad \land \quad \mathrm{SSIM}(I(g_i),I(p_j)) \ge \gamma
\]

\[
\text{HitRate} = \frac{\#\{ g_i \text{ 被命中} \}}{m}
\]

可同时报告 **CoverageScore**（连续）与 **HitRate**（离散）。

### 1.7 与「纯语义 judge」的关系

在本设定下，**信息价值由人工 GT 帧承担**；自动指标负责「是否在时空上对齐到这些证据锚点」。可选再增加 VLM/LLM-as-judge 作为**附加分析列**，不必与主分混用，以免与标注语义重复裁判。

---

## 二、`val_tstar_results.py` 中的指标（论文配套实现）

脚本路径：`/mnt/sda/TStar/LVHaystackBench/val_tstar_results.py`。

脚本说明：从 JSON 读取 `video_path`、`keyframe_timestamps`（预测，**秒**）、`gt_frame_index`（GT **帧号**），计算 **Temporal PRF** 与 **SSIM P/R/F1**。

### 2.1 时间对齐：GT 与预测统一到秒

- GT：`gt_frame_index` 除以视频真实 FPS 得 **秒**。
- 预测：`keyframe_timestamps` 视为 **秒**。
- **Temporal PRF** 在 **秒空间**上计算；CLI 默认 `--threshold 5` 表示 **5 秒** 容忍（需与论文/协议一致，勿与「5 帧」混淆）。

### 2.2 Temporal Precision / Recall / F1（最近邻 + 阈值命中）

对每个视频样本，设 GT 时刻集合 \(G\)（秒）、预测时刻集合 \(P\)（秒），阈值 \(\tau\)（代码中为 `threshold`）。

对每个 GT \(g\)：\(d_{g\to P}(g) = \min_{p\in P}|g-p|\)。  
对每个预测 \(p\)：\(d_{p\to G}(p) = \min_{g\in G}|p-g|\)。

- **Recall**：\(|\{g \in G : d_{g\to P}(g)\le \tau\}| / |G|\)
- **Precision（soft）**：\(|\{p \in P : d_{p\to G}(p)\le \tau\}| / |P|\)
- **F1**：\(2PR/(P+R)\)（单视频），再在数据集上对多视频 **分别对 P、R、F1 取算术平均**（macro over videos）。

**特点**：

- **非一对一**：多个预测可同时「命中」同一 GT 段，Precision 可被重复预测抬高。
- **无 SSIM**：只看时间是否落入 \(\tau\) 内，不判画面是否一致。

### 2.3 SSIM Precision / Recall / F1（像素层、非 query-aware）

流程概要：

1. 按帧从视频解码 GT 帧与预测帧（预测时刻先经 `timestamp × fps_video` 等转为帧索引，再经 `--fps` 缩放；默认 `fps=1.0` 时需与数据管线对齐）。
2. 计算所有 GT 图与所有预测图的 **pairwise SSIM** 矩阵 \(S \in \mathbb{R}^{|G|\times|P|}\)。
3. **SSIM Precision**：对每个预测列取 \(\max\) 行再平均：\(\frac{1}{|P|}\sum_j \max_i S_{ij}\)。
4. **SSIM Recall**：对每个 GT 行取 \(\max\) 列再平均：\(\frac{1}{|G|}\sum_i \max_j S_{ij}\)。
5. 数据集上对各视频的上述量平均后，**SSIM F1** 由平均 P、平均 R 调和得到。

**特点**：

- 衡量 **像素结构相似**，不直接绑定「当前子问题」；与「人工高信息帧」语义不必一致。
- 与 Temporal PRF **并列**：时间对但画面差时，Temporal 可高、SSIM 可低。

### 2.4 ANND（实现但未接入主输出）

脚本中 `calculate_annd` 计算平均最近邻时间距离（预测→GT、GT→预测），**当前 `calculate_metrics` 未写入最终 metrics**，若论文或附录使用需自行对齐。

---

## 三、两类方法对比总表

| 维度 | 自拟：状态内关键帧分（\(\tau\) + SSIM + max + 冗余） | `val_tstar_results.py`：Temporal PRF + SSIM P/R |
|------|------------------------------------------------------|--------------------------------------------------|
| **评估对象** | 子状态「关键帧提取」；GT 为 **人工高信息帧** | 全视频级关键帧检索；GT 为 **帧索引/时刻标注** |
| **时间** | 硬门控：先 \(|t_g-t_p|\le\tau\) 再算 SSIM | 软命中：仅 \(|t|\le\tau\)，无门控后内容 |
| **画面** | SSIM 进入 **match 乘积**，且多帧 **argmax** | SSIM 单独成 **P/R**，与时间 PRF **分开聚合** |
| **GT 多帧** | 对每个 \(g_i\) 分别 \(\max_j\) 再对 \(i\) 平均 | Temporal 按集合命中；SSIM 矩阵对行列取 max 再平均 |
| **冗余** | 显式 **Redundancy** 惩罚 | 无冗余项；soft Precision 易被同段多帧抬高 |
| **匹配唯一性** | 每 GT 取最佳预测（GT-centric best match） | Temporal：多预测可同时算命中；非严格一对一 |
| **子问题 / 状态** | 显式状态池、可独立路由分 | 无状态机；整段 JSON 一条视频一评 |
| **单位与陷阱** | \(\tau,\delta\) 与 \(t\) 单位需在协议中统一 | PRF 在 **秒**；`threshold` 默认 5 **秒**；注释中「frame」易误解 |

---

## 四、选用建议（写进 benchmark 说明即可）

1. **自拟指标**适合：有 **子问题分解 + 人工证据帧** 的 **video deep research 过程评估**；语义价值在标注，自动分负责「是否对齐到证据锚点」并惩罚冗余。
2. **LV-Haystack 式 Temporal PRF + SSIM** 适合：与 **T* / LV-Haystack 论文** 对齐的 **全局长视频检索** 横向对比；注意阈值单位与是否改为一对一匹配。
3. 两者可 **并列报告**：前者为 **过程/子状态** 分，后者为 **与社区可比的长视频检索** 分，避免混为一谈。

---

## 五、参考代码路径

- 论文配套实现：`/mnt/sda/TStar/LVHaystackBench/val_tstar_results.py`
  - Temporal PRF：`calculate_prf`
  - SSIM：`pairwise_ssim`、`calculate_ssim_scores`
  - ANND：`calculate_annd`（未并入 `calculate_metrics` 返回字典）

---

## 六、本文档位置

- `rsagent-v3/frame_selection_metrics_comparison.md`
