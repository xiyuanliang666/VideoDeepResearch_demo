# Video-Grounded Deep Research Framework 结构设计

## 1. 设计目标

该框架的主目标是构建一个真正面向 `VideoDR` 类任务的 `video-grounded deep research framework`。核心关注点不是“支持多模态输入”本身，而是提升以下能力：

- 长链交互中的 `goal drift` 控制
- 视频线索的持续保持与重复注入
- 音频与语音线索的有效利用
- 视频证据与网页证据之间的显式绑定
- 多步检索下的稳定推理与可追溯输出

同时，考虑到该方向 benchmark 仍不丰富，框架需要天然兼容以下次级设定：

- `单图`
- `多图`
- `长报告生成`

因此，这个框架应被理解为一个 **以 VideoDR 为中心、但具备跨 benchmark 复用能力的 grounded research runtime**。

## 2. 与 VideoDeepResearch 项目结构的对齐原则

参考 `VideoDeepResearch` 一类项目结构时，建议采用“**工程分层部分对齐，方法主干不照搬**”的原则。

建议对齐的部分：

- 编码器与表征层独立组织
- `eval/` 独立成层
- `train/` 提前预留
- 视频工具函数单独封装
- 音频与 ASR 工具能力单独封装

不建议直接照搬的部分：

- 不应让 `retriever` 成为项目主入口
- 不应把 `prompt` 与 `preprocess` 仅视为评估附属物
- 不应让项目结构看起来像“检索工程外挂推理”

因此，最终目录既可以吸收 `VideoDeepResearch` 的优点，也必须保留你的方法主线：

- `orchestration`
- `anchors`
- `grounding`
- `memory`
- `compilers`

## 3. 优化后的总体架构

当前版本建议拆成六层：

1. `Task & Orchestration Layer`
2. `Tool Library Layer`
3. `Preprocessing & Representation Layer`
4. `Research Core Layer`
5. `Evidence / Output / Evaluation Layer`
6. `Training Interface Layer`

这样拆分的原因是：

- 将帧采样、切片、字幕、音频、ASR、编码、缓存等重工程模块从核心方法中解耦
- 将 prompt、评估、训练接口从主循环中抽离，避免后期扩展时结构失控
- 让框架既适合做方法论文，也适合后续逐步演化成可复用项目骨架

## 4. 各层职责

### 4.1 Task & Orchestration Layer

这一层负责定义任务配置、benchmark profile、运行模式和主数据流编排。

建议职责：

- 定义 `TaskProfile`
- 选择输入模式：`image / image_set / video`
- 配置输出模式：`short answer / long report / traces`
- 配置搜索轮数、检索预算、缓存策略
- 配置是否启用 ASR
- 编排端到端运行顺序

这一层不直接处理模态细节，而是负责“调度谁做什么”。

### 4.2 Tool Library Layer

这一层用于封装通用工具能力，尤其是视频帧处理、切片、字幕、音频、ASR、检验等操作。建议单独放到 `src/tools/` 目录中，按工具类型拆分文件，而不是散落在各模块内部。

建议至少包含以下工具库：

- `video_tools.py`
  - `cut_video`
  - `extract_frames`
  - `sample_frames`
  - `slice_clips`
  - `validate_video`
- `subtitle_tools.py`
  - 字幕抽取
  - 字幕清洗
  - 字幕时间戳对齐
- `audio_tools.py`
  - 音频抽取
  - 音频切片
  - 音频质量检查
- `asr_tools.py`
  - ASR 转写
  - transcript 清洗
  - 语音文本时间戳对齐
- `image_tools.py`
  - resize
  - normalize
  - format convert
- `embedding_tools.py`
  - embedding 生成
  - 缓存加载
  - 相似度检索
- `cache_tools.py`
  - pkl/json/parquet 缓存写入与读取
- `io_tools.py`
  - 路径组织
  - 运行产物落盘
  - 索引文件维护
- `sanity_tools.py`
  - 检查帧数量
  - 检查 clip 是否损坏
  - 检查 embedding 是否缺失
  - 检查 transcript 是否为空

这一层的目标是让视频处理链路成为一个独立、可复用、可替换的工具系统，而不是写死在主 pipeline 中。

### 4.3 Preprocessing & Representation Layer

这一层负责把原始输入转换成统一的可检索、可推理表示，是工程上非常关键的一层。

主要职责：

- 视频切片
- 帧提取与采样
- 音频抽取与 ASR
- 多模态标准化处理
- 字幕抽取与对齐
- transcript 抽取与对齐
- embedding 编码与缓存
- clip/frame/path 映射管理

此外，建议在这一层预留“可插拔模态扩展接口”，用于未来接入：

- `depth`
- `thermal`
- 其他传感器模态

但这些扩展模态当前仅保留接口，不进入第一版主实验与主数据流。

#### 4.3.1 视频处理阶段

建议采用以下主流程：

- 输入视频：`MP4 / MKV`
- 使用 `cut_video`
- 默认 `30 秒` 一个片段，可配置
- 输出到 `data/clips/`

#### 4.3.2 帧、音频与文本阶段

对每个片段执行：

- 使用 `extract_frames`
- 采样率默认 `2fps`，可配置
- 每个片段保留 `8-32` 帧
- 输出到 `data/dense_frames/`
- 从视频中抽取音频
- 调用 ASR 生成 transcript
- 抽取字幕并按时间范围对齐到 clip
- 将 transcript 按时间范围对齐到 clip

#### 4.3.3 表示与缓存阶段

对帧、字幕、ASR 文本、文本 query 进行统一表征：

- 视频帧 resize 到 `224x224`
- 图像归一化
- tokenization
- 通过 `LanguageBind` 或同类模型编码视频与图像
- 通过 `BGEM3` 或同类模型编码字幕、ASR 文本与 query
- 统一缓存到 `data/embeddings/`

建议缓存结构：

- `embeddings/{clip_duration}/{retriever_type}/{video_name}.pkl`
- `embeddings/subtitle/{retriever_type}/{video_name}_subtitle.pkl`
- `embeddings/asr/{retriever_type}/{video_name}_asr.pkl`
- `indexes/clip_paths.pkl`

这层的意义不只是“预处理”，而是为后续检索、anchor 提取、失败分析和训练数据回放提供稳定中间表示。

如果未来扩展到深度图或热红外输入，可额外预留：

- `embeddings/depth/{retriever_type}/{sample_name}_depth.pkl`
- `embeddings/thermal/{retriever_type}/{sample_name}_thermal.pkl`

但当前不建议将其纳入第一阶段实现目标。

### 4.4 Research Core Layer

这一层是方法创新的核心，主要围绕 `anchor-centric loop` 展开。

建议包含以下模块：

1. `Anchor Extractor`
2. `Anchor Selector`
3. `Query Planner`
4. `Local Retriever`
5. `Open-Web Retriever`
6. `Reranker / Filter`
7. `Cross-Modal Binder`
8. `Memory Updater`
9. `Loop Controller`
10. `Final Reasoner`

这一层和预处理层的关系是：

- 预处理层负责把原始视频变成可操作的表示
- Research Core 负责在这些表示上做深度研究
- 其中音频模态在第一版优先通过 `ASR -> 文本化 -> 时间对齐` 的方式进入 research loop，而不是一开始就引入完整音频 embedding 主线

### 4.5 Evidence / Output / Evaluation Layer

这一层统一负责证据存储、结果编译和评估汇总，不应散落在不同脚本里临时实现。

建议职责：

- 维护 `Evidence Store`
- 输出短答案与长报告
- 输出工具调用链、证据链、推理摘要
- 汇总 benchmark 结果
- 记录失败案例
- 统计 ASR 质量与利用率
- 执行 `LLM-as-a-Judge` 评估
- 输出可复现实验报告

### 4.6 Training Interface Layer

虽然训练模块不是第一阶段重点，但架构上必须预留接口，避免后续推翻重来。

建议预留以下接口：

- `SFTDatasetBuilder`
- `PreferencePairBuilder`
- `TrajectoryExporter`
- `RewardSignalBuilder`
- `TrainerAdapter`

预留支持的训练范式：

- `SFT`
- `DPO`
- `DRPO`

这些模块在第一版可以只是接口和数据导出能力，不必立即实现完整训练流程。

## 5. 核心数据对象

参考 `VideoDeepResearch` 一类项目中较完整的配置对象、检索对象与缓存对象设计，当前框架的 schema 建议进一步完善为“**配置对象 + 运行时对象 + 结果对象 + 评估对象**”四类，而不只是保留研究流程中的抽象概念字段。

### 5.1 ModelConfig

用于统一管理编码器、检索器、LLM 与 Judge 模型配置。

建议字段：

- `model_name`
- `model_type`
- `hidden_size`
- `projection_dim`
- `num_hidden_layers`
- `num_attention_heads`
- `image_size`
- `patch_size`
- `num_frames`
- `add_time_attn`
- `lora_r`
- `lora_alpha`
- `lora_dropout`
- `video_decode_backend`
- `text_max_length`
- `embedding_dim`
- `judge_prompt_version`
- `judge_temperature`

### 5.2 RetrievalConfig

用于统一管理本地检索与网页检索配置。

建议字段：

- `retriever_type`
- `clip_duration`
- `frame_sampling_fps`
- `max_frames_per_clip`
- `topk_per_query`
- `score_threshold`
- `similarity_metric`
- `use_subtitle`
- `enable_asr`
- `enable_depth`
- `enable_thermal`

### 5.3 RuntimeState

用于表示一次运行中的核心状态。

建议字段：

- `run_id`
- `task_id`
- `task_profile`
- `active_step`
- `device`
- `cache_refs`
- `errors`
- `warnings`
- `status`

### 5.4 EmbeddingCacheRef

用于统一记录 embedding 缓存位置，而不是只在字段中写字符串路径。

建议字段：

- `cache_id`
- `modality`
- `cache_path`
- `embedding_dim`
- `num_items`
- `created_at`
- `version`

### 5.5 ObservationUnit

表示统一后的基础观察单元。

建议字段：

- `observation_id`
- `source_type`
- `source_path`
- `order_index`
- `timestamp_start`
- `timestamp_end`
- `frame_paths`
- `subtitle_text`
- `transcript_text`
- `ocr_text`
- `audio_path`
- `metadata`
- `quality_flags`
- `embedding_refs`

其中 `source_type` 在当前版本主要包括：

- `image`
- `image_set`
- `video`
- `audio_text`

为未来扩展可预留：

- `depth`
- `thermal`

### 5.6 ClipUnit

建议新增 `ClipUnit` 作为视频预处理与检索阶段的中间对象，而不仅仅依赖 ObservationUnit。

建议字段：

- `clip_id`
- `video_name`
- `clip_path`
- `start_time`
- `end_time`
- `start_time_str`
- `end_time_str`
- `frame_dir`
- `frame_count`
- `fps`
- `audio_segment_path`
- `subtitle_segment`
- `transcript_segment`
- `embedding_ref`
- `quality_flags`

这个对象可以显著简化视频检索、缓存与回溯分析。

### 5.7 Anchor

`Anchor` 是整个方法的核心研究单元。

建议字段：

- `anchor_id`
- `source_clip_ids`
- `source_observation_ids`
- `time_span`
- `anchor_type`
- `entities`
- `actions`
- `scene_summary`
- `ocr_clues`
- `subtitle_clues`
- `speech_clues`
- `temporal_clues`
- `search_queries`
- `confidence`
- `status`
- `priority_score`
- `evidence_ids`
- `open_slots`

不同模态下的含义：

- image：对象/实体/场景 anchor
- image_set：跨图实体或跨图事件 anchor
- video：与一个或多个时间片段绑定的 temporal anchor

### 5.8 QueryUnit

表示一次由 anchor 驱动的检索行为。

建议字段：

- `query_id`
- `anchor_id`
- `query_text`
- `query_type`
- `step_index`
- `motivation`
- `query_source`
- `query_embedding_ref`
- `rewrite_history`

### 5.9 RetrievalCandidate

建议将“相似度检索结果”和“网页检索结果”区分开来，新增统一对象：

- `candidate_id`
- `candidate_type`
- `source_query_id`
- `source_ref`
- `score`
- `rank`
- `metadata`
- `retriever_name`
- `normalized_score`
- `keep_label`
- `keep_reason`

可选类型：

- `clip_candidate`
- `subtitle_candidate`
- `asr_candidate`
- `web_candidate`

这样可以把“本地 embedding 检索”和“开放网页检索”统一到同一候选层。

### 5.10 Evidence

表示已经进入 grounded state 的正式证据。

建议字段：

- `evidence_id`
- `evidence_type`
- `source_ref`
- `content_summary`
- `span`
- `confidence`
- `source_url`
- `source_clip_id`
- `source_timestamp`
- `raw_excerpt`
- `embedding_ref`

可选类型：

- `video`
- `image`
- `ocr`
- `subtitle`
- `asr`
- `web`

为未来扩展可预留：

- `depth`
- `thermal`

### 5.11 Binding

表示 anchor 与 evidence 之间的显式绑定关系。

建议字段：

- `binding_id`
- `anchor_id`
- `evidence_id`
- `relation`
- `matched_entities`
- `matched_actions`
- `matched_temporal_clues`
- `reason`
- `confidence`
- `judgeable_claim`

关系类型：

- `supports`
- `contradicts`
- `related`
- `uncertain`

### 5.12 Claim

表示中间或最终的 grounded proposition。

建议字段：

- `claim_id`
- `statement`
- `supporting_bindings`
- `status`
- `confidence`
- `last_updated_step`
- `conflict_evidence_ids`

状态类型：

- `tentative`
- `supported`
- `contested`

### 5.13 EvidenceStore

建议把 `Evidence Store` 从概念层显式落实为核心数据对象。

建议字段：

- `store_id`
- `anchors`
- `observations`
- `queries`
- `retrieval_candidates`
- `evidences`
- `bindings`
- `claims`
- `open_questions`
- `entity_aliases`
- `step_summaries`

### 5.14 ToolCallRecord

用于统一记录工具调用链。

建议字段：

- `tool_call_id`
- `tool_name`
- `tool_input`
- `tool_output_ref`
- `start_time`
- `end_time`
- `status`
- `error_message`

### 5.15 ReasoningTrace

用于统一记录研究循环中的中间推理痕迹。

建议字段：

- `trace_id`
- `step_index`
- `active_anchor_ids`
- `selected_queries`
- `retrieval_summary`
- `binding_summary`
- `claim_updates`
- `notes`

### 5.16 FinalAnswerBundle

适用于 VQA 或短答案任务的标准输出对象。

建议字段：

- `task_id`
- `question`
- `final_answer`
- `confidence`
- `supporting_video_evidence`
- `supporting_web_evidence`
- `tool_trace_refs`
- `reasoning_trace_refs`

### 5.17 ReportBundle

适用于长报告任务的标准输出对象。

建议字段：

- `task_id`
- `title`
- `executive_summary`
- `sectioned_report`
- `citations`
- `evidence_table`
- `tool_trace_refs`
- `reasoning_trace_refs`

### 5.18 JudgeInputBundle

用于 `LLM-as-a-Judge` 的标准化输入包。考虑到长报告评估经常需要拼接任务、参考材料、工具链和推理链，建议将其显式对象化。

建议字段：

- `judge_input_id`
- `task_input`
- `model_output`
- `reference_answer`
- `reference_materials`
- `reasoning_trace`
- `evidence_chain`
- `tool_trace`
- `rubric`
- `metadata`

### 5.19 JudgeResult

用于记录 Judge 模块输出。

建议字段：

- `judge_result_id`
- `judge_model`
- `judge_prompt_version`
- `task_type`
- `overall_score`
- `dimension_scores`
- `verdict`
- `explanation`
- `failure_tags`
- `raw_judge_output`

## 6. 端到端数据流

基于当前需求，建议把主 pipeline 明确写成下面这条链路：

1. `输入层`
- 输入视频文件 `MP4 / MKV`

2. `视频切分`
- 调用 `cut_video`
- 默认每 `30 秒` 生成一个 clip
- 输出到 `data/clips/`

3. `帧提取`
- 调用 `extract_frames`
- 默认 `2fps`
- 每个 clip 提取 `8-32` 帧
- 输出到 `data/dense_frames/`

4. `音频与语音处理`
- 从视频中抽取音频
- 调用 ASR 生成 transcript
- 将 transcript 按 clip 时间戳对齐
- 输出到 `data/audio/` 与 `data/transcripts/`

5. `多模态预处理`
- resize 到 `224x224`
- normalize
- tokenization
- 字幕清洗与时间对齐
- transcript 清洗与时间对齐

6. `编码与缓存`
- 视频编码器生成视频 embedding
- 图像编码器生成图像 embedding
- 文本编码器生成字幕、ASR 文本或 query embedding
- 统一落盘到 `data/embeddings/`

7. `本地检索层`
- 文本、ASR 文本或视频 query 编码
- L2 normalization
- cosine similarity
- Top-K 或 threshold filtering
- 输出候选 clip 列表

8. `Anchor 提取层`
- 从 clip、frame、subtitle、transcript 中抽取 anchors
- 生成结构化 anchor cards

9. `Research Loop`
- 选取 active anchors
- 生成 web queries
- 调用网页检索
- 做 rerank/filter
- 执行 evidence binding
- 更新 memory 与 claims

10. `输出层`
- 输出短答案
- 或输出长报告
- 同时输出 traces、evidence chain 与引用

11. `Judge 评估层`
- 将输入任务、模型输出、参考材料、推理链、证据链、工具调用链打包为 `JudgeInputBundle`
- 调用 Judge LLM 进行短答案或长报告评估
- 输出 `JudgeResult`

这个版本的好处是：本地视频检索、语音文本检索和开放网页检索被纳入同一数据流，而不是彼此孤立。

## 7. 双检索架构建议

基于 embedding 检索链路，当前框架建议显式支持“双检索”：

- `Local Retrieval`
  - 面向 clip、frame、subtitle、ASR transcript
  - 解决视频内部定位与 recall 问题
- `Open-Web Retrieval`
  - 面向外部网页搜索
  - 解决开放世界补全与证据获取问题

这两者之间的推荐关系是：

- Local Retrieval 先帮助模型在视频内部缩小范围
- Open-Web Retrieval 再围绕高价值 anchors 扩展外部证据

对于音频模态，第一版建议采用：

- `ASR -> 文本化 -> 时间对齐 -> 本地文本检索 -> anchor/evidence`

而不是一开始就引入完整音频编码器主线。这样可以以较低复杂度获得较高收益，同时保留后续扩展到音频 embedding 的接口。

## 8. Prompt 模板管理

Prompt 不建议散落在代码内部，应单独形成模板管理模块。

建议建立：

- `prompts/anchor_extraction/`
- `prompts/query_planning/`
- `prompts/rerank/`
- `prompts/evidence_binding/`
- `prompts/final_reasoning/`
- `prompts/report_generation/`
- `prompts/asr_cleanup/`

每个模板建议拆成：

- `system`
- `user`
- `output_schema`
- `few_shot_examples`

这样做的好处是：

- 方便不同 benchmark 之间切换
- 方便做 prompt ablation
- 方便后续导出训练数据

## 9. 评估与结果汇总

评估不能只停留在“最终分数”，建议建立单独的评估与汇总模块。

建议能力包括：

- benchmark 指标汇总
- baseline 对比表
- ablation 对比表
- failure case 索引
- ASR 质量与召回分析
- `LLM-as-a-Judge` 评分汇总
- 每步检索和绑定的中间日志
- 最终结果导出为 csv/json/md

### 9.1 LLM-as-a-Judge 模块

考虑到：

- VQA 短答案不总能被简单 EM/F1 完全刻画
- 长报告评估通常需要综合任务输入、参考材料、推理链、证据链、工具调用链与最终输出

建议正式加入 `LLM-as-a-Judge` 模块，而不是把它作为临时脚本。

Judge 模块建议支持两类模式：

- `judge_vqa`
  - 面向短答案
  - 重点判断答案正确性、证据一致性、是否真的同时利用了视频与网页线索
- `judge_report`
  - 面向长报告
  - 重点判断事实性、引用充分性、证据绑定质量、结构完整性与推理可信度

Judge 的标准输入应统一使用 `JudgeInputBundle`，其中可包含：

- 输入任务
- 模型输出
- 参考答案或参考材料
- 推理链摘要
- 证据链
- 工具调用链
- 评分 rubric

Judge 的输出应统一写入 `JudgeResult`，并支持后续：

- 聚合统计
- case study
- 失败模式分析
- 人工复核抽样

建议单独建立：

- `src/evaluation/metrics.py`
- `src/evaluation/aggregate.py`
- `src/evaluation/case_study.py`
- `src/evaluation/reporting.py`
- `src/evaluation/judge.py`
- `src/evaluation/judge_schemas.py`
- `eval/run_eval.py`
- `eval/demo.py`
- `eval/concat_result.py`
- `eval/judge_vqa.py`
- `eval/judge_report.py`
- `eval/eval_result/`

这会直接帮助后续写论文实验部分。

## 10. 训练接口预留

第一版不做训练，但架构上必须预留和 research loop 对接的数据导出能力。

建议至少保留这些数据出口：

- `export_sft_examples`
- `export_preference_pairs`
- `export_trajectory_records`
- `export_reward_annotations`

可支持的训练对象包括：

- anchor extraction
- query planning
- reranking
- evidence binding
- final reasoning
- ASR 后处理与 speech clue extraction

潜在训练范式：

- `SFT`
- `DPO`
- `DRPO`

这样后续如果框架有效，可以自然从工程系统过渡到训练系统。

## 11. TaskProfile 与 Benchmark 兼容

框架应通过 `TaskProfile` 管理 benchmark 适配，而不是在代码中写大量 if-else。

建议字段：

- `task_name`
- `input_mode`
- `output_mode`
- `clip_duration`
- `frame_sampling_fps`
- `max_frames_per_clip`
- `enable_asr`
- `asr_backend`
- `enable_depth`
- `enable_thermal`
- `enable_llm_judge`
- `judge_model`
- `judge_prompt_version`
- `retriever_type`
- `max_steps`
- `retrieval_budget`
- `trace_level`
- `required_outputs`

示例 profile：

- `vdr_bench_profile`
- `mmsearch_plus_profile`
- `videodr_profile`
- `browsecomp_vl_profile`
- `mmdeepresearch_report_profile`

## 12. 第一版最小实现建议

第一版依然建议控制范围，但比之前更工程化：

- 支持 `video -> clip -> frame/audio -> embedding cache + asr transcript -> local retrieval -> anchor -> web retrieval -> evidence binding -> short answer`
- 固定步数循环
- 单一 search API
- 单一 MLLM
- 单一 embedding 配置
- 音频优先采用 ASR 文本化，不强制引入独立音频 embedding 主线
- 先不引入训练，只保留接口

也就是说，第一版最重要的是把下面四条链同时打通：

- 视频处理链
- 音频/ASR 文本化链
- 检索缓存链
- grounded research 主循环

## 13. 建议目录结构

下面这个版本是“**部分对齐 VideoDeepResearch 风格，但主干围绕你的方法创新**”的推荐结构：

```text
VideoGroundedResearch/
├── README.md
├── requirements.txt
├── proposal.md
├── proposalcn.md
├── framework.md
├── frameworkcn.md
├── configs/
│   ├── task_profiles/
│   ├── model_profiles/
│   └── retriever_profiles/
├── prompts/
│   ├── anchor_extraction/
│   ├── query_planning/
│   ├── rerank/
│   ├── evidence_binding/
│   ├── final_reasoning/
│   ├── report_generation/
│   └── asr_cleanup/
├── src/
│   ├── orchestration/
│   ├── tools/
│   │   ├── video_tools.py
│   │   ├── subtitle_tools.py
│   │   ├── audio_tools.py
│   │   ├── asr_tools.py
│   │   ├── image_tools.py
│   │   ├── embedding_tools.py
│   │   ├── cache_tools.py
│   │   ├── io_tools.py
│   │   └── sanity_tools.py
│   ├── preprocessing/
│   ├── encoders/
│   │   ├── languagebind/
│   │   ├── text_encoder/
│   │   ├── asr_adapter/
│   │   ├── depth_encoder/
│   │   └── thermal_encoder/
│   ├── retrieval/
│   │   ├── local_retriever.py
│   │   ├── web_retriever.py
│   │   └── reranker.py
│   ├── anchors/
│   ├── grounding/
│   ├── memory/
│   ├── reasoning/
│   ├── compilers/
│   ├── evaluation/
│   │   ├── judge.py
│   │   └── judge_schemas.py
│   ├── training/
│   └── schemas/
├── eval/
│   ├── run_eval.py
│   ├── demo.py
│   ├── concat_result.py
│   ├── judge_vqa.py
│   ├── judge_report.py
│   └── eval_result/
├── train/
│   ├── sft/
│   ├── dpo/
│   ├── drpo/
│   └── examples/
├── data/
│   ├── raw_videos/
│   ├── clips/
│   ├── dense_frames/
│   ├── audio/
│   ├── subtitles/
│   ├── transcripts/
│   ├── embeddings/
│   └── indexes/
└── outputs/
    ├── traces/
    ├── answers/
    ├── reports/
    └── eval/
```

这个结构与 `VideoDeepResearch` 对齐的部分包括：

- 编码器独立目录
- `eval/` 独立目录
- `train/` 独立目录
- 工具模块抽离
- 音频/ASR 模块抽离

这个结构保留你自己方法主线的部分包括：

- `orchestration`
- `anchors`
- `grounding`
- `memory`
- `compilers`
- `prompts`

同时，也为未来模态扩展预留了接口：

- `depth_encoder/`
- `thermal_encoder/`
- `enable_depth`
- `enable_thermal`

这些接口当前仅作为未来扩展位，不代表第一阶段必须实现 RGB-D 或热红外处理链。

## 14. 框架总结

优化后的 `frameworkcn.md` 不再只是一个偏研究概念图，而是一个兼顾方法创新与工程落地的体系：

- `Tool Library Layer` 负责把帧处理、切片、字幕、音频、ASR、校验等操作标准化
- `Preprocessing & Representation Layer` 负责将视频、音频与文本线索转成可缓存、可检索、可训练的中间表示
- `Research Core Layer` 负责围绕 anchors 进行多步深度研究
- `Evidence / Output / Evaluation Layer` 负责输出、评估与论文分析
- `Training Interface Layer` 为后续 `SFT / DPO / DRPO` 预留接口

其中最关键的主线仍然不变：

- `video-grounded`
- `speech-aware`
- `anchor-centric`
- `evidence-bound`

对 image、multi-image 与 report setting 的兼容性应服务于这条主线，而不是反过来稀释 VideoDR 方法创新。

同样，`depth` 与 `thermal` 等模态当前只保留扩展接口，未来可在特定领域场景下接入，但不应影响当前围绕 VideoDR 展开的主线设计。

此外，当前框架的评估层不应只依赖传统自动指标，而应正式纳入 `LLM-as-a-Judge`，以支持：

- VQA 短答案的语义正确性判断
- 长报告的多维度质量评估
- 基于任务输入、参考材料、推理链、证据链与工具链的综合判断
