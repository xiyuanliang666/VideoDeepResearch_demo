# Hybrid VideoDeepResearch Framework v1

日期：2026-05-25

## 目标

本文整理一个面向 Video Deep Research benchmark 的混合框架方案。核心目标不是单纯提升关键帧抽取质量，而是把视频、网页和候选假设组织成可审计的结构化证据链：

```text
temporal state -> supporting frame set -> visual predicates -> retrieval anchors
               -> query branch -> web evidence -> candidate support/refute -> answer
```

benchmark 侧真正评估的是：

- 是否定位到必要视频状态和证据时间段。
- 是否检索并使用了必要网页证据。
- 是否覆盖 required support targets。
- 是否显式完成候选消歧和 hard negative 避免。
- 最终答案是否由视频证据和网页证据共同支撑。

因此，本框架输出的内部信息量应大于最终评估所需信息量，再通过 projection layer 映射到 benchmark submission schema。

## 总体 Workflow

```text
preprocess
-> state proposal
-> evidence-oriented frame selection
-> coverage control
-> anchor extraction
-> hypothesis generation
-> retrieval planning
-> iterative retrieval
-> evidence binding
-> candidate support/refute
-> reasoning
-> judge
-> benchmark schema projection
```

职责划分：

- workflow 负责状态一致性、时序约束、证据绑定和可审计性。
- agentic 模块负责局部探索、query refinement、hypothesis branching 和 retrieval budget 调度。
- benchmark projection 层负责把内部 rich graph 映射到严格 submission schema，而不是把内部字段原样提交。

## 1. Preprocess

preprocess 由本地 worker 完成，目标是在较低成本下建立全局 observation timeline。

本地并行执行：

- `ffmpeg` dense low-cost sampling。
- PySceneDetect 场景变化检测，替换原 histogram threshold。
- PaddleOCR 提取画面文字和 scorebug / title card / caption。
- `open_clip` MobileCLIP 生成 frame / crop embedding。
- YOLO11n 做轻量目标检测。
- 可选 ASR：默认休眠，仅当题目元数据或 query planner 判断音频必要时启用 Qwen3-ASR 1.7B。

在 PySceneDetect 与 VLM observation 之间加入 temporal semantic consolidation layer：

- PySceneDetect 只产生 editing-cut proposal，不直接作为最终 observation boundary。
- 对相邻 proposal segments 的 1FPS frames 提取 MobileCLIP embedding，并结合 OCR / object / visual tag overlap。
- 仅在时间邻近窗口内计算相似度，避免跨远距离片段误合并。
- 对高相似相邻片段做 agglomerative merge，得到 semantic segment。
- 对 semantic segment 抽取 representative keyframes，再调用 VLM 生成 event-oriented observation。
- trace 中保留 proposal segment ids、merge scores、merge signals、source frame indices 与 representative frame indices，方便审计。

这一步的目标是把 pipeline 从 editing-cut driven 转向 semantic-state driven，避免同一个 semantic state 被多个 shot fragment 重复描述，并稳定后续 state proposal、evidence binding 与 retrieval query generation。

低成本帧还会输入 qwen3-vl-flash，生成简短、事件导向的 frame observations。observation 应包含：

- timestamp / scene boundary。
- caption-like visual observation。
- OCR text。
- object tags。
- scene tags。
- action or event hints。
- clip / frame path。

注意：state constructor 不应只依赖少量代表帧。必须同时输入全局低成本 observation timeline，避免早期 keyframe 选择错误被固化。

## 2. State Proposal

state constructor 使用轻量 reasoning model，例如 GPT-5，根据以下输入生成候选 temporal states：

- question text。
- dense sampling 后的 frame observations。
- OCR / object tags / scene tags。
- timestamp / scene boundary。
- 少量代表帧。
- 可选 ASR transcript。

state 的目的不是复述视频，而是推断“为了回答这个问题，视频中可能有哪些关键时序状态需要被验证”。

内部 state 建议结构：

```json
{
  "state_id": "s1",
  "state_question": "Does the first clip show an initial defensive block followed by a loose second ball?",
  "expected_predicates": [
    "defenders positioned in low box",
    "initial contact or block on ball",
    "ball remains loose in dangerous area"
  ],
  "temporal_hint": {"start_sec": 20.0, "end_sec": 45.0},
  "required_support": "video",
  "candidate_links": ["c_first_clip"]
}
```

这些 state 后续应尽量映射到 benchmark 的 `state_pool_gt.states[*]` 语义：每个 state 是回答问题所需的一个必要视频子判断。

## 3. Evidence-Oriented Frame Selection

frame selection 不再是静态抽帧，而是围绕 state 寻找 supporting frame set。

每个 state 的 selector 输入：

- state expected predicates。
- dense observations。
- scene boundaries。
- OCR / object detections。
- CLIP similarity。
- temporal constraints。
- representative frames。

输出应是短时间 evidence units，而不只是单帧：

```json
{
  "evidence_id": "vid_s1_e1",
  "state_id": "s1",
  "start_sec": 28.5,
  "end_sec": 36.0,
  "frame_paths": [".../frame_0012.jpg", ".../frame_0015.jpg"],
  "visual_predicates": [
    "white-shirted defenders are inside the box",
    "red attacker receives loose ball",
    "shot angle remains open"
  ],
  "confidence": 0.78
}
```

内部可以保留 frame set 和 crop set；提交给 benchmark 时应压成 `start_sec` / `end_sec` 区间，因为评估会统一按 canonical interval 或秒级区间匹配。

## 4. Coverage Controller

coverage controller 先做规则检查，再做 lightweight consistency check。

规则检查：

- state 是否覆盖题目所需时间跨度。
- 每个 state 是否存在 supporting frames。
- 是否缺少关键 predicate。
- 是否只覆盖了正候选，忽略了 distractor / hard negative。
- 是否有状态之间的顺序冲突。

轻量一致性检查：

- 默认使用 claude-haiku-4-5-20251001。
- 如果本地允许，可使用 Qwen2.5-7B-Instruct。

coverage controller 不直接生成最终答案，它只决定是否需要重新选择帧、扩展 state、补充 query 或触发反证搜索。

## 5. Structured Evidence Graph

内部 structured evidence graph 是后续 retrieval 和 reasoning 的主输入：

```json
{
  "states": [
    {
      "state_id": "s1",
      "supporting_frame_set": ["vid_s1_e1", "vid_s1_e2"],
      "visual_predicates": [
        "initial block",
        "second ball not cleared",
        "attacker turns inside box"
      ],
      "temporal_interval": [28.5, 42.0]
    }
  ],
  "edges": [
    {
      "from": "vid_s1_e1",
      "to": "s1",
      "relation": "supports_visual_predicate"
    }
  ]
}
```

后续 retrieval 和 reasoning 不直接基于原始帧自由推理，而是基于这些结构化 states 和 predicates 生成 query、做 candidate binding 和证据验证。

## 6. Anchor Extraction

anchor extractor 从关键帧和 supporting frame set 中提取 retrieval-ready visual anchors。anchor 不是简单物体列表，而是可用于检索、消歧和网页对齐的压缩语义线索。

anchor 类型：

- 服装和颜色：队服颜色、号码、姓名、角色服装。
- 环境特点：场馆、地图、建筑、画作、室内布置、标识牌。
- 空间结构：禁区内、门口、桌边、展厅、低位防守区域。
- 人物关系：visitor vs owner、attacker vs defender、speaker vs listener。
- 动作机制：interception, second ball, turn, low shot, transition attack。
- 时间变化：first clip vs second clip, before/after, repeated attempt。
- OCR anchor：scorebug、match title、字幕、地名、姓名。

建议结构：

```json
{
  "anchor_id": "a_s1_01",
  "state_id": "s1",
  "source_evidence_id": "vid_s1_e1",
  "anchor_text": "Belgium in red versus USA in white, low-box defensive sequence, De Bruyne #7 later shown",
  "anchor_type": "visual_event_identity",
  "retrieval_use": ["candidate_identity", "event_record", "mechanism_condition"]
}
```

## 7. Hypothesis Generation

candidate hypothesis generator 生成多个可能解释路径 query branches。每个 branch 必须绑定 evidence ledger，避免 goal drift。

ledger 至少记录：

- query branch 来自哪个 state。
- 使用了哪个 anchor。
- 要验证哪个 hypothesis。
- 检索结果支持或反驳哪个 candidate。
- 该证据对应哪个 support target。

示例：

```json
{
  "branch_id": "b1",
  "hypothesis": "The first clip is Belgium vs USA 2014, De Bruyne 93', low-box second-ball goal.",
  "source_state_ids": ["s1", "s3", "s4"],
  "source_anchor_ids": ["a_s1_01", "a_s4_02"],
  "candidate_id": "c_first_clip_debruyne_2014",
  "target_support": ["event_record", "mechanism_condition", "candidate_difference"],
  "status": "open"
}
```

## 8. Retrieval Planning and Query Composition

query planner 读取：

- question。
- state pool。
- keyframe observations。
- OCR。
- visual anchors。
- candidate hypotheses。
- optional ASR。

然后生成 retrieval intents。每个 retrieval intent 必须声明它要服务的 support target：

- `candidate_identity`
- `candidate_attribute`
- `candidate_difference`
- `event_record`
- `temporal_condition`
- `mechanism_condition`
- `rule_condition`
- `candidate_elimination`
- `final_answer`

query composer 再把以下信息组合成具体 query：

- 时间约束。
- 视觉锚点。
- 候选假设。
- 问题目标。
- support target。

不建议直接把原问题整体塞进 search。当前 raw_vdr_demo 的失败模式之一就是 query 过于接近原问题，导致网页结果不服务 state/candidate 绑定。

## 9. Iterative Retrieval

retrieval manager 决定：

- 是否继续搜索。
- 是否扩展 query。
- 是否 rerank 来源。
- 是否转入 image-conditioned retrieval。
- 是否触发反证搜索。
- 是否停止并进入 evidence binding。

每轮检索必须绑定到具体 state 或 hypothesis。没有绑定的自由探索只能作为 debug trace，不应进入最终 used evidence。

文本检索输出：

```json
{
  "query_id": "q_b1_02",
  "branch_id": "b1",
  "state_ids": ["s1", "s3"],
  "candidate_id": "c_first_clip_debruyne_2014",
  "support_target": "event_record",
  "query": "Belgium USA 2014 De Bruyne 93 goal loose ball low shot round of 16",
  "results": []
}
```

## 10. Image-Conditioned Retrieval

部分问题需要视频帧和网页图片/截图比较，例如画作、角色、建筑、球衣、地图、物品或场景匹配。

至少实现路径：

1. 从 supporting frame set 中生成 keyframe crops。
2. 收集网页图片、搜索结果图片或网页截图。
3. 使用 CLIP / MobileCLIP 做初筛。
4. 使用 VLM 判断 visual alignment。
5. 将 alignment verdict 绑定回 state、candidate 和 support target。

内部结构：

```json
{
  "alignment_id": "img_align_01",
  "video_crop_id": "crop_s3_face_owner",
  "web_image_id": "web_img_portrait_tanguy",
  "state_id": "s3",
  "candidate_id": "c_pere_tanguy",
  "support_target": "candidate_identity",
  "verdict": "supports",
  "rationale": "The coat, hat, beard, and portrait composition align with the video owner character."
}
```

提交层不直接新增 image alignment 字段，而是折叠进：

- `used_web_evidence.evidence_snippet`
- `used_web_evidence.support_target`
- `evidence_links`
- `dependency_graph`
- `reasoning_trace_summary`

## 11. Evidence Binding

evidence binding 是整个系统最关键的部分。它负责把网页证据重新映射回视频中的 temporal state、关键帧、候选解释。

每条 used web evidence 必须回答：

- 支持或反驳哪个 hypothesis。
- 对应哪个 candidate。
- 补足哪个 video state。
- 属于哪个 support target。
- 是否可能是 hard negative。

网页证据不能替代视频 state。网页可以验证身份、时间、规则、赛事记录或机制描述，但视频中的动作、状态变化和视觉 predicates 仍需由 video evidence 支撑。

## 12. Candidate Support / Refute

candidate support/refute 显式记录每条证据如何影响候选。

建议内部状态：

```json
{
  "candidate_id": "c_first_clip_debruyne_2014",
  "status": "supported",
  "supporting_evidence": ["vid_s1_e1", "web_event_01", "web_mech_01"],
  "refuting_evidence": [],
  "satisfied_targets": ["event_record", "mechanism_condition", "candidate_difference"],
  "open_targets": []
}
```

对于 hard negative，必须记录为什么看起来相关但不应被用于最终答案。例如同一球员、同一赛事或同一队伍，但机制、时间、进球者、阶段或候选差异不满足题目条件。

## 13. Reasoning and Judge

reasoning 只在结构化 grounding 完成后进行。最终 answer generator 输入：

- selected candidate。
- covered states。
- supporting video evidence。
- used web evidence。
- support/refute ledger。
- unresolved gaps。

judge 做两层检查：

1. 规则检查：
   - final answer 是否包含题目要求字段。
   - necessary states 是否有 video evidence。
   - required support targets 是否有 used web evidence。
   - 是否使用 hard negative 作为关键依据。
2. 轻量模型一致性检查：
   - evidence 是否真的支持 answer。
   - candidate elimination 是否合理。
   - video predicates 和网页 claim 是否冲突。

## 14. Benchmark Schema Projection

内部 graph 不能原样作为 submission 顶层字段提交。pilot v1 schema 使用 `additionalProperties: false`，sample 层只允许指定字段。

内部字段到 benchmark submission 的映射：

| Internal | Submission field |
|---|---|
| selected answer | `final_answer.answer_text` |
| selected candidate | `predicted_candidate_id` |
| state supporting frame set | `used_video_evidence[*]` |
| visual predicates | `used_video_evidence[*].description` |
| state id | `used_video_evidence[*].linked_state_ids` |
| text search results | `retrieved_web_results[*]` |
| actually used web snippets | `used_web_evidence[*]` |
| support target | `used_web_evidence[*].support_target` |
| candidate binding | `used_web_evidence[*].linked_candidate_id` |
| state binding | `used_web_evidence[*].linked_state_ids` |
| support/refute links | `evidence_links[*]` |
| reasoning dependency | `dependency_graph[*]` |
| compact audit summary | `reasoning_trace_summary` |

`used_video_evidence` projection example:

```json
{
  "evidence_id": "vid_s1_e1",
  "start_sec": 28.5,
  "end_sec": 36.0,
  "description": "Supports s1: defenders are positioned in a low box; an initial block/contact occurs; the second ball remains available to the red attacker.",
  "linked_state_ids": ["s1"],
  "confidence": 0.78
}
```

`used_web_evidence` projection example:

```json
{
  "evidence_id": "web_b1_event_01",
  "url": "https://example.org/match-report",
  "title": "Belgium v USA match report",
  "evidence_snippet": "Confirms Belgium v USA, De Bruyne goal at 93', and the relevant match context.",
  "support_target": "event_record",
  "linked_candidate_id": "c_first_clip_debruyne_2014",
  "linked_state_ids": ["s1", "s4"],
  "confidence": 0.86
}
```

`evidence_links` projection example:

```json
{
  "source_type": "video_evidence",
  "source_id": "vid_s1_e1",
  "target_type": "state",
  "target_id": "s1",
  "relation": "supports_visual_predicate"
}
```

## 15. Implementation Notes

- Keep internal graph rich, but keep submission schema strict.
- Prefer short evidence intervals over isolated frames for benchmark compatibility.
- Every query should carry `state_id`, `candidate_id` if known, and `support_target`.
- Every used evidence item should declare whether it supports or refutes a candidate.
- Distinguish `retrieved_web_results` from `used_web_evidence`; retrieved results are search output, used evidence is what the answer actually depends on.
- Do not let representative frames replace the dense observation timeline in state proposal.
- Trigger ASR only when question metadata or planner indicates audio relevance.
- Use image-conditioned retrieval only when it can produce a binding verdict that maps back to state/candidate/support target.

## 16. Expected Impact

This design directly addresses observed raw_vdr_demo / ablation weaknesses:

- coarse 30-second clip evidence becomes state-linked short evidence units.
- raw question search becomes state/candidate/support-target retrieval.
- web results become used evidence with explicit roles.
- distractors and hard negatives are handled through support/refute ledger.
- final reasoning runs on grounded states and predicates rather than free-form frame captions.

The design should improve alignment with:

- Frame Evidence F1 / Recall.
- Web Evidence Recall@K.
- Web Evidence Usage Coverage.
- Hard Negative Misuse Rate.
- Candidate Elimination Accuracy.
- Cross-modal Fusion Consistency.
- Strict Grounded Accuracy.
- TraceConsistency / DependencyValidity when dependency graph is available.

## 17. Minimal v1 Milestones

1. Add local preprocess artifacts: dense observations, OCR, scene boundaries, object tags, embeddings.
2. Implement state proposal from question plus global observation timeline.
3. Implement state-aware frame selector and coverage controller.
4. Implement anchor extraction and retrieval intent generation with support targets.
5. Implement evidence ledger for query branches.
6. Implement evidence binding to state/candidate/support target.
7. Implement projection from internal graph to current benchmark submission schema.
8. Add judge checks for missing states, missing required support targets, and hard negative misuse.
