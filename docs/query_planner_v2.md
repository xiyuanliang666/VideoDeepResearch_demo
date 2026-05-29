# Query Planner v2: Observation-Driven Iterative Retrieval Design

日期：2026-05-28

## 结论

当前 baseline 的核心问题不是检索次数不够，而是 query planner 缺少"检索结果驱动下一跳 query"的闭环。在多跳推理场景中，上一轮搜索得到的事实线索应进入下一轮搜索，而不是一次性生成所有 query 后把结果交给最终 reasoner。

当前 baseline 已有 Phase B（structured retrieval）结构，但还不是 observation-driven retrieval：`run_retrieval_for_intents()` 只执行每个 intent 的第一条 query，仅在全 fallback 时才 retry 第二条 query。且当前的 evidence binding 是一次性 LLM 调用，缺少逐 slot 验证和候选排除机制。

因此 v2 的目标是：在保持 Phase B 结构的基础上，增加 slot-conditioned、constraint-aware、observation-driven 的 query planning，使检索过程具备闭环验证和基于中间结果的 query 精炼能力。

## 设计原则

### 1. 视频状态与网页事实分离

从视频分析（Phase A）得到的状态（temporal states）描述"视频里发生了什么"——时间区间、视觉事件、实体线索、场景锚点。这些是视频侧的必要子判断。

网页检索（Phase B）负责补全视频无法回答的部分——实体身份、外部知识、事件记录、数值属性。这些不应被回写成视频状态。

关键区分：

- **视频状态**：可从视频帧/音频直接验证的观察（如"三个角色进入了同一扇门"、"画面呈现几何碎片化风格"）
- **检索槽位（retrieval slot）**：需要通过外部网页补全的信息缺口（如"这种几何碎片化风格对应哪个艺术流派"、"受此风格影响的 2018 年动画电影是什么"）

两者之间的依赖关系由 `dependency_graph` 显式记录。

### 2. 运行时结构不改变对外接口

planner 在运行时可以维护内部 slot、memory、candidate ledger 等结构，但这些最终只投影到通用的输出字段：

- 检索结果（每条搜索返回的网页结果，含 URL、标题、摘要、排名）
- 采纳证据（被 verifier 判定为相关且满足约束的网页证据）
- 证据链接（网页证据与视频状态/候选/最终答案的关系边）
- 依赖图（state → slot → query → result → evidence 的可审计链路）
- 诊断/过程记录（planner 的完整运行 trace，用于分析和消融实验）

### 3. 通用多跳检索框架，非特定评测适配

本设计解决的是通用的 video deep research 多跳检索问题：从视频锚点出发，经过多轮 web 检索逐步获取缺失信息，最终汇聚为有证据支撑的答案。任何具有类似"视频理解 + 多跳网页检索"结构的任务都可以使用这套机制。

具体评测集只是验证框架能力的测试场，不应成为设计的约束条件。

## 当前 Baseline 中应复用的结构

当前 workflow 和 agentic mode 在开启 `enable_state_proposal` 时都走类似 Phase B：

```text
generate_hypotheses()
-> plan_retrieval_intents()
-> run_retrieval_for_intents()
-> bind_evidence_to_states()
-> evaluate_candidates()
```

已有可复用对象：

- `RetrievalIntent`: 已有 `support_target`、`target_type`、`target_id`、`linked_candidate_id`、`query_templates`。
- `QueryUnit`: Pydantic extra=allow，可加入 `slot_id`、`support_target`、`hard_constraints`、`rewrite_history` 等 runtime 字段。
- `RetrievalCandidate`: 可在 `metadata` 写入 `slot_id`、`observation_label`、`constraint_matches`、`is_negative_candidate`。
- `Evidence`: 可在 `metadata` 写入 `support_target`、`bound_to_id`、`linked_candidate_id`、`is_fallback`。
- `dependency_graph`: 已能记录 intent/query/candidate/binding 链路，可扩展 relation 名称。

因此 v2 不需要推翻当前 baseline，而是扩展 Phase B。

## 核心抽象

### Retrieval Slot

`RetrievalSlot` 是运行时结构，表示当前要通过网页补全的一个信息槽。它不是视频状态，而是"现在需要从网上找到什么"的结构化描述。

```json
{
  "slot_id": "slot_002",
  "slot_name": "target_entity_identity",
  "support_target": "candidate_identity",
  "expected_answer_type": "entity name with temporal and categorical constraints",
  "source_state_ids": ["s3"],
  "source_anchor_ids": ["anchor_0002"],
  "depends_on": ["slot_000", "slot_001"],
  "hard_constraints": {
    "year": "2018",
    "medium": "animated feature film",
    "style_relation": "specific-art-style-influenced"
  },
  "soft_clues": ["geometric fragmented visual style"],
  "verification_rule": "Candidate must satisfy year, medium, and style relation constraints before downstream lookup.",
  "status": "pending"
}
```

Slot 合法来源（运行时动态生成）：

- 问题文本中的显式约束
- Phase A 产生的视频状态和视觉锚点
- 候选假设
- 已完成的上一跳 slot 输出

Slot 禁止来源：

- 外部标注的正确答案
- 预设的网页证据 URL
- 预先标注的错误候选标签

### Runtime Planner Memory

第一版只保留最小 memory：

```json
{
  "task_constraints": {},
  "slot_outputs": {},
  "candidate_ledger": {},
  "failed_candidates": [],
  "failed_queries": [],
  "retrieval_history": [],
  "budget": {
    "max_slots": 6,
    "max_rounds_per_slot": 2,
    "topk_web": 5
  }
}
```

含义：

- `task_constraints`: 从题面和视频 anchor 解析出的硬约束（年份、媒介类型、数值范围等）。
- `slot_outputs`: 已验证 slot 的输出，供下游 slot 使用。
- `candidate_ledger`: 候选实体、候选数值及其满足/违反的约束，每条有状态（pending/verified/refuted）。
- `failed_candidates`: 已被明确排除的候选及排除原因。
- `failed_queries`: 低质量、重复或产生错误候选的 query。
- `retrieval_history`: 每轮 query、结果、verdict、next action 的完整记录。

## Planner 流程

### Step 1. Slot Planner

输入：`question`、predicted `TemporalState` 列表、anchors、hypotheses

输出：有依赖顺序的 `RetrievalSlot[]`

规则：

- 视频状态只作为 slot 的 source，不在此阶段被改写成网页事实。
- 每个 slot 必须有 `support_target`，且从通用分类中选择：`final_answer`、`candidate_identity`、`candidate_attribute`、`candidate_difference`、`event_record`、`temporal_condition`、`mechanism_condition`、`rule_condition`、`candidate_elimination`。
- 每个 slot 必须有 `expected_answer_type` 和 `verification_rule`。
- 如果 slot 依赖前一跳结果，应显式声明 `depends_on`。

### Step 2. Query Candidate Generation

输入：当前 slot、runtime memory、source video anchors、previous verified slot outputs

输出：3-5 个 query candidates

每个 query candidate 建议记录：

```json
{
  "query_text": "2018 crime animated film [art-style] influence",
  "intent": "discover_target_entity",
  "included_constraints": ["2018", "crime", "animated film", "[art-style]"],
  "target_answer_type": "entity name",
  "expected_support_target": "candidate_identity",
  "avoid_terms": [],
  "fallback_strategy": "If no candidate satisfies all constraints, relax soft constraints and search with remaining hard constraints."
}
```

硬规则：

- 不允许直接复制完整问题文本作为 query。
- 不允许在 query 生成逻辑中硬编码针对特定问题或领域的 heuristic。
- 每个 query 必须服务一个明确的 slot。
- 对 final-answer slot，query 必须包含已验证的候选实体，而不是从题面直接猜测。
- 对候选发现 slot，query 应包含足够硬约束以控制召回精度。

### Step 3. Query Utility Ranking

第一版用规则排序，不引入 reward model。

评分维度（通用）：

- 是否覆盖当前 slot 的关键硬约束
- 是否包含上一跳 verified output
- 是否避免了已知的 failed candidate / failed query
- 是否具有候选空间收缩能力（避免过宽召回）
- 目标类型是否明确
- 是否过长、过泛或接近完整题干

输出最高分 query。若最高分缺少关键硬约束，先 rewrite 再搜索。

### Step 4. Retrieval Execution

扩展当前 `run_retrieval_for_intents()`：

当前行为：

```text
intent -> first query -> search
if all fallback -> optional second query retry
```

v2 行为：

```text
slot -> ranked query -> search -> observation classifier
if support and verification passes -> advance slot
if support but missing fields -> complete evidence
if contradict -> update failed candidate and refine query
if low relevance / type mismatch / empty -> rewrite query
```

第一阶段不需要打开全文网页（full page reader）；先基于 title/snippet/url 做 lightweight classifier。后续再加入 reader。

### Step 5. Observation Classifier

每个 retrieval result 在进入 reasoner 之前先被打标签。

标签（通用分类）：

- `support`: 支持当前 slot，并满足核心约束。
- `insufficient`: 相关但缺字段或信息不完整，需要补证据。
- `contradict`: 包含与硬约束明确冲突的信息。
- `type_mismatch`: 结果类型与 slot 期望类型不匹配（如需要数值却返回描述文本）。
- `low_relevance`: 语义相关度不足以用于当前 slot。
- `empty_candidate`: 未形成可用候选。

第一版实现方式：

- 规则优先：年份、数值、URL domain、标题关键词、已知实体匹配、数值类型检查。
- 小模型只处理规则无法判断的语义关系（如风格关联、间接引用）。
- fallback 结果永远不进入最终采纳证据，只进入 diagnostics。

### Step 6. Control Router

根据 observation label 决定下一步：


| Label                             | Action                                   |
| --------------------------------- | ---------------------------------------- |
| `support` + verification complete | `ADVANCE_SLOT`                           |
| `support` + missing fields        | `COMPLETE_EVIDENCE`                      |
| `insufficient`                    | `COMPLETE_EVIDENCE`                      |
| `contradict`                      | `ELIMINATE_CANDIDATE -> REFINE_QUERY`    |
| `type_mismatch`                   | `REWRITE_QUERY`                          |
| `low_relevance`                   | `REWRITE_QUERY`                          |
| `empty_candidate`                 | `REWRITE_QUERY_OR_RELAX_SOFT_CONSTRAINT` |


注意：`contradict` 不是普通 rewrite。必须先把冲突写入 `candidate_ledger` 和 `failed_candidates`，再生成带冲突记忆的新 query。

### Step 7. Verification Gate

每个 slot 只有满足其 `verification_rule` 才能标记为 verified，其输出才能被下游 slot 引用。

Verification rule 的通用模板（以实体发现 slot 为例）：

```text
candidate = <observed entity candidate>
must satisfy:
- temporal constraint (year/date range)
- categorical constraint (medium/type/domain)
- relational constraint (style/influence/association with upstream verified output)
- source quality constraint (authoritative source preferred)
```

以数值查询 slot 为例：

```text
candidate = <verified entity from previous slot>
source = <required source type>
field = <required field>
value type = <expected type>
reject:
- wrong entity pages (same source type but wrong subject)
- wrong field values from same entity (correct source, wrong field/section)
- incomplete or partial values when complete value is requested
```

## 通用多跳检索示例

以下是框架在典型多跳"视频锚点 → 外部知识链 → 精确数值"场景下的期望运行轨迹。**这不是针对特定样本的硬编码路径，而是说明 slot-conditioned retrieval 的通用机制**。具体约束值由 Phase A + slot planner 在运行时从实际输入中动态提取。

假设场景：视频包含某种视觉风格的特征画面，问题要求经过"视觉锚点→风格归属→受影响实体→该实体的某属性值"的多跳链找到答案。

### 视频状态（Phase A 产出）

视频分析产生若干 temporal state，每个描述视频中可直接验证的观察：

- 某时间段内画面呈现特定的几何碎片化特征
- 该特征可作为后续外部检索的视觉锚点

### 运行时 Slot Plan 参考轨迹

```json
[
  {
    "slot_id": "slot_000",
    "slot_name": "visual_style_anchor",
    "support_target": "candidate_attribute",
    "expected_answer_type": "artist/style clue derived from visual anchor",
    "source_state_ids": ["<state with visual style observation>"],
    "hard_constraints": {
      "visual_anchor": "<visual feature description from video>"
    }
  },
  {
    "slot_id": "slot_001",
    "slot_name": "art_movement_or_school",
    "support_target": "candidate_identity",
    "depends_on": ["slot_000"],
    "expected_answer_type": "art movement or school name",
    "hard_constraints": {
      "artist_or_founder_clue": "<entities identified from slot_000>"
    }
  },
  {
    "slot_id": "slot_002",
    "slot_name": "target_entity_identity",
    "support_target": "candidate_identity",
    "depends_on": ["slot_001"],
    "expected_answer_type": "entity with temporal, medium, and style constraints",
    "hard_constraints": {
      "year": "<temporal constraint from question>",
      "medium": "<medium constraint from question>",
      "style_relation": "<art movement from slot_001>-influenced"
    }
  },
  {
    "slot_id": "slot_003",
    "slot_name": "target_attribute_value",
    "support_target": "final_answer",
    "depends_on": ["slot_002"],
    "expected_answer_type": "numeric or string attribute value",
    "hard_constraints": {
      "entity": "<verified entity from slot_002>",
      "source": "<required source type from question>",
      "field": "<required field from question>"
    }
  }
]
```

### 典型检索链

1. **视觉锚点 slot**：query 包含视频名称/场景特征 + 视觉风格关键词，找到可能的艺术家/风格线索。
2. **风格归属 slot**：query 携带上游艺术家线索 + "co-founded movement" 等关系词，确定具体流派名称。
3. **受影响的实体 slot**：query 组合年份、媒介、风格约束，从搜索结果中识别候选实体，由 verifier 确认是否满足全部约束。
4. **属性值 slot**：query 使用已验证实体 + 目标数据源 + 字段名，精确检索目标数值。
  - 在此 slot 中，同一数据源的其他实体页面（wrong entity）和同一实体的其他字段值（wrong field）会被 observation classifier 标记为 `contradict` 或 `type_mismatch`，进入 `failed_candidates` 并触发 query refinement。

### 负候选处理

负候选（看起来相关但实际错误的搜索结果）的处理完全由 observation classifier 和 verification gate 在运行时完成，不依赖外部标签：

- **类型 A（同源错实体）**：数据源类型正确但主题实体错误 → classifier 标记为 `contradict`，理由为 `wrong_entity_for_constraints`。
- **类型 B（同实体错字段）**：实体正确但返回的数值字段不对（如返回周末票房而非全球票房）→ classifier 标记为 `insufficient` 或 `contradict`，取决于 slot 的字段约束。
- 所有负候选写入 `candidate_ledger` 并标记排除原因，供后续 trace 分析。

## 输出与证据追踪

Planner v2 的内部字段不直接作为对外输出，而是投影到通用证据追踪结构：

### 检索结果

每条搜索结果记录：

- `query`：实际搜索词
- `hop_id`：query id 或 slot id
- `support_target`：该结果所服务的目标类型
- `linked_state_ids`：关联的视频状态
- `linked_candidate_id`：关联的候选实体（如有）
- `metadata.slot_id`、`metadata.observation_label`、`metadata.constraint_matches`

### 采纳证据

只有通过 verifier 的 evidence 才标记为采纳：

- `support_target`、`linked_state_ids`、`linked_candidate_id`
- `confidence`
- `metadata.slot_id`、`metadata.verification_status`

### 证据链接

建议的 relation 类型（通用）：

- `supports`：证据支持某个 state/candidate
- `refutes`：证据否定某个 candidate
- `verifies_slot`：证据使某个 slot 达到 verified 状态
- `eliminates_candidate`：证据触发了候选排除
- `supports_final_answer`：证据为最终答案提供直接支撑

### 依赖图

记录完整的多跳链路：

```text
state -> slot -> query -> retrieval_result -> evidence -> candidate/final_answer
```

这样后续分析不需要理解 planner 内部对象，也能从 trace 中完整观察多跳推理过程。

## 实现路线

### Phase 1: Slot-Aware One-Pass Planner

目标：最小改动，先让 query 明确服务 slot。

改动：

- 新增 `RetrievalSlot` runtime class 或 dict。
- `plan_retrieval_intents()` 内部先生成 slots，再把 slots 投影为现有 `RetrievalIntent`。
- `RetrievalIntent` 增加 runtime 字段：`slot_id`、`expected_answer_type`、`hard_constraints`、`verification_rule`。
- `QueryUnit` 写入同样字段，方便 trace 和投影。

效果：解决"query 服务整个问题、容易复制题干"的问题。让 `support_target` 更稳定。让后续 classifier 有结构化输入。

### Phase 2: Observation Classifier + Retry Policy

目标：把当前 fallback-only retry 改成 observation-driven retry。

改动：

- 在 `run_retrieval_for_intents()` 中加入 `_classify_observation()`。
- 对每个 intent 至少支持 `max_rounds_per_slot=2`。
- 当 top results 为 `low_relevance/type_mismatch/empty_candidate` 时，用下一个 query candidate 或 rewrite query。
- 当出现 `contradict` 时，把 candidate 写入 `failed_candidates`，再 refine。

效果：错误候选不会直接污染后续 reasoning。query refinement 有明确触发条件。

### Phase 3: Verification Gate + Candidate Ledger

目标：只有 slot 通过验证才进入下一跳。

改动：

- 新增 `candidate_ledger` 到 raw trace。
- slot 输出必须有 `verified / partially_verified / rejected` 状态。
- `bind_evidence_to_states()` 优先绑定 verified evidence。
- `evaluate_candidates()` 使用 candidate ledger，而不是只数 binding 数量。

效果：防止错误实体/错误字段被当作最终答案。可以显式解释为什么某些候选被排除。

### Phase 4: Query Utility Ranking

目标：减少低质量 query 和重复 query。

改动：

- 给每个 query candidate 打规则分。
- 记录 `query_utility_score` 和扣分原因。
- diagnostics 增加通用指标：`valid_query_rate`、`degenerate_query_count`、`stale_query_count`、`constraint_coverage_rate`、`redundant_query_count`。

效果：planner 质量可观测，后续能做消融实验。

## Prompt 约束

Query planner prompt 应明确以下通用原则：

```text
Video states describe what the video shows — do not convert web facts into states.
Plan retrieval slots, not final reasoning prose.
Each slot must have one valid support_target.
Each query must target exactly one slot.
Do not copy the full question as a query.
For downstream slots, inherit verified outputs from previous slots.
For final-answer slots, include the verified candidate entity and requested field.
If a query retrieves a candidate that violates hard constraints, mark it as failed and refine.
Return JSON only.
```

## 过程指标

建议先记录（不强制第一版参与评分），用于框架自身的可观测性：

- `valid_query_rate`: query 是否绑定 slot 且包含关键约束。
- `slot_verification_rate`: slot 是否被 verified 后才进入下一跳。
- `constraint_violation_recovery_rate`: 遇到冲突候选后是否能 refine。
- `support_target_coverage`: 是否覆盖了任务所需的各类 support target。
- `stale_query_rate`: 是否生成与当前 slot 无关的 query。
- `redundant_query_rate`: 是否重复搜索已失败的路径。
- `candidate_elimination_count`: 是否显式排除了冲突候选。

## 与 v1 的主要差异


| v1 设计                            | v2 调整                      |
| -------------------------------- | -------------------------- |
| 完整 Runtime Retrieval Memory 一步到位 | 先做最小 memory，逐步扩展           |
| slot/state/memory 边界模糊           | 明确区分：视频状态 vs 检索槽位 vs 运行时记忆 |
| 偏论文综述和完整 agent 设计                | 偏当前 baseline 可实施路线         |
| 强调 general agentic loop          | 绑定当前 Phase B 结构做增量改进       |
| 没有明确投影机制                         | 明确内部结构到通用证据追踪的投影           |
| 没有分阶段工程计划                        | Phase 1-4 可逐步实现和消融         |


## 推荐近期实现顺序

1. 先实现 Phase 1：slot-aware planner，仍复用现有 retrieval manager。
2. 再实现 Phase 2：observation classifier 和非 fallback 情况下的 retry/refine。
3. 用多跳链式问题（视频锚点 → 实体发现 → 属性查找）作为 probe，检查是否生成完整的 slot 依赖链，以及 verified entity 是否正确传递到下游 slot。
4. 检查冲突候选是否进入 `retrieved_web_results` 但不进入 verified evidence。
5. 最后把 `retrieval_history` 和 `candidate_ledger` 加入 diagnostics/raw_trace，用于后续分析和消融实验。





开始改代码前，最需要先确认的是“这次到底改到哪一层闭环”：图片里的问题已经不只是 query planner，而是从 state proposal、anchor extraction、retrieval、observation classifier、control router、candidate ledger 到 evidence binder 的整条链路，所以要先划定本轮 scope，是只做 Phase 2 的最小闭环，还是连 state/anchor/evidence binder 一起动；其次要确认 runtime memory 的统一数据结构，包括 slot_outputs / failed_candidates / candidate_ledger / missing_evidence / retrieval_history 放在哪里、怎么跨模块传递、怎么写入 trace；还要确认哪些判断由 VLM 做、哪些由规则做、哪些由 LLM 做，避免又变成一次 LLM 判断到底；最后要确认评估和回归方式，用哪几个样本做 smoke、哪些旧 ablation 不重跑、成功标准是什么，比如是否能触发 rewrite、是否避免 fallback-only retry、是否保留更多 web candidates、是否能在 raw trace 里看到闭环决策。

结合你这周的目标，下一步不应该全面铺开改 state/anchor/evidence binder，而应先做 **Phase 2 最小闭环**，但要把接口预留给后面三块。原因是你当前最紧急的是跑出“合格 submission + 可审计 full-vdr trace”，而不是一次性把整条 pipeline 重构完；Phase 2 最小闭环可以优先解决最影响 full-vdr 的问题：检索结果不触发 rewrite、错误候选不回退、fallback-only retry、没有 `candidate_ledger / retrieval_history`。state/anchor/evidence binder 目前确实也有问题，但建议先只做必要的小修，比如 anchor 不泄漏强答案线索、binder 不丢弃太多结果、trace 保留中间状态；不要重写这些模块。等 Phase 2 能稳定产出合格 submission 后，再根据 full-vdr trace 反推哪些样本是 state extraction 问题、哪些是 anchor/query 问题、哪些是 evidence binding 问题，这样回修样本和规则会更有依据。

**Runtime memory 放哪里**  

建议新增一个轻量 `RetrievalMemory` runtime dict/class，放在 Phase B 内部，由 `run_retrieval_for_intents()` 或新建的 `retrieval_controller.py` 持有并返回。不要塞进 benchmark schema，也不要分散写在 `intent/query/candidate` 各处。核心字段保留`verified_outputscandidate_ledgerfailed_candidatesfailed_constraintsmissing_evidenceretrieval_history`。最终 raw trace 顶层写一份 `retrieval_memory`，每个 query/candidate metadata 里只写必要引用，如 `memory_step_idcandidate_idobservation_label`。

**怎么跨模块传递**  

建议 Phase 2 先让 `retrieval_controller` 串起 `query_planner -> retrieval_manager -> observation_classifier -> memory_update -> refinement`，而不是让每个模块互相调用`query_planner` 只根据当前 memory 生成下一轮 query candidates`retrieval_manager` 只负责搜索`observation_classifier` 只打抽象标签`memory_updater` 只更新 ledger`control_router` 决定 advance/rewrite/refine/eliminate。这样以后 state/anchor/binder 可以逐步接入，不会把逻辑揉成一团。

**怎么写入 trace**  

raw trace 顶层建议加 `retrieval_memory` 和 `retrieval_history`。每一步 history 至少记录`step_idactionquerysource_state_idscandidate_idsobservation_labelcontrol_decisionreasonmemory_delta`。submission 里不需要全部展开，但 `metadata.diagnostics` 可以记录计数：rewrite 次数、contradiction 次数、failed candidate 数、verified output 数、fallback result 数。

**哪些判断用规则**  

规则优先处理结构化、低风险判断：空结果、fallback 结果、重复 query、query 是否复制题干、URL/domain 是否重复、候选是否已经失败、年份/数字/明确 ID 是否冲突、检索结果和 query 的 token overlap 是否极低。这些判断稳定、便宜、可解释，应该先做，不要每次都问 LLM。

**哪些判断用 VLM**  

VLM 只负责视频侧或视觉锚点相关判断：视频片段能否回答 state、anchor 是否真的来自画面、VLM 生成的 visual answer 是否与 CV/OCR 标签一致、关键帧是否覆盖了 state。不要让 VLM 判断网页检索结果，也不要让它决定 query rewrite。它的职责是保证 retrieval 的起点不是幻觉。

**哪些判断用 LLM**  

LLM 用在语义判断和改写上，但要被 memory 限制：判断网页 snippet 是否支持/反驳某个候选、partial evidence 缺什么、多个候选如何竞争、下一轮 query 如何基于 failed constraints 和 missing evidence 改写。LLM 不应“一次判断最终答案”，而是输出结构化 verdict 或 query candidates。

**smoke 样本怎么选**  

建议用 3-4 个代表样本，不要一开始全量跑。至少包括`RzQZc-v9crA/q1` 测 query refinement 和 wrong candidate 回退；一个 audio/subtitle 样本测 video evidence 不足时是否暴露；一个 image-search 样本测 visual anchor 是否进入 retrieval；再选一个相对简单样本测 submission schema 稳定性。这样能覆盖主要失败模式。

**哪些旧 ablation 不重跑**  

`text_only / web_only / video_only` 已经跑过，暂时不要重跑，除非 schema converter 或 sample manifest 改了会影响 submission 格式。当前重点是 full-vdr workflow 和 agentic 的合格 submission。资源先投在 `full_vdr` 单样本 probe，再小批量跑 3-4 个 smoke。

**成功标准是什么**  

Phase 2 最小闭环的成功标准不是答对，而是 trace 合格：能看到 observation label，不再只有 fallback-only retry；遇到低相关/冲突能触发 rewrite 或 eliminate`candidate_ledger` 有候选状态`retrieval_history` 有 control decision；submission 通过 validator；raw trace 能解释为什么某个 query 被改写、某个候选被保留或排除。最终答案可以错，但过程必须可诊断。