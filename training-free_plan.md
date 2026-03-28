# Training-Free Experiment Plan

## 1. Goal

This plan answers one core question:

How can we study and improve a `VideoDR-first` method **without training**, while keeping the framework small enough for rapid innovation and broad enough to test across five benchmarks?

The current stage is not about learning new parameters. It is about testing whether the **inference-time method design** is effective.

The central target is still:

- `VideoDR`

The other benchmarks are used as:

- mechanism warm-up
- generalization support
- upper-bound evaluation support

## 2. Current Experimental Philosophy

The current work should be treated as a:

- `training-free`
- `method-first`
- `trace-heavy`
- `failure-driven`

study.

That means:

- backbone models stay fixed
- no `SFT / DPO / DRPO` yet
- we compare different pipeline choices under similar budgets
- we use the resulting traces to discover the most promising innovation points

The main question is not:

- what did the model learn?

The main question is:

- what part of the method core actually improves grounded video-web reasoning?

## 3. Method Scope For The Training-Free Stage

The experiment plan must align with the new compact framework.

The method core includes only:

- `preprocessing`
- `observation`
- `anchors`
- `retrieval`
- `grounding`
- `reasoning`
- `evaluation`

The current stage intentionally does **not** prioritize:

- multi-agent orchestration
- heavy memory systems
- training
- long complex planners
- large prompt systems

This matters because the first paper-worthy insight will likely come from:

- keyframe selection
- low-risk observation construction
- visual anchor building
- video-web evidence binding

not from scaling the engineering surface area.

## 4. What Training-Free Means Here

Even without training, the system can already test:

- preprocessing quality
- low-risk observation design
- anchor usefulness
- local retrieval usefulness
- ASR usefulness
- web retrieval usefulness
- evidence binding quality
- answer grounding quality

So the training-free phase is sufficient to answer:

- is the method structure itself useful?
- which module contributes most?
- where does the system drift or fail?

## 5. Main Experimental Hypothesis

The training-free plan tests the following hypothesis:

> A `low-risk observation -> anchor -> retrieval -> binding` pipeline is better for VideoDR than a naive `global summary + free web search` pipeline.

This hypothesis decomposes into smaller claims:

- low-risk observations are better than a single global summary
- anchors are better than unstructured history for controlling search
- local retrieval before web search improves grounding
- explicit binding is better than naive concatenation

## 6. Benchmark Strategy

The experiment plan must support five benchmarks, but through `thin adapters`, not a large core.

### 6.1 Mechanism Warm-Up

Benchmarks:

- `MMSearch-Plus`
- `VDR-Bench`

Purpose:

- verify whether the method core works at all
- test whether low-risk observations and anchors improve retrieval behavior
- test whether explicit binding helps over naive multimodal fusion

These are not the final target, but they are the fastest place to see whether the method is structurally promising.

### 6.2 Main Target Validation

Benchmark:

- `VideoDR`

Purpose:

- evaluate the real target setting
- test whether the method reduces `goal drift`
- test whether video clues are preserved more effectively
- test whether explicit video-web grounding improves end-to-end reasoning

This is the benchmark that should define the main claim of the project.

### 6.3 Generalization / Upper Bound

Benchmarks:

- `BrowseComp-VL`
- `MMDeepResearch-Bench`

Purpose:

- test whether the method is overly specialized to VideoDR
- test whether the same evidence pipeline can support longer outputs

These are supporting experiments, not the main battlefield of the first stage.

## 7. How Observation Enters The Experiment Plan

Because observation can both help and hurt, the plan treats it as a first-class experimental object.

Observation should be tested as:

- `low-risk observation`
- `global summary`
- `no observation layer`

This lets us ask directly:

- does observation reduce or amplify drift?
- does question-conditioned observation help?
- does observation refresh matter?

This is one of the most important design choices in the whole training-free stage.

## 8. First-Round Experimental Targets

Do not start with full benchmark runs.

Use a small warm-up package first.

### 8.1 Smoke-Test Package

Suggested size:

- `MMSearch-Plus`: 10
- `VDR-Bench`: 10
- `VideoDR`: 10

Goals:

- verify that adapters work
- verify that the pipeline closes end-to-end
- verify that traces are complete
- inspect whether low-risk observations look reasonable

### 8.2 Development Package

Suggested size:

- `MMSearch-Plus`: 50
- `VDR-Bench`: 50
- `VideoDR`: 30

Goals:

- compare baselines
- run ablations
- collect stable failure patterns

### 8.3 Expansion Package

After the development package stabilizes:

- `MMSearch-Plus`: 100-200
- `VDR-Bench`: 100-200
- `VideoDR`: full available split

Only after this stage should broader generalization experiments become a priority.

## 9. Baseline Design

The baseline set should stay small but diagnostic.

### 9.1 Unimodal Controls

- `vision-only`
- `vision+asr-only`
- `web-only`

Purpose:

- test whether the task truly needs multimodal grounding
- identify how much value ASR adds on its own

### 9.2 Weak Fusion Controls

- `vision+web naive concat`
- `general deep research agent + visual pre-summary`

Purpose:

- test whether gains come from actual grounding structure rather than just more context

### 9.3 Intermediate Control

- `local retrieval + naive reasoning`

Purpose:

- test whether local retrieval alone is enough
- separate local recall gains from anchor/binding gains

### 9.4 Main Method

- `low-risk observation + anchors + local retrieval + web retrieval + binding`

This is the proposed method core.

## 10. Main Ablations

The ablation plan should follow the compact framework directly.

### 10.1 Observation Ablations

- replace low-risk observations with `global summary`
- remove `question-conditioned observation`
- remove `observation refresh`

### 10.2 Anchor Ablations

- remove `anchor extraction`
- replace multiple anchors with a single active anchor

### 10.3 Retrieval Ablations

- remove `local retrieval`
- remove `ASR / speech clues`
- replace multi-step web search with one-step web search

### 10.4 Grounding Ablations

- remove `evidence binding`
- keep web results but do not bind them back to time spans

These ablations should reveal whether the method’s benefit comes from:

- better local video access
- better structured control
- better evidence grounding

## 11. Control Variables

To make conclusions credible, the training-free stage should keep these fixed whenever possible:

- same main MLLM
- same search API
- same retrieval budget
- same maximum number of steps
- same benchmark split

Otherwise the results risk being explained by:

- stronger models
- larger search budgets
- more context

rather than the method itself.

## 12. Process Metrics

The project should not rely only on final benchmark scores.

It should also record process-level metrics.

### 12.1 Observation Metrics

- `Observation Coverage`
- `Observation Precision`
- `Observation Drift Rate`

Questions:

- do observations cover useful clues?
- do they contain too much irrelevant information?
- do they already drift before anchor selection?

### 12.2 Anchor Metrics

- `Anchor Coverage`
- `Anchor Precision`
- `Active Anchor Stability`

Questions:

- are the selected anchors actually relevant?
- do they stay useful across steps?

### 12.3 Retrieval Metrics

- `Local Retrieval Recall@K`
- `ASR Retrieval Recall@K`
- `Useful Web Retrieval Rate`

### 12.4 Grounding Metrics

- `Binding Precision`
- `Timestamp Alignment Quality`
- `Claim Support Rate`

### 12.5 Outcome Metrics

- benchmark task score
- judge score
- answer confidence
- evidence count

## 13. What Must Be Saved Per Run

To support failure analysis and innovation discovery, every run should save:

- task input
- benchmark metadata
- observations
- anchors
- local retrieval results
- web retrieval results
- evidence objects
- bindings
- final answer or report
- judge input
- judge output
- error state if the run fails

This is essential because the project is trying to learn from the trajectory, not just from final scores.

## 14. Judge Usage

Judge should be used as a supplement, not a replacement for benchmark metrics.

### 14.1 Short-Answer Tasks

Judge should focus on:

- semantic correctness
- whether video evidence was really used
- whether web evidence was really used
- whether the answer is grounded in the evidence chain

### 14.2 Long-Form Tasks

Judge should focus on:

- factuality
- citation quality
- evidence alignment
- reasoning consistency
- output structure

### 14.3 Judge Input Packaging

Judge input should include:

- task input
- model output
- reference answer or materials
- evidence chain
- tool trace
- rubric

This keeps the evaluation analyzable even when exact-match metrics are insufficient.

## 15. First Experimental Sequence

The recommended order is:

### Step 1: Pipeline Validation

Run the smoke-test package.

Questions:

- do adapters load correctly?
- do observations look usable?
- does the pipeline finish?
- are traces complete?

### Step 2: Baseline Comparison

Run the development package with:

- unimodal controls
- weak fusion controls
- main method

Questions:

- does the main method beat naive baselines?
- does low-risk observation help?

### Step 3: Ablation Study

Run observation, anchor, retrieval, and grounding ablations.

Questions:

- what is the main source of improvement?
- what is replaceable?
- what is indispensable?

### Step 4: Failure Taxonomy

Bucket failures into categories such as:

- `bad_observation`
- `missed_anchor`
- `bad_local_retrieval`
- `bad_asr`
- `bad_web_retrieval`
- `bad_binding`
- `goal_drift`
- `unsupported_answer`

### Step 5: Innovation Extraction

Turn recurring failure modes into method candidates.

Examples:

- too much irrelevant observation
  - better observation pruning
- missed key visual clue
  - better keyframe / event selection
- anchor becomes stale after search
  - anchor refresh or re-grounding
- web evidence cannot be aligned back to video
  - timestamp-aware binding verifier

## 16. How Innovation Should Emerge

The training-free stage is not only for benchmarking. It is the main discovery engine for innovation.

The expected pattern is:

- run traces
- inspect failures
- identify which layer fails first
- turn that failure into a sharper method proposal

This is especially important because the likely innovation points are still open:

- keyframe selection
- low-risk observation construction
- visual anchor construction
- temporal evidence binding

The experiments should therefore be designed to make these differences visible.

## 17. Expected Outputs Of The Training-Free Stage

By the end of the first stage, the project should produce:

- a stable small-core benchmark runner
- baseline comparison tables
- ablation tables
- process-metric tables
- failure taxonomy
- judge-based qualitative cases
- a ranked list of next-step innovation candidates

These outputs are enough to support:

- method iteration
- proposal refinement
- experimental planning for a paper
- later transition into training

## 18. One-Sentence Summary

The training-free stage should be a `VideoDR-first` method study built around `low-risk observations`, `anchor-driven retrieval`, and `explicit evidence binding`, using small but carefully structured experiments to identify the most promising innovation point before expanding the engineering scope or adding training.
