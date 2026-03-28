# Video-Grounded Deep Research Framework Proposal

## 1. Project Positioning

This project targets a new but important setting: `video-grounded deep research`.

In this setting, an agent must:

- extract useful clues from video
- search the open web
- bind external evidence back to video segments
- answer with traceable support

The primary goal is **not** to build a large general-purpose multimodal platform. The primary goal is to propose a method that improves performance on `VideoDR`-style tasks.

At the same time, because the area is still young and benchmark coverage is limited, the framework must remain compatible with:

- `single-image` settings
- `multi-image` settings
- `long-form report` settings

These additional settings are used for:

- mechanism warm-up
- generalization support
- upper-bound capability display

They do not replace `VideoDR` as the main target.

## 2. Research Objective

The project aims to answer two questions at once:

1. Can we improve `VideoDR` by designing a better video-grounded research loop?
2. Can the same method core remain testable across five relevant benchmarks?

The five benchmark targets are:

- `MMSearch-Plus`
- `VDR-Bench`
- `VideoDR`
- `BrowseComp-VL`
- `MMDeepResearch-Bench`

The central focus remains `VideoDR`.

## 3. Problem Statement

Current multimodal research agents often fail on VideoDR-like tasks because of four linked issues:

- `goal drift` during multi-step web search
- weak preservation of video clues across the loop
- poor local grounding before web retrieval
- weak binding between video evidence and web evidence

If we directly compress a long video into one global summary, the system may drift early and then amplify its own mistake.

This proposal therefore takes a different view:

- long videos should first be broken into local observations
- observations should remain low-risk and revisable
- anchors should be built from observations, not from a single one-shot summary
- web retrieval should be driven by active anchors, not free-form history alone

## 4. Main Hypothesis

The central hypothesis is:

> A `low-risk observation -> anchor -> retrieval -> binding` pipeline is better suited to VideoDR than a naive `global video summary + free web search` pipeline.

More specifically:

- local observations reduce the burden of carrying the entire video in context
- low-risk observations reduce the chance of early overcommitment
- anchor selection provides a compact working state for search
- local retrieval improves in-video localization before web expansion
- explicit evidence binding reduces uncontrolled drift

If this hypothesis is correct, the method should:

- improve evidence quality
- improve traceability
- reduce drift
- better utilize video information in long-horizon search

## 5. Core Method Idea

The proposed method has three core ideas.

### 5.1 Low-Risk Observation Layer

Instead of forcing the system to summarize the whole video at once, the framework first builds a list of `observation candidates`.

An observation is:

- local
- question-conditioned
- weakly semantic
- refreshable
- uncertain by design

It should record:

- scene clues
- speech clues
- OCR clues
- candidate entities
- candidate actions
- time range

It should avoid turning uncertain local evidence into strong conclusions too early.

This is important because a wrong early summary can amplify downstream drift.

### 5.2 Anchor-Centric Retrieval Loop

Observations are not the final working state. The system uses them to build `anchors`.

An anchor is the current research unit that drives:

- local retrieval
- web retrieval
- evidence binding
- claim update

The loop is intentionally simple in v1:

- fixed number of steps
- small search budget
- one main controller
- no heavy planner

This keeps the method analyzable while still allowing structured multi-step search.

### 5.3 Cross-Modal Evidence Binding

Retrieved web results are not simply appended back into the prompt.

Instead, each useful result must be explicitly bound to:

- which anchor it supports
- which entities or actions it matches
- which temporal clue in the video it corresponds to

This is the part that most directly distinguishes the method from naive multimodal concatenation.

## 6. Why Observation Must Be Low-Risk

The proposal explicitly treats observation as a potential source of error amplification.

If observation is:

- single-shot
- overconfident
- not refreshable
- directly treated as fact

then it can make `goal drift` worse rather than better.

Therefore the proposal adopts the following design rules:

- `observation != anchor`
- observation is a candidate layer, not the final reasoning layer
- multiple observations can coexist
- observation can be downweighted or refreshed
- observation stores uncertainty, not just compressed meaning

This is not only a safety choice. It is also a likely source of method innovation.

## 7. Method Scope

To keep the project focused on innovation rather than framework sprawl, the method core is intentionally small.

The proposal keeps only:

- `preprocessing`
- `observation`
- `anchors`
- `retrieval`
- `grounding`
- `reasoning`
- `evaluation`

The proposal intentionally postpones:

- multi-agent orchestration
- heavy memory systems
- training
- complex planner stacks
- large prompt infrastructure
- depth / thermal pipelines

## 8. Benchmark Compatibility Strategy

The framework must still support five benchmarks, but compatibility should come from `thin adapters`, not from inflating the method core.

The strategy is:

- keep one shared method core
- add one lightweight adapter per benchmark
- let output mode and evaluation mode be configurable

This means:

- benchmark differences stay outside the core loop
- the same method can be compared across tasks
- the project stays small enough for fast iteration

## 9. Benchmark Roles

### 9.1 Mechanism Warm-Up

Benchmarks:

- `MMSearch-Plus`
- `VDR-Bench`

Purpose:

- verify whether the mechanism itself is useful
- test whether low-risk observations and anchors help retrieval
- test whether binding is better than naive multimodal concatenation

### 9.2 Target Task Validation

Benchmark:

- `VideoDR`

Purpose:

- test the actual target task
- verify whether the method reduces drift
- verify whether video clues are better preserved and used
- verify whether the method improves video-web reasoning quality

### 9.3 Generalization and Upper Bound

Benchmarks:

- `BrowseComp-VL`
- `MMDeepResearch-Bench`

Purpose:

- show that the method is not only a narrow VideoDR trick
- test whether the same evidence pipeline can support longer outputs

These benchmarks are supporting evidence, not the primary source of the paper’s main claim.

## 10. Baseline Design

The proposal uses simple but important control baselines:

- `vision-only`
- `web-only`
- `vision+web naive concat`
- `vision+asr-only`
- `local retrieval + naive reasoning`
- `general deep research agent + visual pre-summary`

These baselines are chosen to answer:

- Is the method better than unimodal control?
- Is the gain really from structured grounding?
- Is local retrieval alone enough?
- Does explicit binding matter?

## 11. Ablation Design

Key ablations should include:

- remove `observation refresh`
- remove `anchor extraction`
- remove `local retrieval`
- remove `ASR / speech clues`
- remove `evidence binding`
- replace multi-step web search with one-step search
- replace local observations with a single global summary

These ablations are critical because they directly test the method hypothesis.

## 12. Expected Innovation Points

The proposal does not assume the final innovation must come from the full pipeline equally.

Instead, it deliberately leaves room for innovation in the most promising areas:

- `query-conditioned keyframe selection`
- `low-risk observation construction`
- `visual anchor extraction`
- `cross-frame anchor maintenance`
- `video-web temporal binding`

Among these, the most likely high-value direction is:

- keyframe / visual anchor handling

because it directly affects both:

- recall of useful video clues
- downstream drift behavior

## 13. Evaluation Plan

Evaluation should include both:

- benchmark-level performance
- process-level diagnostics

Beyond final answer metrics, the proposal recommends tracking:

- local retrieval recall
- anchor count and quality
- evidence count
- binding quality
- judge score
- failure tags

This is important because the main value of the framework is not only higher final accuracy, but also better grounded behavior.

## 14. Failure Analysis

Failure analysis is central to this project.

The proposal expects to explicitly inspect failures such as:

- wrong keyframe selection
- weak observation construction
- anchor drift
- local retrieval miss
- bad ASR clue propagation
- web evidence not binding back to the right time span
- answer looks plausible but is weakly grounded

These failures are not only evaluation artifacts. They are the main source from which stronger method innovations can emerge.

## 15. Expected Contributions

This project is expected to contribute:

- a `VideoDR-first` grounded research framework with a small method core
- a low-risk observation design that avoids early overcommitment
- an anchor-centric retrieval and binding loop for video-web reasoning
- a benchmark-compatible evaluation setup through thin adapters
- an analyzable experiment pipeline for studying drift and grounding failure

## 16. Phase Plan

### Phase A: Method Warm-Up

- build the minimal pipeline
- run `MMSearch-Plus` and `VDR-Bench`
- verify whether observations, anchors, and binding help

### Phase B: VideoDR Main Study

- run `VideoDR`
- compare against control baselines
- perform ablation
- inspect failure taxonomy

### Phase C: Generalization and Upper Bound

- test `BrowseComp-VL`
- support report-mode evaluation on `MMDeepResearch-Bench`
- study whether the same evidence pipeline scales to long-form output

## 17. One-Sentence Summary

This proposal studies a `VideoDR-first` method for video-grounded deep research, centered on `low-risk observations`, `anchor-driven retrieval`, and `explicit evidence binding`, while keeping the project small enough for fast innovation and broad enough to test across five relevant benchmarks.
