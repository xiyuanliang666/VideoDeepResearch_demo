# VideoGroundedResearch v0

This repository is a `training-free` v0 scaffold for a video-grounded deep research system.

Current scope:

- single-case debugging
- small-benchmark experimentation
- video / frame / audio preprocessing
- low-risk observation construction
- ASR-assisted local retrieval
- anchor extraction
- web retrieval with fallback behavior
- evidence binding
- final answer generation
- heuristic Judge output for experiment tracing
- optional online multimodal observation enhancement (provider-routed)

Related docs:

- `proposal.md`
- `framework.md`
- `training-free_plan.md`

## Current Architecture

The project now follows a compact `method-first` layout:

- `src/core/`
  - the main method core
  - `preprocessing -> observation -> anchors -> retrieval -> grounding -> reasoning`
- `src/adapters/`
  - thin benchmark adapters for:
    - `MMSearch-Plus`
    - `VDR-Bench`
    - `VideoDR`
    - `BrowseComp-VL`
    - `MMDeepResearch-Bench`
- `src/schemas/`
  - shared intermediate objects such as `Sample`, `Observation`, `Anchor`, and `Prediction`
- `src/tools/`
  - reusable low-level utilities

Execution mode:

- `workflow` (default): fixed stage pipeline
- `agentic`: LLM planner decides each web-search step (search/finalize loop)

## Quick Start

### 1. Install dependencies

Create a Python environment first, then install:

```bash
pip install -r requirements.txt
```

If you want live online inference + web retrieval, create `.env` from `.env.example`.

Example:

```bash
cp .env.example .env
# then edit .env with your real keys/base_url/model
```

Prompts for multimodal observation are externalized under:

- `prompts/observation_multimodal/system.txt`
- `prompts/observation_multimodal/user.txt`

Recommended config pattern:

- Use one unified gateway: `LLM_API_KEY`, `LLM_BASE_URL`
- Route runtime roles by model name only:
  - `REASONING_MODEL`
  - `JUDGE_MODEL`
  - `VISION_MODEL`
- `LLM_MODEL` is an optional generic fallback

Notes:

- `ffmpeg` / `ffprobe` are recommended for real video slicing and frame extraction.
- If `ffmpeg` is missing, the current v0 pipeline will fall back to a single-clip mode.
- If `TAVILY_API_KEY` is missing, web retrieval will return structured fallback candidates so the pipeline can still be debugged.
- If `LLM_API_KEY` or `LLM_BASE_URL` is missing, online inference automatically falls back to local heuristic logic.
- Judge supports online mode via `ENABLE_ONLINE_JUDGE=true`. If online judge fails, it falls back to local heuristic judge.

## Run A Single Sample

Use `scripts/run_single_case.py` for smoke tests and single-case debugging.

Example:

```bash
python3 scripts/run_single_case.py \
  --video-path data/raw_videos/demo.mp4 \
  --question "What is happening in the video and what external evidence supports it?" \
  --task-profile configs/task_profiles/videodr_v0.yaml \
  --model-profile configs/model_profiles/default.yaml
```

Optional arguments:

- `--run-id`: manually assign a run id
- `--task-profile`: switch task settings
- `--model-profile`: switch model / backend settings

What this script does:

- preprocesses the video into clips / frames / transcript artifacts
- runs local retrieval
- extracts anchors
- performs web retrieval
- binds web evidence back to video anchors
- generates a final answer
- optionally generates a judge result

## Run A Benchmark Split

Use `scripts/run_benchmark.py` to batch-run a small benchmark split.

Supported input formats:

- `.jsonl`: one sample per line
- `.json`: either a top-level list, or a dict containing `samples` or `data`

You can run in two ways:

- pass `--input-file` and `--adapter` directly
- or pass `--benchmark-config` and let the script load the adapter and demo input automatically

Each sample should at least contain:

- `video_path` for `videodr`
- `image_path` or `media_paths` for image / multi-image benchmarks
- `question`

Optional fields:

- `sample_id`
- `reference_answer`
- `metadata`

Minimal example sample:

```json
{
  "sample_id": "demo_0001",
  "video_path": "data/raw_videos/demo.mp4",
  "question": "What is happening in the video?",
  "reference_answer": "Optional reference answer"
}
```

Example command:

```bash
python3 scripts/run_benchmark.py \
  --input-file data/benchmarks/videodr_demo.jsonl \
  --benchmark-name videodr \
  --adapter videodr \
  --mode framework \
  --task-profile configs/task_profiles/videodr_v0.yaml \
  --model-profile configs/model_profiles/default.yaml \
  --limit 10
```

Config-driven example:

```bash
python3 scripts/run_benchmark.py \
  --benchmark-config configs/benchmarks/videodr.yaml \
  --limit 1
```

Experiment sweep example (compare multiple reasoning models from `configs/experiment.yaml`):

```bash
python3 scripts/run_benchmark.py \
  --benchmark-config configs/benchmarks/videodr.yaml \
  --experiment-config configs/experiment.yaml \
  --limit 1
```

Useful arguments:

- `--benchmark-name`: label used in output summaries
- `--adapter`: selects which benchmark adapter normalizes the raw samples
- `--benchmark-config`: loads benchmark-specific defaults such as adapter and demo input file
- `--mode`: label for baseline / method comparison, such as `framework`, `web_only`, `naive_concat`
  - when set to `workflow` or `agentic`, it also controls execution mode
- `--experiment-config`: optional yaml with `reasoning_models` list for batch comparison
- `--limit`: run only the first N samples
- `--run-id`: manually set the benchmark run id

Behavior:

- the script runs each sample independently
- each sample gets its own trace directory
- if one sample fails, the whole batch does not stop
- failures are recorded into the result rows

You can also set mode in task profile yaml:

```yaml
mode: workflow   # or agentic
max_agent_iterations: 3
```

## Benchmark Configs And Demo Inputs

The repository now includes lightweight benchmark configs and demo input files for:

- `configs/benchmarks/videodr.yaml`
- `configs/benchmarks/vdr_bench.yaml`
- `configs/benchmarks/mmsearch_plus.yaml`
- `configs/benchmarks/browsecomp_vl.yaml`
- `configs/benchmarks/mmdeepresearch_bench.yaml`

Matching demo inputs are stored under:

- `data/benchmarks/`

These files are not the full official datasets. They are runnable schema examples that let you:

- validate each adapter
- validate the benchmark runner
- keep input formats consistent before downloading the full datasets

If your datasets are already downloaded under `/mnt/sda/Datasets`, generate runnable benchmark jsonl files with:

```bash
python3 scripts/prepare_benchmark_inputs.py
```

Generated files are written to `data/benchmarks/real/`, and benchmark configs can point `input_file` to these real inputs.

## Re-Summarize Existing Results

If you already have a `results.jsonl`, you can regenerate the summary files without rerunning the pipeline.

Example:

```bash
python3 scripts/summarize_results.py \
  --results-file outputs/eval/demo_run/results.jsonl
```

Optional arguments:

- `--summary-file`
- `--comparison-csv`

## How To Read Outputs

The current v0 outputs are organized into three main folders.

### `outputs/traces/`

This is the most detailed debugging output.

Structure:

```text
outputs/traces/<run_id>/
outputs/traces/<run_id>/<sample_id>/trace.json
```

For single-case runs:

- `trace.json`
- `question.txt`

What to inspect in `trace.json`:

- `clips`
- `observations`
- `transcript`
- `local_query`
- `local_candidates`
- `anchors`
- `web_queries`
- `web_candidates`
- `evidences`
- `bindings`
- `final_answer`
- `judge_result`

Use this file when you want to answer:

- which clip was retrieved
- what anchor was created
- what web results were kept
- whether evidence was bound back to the right video segment

### `outputs/answers/`

This folder stores answer-oriented outputs.

Files:

- single case: `outputs/answers/<run_id>.json`
- benchmark batch: `outputs/answers/<run_id>.jsonl`

Use this when you want to quickly inspect:

- final answers
- answer confidence
- supporting video evidence
- supporting web evidence

### `outputs/eval/`

This folder stores benchmark-level experiment summaries.

Structure:

```text
outputs/eval/<run_id>/
  results.jsonl
  summary.json
  compare.csv
```

Meaning of each file:

- `results.jsonl`: one row per sample
- `summary.json`: aggregate statistics for one run
- `compare.csv`: flattened summary row, convenient for combining multiple runs later

Key fields in `results.jsonl`:

- `status`
- `final_answer`
- `answer_confidence`
- `judge_score`
- `judge_verdict`
- `anchor_count`
- `evidence_count`
- `binding_count`
- `error_message`

Key fields in `summary.json`:

- `sample_count`
- `answered_count`
- `answer_rate`
- `avg_judge_score`
- `avg_answer_confidence`
- `avg_anchor_count`
- `avg_evidence_count`
- `avg_binding_count`
- `status_counts`
- `judge_verdict_counts`

## Suggested First Checks

When you run v0 for the first time, inspect these in order:

1. `outputs/traces/<run_id>/.../trace.json`
2. `outputs/answers/<run_id>.jsonl`
3. `outputs/eval/<run_id>/summary.json`

Recommended questions to ask:

1. Did the pipeline finish with `status=answer_ready`?
2. Did local retrieval produce reasonable candidates?
3. Did anchor extraction preserve the relevant video clue?
4. Did web retrieval return analyzable results or fallback candidates?
5. Did bindings connect web evidence back to the correct clip / time span?
6. Does the final answer actually reflect the evidence chain?

## Current Limitations

This is still a v0 scaffold, so a few parts are intentionally simple:

- local retrieval is heuristic, not embedding-based yet
- anchor extraction is heuristic
- evidence binding is heuristic
- final reasoning is deterministic, not MLLM-based yet
- judge is heuristic, not a real LLM judge yet

The current goal is to make the full experiment loop analyzable before replacing these modules with stronger learned or MLLM-based versions.
