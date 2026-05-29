# Stable Baseline Entrypoints

This directory provides stable command-line entrypoints for external benchmark adapters.

The repository now exposes one implementation family through `videodeepresearch.run(...)`. Historical agent snapshots are not benchmark entrypoints.

For benchmark integration, use this directory. It hides whether the run uses `workflow` or `agentic` mode and always writes the same normalized result envelope.

## Single Sample

```bash
python baseline_entrypoints/run_vdr_single.py \
  --sample-id demo_0001 \
  --video-path data/raw_videos/demo.mp4 \
  --question "What is happening in the video?" \
  --out-json outputs/baseline_entrypoints/demo_0001/result.json \
  --execution-mode workflow
```

The output JSON contains:

- `entrypoint`
- `sample_id`
- `execution_mode`
- `trace_path`
- `answer_path`
- `result`

The nested `result` field is the unified `videodeepresearch.run.v1` output. It still carries `raw_trace` for debugging, but benchmark-side adapters should prefer the normalized fields.
