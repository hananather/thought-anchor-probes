# Systematic Sweep Runbook

This runbook documents the staged sweep workflow for high-confidence Qwen-14B campaigns.

## Goals
- Exhaustive Stage-1 structure sweep.
- Stage-2 vertical-attention expansion on shortlisted configs.
- Stage-3 training hyperparameter saturation on finalists.
- Resume-safe execution and deterministic summaries.

## Prerequisites
- Use the project virtual environment.
- Confirm `torch` / `transformers` versions and GPU visibility.

```bash
source .venv/bin/activate
python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
nvidia-smi
```

## Stage 1
Generate Stage-1 manifest and configs.

```bash
python scripts/generate_sweep_configs.py \
  --base-config configs/scaling_qwen_correct.yaml \
  --sweep-root artifacts/sweeps/qwen_ultra \
  --stage stage1 --num-layers 40 \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
  --reuse-cache --skip-failed
```

Run Stage-1.

```bash
python scripts/run_systematic_sweep.py \
  --sweep-root artifacts/sweeps/qwen_ultra \
  --manifest artifacts/sweeps/qwen_ultra/manifest_stage1.jsonl \
  --retries 1 --continue-on-error
```

Summarize and build Stage-2 shortlist.

```bash
python scripts/summarize_sweep.py \
  --sweep-root artifacts/sweeps/qwen_ultra \
  --stage2-top-n-per-target-mode 24 \
  --stage3-top-n 12
```

## Stage 2
Generate Stage-2 configs from shortlist.

```bash
python scripts/generate_sweep_configs.py \
  --base-config configs/scaling_qwen_correct.yaml \
  --sweep-root artifacts/sweeps/qwen_ultra \
  --stage stage2 \
  --seed-configs artifacts/sweeps/qwen_ultra/summary/shortlist_stage2.json \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
  --reuse-cache --skip-failed
```

Run and summarize Stage-2.

```bash
python scripts/run_systematic_sweep.py \
  --sweep-root artifacts/sweeps/qwen_ultra \
  --manifest artifacts/sweeps/qwen_ultra/manifest_stage2.jsonl \
  --retries 1 --continue-on-error

python scripts/summarize_sweep.py \
  --sweep-root artifacts/sweeps/qwen_ultra \
  --stage2-source-stage stage1 \
  --stage3-source-stage stage2 \
  --stage2-top-n-per-target-mode 24 \
  --stage3-top-n 12
```

## Stage 3
Generate Stage-3 configs from shortlist.

```bash
python scripts/generate_sweep_configs.py \
  --base-config configs/scaling_qwen_correct.yaml \
  --sweep-root artifacts/sweeps/qwen_ultra \
  --stage stage3 \
  --seed-configs artifacts/sweeps/qwen_ultra/summary/shortlist_stage3.json \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
  --reuse-cache --skip-failed
```

Run and summarize Stage-3.

```bash
python scripts/run_systematic_sweep.py \
  --sweep-root artifacts/sweeps/qwen_ultra \
  --manifest artifacts/sweeps/qwen_ultra/manifest_stage3.jsonl \
  --retries 1 --continue-on-error

python scripts/summarize_sweep.py \
  --sweep-root artifacts/sweeps/qwen_ultra \
  --stage2-source-stage stage1 \
  --stage3-source-stage stage2
```

## Artifacts
- `artifacts/sweeps/<sweep_id>/manifest*.jsonl`: generated config manifests.
- `artifacts/sweeps/<sweep_id>/run_registry.jsonl`: execution status log.
- `artifacts/sweeps/<sweep_id>/logs/<config_id>/`: per-step stdout/stderr logs.
- `artifacts/sweeps/<sweep_id>/summary/leaderboard.csv`: per-config ranking table.
- `artifacts/sweeps/<sweep_id>/summary/paired_deltas.csv`: paired delta CI summaries.
- `artifacts/sweeps/<sweep_id>/summary/stability_report.csv`: variance/stability diagnostics.
- `artifacts/sweeps/<sweep_id>/summary/shortlist_stage2.json`: stage-2 seed config list.
- `artifacts/sweeps/<sweep_id>/summary/shortlist_stage3.json`: stage-3 seed config list.

## Operational Notes
- `run_systematic_sweep.py` is resume-safe by default: configs with latest `success` are skipped.
- Use `--force` to rerun completed configs.
- Use `--dry-run` to preview what would execute.
- Keep `--run-tripwires` off for large campaigns unless explicitly auditing tripwire behavior.
