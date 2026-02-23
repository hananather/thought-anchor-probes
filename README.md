# Thought Anchor Probes

Train sentence probes that predict counterfactual sentence importance from model activations.

## What You'll Do
- Load the ARENA Thought Anchors rollout dataset from Hugging Face.
- Build sentence labels from `counterfactual_importance_accuracy`.
- Extract one-layer sentence embeddings with mean pooling.
- Train position, text-only, activation-only, and activation+position probes.
- Report top-k recall and precision-recall AUC.

## Why This Matters
This setup gives a cheap first test for Thought Anchor signal.
You can validate signal before harder token-aware probes.

## Technical Requirements
- Python 3.10 or newer.
- `huggingface-hub` for dataset file listing and downloads.
- `transformers` and `torch` for model activations.
- `scikit-learn` for probe training.
- `pandas` and `pyarrow` for metadata tables.

## Quick Start
1. Create a virtual environment and install dependencies.
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```
2. Run the full pilot + full experiment workflow.
```bash
python scripts/run_experiments.py \
  --config configs/experiment.yaml \
  --pilot-config configs/experiment_pilot.yaml \
  --full-config configs/experiment_full.yaml \
  --problem-id 330 \
  --readme-path README.md
```
3. Inspect per-stage artifacts.
```bash
ls artifacts/runs/pilot
ls artifacts/runs/full
```

## Optional Checks
- Run span integrity checks on one problem.
```bash
python scripts/check_spans.py --config configs/experiment.yaml --problem-id 330 --sample-size 20
```
- Recompute counterfactual labels for one problem.
```bash
pip install -e ".[verify]"
python scripts/verify_problem_labels.py --config configs/experiment.yaml --problem-id 330 --counterfactual
```

## MacBook Tips
- Start with `num_problems: 25` in `configs/experiment.yaml`.
- Keep `layer_mode: mid` and `pooling: mean`.
- Use `device: auto` to select MPS when available.
- Avoid loading full rollouts except one verification problem.

## Output Files
- `artifacts/runs/pilot/sentence_embeddings.dat`
- `artifacts/runs/pilot/metrics_seed_*.json`
- `artifacts/runs/pilot/aggregate_metrics.json`
- `artifacts/runs/full/sentence_embeddings.dat`
- `artifacts/runs/full/metrics_seed_*.json`
- `artifacts/runs/full/aggregate_metrics.json`

## Cache Safety Invariants
- `scripts/train_probes.py` validates extraction-time provenance before fitting probes.
- If you change label or activation config (for example `labels.anchor_percentile`, `labels.drop_last_chunk`, or `activations.model_name_or_path`), re-run `scripts/extract_embeddings.py`.
- Training now fails fast when split problem IDs are missing from extracted metadata.
- `scripts/extract_embeddings.py --reuse-cache` skips extraction only when cache provenance matches the active config.
- Implementation details for maintainers: `docs/cache_provenance_guardrail.md`.

## High-Confidence Scaling Matrix
- 4-setting configs are available under `configs/scaling_*.yaml`.
- Default matrix run now focuses on the paper's primary model, Qwen-14B:
```bash
python scripts/run_scaling_grid.py
```
- This default runs:
  - `configs/scaling_qwen_correct.yaml`
  - `configs/scaling_qwen_incorrect.yaml`
- To include Llama-8B replication settings, pass configs explicitly:
```bash
python scripts/run_scaling_grid.py \
  --configs \
  configs/scaling_llama_correct.yaml \
  configs/scaling_llama_incorrect.yaml \
  configs/scaling_qwen_correct.yaml \
  configs/scaling_qwen_incorrect.yaml
```
- RunPod memory: use at least 48 GB VRAM for stable 14B extraction runs.
- RunPod dependency contract (for reproducible VRAM/runtime behavior):
```bash
uv pip install -e . --system -c constraints/runpod.txt
python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
```
- Expected preflight versions: `2.9.1 4.57.3`.
- RunPod execution guide: `docs/runpod_scaling_runbook.md`.

## Repo Layout
- `src/ta_probe/data_loading.py`: dataset listing and fast metadata loading.
- `src/ta_probe/labels.py`: anchor labels from counterfactual scores.
- `src/ta_probe/spans.py`: chunking and sentence token boundaries.
- `src/ta_probe/activations.py`: one-layer hooks and pooled embeddings.
- `src/ta_probe/models.py`: baseline and probe models.
- `src/ta_probe/train.py`: training, evaluation, and tripwire checks.
- `src/ta_probe/aggregate.py`: multi-seed metric aggregation.
- `src/ta_probe/readme_update.py`: deterministic README marker updates.
- `scripts/run_experiments.py`: end-to-end pilot + full orchestration.
- `scripts/run_scaling_grid.py`: four-setting scaling matrix orchestration.
- `tests/`: unit tests for spans, labels, and metrics.

## Experiment Results

<!-- EXPERIMENT_RESULTS_START -->
Last updated: 2026-02-22 17:40:42

### Objective
Run the full Thought Anchor probe plan with pilot and full stages.

### Environment
- Host: MacBook-Pro-2.local
- Platform: macOS-26.2-arm64-arm-64bit
- Python: 3.11.8
- Dataset listing date context: February 22, 2026

### Commands Executed
- `python -m ruff check .` | exit=0 | 0.03s
- `python -m pytest -q` | exit=0 | 3.35s
- `python scripts/build_problem_index.py --config configs/experiment.yaml --refresh` | exit=0 | 0.35s
- `python scripts/verify_problem_labels.py --config configs/experiment.yaml --problem-id 330` | exit=0 | 2.44s
- `python scripts/check_spans.py --config configs/experiment.yaml --problem-id 330 --sample-size 20` | exit=0 | 3.74s
- `python scripts/build_problem_index.py --config configs/experiment_pilot.yaml --refresh` | exit=0 | 0.36s
- `python scripts/build_problem_index.py --config configs/experiment_full.yaml --refresh` | exit=0 | 0.34s
- `python scripts/extract_embeddings.py --config configs/experiment_pilot.yaml --skip-failed --failure-log /Users/hananather/Desktop/log/MATS/probes/thought-anchor-probes/artifacts/runs/pilot/extraction_failures.json` | exit=0 | 403.48s
- `python scripts/train_probes.py --config configs/experiment_pilot.yaml --seed 0 --run-name seed_0` | exit=0 | 1.86s
- `python scripts/train_probes.py --config configs/experiment_pilot.yaml --seed 1 --run-name seed_1` | exit=0 | 1.23s
- `python scripts/train_probes.py --config configs/experiment_pilot.yaml --seed 2 --run-name seed_2` | exit=0 | 1.22s
- `python scripts/aggregate_runs.py --run-root /Users/hananather/Desktop/log/MATS/probes/thought-anchor-probes/artifacts/runs/pilot` | exit=0 | 0.31s
- `python scripts/extract_embeddings.py --config configs/experiment_full.yaml --skip-failed --failure-log /Users/hananather/Desktop/log/MATS/probes/thought-anchor-probes/artifacts/runs/full/extraction_failures.json` | exit=0 | 553.25s
- `python scripts/train_probes.py --config configs/experiment_full.yaml --seed 0 --run-name seed_0` | exit=0 | 3.02s
- `python scripts/train_probes.py --config configs/experiment_full.yaml --seed 1 --run-name seed_1` | exit=0 | 2.73s
- `python scripts/train_probes.py --config configs/experiment_full.yaml --seed 2 --run-name seed_2` | exit=0 | 2.58s
- `python scripts/aggregate_runs.py --run-root /Users/hananather/Desktop/log/MATS/probes/thought-anchor-probes/artifacts/runs/full` | exit=0 | 0.32s

### Verification Gates
- Resampling avg diff: 0.008326
- Resampling pass: True
- Span pass rate: 1.000

### Dataset and Split Summary
- Pilot split sizes: train=3, val=1, test=1
- Full split sizes: train=14, val=3, test=3

### Pilot Per-Seed Test Metrics
| Run | Seed | Model | PR AUC | Spearman | Top-5 | Top-10 |
|---|---:|---|---:|---:|---:|---:|
| seed_0 | 0 | linear_probe | 0.1456 | 0.1315 | 0.0000 | 0.0000 |
| seed_0 | 0 | mlp_probe | 0.1574 | 0.2708 | 0.0000 | 0.0000 |
| seed_0 | 0 | position_baseline | 0.1415 | 0.4686 | 0.0000 | 0.1000 |
| seed_1 | 1 | linear_probe | 0.1456 | 0.1315 | 0.0000 | 0.0000 |
| seed_1 | 1 | mlp_probe | 0.1012 | 0.0968 | 0.0000 | 0.0000 |
| seed_1 | 1 | position_baseline | 0.1415 | 0.4686 | 0.0000 | 0.1000 |
| seed_2 | 2 | linear_probe | 0.1456 | 0.1315 | 0.0000 | 0.0000 |
| seed_2 | 2 | mlp_probe | 0.1649 | 0.0911 | 0.0000 | 0.3000 |
| seed_2 | 2 | position_baseline | 0.1415 | 0.4686 | 0.0000 | 0.1000 |

### Pilot Mean and Std Across Seeds
| Model | PR AUC mean | PR AUC std | Spearman mean | Spearman std | Top-5 mean | Top-5 std | Top-10 mean | Top-10 std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| linear_probe | 0.1456 | 0.0000 | 0.1315 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| position_baseline | 0.1415 | 0.0000 | 0.4686 | 0.0000 | 0.0000 | 0.0000 | 0.1000 | 0.0000 |
| mlp_probe | 0.1412 | 0.0284 | 0.1529 | 0.0834 | 0.0000 | 0.0000 | 0.1000 | 0.1414 |

- Pilot best model by mean PR AUC: `linear_probe`

### Pilot Tripwire Outcomes
| Run | Random-label near chance | Random-label PR AUC | Prevalence | Overfit can memorize | Overfit PR AUC |
|---|---|---:|---:|---|---:|
| seed_0 | True | 0.0877 | 0.1003 | False | 0.7540 |
| seed_1 | True | 0.0841 | 0.1003 | False | 0.7279 |
| seed_2 | True | 0.0982 | 0.1003 | False | 0.8447 |

### Pilot Extraction Failures
- No extraction failures were logged.

### Full Per-Seed Test Metrics
| Run | Seed | Model | PR AUC | Spearman | Top-5 | Top-10 |
|---|---:|---|---:|---:|---:|---:|
| seed_0 | 0 | linear_probe | 0.1766 | 0.1951 | 0.0667 | 0.1667 |
| seed_0 | 0 | mlp_probe | 0.1708 | 0.2506 | 0.0000 | 0.0333 |
| seed_0 | 0 | position_baseline | 0.1625 | 0.5125 | 0.0667 | 0.1000 |
| seed_1 | 1 | linear_probe | 0.1766 | 0.1951 | 0.0667 | 0.1667 |
| seed_1 | 1 | mlp_probe | 0.1901 | 0.2247 | 0.0667 | 0.1333 |
| seed_1 | 1 | position_baseline | 0.1625 | 0.5125 | 0.0667 | 0.1000 |
| seed_2 | 2 | linear_probe | 0.1766 | 0.1951 | 0.0667 | 0.1667 |
| seed_2 | 2 | mlp_probe | 0.1704 | 0.2727 | 0.0000 | 0.0667 |
| seed_2 | 2 | position_baseline | 0.1625 | 0.5125 | 0.0667 | 0.1000 |

### Full Mean and Std Across Seeds
| Model | PR AUC mean | PR AUC std | Spearman mean | Spearman std | Top-5 mean | Top-5 std | Top-10 mean | Top-10 std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp_probe | 0.1771 | 0.0092 | 0.2493 | 0.0196 | 0.0222 | 0.0314 | 0.0778 | 0.0416 |
| linear_probe | 0.1766 | 0.0000 | 0.1951 | 0.0000 | 0.0667 | 0.0000 | 0.1667 | 0.0000 |
| position_baseline | 0.1625 | 0.0000 | 0.5125 | 0.0000 | 0.0667 | 0.0000 | 0.1000 | 0.0000 |

- Full best model by mean PR AUC: `mlp_probe`

### Full Tripwire Outcomes
| Run | Random-label near chance | Random-label PR AUC | Prevalence | Overfit can memorize | Overfit PR AUC |
|---|---|---:|---:|---|---:|
| seed_0 | True | 0.0960 | 0.1009 | False | 0.8994 |
| seed_1 | True | 0.0965 | 0.1009 | True | 0.9553 |
| seed_2 | True | 0.0970 | 0.1009 | True | 0.9381 |

### Full Extraction Failures
- No extraction failures were logged.

### Methodology Fidelity
- Planned and executed: resampling verification, span checks, pilot gate, full run, and three seeds.
- Planned and executed: position baseline, linear probe, and MLP probe.
- Planned and executed: problem-level train, validation, and test splits.

### Deviations
#### Minor Changes
- Verification uses fixed problem ID `330` for determinism.
- Full config uses `num_problems: 9999` to include all available IDs automatically.

#### Major Methodology Changes
1. Three-seed training and aggregation added.
2. Skip-and-log extraction policy added.
3. Pilot gate before full run added.
<!-- EXPERIMENT_RESULTS_END -->
