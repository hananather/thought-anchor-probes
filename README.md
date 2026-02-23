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
- Default matrix run executes the full four-setting replication grid:
```bash
python scripts/run_scaling_grid.py
```
- This default runs:
  - `configs/scaling_llama_correct.yaml`
  - `configs/scaling_llama_incorrect.yaml`
  - `configs/scaling_qwen_correct.yaml`
  - `configs/scaling_qwen_incorrect.yaml`
- To run only Qwen-14B settings, pass configs explicitly:
```bash
python scripts/run_scaling_grid.py \
  --configs \
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
Last updated: 2026-02-23 01:24:26

### Objective
Run the full Thought Anchor probe plan with pilot and full stages.

### Environment
- Host: 5dcc8e944ec4
- Platform: Linux-6.8.0-60-generic-x86_64-with-glibc2.35
- Python: 3.11.10
- Dataset listing date context: February 22, 2026

### Commands Executed
- `python -m ruff check .` | exit=0 | 0.12s
- `python -m pytest -q` | exit=0 | 8.12s
- `python scripts/build_problem_index.py --config configs/experiment.yaml --refresh` | exit=0 | 0.59s
- `python scripts/verify_problem_labels.py --config configs/experiment.yaml --problem-id 330` | exit=0 | 7.48s
- `python scripts/check_spans.py --config configs/experiment.yaml --problem-id 330 --sample-size 20` | exit=0 | 7.07s
- `python scripts/build_problem_index.py --config configs/experiment_pilot.yaml --refresh` | exit=0 | 0.60s
- `python scripts/build_problem_index.py --config configs/experiment_full.yaml --refresh` | exit=0 | 0.60s
- `python scripts/extract_embeddings.py --config configs/experiment_pilot.yaml --skip-failed --failure-log /workspace/thought-anchor-probes/artifacts/runs/pilot/extraction_failures.json --reuse-cache` | exit=0 | 33.27s
- `python scripts/train_probes.py --config configs/experiment_pilot.yaml --seed 0 --run-name seed_0` | exit=0 | 10.95s
- `python scripts/train_probes.py --config configs/experiment_pilot.yaml --seed 1 --run-name seed_1` | exit=0 | 11.10s
- `python scripts/train_probes.py --config configs/experiment_pilot.yaml --seed 2 --run-name seed_2` | exit=0 | 11.31s
- `python scripts/aggregate_runs.py --run-root /workspace/thought-anchor-probes/artifacts/runs/pilot` | exit=0 | 0.67s
- `python scripts/extract_embeddings.py --config configs/experiment_full.yaml --skip-failed --failure-log /workspace/thought-anchor-probes/artifacts/runs/full/extraction_failures.json --reuse-cache` | exit=0 | 37.97s
- `python scripts/train_probes.py --config configs/experiment_full.yaml --seed 0 --run-name seed_0` | exit=0 | 25.87s
- `python scripts/train_probes.py --config configs/experiment_full.yaml --seed 1 --run-name seed_1` | exit=0 | 24.72s
- `python scripts/train_probes.py --config configs/experiment_full.yaml --seed 2 --run-name seed_2` | exit=0 | 25.08s
- `python scripts/aggregate_runs.py --run-root /workspace/thought-anchor-probes/artifacts/runs/full` | exit=0 | 0.63s

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
| seed_0 | 0 | activations_plus_position | 0.1472 | 0.1386 | 0.0000 | 0.0000 |
| seed_0 | 0 | linear_probe | 0.1478 | 0.1380 | 0.0000 | 0.0000 |
| seed_0 | 0 | mlp_probe | 0.1565 | 0.2705 | 0.0000 | 0.0000 |
| seed_0 | 0 | position_baseline | 0.1415 | 0.4686 | 0.0000 | 0.1000 |
| seed_0 | 0 | text_only_baseline | 0.0955 | -0.0274 | 0.0000 | 0.0000 |
| seed_1 | 1 | activations_plus_position | 0.1472 | 0.1386 | 0.0000 | 0.0000 |
| seed_1 | 1 | linear_probe | 0.1478 | 0.1380 | 0.0000 | 0.0000 |
| seed_1 | 1 | mlp_probe | 0.0993 | 0.0944 | 0.0000 | 0.0000 |
| seed_1 | 1 | position_baseline | 0.1415 | 0.4686 | 0.0000 | 0.1000 |
| seed_1 | 1 | text_only_baseline | 0.0955 | -0.0274 | 0.0000 | 0.0000 |
| seed_2 | 2 | activations_plus_position | 0.1472 | 0.1386 | 0.0000 | 0.0000 |
| seed_2 | 2 | linear_probe | 0.1478 | 0.1380 | 0.0000 | 0.0000 |
| seed_2 | 2 | mlp_probe | 0.1866 | 0.2109 | 0.2000 | 0.2000 |
| seed_2 | 2 | position_baseline | 0.1415 | 0.4686 | 0.0000 | 0.1000 |
| seed_2 | 2 | text_only_baseline | 0.0955 | -0.0274 | 0.0000 | 0.0000 |

### Pilot Mean and Std Across Seeds
| Model | PR AUC mean | PR AUC std | Spearman mean | Spearman std | Top-5 mean | Top-5 std | Top-10 mean | Top-10 std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| linear_probe | 0.1478 | 0.0000 | 0.1380 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| mlp_probe | 0.1475 | 0.0362 | 0.1919 | 0.0731 | 0.0667 | 0.0943 | 0.0667 | 0.0943 |
| activations_plus_position | 0.1472 | 0.0000 | 0.1386 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| position_baseline | 0.1415 | 0.0000 | 0.4686 | 0.0000 | 0.0000 | 0.0000 | 0.1000 | 0.0000 |
| text_only_baseline | 0.0955 | 0.0000 | -0.0274 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

- Pilot best model by mean PR AUC: `linear_probe`

### Pilot Tripwire Outcomes
| Run | Random-label near chance | Random-label PR AUC | Prevalence | Overfit can memorize | Overfit PR AUC |
|---|---|---:|---:|---|---:|
| seed_0 | True | 0.0875 | 0.1003 | False | 0.7571 |
| seed_1 | True | 0.0841 | 0.1003 | False | 0.7274 |
| seed_2 | True | 0.0981 | 0.1003 | False | 0.8439 |

### Pilot Bootstrap CI Summary
| Comparison | Point delta mean | Point delta std | Bootstrap delta mean | Bootstrap delta std | Seeds CI excludes 0 | Seeds |
|---|---:|---:|---:|---:|---:|---:|
| score_activations_plus_position_minus_score_position_baseline | 0.0057 | 0.0000 | 0.0057 | 0.0000 | 3 | 3 |
| score_activations_plus_position_minus_score_text_only_baseline | 0.0518 | 0.0000 | 0.0518 | 0.0000 | 3 | 3 |

### Pilot Extraction Failures
- No extraction failures were logged.

### Full Per-Seed Test Metrics
| Run | Seed | Model | PR AUC | Spearman | Top-5 | Top-10 |
|---|---:|---|---:|---:|---:|---:|
| seed_0 | 0 | activations_plus_position | 0.1679 | 0.2286 | 0.0667 | 0.1333 |
| seed_0 | 0 | linear_probe | 0.1670 | 0.2261 | 0.0667 | 0.1667 |
| seed_0 | 0 | mlp_probe | 0.1523 | 0.2560 | 0.0000 | 0.0667 |
| seed_0 | 0 | position_baseline | 0.1625 | 0.5125 | 0.0667 | 0.1000 |
| seed_0 | 0 | text_only_baseline | 0.1032 | -0.0135 | 0.0667 | 0.0667 |
| seed_1 | 1 | activations_plus_position | 0.1679 | 0.2286 | 0.0667 | 0.1333 |
| seed_1 | 1 | linear_probe | 0.1670 | 0.2261 | 0.0667 | 0.1667 |
| seed_1 | 1 | mlp_probe | 0.1354 | 0.1917 | 0.0667 | 0.0667 |
| seed_1 | 1 | position_baseline | 0.1625 | 0.5125 | 0.0667 | 0.1000 |
| seed_1 | 1 | text_only_baseline | 0.1032 | -0.0135 | 0.0667 | 0.0667 |
| seed_2 | 2 | activations_plus_position | 0.1679 | 0.2286 | 0.0667 | 0.1333 |
| seed_2 | 2 | linear_probe | 0.1670 | 0.2261 | 0.0667 | 0.1667 |
| seed_2 | 2 | mlp_probe | 0.1911 | 0.3234 | 0.0667 | 0.1000 |
| seed_2 | 2 | position_baseline | 0.1625 | 0.5125 | 0.0667 | 0.1000 |
| seed_2 | 2 | text_only_baseline | 0.1032 | -0.0135 | 0.0667 | 0.0667 |

### Full Mean and Std Across Seeds
| Model | PR AUC mean | PR AUC std | Spearman mean | Spearman std | Top-5 mean | Top-5 std | Top-10 mean | Top-10 std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| activations_plus_position | 0.1679 | 0.0000 | 0.2286 | 0.0000 | 0.0667 | 0.0000 | 0.1333 | 0.0000 |
| linear_probe | 0.1670 | 0.0000 | 0.2261 | 0.0000 | 0.0667 | 0.0000 | 0.1667 | 0.0000 |
| position_baseline | 0.1625 | 0.0000 | 0.5125 | 0.0000 | 0.0667 | 0.0000 | 0.1000 | 0.0000 |
| mlp_probe | 0.1596 | 0.0233 | 0.2570 | 0.0538 | 0.0444 | 0.0314 | 0.0778 | 0.0157 |
| text_only_baseline | 0.1032 | 0.0000 | -0.0135 | 0.0000 | 0.0667 | 0.0000 | 0.0667 | 0.0000 |

- Full best model by mean PR AUC: `activations_plus_position`

### Full Tripwire Outcomes
| Run | Random-label near chance | Random-label PR AUC | Prevalence | Overfit can memorize | Overfit PR AUC |
|---|---|---:|---:|---|---:|
| seed_0 | True | 0.0767 | 0.1009 | True | 0.9029 |
| seed_1 | True | 0.0788 | 0.1009 | True | 0.9553 |
| seed_2 | True | 0.1016 | 0.1009 | True | 0.9495 |

### Full Bootstrap CI Summary
| Comparison | Point delta mean | Point delta std | Bootstrap delta mean | Bootstrap delta std | Seeds CI excludes 0 | Seeds |
|---|---:|---:|---:|---:|---:|---:|
| score_activations_plus_position_minus_score_position_baseline | 0.0055 | 0.0000 | 0.0074 | 0.0005 | 0 | 3 |
| score_activations_plus_position_minus_score_text_only_baseline | 0.0647 | 0.0000 | 0.0635 | 0.0004 | 3 | 3 |

### Full Extraction Failures
- No extraction failures were logged.

### Methodology Fidelity
- Planned and executed: resampling verification, span checks, pilot gate, full run, and three seeds.
- Planned and executed: position baseline, text-only baseline, linear probe, MLP probe, and activations+position probe.
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

## Scaling Grid Results

Four-setting replication matrix run on an NVIDIA L40S (44.4 GB VRAM), 5 seeds per setting, 20 shared problems (train=14, val=3, test=3).

### Summary

| Setting | Best Model | Mean PR-AUC | Spearman |
|---|---|---:|---:|
| Llama-8B correct | mlp_probe | 0.1682 | 0.2545 |
| Llama-8B incorrect | position_baseline | 0.1892 | 0.2155 |
| Qwen-14B correct | position_baseline | 0.1801 | 0.4799 |
| Qwen-14B incorrect | mlp_probe | 0.1411 | 0.1520 |

### Per-Setting Model Comparison (Test Set, 5-seed mean)

#### Llama-8B Correct
| Model | PR-AUC | std | Spearman |
|---|---:|---:|---:|
| mlp_probe | 0.1682 | 0.0202 | 0.2545 |
| activations_plus_position | 0.1679 | 0.0000 | 0.2250 |
| linear_probe | 0.1664 | 0.0000 | 0.2157 |
| position_baseline | 0.1625 | 0.0000 | 0.5125 |
| text_only_baseline | 0.1032 | 0.0000 | -0.0135 |

#### Llama-8B Incorrect
| Model | PR-AUC | std | Spearman |
|---|---:|---:|---:|
| position_baseline | 0.1892 | 0.0000 | 0.2155 |
| activations_plus_position | 0.1319 | 0.0000 | -0.0171 |
| linear_probe | 0.1272 | 0.0000 | -0.0330 |
| mlp_probe | 0.1265 | 0.0122 | 0.0573 |
| text_only_baseline | 0.0998 | 0.0000 | -0.0367 |

#### Qwen-14B Correct
| Model | PR-AUC | std | Spearman |
|---|---:|---:|---:|
| position_baseline | 0.1801 | 0.0000 | 0.4799 |
| mlp_probe | 0.1436 | 0.0272 | 0.2601 |
| activations_plus_position | 0.1190 | 0.0000 | 0.1773 |
| linear_probe | 0.1187 | 0.0000 | 0.1703 |
| text_only_baseline | 0.0961 | 0.0000 | 0.0876 |

#### Qwen-14B Incorrect
| Model | PR-AUC | std | Spearman |
|---|---:|---:|---:|
| mlp_probe | 0.1411 | 0.0083 | 0.1520 |
| linear_probe | 0.1249 | 0.0000 | 0.0591 |
| activations_plus_position | 0.1249 | 0.0000 | 0.0594 |
| text_only_baseline | 0.1186 | 0.0000 | 0.0960 |
| position_baseline | 0.1107 | 0.0000 | 0.1379 |

### Bootstrap CI: activations_plus_position vs Baselines

| Setting | vs position_baseline (delta) | CI excludes 0 | vs text_only_baseline (delta) | CI excludes 0 |
|---|---:|---|---:|---|
| Llama-8B correct | +0.0070 | 0/5 seeds | +0.0628 | 5/5 seeds |
| Llama-8B incorrect | -0.0560 | 5/5 seeds | +0.0289 | 0/5 seeds |
| Qwen-14B correct | -0.0768 | 0/5 seeds | +0.0240 | 0/5 seeds |
| Qwen-14B incorrect | +0.0116 | 5/5 seeds | +0.0133 | 0/5 seeds |
