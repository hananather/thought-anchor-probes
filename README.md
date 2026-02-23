# Thought Anchor Probes

Train sentence probes that predict counterfactual sentence importance from model activations.

## What You'll Do
- Load the ARENA Thought Anchors rollout dataset from Hugging Face.
- Build sentence labels from `counterfactual_importance_accuracy`.
- Extract one-layer sentence embeddings with mean pooling (default) or token-level ragged spans.
- Optionally extract sentence-level vertical attention scores with depth control (`off` / `light` / `full`).
- Train position, text-only, activation-only, activation+position, and vertical-attention baselines.
- Train token-aware `attention_probe` and `multimax_probe` when `activations.pooling: tokens`.
- Simulate cascade/deferral thresholds to plan efficient future labeling.
- Report top-k recall and precision-recall AUC.

## Why This Matters
This setup gives a cheap first test for Thought Anchor signal.
But sparse token-local evidence can be diluted by sentence-level averaging on long contexts.
Token-aware aggregation lets probes focus on the few relevant tokens instead of washing them out.

## Multi-token Probe Motivation
- Mean pooling can erase sparse signal: if only a few tokens carry anchor evidence, averaging over many neutral tokens shrinks the signal.
- Attention probes score transformed token activations per head and compute a weighted sum of per-token values.
- MultiMax replaces softmax averaging with per-head hard max over tokens, reducing long-context dilution.
- This PR-level design follows `Building Production-Ready Probes For Gemini` and Neel Nanda's repeated warning that tokenization-level confounds make fragile single-token interpretations easy to overread.
- The receiver-head/vertical-attention framing is aligned with ARENA-style interpretability workflows and the Thought Anchors analysis.
- References:
  - Kramár et al. (2026), *Building Production-Ready Probes For Gemini*: https://arxiv.org/abs/2601.11516
  - Neel Nanda (2024), tokenization confound cautionary example: https://www.neelnanda.io/mechanistic-interpretability/emergent-pos
  - Neel Nanda interview (2025), probe-utility caution in deployment contexts: https://80000hours.org/podcast/episodes/neel-nanda-mechanistic-interpretability/
  - ARENA Chapter 1 (transformer interpretability tutorials): https://github.com/callummcdougall/ARENA_3.0/tree/main/chapter1_transformer_interp
  - *Thought Anchors: Which LLM Reasoning Steps Matter?* (receiver heads and vertical attention): https://openreview.net/pdf/e2eb4189bc2250be1718a88fcb63c2423b280109.pdf

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
- Estimate disk usage for extracted activations (including ragged token mode).
```bash
python scripts/estimate_storage.py \
  --shape-json artifacts/runs/pilot/sentence_embeddings_shape.json \
  --metadata-parquet artifacts/runs/pilot/sentence_metadata.parquet
```

## MacBook Tips
- Start with `num_problems: 25` in `configs/experiment.yaml`.
- Keep `layer_mode: mid` and `pooling: mean` for low-memory runs.
- Use `device: auto` to select MPS when available.
- Avoid loading full rollouts except one verification problem.

## Output Files
- `artifacts/runs/pilot/sentence_embeddings.dat`
- `artifacts/runs/pilot/token_embeddings.dat` (when `activations.pooling: tokens`)
- `artifacts/runs/pilot/metrics_seed_*.json`
- `artifacts/runs/pilot/aggregate_metrics.json`
- `artifacts/runs/full/sentence_embeddings.dat`
- `artifacts/runs/full/token_embeddings.dat` (when `activations.pooling: tokens`)
- `artifacts/runs/full/metrics_seed_*.json`
- `artifacts/runs/full/aggregate_metrics.json`

## Cache Safety Invariants
- `scripts/train_probes.py` validates extraction-time provenance before fitting probes.
- If you change label or activation config (for example `labels.anchor_percentile`, `labels.drop_last_chunk`, or `activations.model_name_or_path`), re-run `scripts/extract_embeddings.py`.
- Training now fails fast when split problem IDs are missing from extracted metadata.
- `scripts/extract_embeddings.py --reuse-cache` skips extraction only when cache provenance matches the active config.
- Implementation details for maintainers: `docs/cache_provenance_guardrail.md`.

## Vertical Attention Baseline
- Vertical score definition: aggregate how strongly later tokens attend to each sentence span.
- `depth_control: true` normalizes by the number of later queries so early sentences are not over-favored just because they have more future tokens.
- Modes:
  - `light`: attention from only the final `N` reasoning tokens.
  - `full`: full sentence-to-sentence attention when `seq_len <= full_max_seq_len`; otherwise light fallback.
- Config snippet:
```yaml
activations:
  vertical_attention:
    mode: light
    depth_control: true
    light_last_n_tokens: 4
    full_max_seq_len: 1024
```
- When vertical scores are extracted, training adds:
  - `vertical_attention_baseline`
  - `vertical_attention_plus_position`

## Deferral Planning
- `scripts/plan_deferrals.py` tunes two thresholds on validation scores:
  - `score <= t_neg` => confident negative
  - `score >= t_pos` => confident positive
  - otherwise => defer
- Objectives:
  - `budget`: minimize accepted-set error subject to a max deferral budget.
  - `error`: minimize deferral while meeting a target accepted-set error.
- `scripts/train_probes.py` predictions now include `split` (`val` / `test`) so one parquet can drive tuning + projection directly.
- Example:
```bash
python scripts/plan_deferrals.py \
  --predictions artifacts/scaling/qwen_correct/predictions_seed_0.parquet \
  --score-column score_linear_probe \
  --objective budget \
  --deferral-budget 0.20 \
  --output artifacts/scaling/qwen_correct/deferral_plan_seed_0.json
```

## High-Confidence Scaling Matrix
- 4-setting configs are available under `configs/scaling_*.yaml`.
- Primary paper model for reporting: `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`.
- Primary metric is target-mode aware for selection and LOPO deltas:
  - `anchor_binary` -> `pr_auc`
  - `importance_abs` / `importance_signed` -> `spearman`
- Default matrix run executes the full four-setting replication grid:
```bash
python scripts/run_scaling_grid.py --seeds 0 1 2 3 4 --no-reuse-cache
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
  configs/scaling_qwen_incorrect.yaml \
  --seeds 0 1 2 3 4 --no-reuse-cache
```
- RunPod memory: use at least 48 GB VRAM for stable 14B extraction runs.
- RunPod dependency contract (for reproducible VRAM/runtime behavior):
```bash
uv pip install -e . --system -c constraints/runpod.txt
python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
```
- Expected preflight versions: `2.9.1 4.57.3`.
- RunPod execution guide: `docs/runpod_scaling_runbook.md`.

## Systematic Sweep Campaign
- For large GPU-backed campaigns, use the staged sweep tooling:
  - `scripts/generate_sweep_configs.py`: generate Stage-1/2/3 config manifests.
  - `scripts/run_systematic_sweep.py`: execute manifest entries with retry + resume.
  - `scripts/summarize_sweep.py`: produce global leaderboards and shortlist files.
- Suggested flow:
```bash
python scripts/generate_sweep_configs.py \
  --base-config configs/scaling_qwen_correct.yaml \
  --sweep-root artifacts/sweeps/qwen_ultra \
  --stage stage1 --num-layers 40 \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
  --reuse-cache --skip-failed

python scripts/run_systematic_sweep.py \
  --sweep-root artifacts/sweeps/qwen_ultra \
  --manifest artifacts/sweeps/qwen_ultra/manifest_stage1.jsonl \
  --retries 1 --continue-on-error

python scripts/summarize_sweep.py \
  --sweep-root artifacts/sweeps/qwen_ultra \
  --stage2-top-n-per-target-mode 24 \
  --stage3-top-n 12
```
- Detailed runbook: `docs/systematic_sweep_runbook.md`.
- Current execution snapshot and handoff notes: `docs/repo_status_2026-02-23.md`.

## Repo Layout
- `src/ta_probe/data_loading.py`: dataset listing and fast metadata loading.
- `src/ta_probe/labels.py`: anchor labels from counterfactual scores.
- `src/ta_probe/spans.py`: chunking and sentence token boundaries.
- `src/ta_probe/activations.py`: one-layer hooks, pooled embeddings, and ragged token storage.
- `src/ta_probe/models.py`: sklearn baselines and mean-pooled probes.
- `src/ta_probe/token_probes.py`: torch attention and MultiMax token-level probes.
- `src/ta_probe/deferrals.py`: threshold tuning and projection for deferral simulation.
- `src/ta_probe/storage.py`: activation cache disk-usage estimation helpers.
- `src/ta_probe/train.py`: training, evaluation, and tripwire checks.
- `src/ta_probe/aggregate.py`: multi-seed metric aggregation.
- `src/ta_probe/readme_update.py`: deterministic README marker updates.
- `scripts/run_experiments.py`: end-to-end pilot + full orchestration.
- `scripts/run_scaling_grid.py`: four-setting scaling matrix orchestration.
- `scripts/generate_sweep_configs.py`: Stage-1/2/3 config and manifest generation.
- `scripts/run_systematic_sweep.py`: resumable execution over generated sweep manifests.
- `scripts/summarize_sweep.py`: sweep-wide leaderboard, deltas, stability, and shortlists.
- `scripts/plan_deferrals.py`: cascade/deferral simulation from saved prediction scores.
- `scripts/estimate_storage.py`: disk usage estimates from extracted metadata.
- `docs/why_mean_pooling_can_erase_sparse_signal.md`: mean-pooling dilution and MultiMax rationale.
- `docs/systematic_sweep_runbook.md`: operations guide for large staged Qwen sweeps.
- `docs/repo_status_2026-02-23.md`: current implementation/runtime status snapshot for replanning.
- `tests/`: unit tests for spans, labels, and metrics.

## Experiment Results

<!-- EXPERIMENT_RESULTS_START -->
Last updated: 2026-02-23 14:00:31

### Objective
Run the full Thought Anchor probe plan with pilot and full stages.

### Environment
- Host: 390f5efa15ed
- Platform: Linux-6.8.0-63-generic-x86_64-with-glibc2.35
- Python: 3.11.10
- Dataset listing date context: February 22, 2026

### Commands Executed
- `python -m ruff check .` | exit=0 | 0.09s
- `python -m pytest -q` | exit=0 | 28.81s
- `python scripts/build_problem_index.py --config configs/experiment.yaml --refresh` | exit=0 | 0.88s
- `python scripts/verify_problem_labels.py --config configs/experiment.yaml --problem-id 330` | exit=0 | 8.62s
- `python scripts/check_spans.py --config configs/experiment.yaml --problem-id 330 --sample-size 20` | exit=0 | 21.86s
- `python scripts/build_problem_index.py --config configs/experiment_pilot.yaml --refresh` | exit=0 | 0.93s
- `python scripts/build_problem_index.py --config configs/experiment_full.yaml --refresh` | exit=0 | 1.03s
- `python scripts/extract_embeddings.py --config configs/experiment_pilot.yaml --skip-failed --failure-log /workspace/thought-anchor-probes/artifacts/runs/pilot/extraction_failures.json --reuse-cache` | exit=0 | 46.45s
- `python scripts/train_probes.py --config configs/experiment_pilot.yaml --seed 0 --run-name seed_0` | exit=0 | 17.39s
- `python scripts/train_probes.py --config configs/experiment_pilot.yaml --seed 1 --run-name seed_1` | exit=0 | 17.72s
- `python scripts/train_probes.py --config configs/experiment_pilot.yaml --seed 2 --run-name seed_2` | exit=0 | 18.59s
- `python scripts/aggregate_runs.py --run-root /workspace/thought-anchor-probes/artifacts/runs/pilot` | exit=0 | 1.57s
- `python scripts/extract_embeddings.py --config configs/experiment_full.yaml --skip-failed --failure-log /workspace/thought-anchor-probes/artifacts/runs/full/extraction_failures.json --reuse-cache` | exit=0 | 59.74s
- `python scripts/train_probes.py --config configs/experiment_full.yaml --seed 0 --run-name seed_0` | exit=0 | 28.56s
- `python scripts/train_probes.py --config configs/experiment_full.yaml --seed 1 --run-name seed_1` | exit=0 | 29.87s
- `python scripts/train_probes.py --config configs/experiment_full.yaml --seed 2 --run-name seed_2` | exit=0 | 28.84s
- `python scripts/aggregate_runs.py --run-root /workspace/thought-anchor-probes/artifacts/runs/full` | exit=0 | 1.80s

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
| seed_0 | 0 | activations_plus_position | 0.1450 | 0.1305 | 0.0000 | 0.0000 |
| seed_0 | 0 | linear_probe | 0.1476 | 0.1347 | 0.0000 | 0.0000 |
| seed_0 | 0 | mlp_probe | 0.1581 | 0.2709 | 0.0000 | 0.0000 |
| seed_0 | 0 | position_baseline | 0.1415 | 0.4686 | 0.0000 | 0.1000 |
| seed_0 | 0 | text_only_baseline | 0.0955 | -0.0274 | 0.0000 | 0.0000 |
| seed_1 | 1 | activations_plus_position | 0.1450 | 0.1305 | 0.0000 | 0.0000 |
| seed_1 | 1 | linear_probe | 0.1476 | 0.1347 | 0.0000 | 0.0000 |
| seed_1 | 1 | mlp_probe | 0.0999 | 0.0941 | 0.0000 | 0.0000 |
| seed_1 | 1 | position_baseline | 0.1415 | 0.4686 | 0.0000 | 0.1000 |
| seed_1 | 1 | text_only_baseline | 0.0955 | -0.0274 | 0.0000 | 0.0000 |
| seed_2 | 2 | activations_plus_position | 0.1450 | 0.1305 | 0.0000 | 0.0000 |
| seed_2 | 2 | linear_probe | 0.1476 | 0.1347 | 0.0000 | 0.0000 |
| seed_2 | 2 | mlp_probe | 0.1849 | 0.2125 | 0.2000 | 0.2000 |
| seed_2 | 2 | position_baseline | 0.1415 | 0.4686 | 0.0000 | 0.1000 |
| seed_2 | 2 | text_only_baseline | 0.0955 | -0.0274 | 0.0000 | 0.0000 |

### Pilot Mean and Std Across Seeds
| Model | PR AUC mean | PR AUC std | Spearman mean | Spearman std | Top-5 mean | Top-5 std | Top-10 mean | Top-10 std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp_probe | 0.1476 | 0.0355 | 0.1925 | 0.0735 | 0.0667 | 0.0943 | 0.0667 | 0.0943 |
| linear_probe | 0.1476 | 0.0000 | 0.1347 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| activations_plus_position | 0.1450 | 0.0000 | 0.1305 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| position_baseline | 0.1415 | 0.0000 | 0.4686 | 0.0000 | 0.0000 | 0.0000 | 0.1000 | 0.0000 |
| text_only_baseline | 0.0955 | 0.0000 | -0.0274 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

- Pilot best model by mean PR AUC: `mlp_probe`

### Pilot Tripwire Outcomes
| Run | Random-label near chance | Random-label PR AUC | Prevalence | Overfit can memorize | Overfit PR AUC |
|---|---|---:|---:|---|---:|
| seed_0 | True | 0.0877 | 0.1003 | False | 0.7528 |
| seed_1 | True | 0.0844 | 0.1003 | False | 0.7277 |
| seed_2 | True | 0.0968 | 0.1003 | False | 0.8439 |

### Pilot Bootstrap CI Summary
| Comparison | Point delta mean | Point delta std | Bootstrap delta mean | Bootstrap delta std | Seeds CI excludes 0 | Seeds |
|---|---:|---:|---:|---:|---:|---:|
| score_activations_plus_position_minus_score_position_baseline | 0.0035 | 0.0000 | 0.0035 | 0.0000 | 3 | 3 |
| score_activations_plus_position_minus_score_text_only_baseline | 0.0496 | 0.0000 | 0.0496 | 0.0000 | 3 | 3 |

### Pilot Extraction Failures
- No extraction failures were logged.

### Full Per-Seed Test Metrics
| Run | Seed | Model | PR AUC | Spearman | Top-5 | Top-10 |
|---|---:|---|---:|---:|---:|---:|
| seed_0 | 0 | activations_plus_position | 0.1676 | 0.2245 | 0.0667 | 0.1333 |
| seed_0 | 0 | linear_probe | 0.1680 | 0.2248 | 0.0667 | 0.1667 |
| seed_0 | 0 | mlp_probe | 0.1537 | 0.2600 | 0.0000 | 0.1000 |
| seed_0 | 0 | position_baseline | 0.1625 | 0.5125 | 0.0667 | 0.1000 |
| seed_0 | 0 | text_only_baseline | 0.1023 | -0.0128 | 0.0667 | 0.0667 |
| seed_1 | 1 | activations_plus_position | 0.1676 | 0.2245 | 0.0667 | 0.1333 |
| seed_1 | 1 | linear_probe | 0.1680 | 0.2248 | 0.0667 | 0.1667 |
| seed_1 | 1 | mlp_probe | 0.1368 | 0.1951 | 0.0667 | 0.0667 |
| seed_1 | 1 | position_baseline | 0.1625 | 0.5125 | 0.0667 | 0.1000 |
| seed_1 | 1 | text_only_baseline | 0.1023 | -0.0128 | 0.0667 | 0.0667 |
| seed_2 | 2 | activations_plus_position | 0.1676 | 0.2245 | 0.0667 | 0.1333 |
| seed_2 | 2 | linear_probe | 0.1680 | 0.2248 | 0.0667 | 0.1667 |
| seed_2 | 2 | mlp_probe | 0.1940 | 0.3188 | 0.0667 | 0.1000 |
| seed_2 | 2 | position_baseline | 0.1625 | 0.5125 | 0.0667 | 0.1000 |
| seed_2 | 2 | text_only_baseline | 0.1023 | -0.0128 | 0.0667 | 0.0667 |

### Full Mean and Std Across Seeds
| Model | PR AUC mean | PR AUC std | Spearman mean | Spearman std | Top-5 mean | Top-5 std | Top-10 mean | Top-10 std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| linear_probe | 0.1680 | 0.0000 | 0.2248 | 0.0000 | 0.0667 | 0.0000 | 0.1667 | 0.0000 |
| activations_plus_position | 0.1676 | 0.0000 | 0.2245 | 0.0000 | 0.0667 | 0.0000 | 0.1333 | 0.0000 |
| position_baseline | 0.1625 | 0.0000 | 0.5125 | 0.0000 | 0.0667 | 0.0000 | 0.1000 | 0.0000 |
| mlp_probe | 0.1615 | 0.0240 | 0.2580 | 0.0505 | 0.0444 | 0.0314 | 0.0889 | 0.0157 |
| text_only_baseline | 0.1023 | 0.0000 | -0.0128 | 0.0000 | 0.0667 | 0.0000 | 0.0667 | 0.0000 |

- Full best model by mean PR AUC: `linear_probe`

### Full Tripwire Outcomes
| Run | Random-label near chance | Random-label PR AUC | Prevalence | Overfit can memorize | Overfit PR AUC |
|---|---|---:|---:|---|---:|
| seed_0 | True | 0.0765 | 0.1009 | False | 0.8994 |
| seed_1 | True | 0.0788 | 0.1009 | True | 0.9520 |
| seed_2 | True | 0.1013 | 0.1009 | True | 0.9450 |

### Full Bootstrap CI Summary
| Comparison | Point delta mean | Point delta std | Bootstrap delta mean | Bootstrap delta std | Seeds CI excludes 0 | Seeds |
|---|---:|---:|---:|---:|---:|---:|
| score_activations_plus_position_minus_score_position_baseline | 0.0052 | 0.0000 | 0.0072 | 0.0005 | 0 | 3 |
| score_activations_plus_position_minus_score_text_only_baseline | 0.0653 | 0.0000 | 0.0642 | 0.0004 | 0 | 3 |

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

<!-- SCALING_RESULTS_START -->
# Scaling Matrix Summary

- Shared problems: 20
- Shared splits: train=14, val=3, test=3

| Setting | Best model | Primary metric | Mean primary metric (best) | CI rows | Storage estimate |
|---|---|---|---:|---:|---:|
| deepseek-r1-distill-llama-8b__correct_base_solution | activations_plus_position | pr_auc | 0.1684 | 10 | 31.85 MiB |
| deepseek-r1-distill-llama-8b__incorrect_base_solution | position_baseline | pr_auc | 0.1892 | 10 | 27.71 MiB |
| deepseek-r1-distill-qwen-14b__correct_base_solution | position_baseline | pr_auc | 0.1801 | 10 | 29.17 MiB |
| deepseek-r1-distill-qwen-14b__incorrect_base_solution | mlp_probe | pr_auc | 0.1358 | 10 | 34.84 MiB |
<!-- SCALING_RESULTS_END -->

## Latest Release Gate Status (2026-02-23)
- Bundle: `artifacts/logs/20260223_135420`
- Verdict: `NOT OK` (tripwire invariant failures)
- Stage exits: `ruff_check=0`, `pytest=0`, `run_experiments=0`, `run_scaling_grid=0`
- Tripwire summary:
  - Random-label near-chance check passed `26/26` seed runs.
  - Overfit-one-problem memorization check passed `15/26` and failed `11/26`.
- Extraction failures: `0` across pilot, full, and all four scaling settings.
- Detailed reports:
  - `artifacts/logs/20260223_135420/RUN_SUMMARY.md`
  - `artifacts/logs/20260223_135420/COMPREHENSIVE_REVIEW_REPORT.md`
- Current release blocker: stabilize or recalibrate `overfit_one_problem_test.can_memorize` in `src/ta_probe/train.py`.
