# Thought Anchor Probes: Comprehensive Run Review

Generated on: 2026-02-23 (UTC)  
Repo root: `/workspace/thought-anchor-probes`  
Primary run bundle: `artifacts/logs/20260223_135420`

## 1. Executive Summary

- The pipeline **did run end-to-end** for experiments + scaling:
  - `run_experiments` exit `0`
  - `run_scaling_grid` exit `0`
- The release verdict is `NOT OK` due to a **post-run invariant failure**:
  - Tripwire: `overfit_one_problem_test.can_memorize` failed in 11 seed runs.
- Tripwire split:
  - Random-label sanity passed `26/26` seed runs.
  - Overfit-one-problem memorization passed `15/26`, failed `11/26`.
- Core outputs were produced:
  - Pilot/full/scaling artifacts and aggregate metrics exist.
  - Extraction failures are all zero.
  - Primary-metric summaries are finite (no NaN in aggregate model summaries).
  - README experiment and scaling marker blocks were updated.

Bottom line: this is a **quality-gate failure**, not a crash or missing-artifact failure.

## 2. Request-to-Checklist Mapping

Original execution checklist status:

1. Install dependencies (RunPod constraints + dev deps): **DONE**
   - Installed in `.venv` with `constraints/runpod.txt`.
   - Confirmed `ruff` and `pytest` available.
2. Confirm versions (`torch`, `transformers`, `nvidia-smi`): **DONE**
   - `torch=2.9.1+cu128`, `transformers=4.57.3`
   - GPU visible (RTX 6000 Ada, CUDA 12.8)
3. Run `python scripts/run_release.py`: **DONE**
   - Canonical run: `artifacts/logs/20260223_135420`
4. Audit `RUN_SUMMARY.md` invariants: **DONE**
   - Stage exits, artifacts, metrics finite, README markers checked manually.
5. Emit digest: **DONE**
   - Previously provided; expanded in this report.

Fail-fast conditions:

- `ruff` / `pytest` failure: **NOT triggered**.
- Extraction failures non-empty: **NOT triggered** (all zero).
- NaN primary metrics in best-model summaries: **NOT triggered**.
- Release-specific tripwire invariant: **TRIGGERED** (reason for final `NOT OK` verdict).

## 3. Codebase Overview

### 3.1 Purpose

This repository trains sentence-level probes to predict Thought Anchor relevance from model activations.  
Reference: `README.md`.

### 3.2 Pipeline Scripts (entrypoints)

From `scripts/*.py` docstrings:

- `scripts/run_release.py`: strict release harness (lint, tests, experiments, scaling, invariants, logs).
- `scripts/run_experiments.py`: pilot + full orchestration plus README experiment results update.
- `scripts/run_scaling_grid.py`: 4-setting scaling matrix orchestration plus README scaling update.
- `scripts/extract_embeddings.py`: activation extraction and cache writing.
- `scripts/train_probes.py`: seed-level training/eval/tripwires.
- `scripts/aggregate_runs.py`: multi-seed aggregate metrics.

Other utility scripts include span checks, storage estimation, deferral planning, LOPO CV, and verification.

### 3.3 Core Library Modules

From `src/ta_probe/*.py` docstrings:

- `activations.py`: extraction/pooling/cache creation.
- `train.py`: training + evaluation + tripwire checks.
- `aggregate.py`: aggregate summaries.
- `config.py`: config models and loaders.
- `data_loading.py`: dataset listing, loading, split utilities.
- `metrics.py`, `models.py`, `token_probes.py`: model and metric internals.
- `readme_update.py`: deterministic marker block updates.

### 3.4 Test Coverage Snapshot

- Test files: `17`
- Test functions: `66`
- Test command: `python -m pytest -q` passed in release run.

## 4. Config + Data Context Used

### 4.1 Dataset and model context

- Dataset repo: `uzaymacar/math-rollouts`
- Sampling temp dir: `temperature_0.6_top_p_0.95`
- Primary paper model: Qwen-14B (`deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`)
- Optional model in scaling grid: Llama-8B (`deepseek-ai/DeepSeek-R1-Distill-Llama-8B`)

### 4.2 Config families used

- Experiments:
  - `configs/experiment_pilot.yaml` (`num_problems: 5`)
  - `configs/experiment_full.yaml` (`num_problems: 9999`, resolves to shared 20 IDs here)
- Scaling grid:
  - `configs/scaling_llama_correct.yaml`
  - `configs/scaling_llama_incorrect.yaml`
  - `configs/scaling_qwen_correct.yaml`
  - `configs/scaling_qwen_incorrect.yaml`

### 4.3 Effective split/problem IDs (from generated JSON)

- Shared IDs (full/scaling): `20` IDs in `data/problem_ids_scaling_shared.json`
- Shared split (`0.7/0.15/0.15`):
  - train: `14`
  - val: `3`
  - test: `3`
- Pilot:
  - IDs: `5`
  - split sizes: train `3`, val `1`, test `1`

## 5. Environment + Dependency Audit

### 5.1 Runtime snapshot (canonical run)

From `artifacts/logs/20260223_135420/*`:

- Git commit: `a653aa429e21b25a2ed23485356a7894ab38f353`
- Python: `3.11.10`
- Torch/Transformers: `2.9.1+cu128 / 4.57.3`
- GPU: `NVIDIA RTX 6000 Ada`, driver `570.124.06`, CUDA `12.8`

### 5.2 Key package versions

From `artifacts/logs/20260223_135420/pip_freeze.txt`:

- `torch==2.9.1`
- `transformers==4.57.3`
- `huggingface_hub==0.36.2`
- `pytest==9.0.2`
- `ruff==0.15.2`
- `sentence-transformers==5.2.3`

## 6. Execution Chronology (Important for Review)

Three release attempts were run:

1. `20260223_134421` (failed)
   - `run_experiments` failed due `httpx.ConnectError` during dataset listing refresh.
2. `20260223_134938` (failed)
   - `run_experiments` failed because nested `python` call resolved to `/usr/bin/python` without `ruff`.
3. `20260223_135420` (canonical full run)
   - `ruff` pass, `pytest` pass, `run_experiments` pass, `run_scaling_grid` pass.
   - Final verdict `NOT OK` due tripwire invariant failures.

Canonical run should be treated as the audited source of truth.

## 7. Canonical Run Details (`20260223_135420`)

### 7.1 Stage command results

From `artifacts/logs/20260223_135420/RUN_SUMMARY.md`:

- `ruff_check`: exit `0`, `0.23s`
- `pytest`: exit `0`, `35.08s`
- `run_experiments`: exit `0`, `313.18s`
- `run_scaling_grid`: exit `0`, `941.92s`

Approx stage total: `1290.41s` (~`21m 30s`).

### 7.2 Test/lint outputs

- Ruff: `All checks passed!`
- Pytest: all tests passed (`[100%]`) with warnings:
  - MLP convergence warning in one training-guard test.
  - Spearman constant-input warnings in one test.

### 7.3 Artifact existence checks

All required release artifacts exist for:

- pilot (`artifacts/runs/pilot/*`)
- full (`artifacts/runs/full/*`)
- scaling llama/qwen correct/incorrect (`artifacts/scaling/*/*`)

No missing artifact failures.

### 7.4 Extraction-failure logs

`skipped_problem_ids` counts:

- pilot: `0`
- full: `0`
- scaling:llama_correct: `0`
- scaling:llama_incorrect: `0`
- scaling:qwen_correct: `0`
- scaling:qwen_incorrect: `0`

### 7.5 README marker update checks

Manual marker verification in `README.md`:

- Experiment markers present and block non-empty (`Last updated:` present).
- Scaling markers present and block non-empty (`# Scaling Matrix Summary` present).

## 8. Metrics Summary

### 8.1 Best model per setting (primary metric)

Primary metric is `pr_auc` for all active settings (`anchor_binary` mode):

- pilot: `mlp_probe` (`0.1476`)
- full: `linear_probe` (`0.1680`)
- scaling:llama_correct: `activations_plus_position` (`0.1684`)
- scaling:llama_incorrect: `position_baseline` (`0.1892`)
- scaling:qwen_correct: `position_baseline` (`0.1801`)
- scaling:qwen_incorrect: `mlp_probe` (`0.1358`)

### 8.2 Top-model ranking snapshots

- Pilot: top 3 by PR AUC mean
  - `mlp_probe` `0.1476`
  - `linear_probe` `0.1476` (very close)
  - `activations_plus_position` `0.1450`
- Full: top 3
  - `linear_probe` `0.1680`
  - `activations_plus_position` `0.1676`
  - `position_baseline` `0.1625`
- Scaling highlights
  - Llama/correct: `activations_plus_position` wins.
  - Llama/incorrect: `position_baseline` wins.
  - Qwen/correct: `position_baseline` wins.
  - Qwen/incorrect: `mlp_probe` wins.

### 8.3 Finite-metric audit

Manual audit of all `aggregate_metrics.json` summary rows found:

- No non-finite values (`NaN`/`inf`) in model-level primary metric means.

## 9. Tripwire Failure Analysis (Why Verdict is NOT OK)

### 9.1 Invariant in code

From `src/ta_probe/train.py`:

- `overfit_one_problem_test.can_memorize = (train_pr_auc >= 0.9)`
- Release harness treats `can_memorize=False` as hard failure.

### 9.2 What passed

- Random-label tripwire passed in all examined seed runs:
  - `near_chance` was `True` throughout.
  - Max `(random_pr_auc - prevalence)` observed was `+0.0172` (well below `+0.1` guardrail).

### 9.3 What failed

11 seed runs failed `can_memorize`:

- pilot: `3/3` failed
  - seed_0 `0.7528`
  - seed_1 `0.7277`
  - seed_2 `0.8439`
- full: `1/3` failed
  - seed_0 `0.8994`
- scaling:llama_correct: `1/5` failed
  - seed_3 `0.8264`
- scaling:llama_incorrect: `2/5` failed
  - seed_2 `0.7937`
  - seed_4 `0.4611`
- scaling:qwen_correct: `1/5` failed
  - seed_1 `0.8767`
- scaling:qwen_incorrect: `3/5` failed
  - seed_1 `0.5453`
  - seed_3 `0.8337`
  - seed_4 `0.7624`

### 9.4 Interpretation

- This pattern indicates instability in the one-problem memorization check under current settings/seeds.
- It does **not** indicate extraction failures or random-label leakage.
- Pilot is especially fragile (tiny split sizes), but failures also appear in full/scaling seeds.

### 9.5 Problem-Level Context Used by the Overfit Tripwire

The overfit tripwire does **not** sample multiple problems. It always tests the first training problem in
`train_frame` (see code pointers in 9.6). In this run, that yielded:

| Setting | Tripwire problem_id | Rows in selected problem | Positives | Negatives | Overfit failures |
|---|---:|---:|---:|---:|---:|
| pilot | 1591 | 100 | 10 | 90 | 3/3 |
| full | 330 | 169 | 17 | 152 | 1/3 |
| scaling:llama_correct | 330 | 169 | 17 | 152 | 1/5 |
| scaling:llama_incorrect | 330 | 25 | 3 | 22 | 2/5 |
| scaling:qwen_correct | 330 | 138 | 14 | 124 | 1/5 |
| scaling:qwen_incorrect | 330 | 230 | 23 | 207 | 3/5 |

Diagnostic implication:

- The selected overfit problem is consistently class-imbalanced (~10% positives).
- One setting (`scaling:llama_incorrect`) is especially tiny (`25` rows, `3` positives), which increases seed sensitivity.

### 9.6 Exact Code Pointers for the Failing Invariant

Tripwire implementation:

- Problem selection: `src/ta_probe/train.py:865`
  - `one_problem = int(train_frame["problem_id"].iloc[0])`
- Random-label criterion: `src/ta_probe/train.py:862`
  - `near_chance = chance_pr_auc <= prevalence + 0.1`
- Overfit criterion (hard gate): `src/ta_probe/train.py:879`
  - `can_memorize = train_pr_auc >= 0.9`

Tripwire model used:

- `make_mlp_probe(...)` in `src/ta_probe/models.py:99`
- Uses `MLPClassifier(..., early_stopping=True, max_iter=max_iter)` at `src/ta_probe/models.py:113`.

Release gate order:

- `scripts/run_release.py:828` runs `_validate_tripwires(...)` **before**
  - primary metric NaN validation (`scripts/run_release.py:829`),
  - README marker validation (`scripts/run_release.py:830`),
  - extraction-failure collection (`scripts/run_release.py:831`).

Consequence:

- A tripwire failure short-circuits subsequent sections in `RUN_SUMMARY.md` key-metrics/readme-check tables.
- Those checks were still performed manually in this report.

## 10. Artifact Trail (Review Entry Points)

Primary review files:

- Release summary: `artifacts/logs/20260223_135420/RUN_SUMMARY.md`
- Stage logs:
  - `artifacts/logs/20260223_135420/01_ruff_check.log`
  - `artifacts/logs/20260223_135420/02_pytest.log`
  - `artifacts/logs/20260223_135420/03_run_experiments.log`
  - `artifacts/logs/20260223_135420/04_run_scaling_grid.log`
- Environment snapshot:
  - `artifacts/logs/20260223_135420/pip_freeze.txt`
  - `artifacts/logs/20260223_135420/nvidia_smi.txt`
  - `artifacts/logs/20260223_135420/torch_transformers_versions.txt`
- Experiment outputs:
  - `artifacts/runs/pilot/aggregate_metrics.json`
  - `artifacts/runs/full/aggregate_metrics.json`
- Scaling outputs:
  - `artifacts/scaling/scaling_summary.md`
  - `artifacts/scaling/llama_correct/aggregate_metrics.json`
  - `artifacts/scaling/llama_incorrect/aggregate_metrics.json`
  - `artifacts/scaling/qwen_correct/aggregate_metrics.json`
  - `artifacts/scaling/qwen_incorrect/aggregate_metrics.json`
- Split/problem ID provenance:
  - `data/problem_ids_pilot.json`
  - `data/splits_pilot.json`
  - `data/problem_ids_full.json`
  - `data/splits_full.json`
  - `data/problem_ids_scaling_shared.json`
  - `data/splits_scaling_shared.json`

## 11. Disk Usage Snapshot

- `du -sh artifacts` -> `228M`
- `du -sh artifacts/logs/20260223_135420` -> `998K`
- Per major run roots:
  - `artifacts/runs/pilot`: `21M`
  - `artifacts/runs/full`: `65M`
  - `artifacts/scaling/llama_correct`: `35M`
  - `artifacts/scaling/llama_incorrect`: `31M`
  - `artifacts/scaling/qwen_correct`: `32M`
  - `artifacts/scaling/qwen_incorrect`: `38M`

## 12. Workspace Delta Produced by This Run

`git status --short` shows:

- `README.md` modified (experiment + scaling result blocks refreshed).
- Many files under `artifacts/runs/*` and `artifacts/scaling/*` modified.
- New logs under `artifacts/logs/` (including this report).
- `src/thought_anchor_probes.egg-info/PKG-INFO` updated from dependency install context.

## 13. Suggested Reviewer Focus

1. Tripwire policy vs. practical stability:
   - Is `train_pr_auc >= 0.9` appropriate for all seed/problem settings?
2. Pilot reliability:
   - Pilot’s `3/1/1` split is very small and appears unstable for overfit check.
3. Seed sensitivity:
   - Check failed seeds in full/scaling for reproducibility and whether deterministic controls need tightening.
4. Release gate behavior:
   - Current failure path prevents key-metrics table population in `RUN_SUMMARY.md` after tripwire exception; manual audit was required.
5. Environment assumptions:
   - Nested `python` calls in scripts rely on active virtualenv; keep activation explicit in automation wrappers.

## 14. Reproduction Commands (Canonical)

```bash
source .venv/bin/activate
python -m pip install -e '.[dev,verify]' -c constraints/runpod.txt
python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
nvidia-smi
python scripts/run_release.py
```

For strict review, use bundle:

- `artifacts/logs/20260223_135420`

## 15. Fast Diagnostic Playbook for Coworker

These commands give a minimal, high-signal path to diagnose the failure.

1. Confirm release failure reason

```bash
sed -n '1,220p' artifacts/logs/20260223_135420/RUN_SUMMARY.md
```

2. Inspect full scaling-stage payload

```bash
sed -n '1,260p' artifacts/logs/20260223_135420/04_run_scaling_grid.log
```

3. Print overfit tripwire outcomes by seed

```bash
python - <<'PY'
import json
from pathlib import Path
roots = [
    Path('artifacts/runs/pilot'),
    Path('artifacts/runs/full'),
    Path('artifacts/scaling/llama_correct'),
    Path('artifacts/scaling/llama_incorrect'),
    Path('artifacts/scaling/qwen_correct'),
    Path('artifacts/scaling/qwen_incorrect'),
]
for root in roots:
    print(f'== {root} ==')
    for p in sorted(root.glob('metrics_seed_*.json')):
        j = json.loads(p.read_text())
        over = j.get('tripwires', {}).get('overfit_one_problem_test', {})
        print(p.name, 'problem', over.get('problem_id'),
              'train_pr_auc', over.get('train_pr_auc'),
              'can_memorize', over.get('can_memorize'))
PY
```

4. Re-run a single failing seed for targeted debugging (example)

```bash
source .venv/bin/activate
python scripts/train_probes.py \
  --config configs/scaling_qwen_incorrect.yaml \
  --seed 1 \
  --run-name debug_seed_1
```

5. Compare with a passing seed in same setting

```bash
source .venv/bin/activate
python scripts/train_probes.py \
  --config configs/scaling_qwen_incorrect.yaml \
  --seed 2 \
  --run-name debug_seed_2
```

## 16. Standalone Results Appendix (No JSON Required)

This section mirrors the aggregate probe tables so a reviewer can work from this document alone.

### pilot

| Model | PR-AUC mean | PR-AUC std | Spearman mean | Top-5 recall | Top-10 recall |
|---|---:|---:|---:|---:|---:|
| mlp_probe | 0.1476 | 0.0355 | 0.1925 | 0.0667 | 0.0667 |
| linear_probe | 0.1476 | 0.0000 | 0.1347 | 0.0000 | 0.0000 |
| activations_plus_position | 0.1450 | 0.0000 | 0.1305 | 0.0000 | 0.0000 |
| position_baseline | 0.1415 | 0.0000 | 0.4686 | 0.0000 | 0.1000 |
| text_only_baseline | 0.0955 | 0.0000 | -0.0274 | 0.0000 | 0.0000 |

### full

| Model | PR-AUC mean | PR-AUC std | Spearman mean | Top-5 recall | Top-10 recall |
|---|---:|---:|---:|---:|---:|
| linear_probe | 0.1680 | 0.0000 | 0.2248 | 0.0667 | 0.1667 |
| activations_plus_position | 0.1676 | 0.0000 | 0.2245 | 0.0667 | 0.1333 |
| position_baseline | 0.1625 | 0.0000 | 0.5125 | 0.0667 | 0.1000 |
| mlp_probe | 0.1615 | 0.0240 | 0.2580 | 0.0444 | 0.0889 |
| text_only_baseline | 0.1023 | 0.0000 | -0.0128 | 0.0667 | 0.0667 |

### scaling_llama_correct

| Model | PR-AUC mean | PR-AUC std | Spearman mean | Top-5 recall | Top-10 recall |
|---|---:|---:|---:|---:|---:|
| activations_plus_position | 0.1684 | 0.0000 | 0.2302 | 0.0667 | 0.1333 |
| linear_probe | 0.1664 | 0.0000 | 0.2228 | 0.0667 | 0.1667 |
| mlp_probe | 0.1630 | 0.0178 | 0.2523 | 0.0533 | 0.0933 |
| position_baseline | 0.1625 | 0.0000 | 0.5125 | 0.0667 | 0.1000 |
| text_only_baseline | 0.1023 | 0.0000 | -0.0128 | 0.0667 | 0.0667 |

### scaling_llama_incorrect

| Model | PR-AUC mean | PR-AUC std | Spearman mean | Top-5 recall | Top-10 recall |
|---|---:|---:|---:|---:|---:|
| position_baseline | 0.1892 | 0.0000 | 0.2155 | 0.0000 | 0.1000 |
| activations_plus_position | 0.1336 | 0.0000 | -0.0141 | 0.0000 | 0.0333 |
| linear_probe | 0.1272 | 0.0000 | -0.0347 | 0.0000 | 0.0333 |
| mlp_probe | 0.1263 | 0.0124 | 0.0538 | 0.0667 | 0.0667 |
| text_only_baseline | 0.0998 | 0.0000 | -0.0367 | 0.0000 | 0.0000 |

### scaling_qwen_correct

| Model | PR-AUC mean | PR-AUC std | Spearman mean | Top-5 recall | Top-10 recall |
|---|---:|---:|---:|---:|---:|
| position_baseline | 0.1801 | 0.0000 | 0.4799 | 0.0667 | 0.0667 |
| mlp_probe | 0.1473 | 0.0230 | 0.2662 | 0.0533 | 0.1067 |
| activations_plus_position | 0.1188 | 0.0000 | 0.1765 | 0.0000 | 0.0333 |
| linear_probe | 0.1187 | 0.0000 | 0.1698 | 0.0000 | 0.0667 |
| text_only_baseline | 0.0961 | 0.0000 | 0.0876 | 0.0000 | 0.0333 |

### scaling_qwen_incorrect

| Model | PR-AUC mean | PR-AUC std | Spearman mean | Top-5 recall | Top-10 recall |
|---|---:|---:|---:|---:|---:|
| mlp_probe | 0.1358 | 0.0050 | 0.1403 | 0.0400 | 0.0933 |
| activations_plus_position | 0.1251 | 0.0000 | 0.0603 | 0.0000 | 0.1000 |
| linear_probe | 0.1250 | 0.0000 | 0.0583 | 0.0000 | 0.1000 |
| text_only_baseline | 0.1186 | 0.0000 | 0.0960 | 0.0000 | 0.0333 |
| position_baseline | 0.1107 | 0.0000 | 0.1379 | 0.0000 | 0.0333 |
