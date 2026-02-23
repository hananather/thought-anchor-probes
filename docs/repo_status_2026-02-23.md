# Repo Status Snapshot — 2026-02-23

This document captures the current state after implementing and testing the sweep tooling, and after stopping the large Stage-1 run.

## 1. What Was Implemented

### New sweep infrastructure (Codex)
- `src/ta_probe/sweep.py`
- `scripts/generate_sweep_configs.py`
- `scripts/run_systematic_sweep.py`
- `scripts/summarize_sweep.py`
- `docs/systematic_sweep_runbook.md`

### Existing runner enhancement
- `scripts/run_scaling_grid.py`
  - Added `--no-tripwires` passthrough into both single-split and LOPO paths.

### README updates
- Added systematic sweep workflow and runbook references.
- Added latest release-gate status context and scaling command updates.

### Additional support modules/tests (coworker stream)
- `src/ta_probe/sweep_schema.py`
- `src/ta_probe/sweep_factors.py`
- `tests/test_sweep_schema.py`
- `tests/test_sweep_factors.py`
- `tests/test_sweep_smoke.py`

### New tests (Codex)
- `tests/test_sweep.py`
- `tests/test_systematic_sweep_scripts.py`
- `tests/test_run_scaling_grid.py` (new arg test for `--no-tripwires`)

## 2. Validation Status

- `./.venv/bin/python -m pytest -q` passed for the main suite during implementation.
- Focused sweep tests (`test_sweep*`, script smoke tests) pass.
- Full-repo `ruff` status is clean for Codex-added files; coworker additions required minor lint cleanup (applied in this branch).

## 3. Sweep Execution Attempt Summary

Sweep root used:
- `artifacts/sweeps/qwen_ultra`

Stage-1 manifest:
- `artifacts/sweeps/qwen_ultra/manifest_stage1.jsonl`
- Total configs: `768`

Runtime outcome before stop:
- No full config completed.
- One config (`stage1_00293cc93cb1`) ran partially.
- Registry snapshot at stop:
  - `latest_configs_seen`: `1`
  - latest status for that config: `running` (stale after manual stop)
  - historical row counts included earlier network-related failures + multiple starts.

Partial artifacts from first config:
- Run root: `artifacts/sweeps/qwen_ultra/runs/stage1/stage1_00293cc93cb1`
- `metrics.json` files observed: `5`
- `predictions.parquet` files observed: `5`
- No `aggregate_lopo_metrics.json` yet.

Disk footprint snapshot:
- `artifacts/sweeps/qwen_ultra`: ~`753M`
- First config run root: ~`736M`
- `token_embeddings.dat` in first config: ~`728M`

## 4. Early Observations From Partial Results

Source: first config (`stage1_00293cc93cb1`), fold `330`, seeds `0..4`.

Mean test PR-AUC in this slice:
- `position_baseline`: `0.6755`
- `attention_probe`: `0.4813`
- `multimax_probe`: `0.4376`
- `activations_plus_position`: `0.2417`
- `linear_probe`: `0.2383`

Interpretation (limited, non-conclusive):
- Token probes outperformed linear activation probes in this narrow slice.
- Position baseline remained strongest in this slice.
- This is not sufficient evidence for broad conclusions because the config was not completed across all folds/seeds, and Stage-1 was not completed.

## 5. Operational Constraints Observed

- Network access to Hugging Face can fail in sandboxed runs (DNS/connection failures).
- Detached `nohup` behavior was unreliable in this environment for long-running sweeps.
- Persistent sessions worked more reliably than daemonized background launches.

## 6. Why the Large Plan Was Stopped

The original Stage-1 plan (768 configs × LOPO × 20 seeds) implied an impractically large number of seed-fold executions on a serial lane.

Given partial throughput observed on the first config, expected wall-clock was far beyond reasonable execution time.

## 7. Recommended Handoff Inputs for New Plan

Use this as ground truth for re-planning:
- Sweep tooling exists and works end-to-end.
- Stage-1 manifest generation is complete and reproducible.
- Runner/resume logic exists, but registry has stale running rows from interrupted sessions.
- Partial empirical signal suggests position remains a strong baseline.
- A narrower, decision-focused sweep is strongly preferred before any broad expansion.

## 8. Practical Reset Notes

Before launching a new strategy:
1. Start a new sweep root (e.g., `artifacts/sweeps/qwen_decisive_v1`) instead of reusing `qwen_ultra`.
2. Keep previous artifacts as exploratory history.
3. Rebuild manifest with reduced config count and lower seed budget for screening.
4. Use LOPO, but restrict factorial breadth first.
