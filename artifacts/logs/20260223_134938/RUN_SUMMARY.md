# Release Run Summary

- Timestamp (UTC): 20260223_134938
- Log bundle: `/workspace/thought-anchor-probes/artifacts/logs/20260223_134938`
- README: `/workspace/thought-anchor-probes/README.md`

## Verdict: NOT OK

### Failure
Command failed (stage='run_experiments', exit=1): /workspace/thought-anchor-probes/.venv/bin/python scripts/run_experiments.py --config configs/experiment.yaml --pilot-config configs/experiment_pilot.yaml --full-config configs/experiment_full.yaml --problem-id 330 --readme-path README.md. See log: /workspace/thought-anchor-probes/artifacts/logs/20260223_134938/03_run_experiments.log

## Environment Snapshot
- Git commit: `a653aa429e21b25a2ed23485356a7894ab38f353`
- Python: `Python 3.11.10`
- Torch/Transformers: `torch=2.9.1+cu128
transformers=4.57.3`
- Files: `git_commit.txt`, `python_version.txt`, `pip_freeze.txt`, `nvidia_smi.txt`, `torch_transformers_versions.txt`

## Commands Executed
| Stage | Exit | Duration (s) | Log File | Command |
|---|---:|---:|---|---|
| ruff_check | 0 | 0.23 | `01_ruff_check.log` | `/workspace/thought-anchor-probes/.venv/bin/python -m ruff check .` |
| pytest | 0 | 34.49 | `02_pytest.log` | `/workspace/thought-anchor-probes/.venv/bin/python -m pytest -q` |
| run_experiments | 1 | 0.37 | `03_run_experiments.log` | `/workspace/thought-anchor-probes/.venv/bin/python scripts/run_experiments.py --config configs/experiment.yaml --pilot-config configs/experiment_pilot.yaml --full-config configs/experiment_full.yaml --problem-id 330 --readme-path README.md` |

## Key Metrics (Best by Primary Metric per Setting)
| Setting | Best Model | Primary Metric | Best Primary Metric Value |
|---|---|---|---:|

## Artifact Existence Checks
| Setting | Check | Exists | Path |
|---|---|---|---|

## Tripwire Sanity Checks
- No binary-mode tripwire rows were checked.

## Extraction Failures
| Setting | Failure Log | Missing | Skipped Count | Sample Problem IDs |
|---|---|---|---:|---|

## Disk Usage
- `du -sh artifacts/`: 225M	artifacts
