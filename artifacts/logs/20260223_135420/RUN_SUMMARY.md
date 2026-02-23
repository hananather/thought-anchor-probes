# Release Run Summary

- Timestamp (UTC): 20260223_135420
- Log bundle: `/workspace/thought-anchor-probes/artifacts/logs/20260223_135420`
- README: `/workspace/thought-anchor-probes/README.md`

## Verdict: NOT OK

### Failure
Tripwire invariant failed:
- pilot metrics_seed_0.json: overfit one-problem test failed to memorize.
- pilot metrics_seed_1.json: overfit one-problem test failed to memorize.
- pilot metrics_seed_2.json: overfit one-problem test failed to memorize.
- full metrics_seed_0.json: overfit one-problem test failed to memorize.
- scaling:scaling_llama_correct metrics_seed_3.json: overfit one-problem test failed to memorize.
- scaling:scaling_llama_incorrect metrics_seed_2.json: overfit one-problem test failed to memorize.
- scaling:scaling_llama_incorrect metrics_seed_4.json: overfit one-problem test failed to memorize.
- scaling:scaling_qwen_correct metrics_seed_1.json: overfit one-problem test failed to memorize.
- scaling:scaling_qwen_incorrect metrics_seed_1.json: overfit one-problem test failed to memorize.
- scaling:scaling_qwen_incorrect metrics_seed_3.json: overfit one-problem test failed to memorize.
- scaling:scaling_qwen_incorrect metrics_seed_4.json: overfit one-problem test failed to memorize.

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
| pytest | 0 | 35.08 | `02_pytest.log` | `/workspace/thought-anchor-probes/.venv/bin/python -m pytest -q` |
| run_experiments | 0 | 313.18 | `03_run_experiments.log` | `/workspace/thought-anchor-probes/.venv/bin/python scripts/run_experiments.py --config configs/experiment.yaml --pilot-config configs/experiment_pilot.yaml --full-config configs/experiment_full.yaml --problem-id 330 --readme-path README.md` |
| run_scaling_grid | 0 | 941.92 | `04_run_scaling_grid.log` | `/workspace/thought-anchor-probes/.venv/bin/python scripts/run_scaling_grid.py --configs configs/scaling_llama_correct.yaml configs/scaling_llama_incorrect.yaml configs/scaling_qwen_correct.yaml configs/scaling_qwen_incorrect.yaml --seeds 0 1 2 3 4 --readme-path README.md` |

## Key Metrics (Best by Primary Metric per Setting)
| Setting | Best Model | Primary Metric | Best Primary Metric Value |
|---|---|---|---:|

## Artifact Existence Checks
| Setting | Check | Exists | Path |
|---|---|---|---|
| pilot | metrics_seed_glob | True | `/workspace/thought-anchor-probes/artifacts/runs/pilot/metrics_seed_*.json` |
| pilot | embeddings_memmap | True | `/workspace/thought-anchor-probes/artifacts/runs/pilot/sentence_embeddings.dat` |
| pilot | embeddings_shape_json | True | `/workspace/thought-anchor-probes/artifacts/runs/pilot/sentence_embeddings_shape.json` |
| pilot | metadata_parquet | True | `/workspace/thought-anchor-probes/artifacts/runs/pilot/sentence_metadata.parquet` |
| pilot | aggregate_metrics_json | True | `/workspace/thought-anchor-probes/artifacts/runs/pilot/aggregate_metrics.json` |
| full | metrics_seed_glob | True | `/workspace/thought-anchor-probes/artifacts/runs/full/metrics_seed_*.json` |
| full | embeddings_memmap | True | `/workspace/thought-anchor-probes/artifacts/runs/full/sentence_embeddings.dat` |
| full | embeddings_shape_json | True | `/workspace/thought-anchor-probes/artifacts/runs/full/sentence_embeddings_shape.json` |
| full | metadata_parquet | True | `/workspace/thought-anchor-probes/artifacts/runs/full/sentence_metadata.parquet` |
| full | aggregate_metrics_json | True | `/workspace/thought-anchor-probes/artifacts/runs/full/aggregate_metrics.json` |
| scaling:scaling_llama_correct | metrics_seed_glob | True | `/workspace/thought-anchor-probes/artifacts/scaling/llama_correct/metrics_seed_*.json` |
| scaling:scaling_llama_correct | embeddings_memmap | True | `/workspace/thought-anchor-probes/artifacts/scaling/llama_correct/sentence_embeddings.dat` |
| scaling:scaling_llama_correct | embeddings_shape_json | True | `/workspace/thought-anchor-probes/artifacts/scaling/llama_correct/sentence_embeddings_shape.json` |
| scaling:scaling_llama_correct | metadata_parquet | True | `/workspace/thought-anchor-probes/artifacts/scaling/llama_correct/sentence_metadata.parquet` |
| scaling:scaling_llama_correct | aggregate_metrics_json | True | `/workspace/thought-anchor-probes/artifacts/scaling/llama_correct/aggregate_metrics.json` |
| scaling:scaling_llama_incorrect | metrics_seed_glob | True | `/workspace/thought-anchor-probes/artifacts/scaling/llama_incorrect/metrics_seed_*.json` |
| scaling:scaling_llama_incorrect | embeddings_memmap | True | `/workspace/thought-anchor-probes/artifacts/scaling/llama_incorrect/sentence_embeddings.dat` |
| scaling:scaling_llama_incorrect | embeddings_shape_json | True | `/workspace/thought-anchor-probes/artifacts/scaling/llama_incorrect/sentence_embeddings_shape.json` |
| scaling:scaling_llama_incorrect | metadata_parquet | True | `/workspace/thought-anchor-probes/artifacts/scaling/llama_incorrect/sentence_metadata.parquet` |
| scaling:scaling_llama_incorrect | aggregate_metrics_json | True | `/workspace/thought-anchor-probes/artifacts/scaling/llama_incorrect/aggregate_metrics.json` |
| scaling:scaling_qwen_correct | metrics_seed_glob | True | `/workspace/thought-anchor-probes/artifacts/scaling/qwen_correct/metrics_seed_*.json` |
| scaling:scaling_qwen_correct | embeddings_memmap | True | `/workspace/thought-anchor-probes/artifacts/scaling/qwen_correct/sentence_embeddings.dat` |
| scaling:scaling_qwen_correct | embeddings_shape_json | True | `/workspace/thought-anchor-probes/artifacts/scaling/qwen_correct/sentence_embeddings_shape.json` |
| scaling:scaling_qwen_correct | metadata_parquet | True | `/workspace/thought-anchor-probes/artifacts/scaling/qwen_correct/sentence_metadata.parquet` |
| scaling:scaling_qwen_correct | aggregate_metrics_json | True | `/workspace/thought-anchor-probes/artifacts/scaling/qwen_correct/aggregate_metrics.json` |
| scaling:scaling_qwen_incorrect | metrics_seed_glob | True | `/workspace/thought-anchor-probes/artifacts/scaling/qwen_incorrect/metrics_seed_*.json` |
| scaling:scaling_qwen_incorrect | embeddings_memmap | True | `/workspace/thought-anchor-probes/artifacts/scaling/qwen_incorrect/sentence_embeddings.dat` |
| scaling:scaling_qwen_incorrect | embeddings_shape_json | True | `/workspace/thought-anchor-probes/artifacts/scaling/qwen_incorrect/sentence_embeddings_shape.json` |
| scaling:scaling_qwen_incorrect | metadata_parquet | True | `/workspace/thought-anchor-probes/artifacts/scaling/qwen_incorrect/sentence_metadata.parquet` |
| scaling:scaling_qwen_incorrect | aggregate_metrics_json | True | `/workspace/thought-anchor-probes/artifacts/scaling/qwen_incorrect/aggregate_metrics.json` |

## Tripwire Sanity Checks
- No binary-mode tripwire rows were checked.

## Extraction Failures
| Setting | Failure Log | Missing | Skipped Count | Sample Problem IDs |
|---|---|---|---:|---|

## Disk Usage
- `du -sh artifacts/`: 228M	artifacts
