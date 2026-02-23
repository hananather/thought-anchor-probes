# RunPod Scaling Runbook

## Purpose
Run the 4-setting high-confidence matrix on RunPod without local disk pressure.

## Pod Setup

1. Launch pod and work from `/workspace` (persistent volume).
2. Start `tmux` immediately so long runs survive disconnects.
3. Set Hugging Face cache before running Python:

```bash
export HF_HOME=/workspace/hf_cache
```

4. Clone repo and install dependencies:

```bash
cd /workspace
git clone <repo-url> thought-anchor-probes
cd thought-anchor-probes
uv pip install -e . --system -c constraints/runpod.txt
```

5. Run the version preflight (required before extraction):

```bash
python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
```

Expected output:
- `2.9.1 4.57.3`

These pins are required for reproducible VRAM/runtime behavior across RunPod runs.

## Execute Scaling Matrix

```bash
python scripts/run_scaling_grid.py
```

This command:
- builds shared problem IDs and shared train/val/test splits,
- runs extraction for each setting (with cache reuse when valid),
- runs 5 seeds per setting,
- aggregates metrics and bootstrap CI summaries,
- writes a global summary to `artifacts/scaling/scaling_summary.md`.

## Optional Controls

- Force fresh dataset listing:

```bash
python scripts/run_scaling_grid.py --refresh-problem-list
```

- Disable cache reuse:

```bash
python scripts/run_scaling_grid.py --no-reuse-cache
```

- Override seeds:

```bash
python scripts/run_scaling_grid.py --seeds 0 1 2
```

## Artifacts to Keep

- `artifacts/scaling/*/aggregate_metrics.json`
- `artifacts/scaling/*/aggregate_metrics.md`
- `artifacts/scaling/scaling_summary.md`
- `data/problem_ids_scaling_shared.json`
- `data/splits_scaling_shared.json`

Do not commit memmap embeddings (`sentence_embeddings.dat`) unless explicitly required.
