# Thought Anchor Probes

Train sentence probes that predict counterfactual sentence importance from model activations.

## What You'll Do
- Load the ARENA Thought Anchors rollout dataset from Hugging Face.
- Build sentence labels from `counterfactual_importance_accuracy`.
- Extract one-layer sentence embeddings with mean pooling.
- Train position, linear, and small neural probes.
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
2. Build problem ID cache and split by problem.
```bash
python scripts/build_problem_index.py --config configs/experiment.yaml --refresh
```
3. Verify stored rollout labels on one problem.
```bash
python scripts/verify_problem_labels.py --config configs/experiment.yaml --problem-id 4682
```
4. Extract sentence embeddings with one forward pass per problem.
```bash
python scripts/extract_embeddings.py --config configs/experiment.yaml
```
5. Train probes and write metrics.
```bash
python scripts/train_probes.py --config configs/experiment.yaml
```
6. Build a markdown report.
```bash
python scripts/build_report.py --config configs/experiment.yaml
```

## Optional Checks
- Run span integrity checks on one problem.
```bash
python scripts/check_spans.py --config configs/experiment.yaml --problem-id 4682
```
- Recompute counterfactual labels for one problem.
```bash
pip install -e ".[verify]"
python scripts/verify_problem_labels.py --config configs/experiment.yaml --problem-id 4682 --counterfactual
```

## MacBook Tips
- Start with `num_problems: 25` in `configs/experiment.yaml`.
- Keep `layer_mode: mid` and `pooling: mean`.
- Use `device: auto` to select MPS when available.
- Avoid loading full rollouts except one verification problem.

## Output Files
- `artifacts/sentence_embeddings.dat`
- `artifacts/sentence_embeddings_shape.json`
- `artifacts/sentence_metadata.parquet`
- `artifacts/metrics.json`
- `artifacts/predictions.parquet`
- `artifacts/report.md`

## Repo Layout
- `src/ta_probe/data_loading.py`: dataset listing and fast metadata loading.
- `src/ta_probe/labels.py`: anchor labels from counterfactual scores.
- `src/ta_probe/spans.py`: chunking and sentence token boundaries.
- `src/ta_probe/activations.py`: one-layer hooks and pooled embeddings.
- `src/ta_probe/models.py`: baseline and probe models.
- `src/ta_probe/train.py`: training, evaluation, and tripwire checks.
- `src/ta_probe/report.py`: markdown report generation.
- `tests/`: unit tests for spans, labels, and metrics.

## Experiment Results
<!-- EXPERIMENT_RESULTS_START -->
Run `python scripts/run_experiments.py --config configs/experiment.yaml` to populate this block.
<!-- EXPERIMENT_RESULTS_END -->
