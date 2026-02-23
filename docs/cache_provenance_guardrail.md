# Cache Provenance Guardrail

## Bug Summary

The training step could silently reuse stale embedding/label caches after config
changes (for example, changing `labels.anchor_percentile` or
`labels.drop_last_chunk`) because `scripts/train_probes.py` previously trusted
whatever was on disk.

This could produce unchanged metrics even though the active config changed.

## Fix

We now store extraction-time provenance in
`sentence_embeddings_shape.json` and validate it at training time.
Key checks include:

- `counterfactual_field`
- `anchor_percentile`
- `drop_last_chunk`
- `model_name_or_path`
- `pooling`
- `layer_mode` / `requested_layer_index`
- dataset path tuple (`repo_id`, `model_dir`, `temp_dir`, `split_dir`)

If any key is missing or mismatched, training fails fast with a clear message to
rerun `scripts/extract_embeddings.py`.

We also enforce split coverage at training time: every split problem ID must
exist in extracted metadata, so skipped extraction failures can no longer pass
silently.

## Engineer Notes

When adding new config knobs that affect extracted features or labels, add them
to both:

1. Extraction provenance payload (`src/ta_probe/activations.py`)
2. Training provenance validation (`src/ta_probe/train.py`)

This keeps cache semantics explicit and prevents stale-artifact regressions.

## Cache Reuse

`scripts/extract_embeddings.py --reuse-cache` now reuses existing artifacts only
when the shape payload provenance matches the active config.
