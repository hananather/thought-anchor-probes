# Aggregated Probe Metrics

## Per-Seed Test Metrics

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

## Mean and Std Across Seeds (Test)

| Model | PR AUC mean | PR AUC std | Spearman mean | Spearman std | Top-5 mean | Top-5 std | Top-10 mean | Top-10 std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp_probe | 0.1476 | 0.0355 | 0.1925 | 0.0735 | 0.0667 | 0.0943 | 0.0667 | 0.0943 |
| linear_probe | 0.1476 | 0.0000 | 0.1347 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| activations_plus_position | 0.1450 | 0.0000 | 0.1305 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| position_baseline | 0.1415 | 0.0000 | 0.4686 | 0.0000 | 0.0000 | 0.0000 | 0.1000 | 0.0000 |
| text_only_baseline | 0.0955 | 0.0000 | -0.0274 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Best-of-K by Validation PR AUC (k=1)

| Model | PR AUC mean | PR AUC std | Spearman mean | Spearman std | Top-5 mean | Top-5 std | Top-10 mean | Top-10 std | Seeds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| mlp_probe | 0.1581 | 0.0000 | 0.2709 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | [0] |
| linear_probe | 0.1476 | 0.0000 | 0.1347 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | [0] |
| activations_plus_position | 0.1450 | 0.0000 | 0.1305 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | [0] |
| position_baseline | 0.1415 | 0.0000 | 0.4686 | 0.0000 | 0.0000 | 0.0000 | 0.1000 | 0.0000 | [0] |
| text_only_baseline | 0.0955 | 0.0000 | -0.0274 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | [0] |

## Bootstrap CI Summary (Test Deltas)

| Comparison | Point delta mean | Point delta std | Bootstrap delta mean | Bootstrap delta std | Seeds CI excludes 0 | Seeds |
|---|---:|---:|---:|---:|---:|---:|
| score_activations_plus_position_minus_score_position_baseline | 0.0035 | 0.0000 | 0.0035 | 0.0000 | 3 | 3 |
| score_activations_plus_position_minus_score_text_only_baseline | 0.0496 | 0.0000 | 0.0496 | 0.0000 | 3 | 3 |

### Per-Seed CIs

| Run | Seed | Comparison | CI low | CI high | Excludes 0 |
|---|---:|---|---:|---:|---|
| seed_0 | 0 | score_activations_plus_position_minus_score_position_baseline | 0.0035 | 0.0035 | True |
| seed_0 | 0 | score_activations_plus_position_minus_score_text_only_baseline | 0.0496 | 0.0496 | True |
| seed_1 | 1 | score_activations_plus_position_minus_score_position_baseline | 0.0035 | 0.0035 | True |
| seed_1 | 1 | score_activations_plus_position_minus_score_text_only_baseline | 0.0496 | 0.0496 | True |
| seed_2 | 2 | score_activations_plus_position_minus_score_position_baseline | 0.0035 | 0.0035 | True |
| seed_2 | 2 | score_activations_plus_position_minus_score_text_only_baseline | 0.0496 | 0.0496 | True |
