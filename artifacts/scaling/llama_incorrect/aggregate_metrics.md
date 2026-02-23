# Aggregated Probe Metrics

## Per-Seed Test Metrics

| Run | Seed | Model | PR AUC | Spearman | Top-5 | Top-10 |
|---|---:|---|---:|---:|---:|---:|
| seed_0 | 0 | activations_plus_position | 0.1336 | -0.0141 | 0.0000 | 0.0333 |
| seed_0 | 0 | linear_probe | 0.1272 | -0.0347 | 0.0000 | 0.0333 |
| seed_0 | 0 | mlp_probe | 0.1406 | 0.0398 | 0.0000 | 0.0667 |
| seed_0 | 0 | position_baseline | 0.1892 | 0.2155 | 0.0000 | 0.1000 |
| seed_0 | 0 | text_only_baseline | 0.0998 | -0.0367 | 0.0000 | 0.0000 |
| seed_1 | 1 | activations_plus_position | 0.1336 | -0.0141 | 0.0000 | 0.0333 |
| seed_1 | 1 | linear_probe | 0.1272 | -0.0347 | 0.0000 | 0.0333 |
| seed_1 | 1 | mlp_probe | 0.1242 | 0.0367 | 0.1333 | 0.0667 |
| seed_1 | 1 | position_baseline | 0.1892 | 0.2155 | 0.0000 | 0.1000 |
| seed_1 | 1 | text_only_baseline | 0.0998 | -0.0367 | 0.0000 | 0.0000 |
| seed_2 | 2 | activations_plus_position | 0.1336 | -0.0141 | 0.0000 | 0.0333 |
| seed_2 | 2 | linear_probe | 0.1272 | -0.0347 | 0.0000 | 0.0333 |
| seed_2 | 2 | mlp_probe | 0.1405 | 0.0975 | 0.2000 | 0.1333 |
| seed_2 | 2 | position_baseline | 0.1892 | 0.2155 | 0.0000 | 0.1000 |
| seed_2 | 2 | text_only_baseline | 0.0998 | -0.0367 | 0.0000 | 0.0000 |
| seed_3 | 3 | activations_plus_position | 0.1336 | -0.0141 | 0.0000 | 0.0333 |
| seed_3 | 3 | linear_probe | 0.1272 | -0.0347 | 0.0000 | 0.0333 |
| seed_3 | 3 | mlp_probe | 0.1135 | 0.0453 | 0.0000 | 0.0333 |
| seed_3 | 3 | position_baseline | 0.1892 | 0.2155 | 0.0000 | 0.1000 |
| seed_3 | 3 | text_only_baseline | 0.0998 | -0.0367 | 0.0000 | 0.0000 |
| seed_4 | 4 | activations_plus_position | 0.1336 | -0.0141 | 0.0000 | 0.0333 |
| seed_4 | 4 | linear_probe | 0.1272 | -0.0347 | 0.0000 | 0.0333 |
| seed_4 | 4 | mlp_probe | 0.1124 | 0.0498 | 0.0000 | 0.0333 |
| seed_4 | 4 | position_baseline | 0.1892 | 0.2155 | 0.0000 | 0.1000 |
| seed_4 | 4 | text_only_baseline | 0.0998 | -0.0367 | 0.0000 | 0.0000 |

## Mean and Std Across Seeds (Test)

| Model | PR AUC mean | PR AUC std | Spearman mean | Spearman std | Top-5 mean | Top-5 std | Top-10 mean | Top-10 std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| position_baseline | 0.1892 | 0.0000 | 0.2155 | 0.0000 | 0.0000 | 0.0000 | 0.1000 | 0.0000 |
| activations_plus_position | 0.1336 | 0.0000 | -0.0141 | 0.0000 | 0.0000 | 0.0000 | 0.0333 | 0.0000 |
| linear_probe | 0.1272 | 0.0000 | -0.0347 | 0.0000 | 0.0000 | 0.0000 | 0.0333 | 0.0000 |
| mlp_probe | 0.1263 | 0.0124 | 0.0538 | 0.0223 | 0.0667 | 0.0843 | 0.0667 | 0.0365 |
| text_only_baseline | 0.0998 | 0.0000 | -0.0367 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Best-of-K by Validation PR AUC (k=1)

| Model | PR AUC mean | PR AUC std | Spearman mean | Spearman std | Top-5 mean | Top-5 std | Top-10 mean | Top-10 std | Seeds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| position_baseline | 0.1892 | 0.0000 | 0.2155 | 0.0000 | 0.0000 | 0.0000 | 0.1000 | 0.0000 | [0] |
| mlp_probe | 0.1405 | 0.0000 | 0.0975 | 0.0000 | 0.2000 | 0.0000 | 0.1333 | 0.0000 | [2] |
| activations_plus_position | 0.1336 | 0.0000 | -0.0141 | 0.0000 | 0.0000 | 0.0000 | 0.0333 | 0.0000 | [0] |
| linear_probe | 0.1272 | 0.0000 | -0.0347 | 0.0000 | 0.0000 | 0.0000 | 0.0333 | 0.0000 | [0] |
| text_only_baseline | 0.0998 | 0.0000 | -0.0367 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | [0] |

## Bootstrap CI Summary (Test Deltas)

| Comparison | Point delta mean | Point delta std | Bootstrap delta mean | Bootstrap delta std | Seeds CI excludes 0 | Seeds |
|---|---:|---:|---:|---:|---:|---:|
| score_activations_plus_position_minus_score_position_baseline | -0.0555 | 0.0000 | -0.0542 | 0.0003 | 5 | 5 |
| score_activations_plus_position_minus_score_text_only_baseline | 0.0338 | 0.0000 | 0.0306 | 0.0003 | 0 | 5 |

### Per-Seed CIs

| Run | Seed | Comparison | CI low | CI high | Excludes 0 |
|---|---:|---|---:|---:|---|
| seed_0 | 0 | score_activations_plus_position_minus_score_position_baseline | -0.1173 | -0.0187 | True |
| seed_0 | 0 | score_activations_plus_position_minus_score_text_only_baseline | -0.0005 | 0.0714 | False |
| seed_1 | 1 | score_activations_plus_position_minus_score_position_baseline | -0.1173 | -0.0187 | True |
| seed_1 | 1 | score_activations_plus_position_minus_score_text_only_baseline | -0.0005 | 0.0714 | False |
| seed_2 | 2 | score_activations_plus_position_minus_score_position_baseline | -0.1173 | -0.0187 | True |
| seed_2 | 2 | score_activations_plus_position_minus_score_text_only_baseline | -0.0005 | 0.0714 | False |
| seed_3 | 3 | score_activations_plus_position_minus_score_position_baseline | -0.1173 | -0.0187 | True |
| seed_3 | 3 | score_activations_plus_position_minus_score_text_only_baseline | -0.0005 | 0.0714 | False |
| seed_4 | 4 | score_activations_plus_position_minus_score_position_baseline | -0.1173 | -0.0187 | True |
| seed_4 | 4 | score_activations_plus_position_minus_score_text_only_baseline | -0.0005 | 0.0714 | False |
