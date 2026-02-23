# Aggregated Probe Metrics

## Per-Seed Test Metrics

| Run | Seed | Model | PR AUC | Spearman | Top-5 | Top-10 |
|---|---:|---|---:|---:|---:|---:|
| seed_0 | 0 | activations_plus_position | 0.1251 | 0.0603 | 0.0000 | 0.1000 |
| seed_0 | 0 | linear_probe | 0.1250 | 0.0583 | 0.0000 | 0.1000 |
| seed_0 | 0 | mlp_probe | 0.1397 | 0.2310 | 0.0000 | 0.1000 |
| seed_0 | 0 | position_baseline | 0.1107 | 0.1379 | 0.0000 | 0.0333 |
| seed_0 | 0 | text_only_baseline | 0.1186 | 0.0960 | 0.0000 | 0.0333 |
| seed_1 | 1 | activations_plus_position | 0.1251 | 0.0603 | 0.0000 | 0.1000 |
| seed_1 | 1 | linear_probe | 0.1250 | 0.0583 | 0.0000 | 0.1000 |
| seed_1 | 1 | mlp_probe | 0.1415 | 0.0592 | 0.0667 | 0.1333 |
| seed_1 | 1 | position_baseline | 0.1107 | 0.1379 | 0.0000 | 0.0333 |
| seed_1 | 1 | text_only_baseline | 0.1186 | 0.0960 | 0.0000 | 0.0333 |
| seed_2 | 2 | activations_plus_position | 0.1251 | 0.0603 | 0.0000 | 0.1000 |
| seed_2 | 2 | linear_probe | 0.1250 | 0.0583 | 0.0000 | 0.1000 |
| seed_2 | 2 | mlp_probe | 0.1295 | 0.0735 | 0.0667 | 0.0667 |
| seed_2 | 2 | position_baseline | 0.1107 | 0.1379 | 0.0000 | 0.0333 |
| seed_2 | 2 | text_only_baseline | 0.1186 | 0.0960 | 0.0000 | 0.0333 |
| seed_3 | 3 | activations_plus_position | 0.1251 | 0.0603 | 0.0000 | 0.1000 |
| seed_3 | 3 | linear_probe | 0.1250 | 0.0583 | 0.0000 | 0.1000 |
| seed_3 | 3 | mlp_probe | 0.1300 | 0.1236 | 0.0667 | 0.1000 |
| seed_3 | 3 | position_baseline | 0.1107 | 0.1379 | 0.0000 | 0.0333 |
| seed_3 | 3 | text_only_baseline | 0.1186 | 0.0960 | 0.0000 | 0.0333 |
| seed_4 | 4 | activations_plus_position | 0.1251 | 0.0603 | 0.0000 | 0.1000 |
| seed_4 | 4 | linear_probe | 0.1250 | 0.0583 | 0.0000 | 0.1000 |
| seed_4 | 4 | mlp_probe | 0.1383 | 0.2142 | 0.0000 | 0.0667 |
| seed_4 | 4 | position_baseline | 0.1107 | 0.1379 | 0.0000 | 0.0333 |
| seed_4 | 4 | text_only_baseline | 0.1186 | 0.0960 | 0.0000 | 0.0333 |

## Mean and Std Across Seeds (Test)

| Model | PR AUC mean | PR AUC std | Spearman mean | Spearman std | Top-5 mean | Top-5 std | Top-10 mean | Top-10 std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp_probe | 0.1358 | 0.0050 | 0.1403 | 0.0707 | 0.0400 | 0.0327 | 0.0933 | 0.0249 |
| activations_plus_position | 0.1251 | 0.0000 | 0.0603 | 0.0000 | 0.0000 | 0.0000 | 0.1000 | 0.0000 |
| linear_probe | 0.1250 | 0.0000 | 0.0583 | 0.0000 | 0.0000 | 0.0000 | 0.1000 | 0.0000 |
| text_only_baseline | 0.1186 | 0.0000 | 0.0960 | 0.0000 | 0.0000 | 0.0000 | 0.0333 | 0.0000 |
| position_baseline | 0.1107 | 0.0000 | 0.1379 | 0.0000 | 0.0000 | 0.0000 | 0.0333 | 0.0000 |

## Best-of-K by Validation PR AUC (k=1)

| Model | PR AUC mean | PR AUC std | Spearman mean | Spearman std | Top-5 mean | Top-5 std | Top-10 mean | Top-10 std | Seeds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| mlp_probe | 0.1397 | 0.0000 | 0.2310 | 0.0000 | 0.0000 | 0.0000 | 0.1000 | 0.0000 | [0] |
| activations_plus_position | 0.1251 | 0.0000 | 0.0603 | 0.0000 | 0.0000 | 0.0000 | 0.1000 | 0.0000 | [0] |
| linear_probe | 0.1250 | 0.0000 | 0.0583 | 0.0000 | 0.0000 | 0.0000 | 0.1000 | 0.0000 | [0] |
| text_only_baseline | 0.1186 | 0.0000 | 0.0960 | 0.0000 | 0.0000 | 0.0000 | 0.0333 | 0.0000 | [0] |
| position_baseline | 0.1107 | 0.0000 | 0.1379 | 0.0000 | 0.0000 | 0.0000 | 0.0333 | 0.0000 | [0] |

## Bootstrap CI Summary (Test Deltas)

| Comparison | Point delta mean | Point delta std | Bootstrap delta mean | Bootstrap delta std | Seeds CI excludes 0 | Seeds |
|---|---:|---:|---:|---:|---:|---:|
| score_activations_plus_position_minus_score_position_baseline | 0.0144 | 0.0000 | 0.0119 | 0.0002 | 5 | 5 |
| score_activations_plus_position_minus_score_text_only_baseline | 0.0065 | 0.0000 | 0.0136 | 0.0004 | 0 | 5 |

### Per-Seed CIs

| Run | Seed | Comparison | CI low | CI high | Excludes 0 |
|---|---:|---|---:|---:|---|
| seed_0 | 0 | score_activations_plus_position_minus_score_position_baseline | 0.0021 | 0.0294 | True |
| seed_0 | 0 | score_activations_plus_position_minus_score_text_only_baseline | -0.0306 | 0.0799 | False |
| seed_1 | 1 | score_activations_plus_position_minus_score_position_baseline | 0.0021 | 0.0294 | True |
| seed_1 | 1 | score_activations_plus_position_minus_score_text_only_baseline | -0.0306 | 0.0799 | False |
| seed_2 | 2 | score_activations_plus_position_minus_score_position_baseline | 0.0021 | 0.0294 | True |
| seed_2 | 2 | score_activations_plus_position_minus_score_text_only_baseline | -0.0306 | 0.0799 | False |
| seed_3 | 3 | score_activations_plus_position_minus_score_position_baseline | 0.0021 | 0.0294 | True |
| seed_3 | 3 | score_activations_plus_position_minus_score_text_only_baseline | -0.0306 | 0.0799 | False |
| seed_4 | 4 | score_activations_plus_position_minus_score_position_baseline | 0.0021 | 0.0294 | True |
| seed_4 | 4 | score_activations_plus_position_minus_score_text_only_baseline | -0.0306 | 0.0799 | False |
