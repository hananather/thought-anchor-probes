# Aggregated Probe Metrics

## Per-Seed Test Metrics

| Run | Seed | Model | PR AUC | Spearman | Top-5 | Top-10 |
|---|---:|---|---:|---:|---:|---:|
| seed_0 | 0 | activations_plus_position | 0.1249 | 0.0594 | 0.0000 | 0.1000 |
| seed_0 | 0 | linear_probe | 0.1249 | 0.0591 | 0.0000 | 0.1000 |
| seed_0 | 0 | mlp_probe | 0.1471 | 0.2276 | 0.0000 | 0.0667 |
| seed_0 | 0 | position_baseline | 0.1107 | 0.1379 | 0.0000 | 0.0333 |
| seed_0 | 0 | text_only_baseline | 0.1186 | 0.0960 | 0.0000 | 0.0333 |
| seed_1 | 1 | activations_plus_position | 0.1249 | 0.0594 | 0.0000 | 0.1000 |
| seed_1 | 1 | linear_probe | 0.1249 | 0.0591 | 0.0000 | 0.1000 |
| seed_1 | 1 | mlp_probe | 0.1420 | 0.0644 | 0.0667 | 0.1333 |
| seed_1 | 1 | position_baseline | 0.1107 | 0.1379 | 0.0000 | 0.0333 |
| seed_1 | 1 | text_only_baseline | 0.1186 | 0.0960 | 0.0000 | 0.0333 |
| seed_2 | 2 | activations_plus_position | 0.1249 | 0.0594 | 0.0000 | 0.1000 |
| seed_2 | 2 | linear_probe | 0.1249 | 0.0591 | 0.0000 | 0.1000 |
| seed_2 | 2 | mlp_probe | 0.1520 | 0.1249 | 0.0667 | 0.1667 |
| seed_2 | 2 | position_baseline | 0.1107 | 0.1379 | 0.0000 | 0.0333 |
| seed_2 | 2 | text_only_baseline | 0.1186 | 0.0960 | 0.0000 | 0.0333 |
| seed_3 | 3 | activations_plus_position | 0.1249 | 0.0594 | 0.0000 | 0.1000 |
| seed_3 | 3 | linear_probe | 0.1249 | 0.0591 | 0.0000 | 0.1000 |
| seed_3 | 3 | mlp_probe | 0.1285 | 0.1242 | 0.0667 | 0.1000 |
| seed_3 | 3 | position_baseline | 0.1107 | 0.1379 | 0.0000 | 0.0333 |
| seed_3 | 3 | text_only_baseline | 0.1186 | 0.0960 | 0.0000 | 0.0333 |
| seed_4 | 4 | activations_plus_position | 0.1249 | 0.0594 | 0.0000 | 0.1000 |
| seed_4 | 4 | linear_probe | 0.1249 | 0.0591 | 0.0000 | 0.1000 |
| seed_4 | 4 | mlp_probe | 0.1361 | 0.2187 | 0.0000 | 0.1000 |
| seed_4 | 4 | position_baseline | 0.1107 | 0.1379 | 0.0000 | 0.0333 |
| seed_4 | 4 | text_only_baseline | 0.1186 | 0.0960 | 0.0000 | 0.0333 |

## Mean and Std Across Seeds (Test)

| Model | PR AUC mean | PR AUC std | Spearman mean | Spearman std | Top-5 mean | Top-5 std | Top-10 mean | Top-10 std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp_probe | 0.1411 | 0.0083 | 0.1520 | 0.0622 | 0.0400 | 0.0327 | 0.1133 | 0.0340 |
| linear_probe | 0.1249 | 0.0000 | 0.0591 | 0.0000 | 0.0000 | 0.0000 | 0.1000 | 0.0000 |
| activations_plus_position | 0.1249 | 0.0000 | 0.0594 | 0.0000 | 0.0000 | 0.0000 | 0.1000 | 0.0000 |
| text_only_baseline | 0.1186 | 0.0000 | 0.0960 | 0.0000 | 0.0000 | 0.0000 | 0.0333 | 0.0000 |
| position_baseline | 0.1107 | 0.0000 | 0.1379 | 0.0000 | 0.0000 | 0.0000 | 0.0333 | 0.0000 |

## Bootstrap CI Summary (Test Deltas)

| Comparison | Point delta mean | Point delta std | Bootstrap delta mean | Bootstrap delta std | Seeds CI excludes 0 | Seeds |
|---|---:|---:|---:|---:|---:|---:|
| score_activations_plus_position_minus_score_position_baseline | 0.0143 | 0.0000 | 0.0116 | 0.0002 | 5 | 5 |
| score_activations_plus_position_minus_score_text_only_baseline | 0.0063 | 0.0000 | 0.0133 | 0.0004 | 0 | 5 |

### Per-Seed CIs

| Run | Seed | Comparison | CI low | CI high | Excludes 0 |
|---|---:|---|---:|---:|---|
| seed_0 | 0 | score_activations_plus_position_minus_score_position_baseline | 0.0020 | 0.0288 | True |
| seed_0 | 0 | score_activations_plus_position_minus_score_text_only_baseline | -0.0303 | 0.0775 | False |
| seed_1 | 1 | score_activations_plus_position_minus_score_position_baseline | 0.0020 | 0.0288 | True |
| seed_1 | 1 | score_activations_plus_position_minus_score_text_only_baseline | -0.0303 | 0.0775 | False |
| seed_2 | 2 | score_activations_plus_position_minus_score_position_baseline | 0.0020 | 0.0288 | True |
| seed_2 | 2 | score_activations_plus_position_minus_score_text_only_baseline | -0.0303 | 0.0775 | False |
| seed_3 | 3 | score_activations_plus_position_minus_score_position_baseline | 0.0020 | 0.0288 | True |
| seed_3 | 3 | score_activations_plus_position_minus_score_text_only_baseline | -0.0303 | 0.0775 | False |
| seed_4 | 4 | score_activations_plus_position_minus_score_position_baseline | 0.0020 | 0.0288 | True |
| seed_4 | 4 | score_activations_plus_position_minus_score_text_only_baseline | -0.0303 | 0.0775 | False |
