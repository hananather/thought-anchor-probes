# Aggregated Probe Metrics

## Per-Seed Test Metrics

| Run | Seed | Model | PR AUC | Spearman | Top-5 | Top-10 |
|---|---:|---|---:|---:|---:|---:|
| seed_0 | 0 | activations_plus_position | 0.1679 | 0.2250 | 0.0667 | 0.1333 |
| seed_0 | 0 | linear_probe | 0.1664 | 0.2157 | 0.0667 | 0.1667 |
| seed_0 | 0 | mlp_probe | 0.1566 | 0.2566 | 0.0000 | 0.0667 |
| seed_0 | 0 | position_baseline | 0.1625 | 0.5125 | 0.0667 | 0.1000 |
| seed_0 | 0 | text_only_baseline | 0.1032 | -0.0135 | 0.0667 | 0.0667 |
| seed_1 | 1 | activations_plus_position | 0.1679 | 0.2250 | 0.0667 | 0.1333 |
| seed_1 | 1 | linear_probe | 0.1664 | 0.2157 | 0.0667 | 0.1667 |
| seed_1 | 1 | mlp_probe | 0.1357 | 0.1937 | 0.0667 | 0.0667 |
| seed_1 | 1 | position_baseline | 0.1625 | 0.5125 | 0.0667 | 0.1000 |
| seed_1 | 1 | text_only_baseline | 0.1032 | -0.0135 | 0.0667 | 0.0667 |
| seed_2 | 2 | activations_plus_position | 0.1679 | 0.2250 | 0.0667 | 0.1333 |
| seed_2 | 2 | linear_probe | 0.1664 | 0.2157 | 0.0667 | 0.1667 |
| seed_2 | 2 | mlp_probe | 0.1925 | 0.3199 | 0.0667 | 0.1000 |
| seed_2 | 2 | position_baseline | 0.1625 | 0.5125 | 0.0667 | 0.1000 |
| seed_2 | 2 | text_only_baseline | 0.1032 | -0.0135 | 0.0667 | 0.0667 |
| seed_3 | 3 | activations_plus_position | 0.1679 | 0.2250 | 0.0667 | 0.1333 |
| seed_3 | 3 | linear_probe | 0.1664 | 0.2157 | 0.0667 | 0.1667 |
| seed_3 | 3 | mlp_probe | 0.1838 | 0.2541 | 0.0667 | 0.1333 |
| seed_3 | 3 | position_baseline | 0.1625 | 0.5125 | 0.0667 | 0.1000 |
| seed_3 | 3 | text_only_baseline | 0.1032 | -0.0135 | 0.0667 | 0.0667 |
| seed_4 | 4 | activations_plus_position | 0.1679 | 0.2250 | 0.0667 | 0.1333 |
| seed_4 | 4 | linear_probe | 0.1664 | 0.2157 | 0.0667 | 0.1667 |
| seed_4 | 4 | mlp_probe | 0.1727 | 0.2483 | 0.0000 | 0.1000 |
| seed_4 | 4 | position_baseline | 0.1625 | 0.5125 | 0.0667 | 0.1000 |
| seed_4 | 4 | text_only_baseline | 0.1032 | -0.0135 | 0.0667 | 0.0667 |

## Mean and Std Across Seeds (Test)

| Model | PR AUC mean | PR AUC std | Spearman mean | Spearman std | Top-5 mean | Top-5 std | Top-10 mean | Top-10 std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp_probe | 0.1682 | 0.0202 | 0.2545 | 0.0401 | 0.0400 | 0.0327 | 0.0933 | 0.0249 |
| activations_plus_position | 0.1679 | 0.0000 | 0.2250 | 0.0000 | 0.0667 | 0.0000 | 0.1333 | 0.0000 |
| linear_probe | 0.1664 | 0.0000 | 0.2157 | 0.0000 | 0.0667 | 0.0000 | 0.1667 | 0.0000 |
| position_baseline | 0.1625 | 0.0000 | 0.5125 | 0.0000 | 0.0667 | 0.0000 | 0.1000 | 0.0000 |
| text_only_baseline | 0.1032 | 0.0000 | -0.0135 | 0.0000 | 0.0667 | 0.0000 | 0.0667 | 0.0000 |

## Bootstrap CI Summary (Test Deltas)

| Comparison | Point delta mean | Point delta std | Bootstrap delta mean | Bootstrap delta std | Seeds CI excludes 0 | Seeds |
|---|---:|---:|---:|---:|---:|---:|
| score_activations_plus_position_minus_score_position_baseline | 0.0055 | 0.0000 | 0.0070 | 0.0003 | 0 | 5 |
| score_activations_plus_position_minus_score_text_only_baseline | 0.0647 | 0.0000 | 0.0628 | 0.0003 | 5 | 5 |

### Per-Seed CIs

| Run | Seed | Comparison | CI low | CI high | Excludes 0 |
|---|---:|---|---:|---:|---|
| seed_0 | 0 | score_activations_plus_position_minus_score_position_baseline | -0.0381 | 0.0703 | False |
| seed_0 | 0 | score_activations_plus_position_minus_score_text_only_baseline | 0.0011 | 0.1092 | True |
| seed_1 | 1 | score_activations_plus_position_minus_score_position_baseline | -0.0381 | 0.0703 | False |
| seed_1 | 1 | score_activations_plus_position_minus_score_text_only_baseline | 0.0011 | 0.1092 | True |
| seed_2 | 2 | score_activations_plus_position_minus_score_position_baseline | -0.0381 | 0.0703 | False |
| seed_2 | 2 | score_activations_plus_position_minus_score_text_only_baseline | 0.0011 | 0.1092 | True |
| seed_3 | 3 | score_activations_plus_position_minus_score_position_baseline | -0.0381 | 0.0703 | False |
| seed_3 | 3 | score_activations_plus_position_minus_score_text_only_baseline | 0.0011 | 0.1092 | True |
| seed_4 | 4 | score_activations_plus_position_minus_score_position_baseline | -0.0381 | 0.0703 | False |
| seed_4 | 4 | score_activations_plus_position_minus_score_text_only_baseline | 0.0011 | 0.1092 | True |
