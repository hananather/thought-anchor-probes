# Aggregated Probe Metrics

## Per-Seed Test Metrics

| Run | Seed | Model | PR AUC | Spearman | Top-5 | Top-10 |
|---|---:|---|---:|---:|---:|---:|
| seed_0 | 0 | activations_plus_position | 0.1684 | 0.2302 | 0.0667 | 0.1333 |
| seed_0 | 0 | linear_probe | 0.1664 | 0.2228 | 0.0667 | 0.1667 |
| seed_0 | 0 | mlp_probe | 0.1521 | 0.2669 | 0.0000 | 0.1000 |
| seed_0 | 0 | position_baseline | 0.1625 | 0.5125 | 0.0667 | 0.1000 |
| seed_0 | 0 | text_only_baseline | 0.1023 | -0.0128 | 0.0667 | 0.0667 |
| seed_1 | 1 | activations_plus_position | 0.1684 | 0.2302 | 0.0667 | 0.1333 |
| seed_1 | 1 | linear_probe | 0.1664 | 0.2228 | 0.0667 | 0.1667 |
| seed_1 | 1 | mlp_probe | 0.1364 | 0.1941 | 0.0667 | 0.0333 |
| seed_1 | 1 | position_baseline | 0.1625 | 0.5125 | 0.0667 | 0.1000 |
| seed_1 | 1 | text_only_baseline | 0.1023 | -0.0128 | 0.0667 | 0.0667 |
| seed_2 | 2 | activations_plus_position | 0.1684 | 0.2302 | 0.0667 | 0.1333 |
| seed_2 | 2 | linear_probe | 0.1664 | 0.2228 | 0.0667 | 0.1667 |
| seed_2 | 2 | mlp_probe | 0.1891 | 0.3175 | 0.0667 | 0.1000 |
| seed_2 | 2 | position_baseline | 0.1625 | 0.5125 | 0.0667 | 0.1000 |
| seed_2 | 2 | text_only_baseline | 0.1023 | -0.0128 | 0.0667 | 0.0667 |
| seed_3 | 3 | activations_plus_position | 0.1684 | 0.2302 | 0.0667 | 0.1333 |
| seed_3 | 3 | linear_probe | 0.1664 | 0.2228 | 0.0667 | 0.1667 |
| seed_3 | 3 | mlp_probe | 0.1677 | 0.2347 | 0.1333 | 0.1333 |
| seed_3 | 3 | position_baseline | 0.1625 | 0.5125 | 0.0667 | 0.1000 |
| seed_3 | 3 | text_only_baseline | 0.1023 | -0.0128 | 0.0667 | 0.0667 |
| seed_4 | 4 | activations_plus_position | 0.1684 | 0.2302 | 0.0667 | 0.1333 |
| seed_4 | 4 | linear_probe | 0.1664 | 0.2228 | 0.0667 | 0.1667 |
| seed_4 | 4 | mlp_probe | 0.1699 | 0.2483 | 0.0000 | 0.1000 |
| seed_4 | 4 | position_baseline | 0.1625 | 0.5125 | 0.0667 | 0.1000 |
| seed_4 | 4 | text_only_baseline | 0.1023 | -0.0128 | 0.0667 | 0.0667 |

## Mean and Std Across Seeds (Test)

| Model | PR AUC mean | PR AUC std | Spearman mean | Spearman std | Top-5 mean | Top-5 std | Top-10 mean | Top-10 std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| activations_plus_position | 0.1684 | 0.0000 | 0.2302 | 0.0000 | 0.0667 | 0.0000 | 0.1333 | 0.0000 |
| linear_probe | 0.1664 | 0.0000 | 0.2228 | 0.0000 | 0.0667 | 0.0000 | 0.1667 | 0.0000 |
| mlp_probe | 0.1630 | 0.0178 | 0.2523 | 0.0405 | 0.0533 | 0.0499 | 0.0933 | 0.0327 |
| position_baseline | 0.1625 | 0.0000 | 0.5125 | 0.0000 | 0.0667 | 0.0000 | 0.1000 | 0.0000 |
| text_only_baseline | 0.1023 | 0.0000 | -0.0128 | 0.0000 | 0.0667 | 0.0000 | 0.0667 | 0.0000 |

## Best-of-K by Validation PR AUC (k=1)

| Model | PR AUC mean | PR AUC std | Spearman mean | Spearman std | Top-5 mean | Top-5 std | Top-10 mean | Top-10 std | Seeds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| mlp_probe | 0.1891 | 0.0000 | 0.3175 | 0.0000 | 0.0667 | 0.0000 | 0.1000 | 0.0000 | [2] |
| activations_plus_position | 0.1684 | 0.0000 | 0.2302 | 0.0000 | 0.0667 | 0.0000 | 0.1333 | 0.0000 | [0] |
| linear_probe | 0.1664 | 0.0000 | 0.2228 | 0.0000 | 0.0667 | 0.0000 | 0.1667 | 0.0000 | [0] |
| position_baseline | 0.1625 | 0.0000 | 0.5125 | 0.0000 | 0.0667 | 0.0000 | 0.1000 | 0.0000 | [0] |
| text_only_baseline | 0.1023 | 0.0000 | -0.0128 | 0.0000 | 0.0667 | 0.0000 | 0.0667 | 0.0000 | [0] |

## Bootstrap CI Summary (Test Deltas)

| Comparison | Point delta mean | Point delta std | Bootstrap delta mean | Bootstrap delta std | Seeds CI excludes 0 | Seeds |
|---|---:|---:|---:|---:|---:|---:|
| score_activations_plus_position_minus_score_position_baseline | 0.0059 | 0.0000 | 0.0075 | 0.0003 | 0 | 5 |
| score_activations_plus_position_minus_score_text_only_baseline | 0.0661 | 0.0000 | 0.0641 | 0.0003 | 5 | 5 |

### Per-Seed CIs

| Run | Seed | Comparison | CI low | CI high | Excludes 0 |
|---|---:|---|---:|---:|---|
| seed_0 | 0 | score_activations_plus_position_minus_score_position_baseline | -0.0367 | 0.0698 | False |
| seed_0 | 0 | score_activations_plus_position_minus_score_text_only_baseline | 0.0021 | 0.1095 | True |
| seed_1 | 1 | score_activations_plus_position_minus_score_position_baseline | -0.0367 | 0.0698 | False |
| seed_1 | 1 | score_activations_plus_position_minus_score_text_only_baseline | 0.0021 | 0.1095 | True |
| seed_2 | 2 | score_activations_plus_position_minus_score_position_baseline | -0.0367 | 0.0698 | False |
| seed_2 | 2 | score_activations_plus_position_minus_score_text_only_baseline | 0.0021 | 0.1095 | True |
| seed_3 | 3 | score_activations_plus_position_minus_score_position_baseline | -0.0367 | 0.0698 | False |
| seed_3 | 3 | score_activations_plus_position_minus_score_text_only_baseline | 0.0021 | 0.1095 | True |
| seed_4 | 4 | score_activations_plus_position_minus_score_position_baseline | -0.0367 | 0.0698 | False |
| seed_4 | 4 | score_activations_plus_position_minus_score_text_only_baseline | 0.0021 | 0.1095 | True |
