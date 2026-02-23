# Aggregated Probe Metrics

## Per-Seed Test Metrics

| Run | Seed | Model | PR AUC | Spearman | Top-5 | Top-10 |
|---|---:|---|---:|---:|---:|---:|
| seed_0 | 0 | activations_plus_position | 0.1188 | 0.1765 | 0.0000 | 0.0333 |
| seed_0 | 0 | linear_probe | 0.1187 | 0.1698 | 0.0000 | 0.0667 |
| seed_0 | 0 | mlp_probe | 0.1123 | 0.2011 | 0.0000 | 0.1333 |
| seed_0 | 0 | position_baseline | 0.1801 | 0.4799 | 0.0667 | 0.0667 |
| seed_0 | 0 | text_only_baseline | 0.0961 | 0.0876 | 0.0000 | 0.0333 |
| seed_1 | 1 | activations_plus_position | 0.1188 | 0.1765 | 0.0000 | 0.0333 |
| seed_1 | 1 | linear_probe | 0.1187 | 0.1698 | 0.0000 | 0.0667 |
| seed_1 | 1 | mlp_probe | 0.1501 | 0.2700 | 0.0667 | 0.0667 |
| seed_1 | 1 | position_baseline | 0.1801 | 0.4799 | 0.0667 | 0.0667 |
| seed_1 | 1 | text_only_baseline | 0.0961 | 0.0876 | 0.0000 | 0.0333 |
| seed_2 | 2 | activations_plus_position | 0.1188 | 0.1765 | 0.0000 | 0.0333 |
| seed_2 | 2 | linear_probe | 0.1187 | 0.1698 | 0.0000 | 0.0667 |
| seed_2 | 2 | mlp_probe | 0.1846 | 0.2449 | 0.1333 | 0.1667 |
| seed_2 | 2 | position_baseline | 0.1801 | 0.4799 | 0.0667 | 0.0667 |
| seed_2 | 2 | text_only_baseline | 0.0961 | 0.0876 | 0.0000 | 0.0333 |
| seed_3 | 3 | activations_plus_position | 0.1188 | 0.1765 | 0.0000 | 0.0333 |
| seed_3 | 3 | linear_probe | 0.1187 | 0.1698 | 0.0000 | 0.0667 |
| seed_3 | 3 | mlp_probe | 0.1478 | 0.2764 | 0.0667 | 0.0667 |
| seed_3 | 3 | position_baseline | 0.1801 | 0.4799 | 0.0667 | 0.0667 |
| seed_3 | 3 | text_only_baseline | 0.0961 | 0.0876 | 0.0000 | 0.0333 |
| seed_4 | 4 | activations_plus_position | 0.1188 | 0.1765 | 0.0000 | 0.0333 |
| seed_4 | 4 | linear_probe | 0.1187 | 0.1698 | 0.0000 | 0.0667 |
| seed_4 | 4 | mlp_probe | 0.1418 | 0.3386 | 0.0000 | 0.1000 |
| seed_4 | 4 | position_baseline | 0.1801 | 0.4799 | 0.0667 | 0.0667 |
| seed_4 | 4 | text_only_baseline | 0.0961 | 0.0876 | 0.0000 | 0.0333 |

## Mean and Std Across Seeds (Test)

| Model | PR AUC mean | PR AUC std | Spearman mean | Spearman std | Top-5 mean | Top-5 std | Top-10 mean | Top-10 std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| position_baseline | 0.1801 | 0.0000 | 0.4799 | 0.0000 | 0.0667 | 0.0000 | 0.0667 | 0.0000 |
| mlp_probe | 0.1473 | 0.0230 | 0.2662 | 0.0448 | 0.0533 | 0.0499 | 0.1067 | 0.0389 |
| activations_plus_position | 0.1188 | 0.0000 | 0.1765 | 0.0000 | 0.0000 | 0.0000 | 0.0333 | 0.0000 |
| linear_probe | 0.1187 | 0.0000 | 0.1698 | 0.0000 | 0.0000 | 0.0000 | 0.0667 | 0.0000 |
| text_only_baseline | 0.0961 | 0.0000 | 0.0876 | 0.0000 | 0.0000 | 0.0000 | 0.0333 | 0.0000 |

## Best-of-K by Validation PR AUC (k=1)

| Model | PR AUC mean | PR AUC std | Spearman mean | Spearman std | Top-5 mean | Top-5 std | Top-10 mean | Top-10 std | Seeds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| mlp_probe | 0.1846 | 0.0000 | 0.2449 | 0.0000 | 0.1333 | 0.0000 | 0.1667 | 0.0000 | [2] |
| position_baseline | 0.1801 | 0.0000 | 0.4799 | 0.0000 | 0.0667 | 0.0000 | 0.0667 | 0.0000 | [0] |
| activations_plus_position | 0.1188 | 0.0000 | 0.1765 | 0.0000 | 0.0000 | 0.0000 | 0.0333 | 0.0000 | [0] |
| linear_probe | 0.1187 | 0.0000 | 0.1698 | 0.0000 | 0.0000 | 0.0000 | 0.0667 | 0.0000 | [0] |
| text_only_baseline | 0.0961 | 0.0000 | 0.0876 | 0.0000 | 0.0000 | 0.0000 | 0.0333 | 0.0000 | [0] |

## Bootstrap CI Summary (Test Deltas)

| Comparison | Point delta mean | Point delta std | Bootstrap delta mean | Bootstrap delta std | Seeds CI excludes 0 | Seeds |
|---|---:|---:|---:|---:|---:|---:|
| score_activations_plus_position_minus_score_position_baseline | -0.0612 | 0.0000 | -0.0771 | 0.0007 | 0 | 5 |
| score_activations_plus_position_minus_score_text_only_baseline | 0.0227 | 0.0000 | 0.0236 | 0.0004 | 0 | 5 |

### Per-Seed CIs

| Run | Seed | Comparison | CI low | CI high | Excludes 0 |
|---|---:|---|---:|---:|---|
| seed_0 | 0 | score_activations_plus_position_minus_score_position_baseline | -0.2206 | 0.0222 | False |
| seed_0 | 0 | score_activations_plus_position_minus_score_text_only_baseline | -0.0192 | 0.0652 | False |
| seed_1 | 1 | score_activations_plus_position_minus_score_position_baseline | -0.2206 | 0.0222 | False |
| seed_1 | 1 | score_activations_plus_position_minus_score_text_only_baseline | -0.0192 | 0.0652 | False |
| seed_2 | 2 | score_activations_plus_position_minus_score_position_baseline | -0.2206 | 0.0222 | False |
| seed_2 | 2 | score_activations_plus_position_minus_score_text_only_baseline | -0.0192 | 0.0652 | False |
| seed_3 | 3 | score_activations_plus_position_minus_score_position_baseline | -0.2206 | 0.0222 | False |
| seed_3 | 3 | score_activations_plus_position_minus_score_text_only_baseline | -0.0192 | 0.0652 | False |
| seed_4 | 4 | score_activations_plus_position_minus_score_position_baseline | -0.2206 | 0.0222 | False |
| seed_4 | 4 | score_activations_plus_position_minus_score_text_only_baseline | -0.0192 | 0.0652 | False |
