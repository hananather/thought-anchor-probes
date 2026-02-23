# Aggregated Probe Metrics

## Per-Seed Test Metrics

| Run | Seed | Model | PR AUC | Spearman | Top-5 | Top-10 |
|---|---:|---|---:|---:|---:|---:|
| seed_0 | 0 | activations_plus_position | 0.1679 | 0.2286 | 0.0667 | 0.1333 |
| seed_0 | 0 | linear_probe | 0.1670 | 0.2261 | 0.0667 | 0.1667 |
| seed_0 | 0 | mlp_probe | 0.1523 | 0.2560 | 0.0000 | 0.0667 |
| seed_0 | 0 | position_baseline | 0.1625 | 0.5125 | 0.0667 | 0.1000 |
| seed_0 | 0 | text_only_baseline | 0.1032 | -0.0135 | 0.0667 | 0.0667 |
| seed_1 | 1 | activations_plus_position | 0.1679 | 0.2286 | 0.0667 | 0.1333 |
| seed_1 | 1 | linear_probe | 0.1670 | 0.2261 | 0.0667 | 0.1667 |
| seed_1 | 1 | mlp_probe | 0.1354 | 0.1917 | 0.0667 | 0.0667 |
| seed_1 | 1 | position_baseline | 0.1625 | 0.5125 | 0.0667 | 0.1000 |
| seed_1 | 1 | text_only_baseline | 0.1032 | -0.0135 | 0.0667 | 0.0667 |
| seed_2 | 2 | activations_plus_position | 0.1679 | 0.2286 | 0.0667 | 0.1333 |
| seed_2 | 2 | linear_probe | 0.1670 | 0.2261 | 0.0667 | 0.1667 |
| seed_2 | 2 | mlp_probe | 0.1911 | 0.3234 | 0.0667 | 0.1000 |
| seed_2 | 2 | position_baseline | 0.1625 | 0.5125 | 0.0667 | 0.1000 |
| seed_2 | 2 | text_only_baseline | 0.1032 | -0.0135 | 0.0667 | 0.0667 |

## Mean and Std Across Seeds (Test)

| Model | PR AUC mean | PR AUC std | Spearman mean | Spearman std | Top-5 mean | Top-5 std | Top-10 mean | Top-10 std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| activations_plus_position | 0.1679 | 0.0000 | 0.2286 | 0.0000 | 0.0667 | 0.0000 | 0.1333 | 0.0000 |
| linear_probe | 0.1670 | 0.0000 | 0.2261 | 0.0000 | 0.0667 | 0.0000 | 0.1667 | 0.0000 |
| position_baseline | 0.1625 | 0.0000 | 0.5125 | 0.0000 | 0.0667 | 0.0000 | 0.1000 | 0.0000 |
| mlp_probe | 0.1596 | 0.0233 | 0.2570 | 0.0538 | 0.0444 | 0.0314 | 0.0778 | 0.0157 |
| text_only_baseline | 0.1032 | 0.0000 | -0.0135 | 0.0000 | 0.0667 | 0.0000 | 0.0667 | 0.0000 |

## Bootstrap CI Summary (Test Deltas)

| Comparison | Point delta mean | Point delta std | Bootstrap delta mean | Bootstrap delta std | Seeds CI excludes 0 | Seeds |
|---|---:|---:|---:|---:|---:|---:|
| score_activations_plus_position_minus_score_position_baseline | 0.0055 | 0.0000 | 0.0074 | 0.0005 | 0 | 3 |
| score_activations_plus_position_minus_score_text_only_baseline | 0.0647 | 0.0000 | 0.0635 | 0.0004 | 3 | 3 |

### Per-Seed CIs

| Run | Seed | Comparison | CI low | CI high | Excludes 0 |
|---|---:|---|---:|---:|---|
| seed_0 | 0 | score_activations_plus_position_minus_score_position_baseline | -0.0363 | 0.0690 | False |
| seed_0 | 0 | score_activations_plus_position_minus_score_text_only_baseline | 0.0009 | 0.1080 | True |
| seed_1 | 1 | score_activations_plus_position_minus_score_position_baseline | -0.0363 | 0.0690 | False |
| seed_1 | 1 | score_activations_plus_position_minus_score_text_only_baseline | 0.0009 | 0.1080 | True |
| seed_2 | 2 | score_activations_plus_position_minus_score_position_baseline | -0.0363 | 0.0690 | False |
| seed_2 | 2 | score_activations_plus_position_minus_score_text_only_baseline | 0.0009 | 0.1080 | True |
