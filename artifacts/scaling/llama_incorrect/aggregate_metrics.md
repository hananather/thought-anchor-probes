# Aggregated Probe Metrics

## Per-Seed Test Metrics

| Run | Seed | Model | PR AUC | Spearman | Top-5 | Top-10 |
|---|---:|---|---:|---:|---:|---:|
| seed_0 | 0 | activations_plus_position | 0.1319 | -0.0171 | 0.0000 | 0.0333 |
| seed_0 | 0 | linear_probe | 0.1272 | -0.0330 | 0.0000 | 0.0333 |
| seed_0 | 0 | mlp_probe | 0.1402 | 0.0619 | 0.0667 | 0.0667 |
| seed_0 | 0 | position_baseline | 0.1892 | 0.2155 | 0.0000 | 0.1000 |
| seed_0 | 0 | text_only_baseline | 0.0998 | -0.0367 | 0.0000 | 0.0000 |
| seed_1 | 1 | activations_plus_position | 0.1319 | -0.0171 | 0.0000 | 0.0333 |
| seed_1 | 1 | linear_probe | 0.1272 | -0.0330 | 0.0000 | 0.0333 |
| seed_1 | 1 | mlp_probe | 0.1249 | 0.0379 | 0.1333 | 0.0667 |
| seed_1 | 1 | position_baseline | 0.1892 | 0.2155 | 0.0000 | 0.1000 |
| seed_1 | 1 | text_only_baseline | 0.0998 | -0.0367 | 0.0000 | 0.0000 |
| seed_2 | 2 | activations_plus_position | 0.1319 | -0.0171 | 0.0000 | 0.0333 |
| seed_2 | 2 | linear_probe | 0.1272 | -0.0330 | 0.0000 | 0.0333 |
| seed_2 | 2 | mlp_probe | 0.1406 | 0.0993 | 0.2000 | 0.1333 |
| seed_2 | 2 | position_baseline | 0.1892 | 0.2155 | 0.0000 | 0.1000 |
| seed_2 | 2 | text_only_baseline | 0.0998 | -0.0367 | 0.0000 | 0.0000 |
| seed_3 | 3 | activations_plus_position | 0.1319 | -0.0171 | 0.0000 | 0.0333 |
| seed_3 | 3 | linear_probe | 0.1272 | -0.0330 | 0.0000 | 0.0333 |
| seed_3 | 3 | mlp_probe | 0.1154 | 0.0421 | 0.0000 | 0.0333 |
| seed_3 | 3 | position_baseline | 0.1892 | 0.2155 | 0.0000 | 0.1000 |
| seed_3 | 3 | text_only_baseline | 0.0998 | -0.0367 | 0.0000 | 0.0000 |
| seed_4 | 4 | activations_plus_position | 0.1319 | -0.0171 | 0.0000 | 0.0333 |
| seed_4 | 4 | linear_probe | 0.1272 | -0.0330 | 0.0000 | 0.0333 |
| seed_4 | 4 | mlp_probe | 0.1115 | 0.0453 | 0.0000 | 0.0333 |
| seed_4 | 4 | position_baseline | 0.1892 | 0.2155 | 0.0000 | 0.1000 |
| seed_4 | 4 | text_only_baseline | 0.0998 | -0.0367 | 0.0000 | 0.0000 |

## Mean and Std Across Seeds (Test)

| Model | PR AUC mean | PR AUC std | Spearman mean | Spearman std | Top-5 mean | Top-5 std | Top-10 mean | Top-10 std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| position_baseline | 0.1892 | 0.0000 | 0.2155 | 0.0000 | 0.0000 | 0.0000 | 0.1000 | 0.0000 |
| activations_plus_position | 0.1319 | 0.0000 | -0.0171 | 0.0000 | 0.0000 | 0.0000 | 0.0333 | 0.0000 |
| linear_probe | 0.1272 | 0.0000 | -0.0330 | 0.0000 | 0.0000 | 0.0000 | 0.0333 | 0.0000 |
| mlp_probe | 0.1265 | 0.0122 | 0.0573 | 0.0225 | 0.0800 | 0.0777 | 0.0667 | 0.0365 |
| text_only_baseline | 0.0998 | 0.0000 | -0.0367 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Bootstrap CI Summary (Test Deltas)

| Comparison | Point delta mean | Point delta std | Bootstrap delta mean | Bootstrap delta std | Seeds CI excludes 0 | Seeds |
|---|---:|---:|---:|---:|---:|---:|
| score_activations_plus_position_minus_score_position_baseline | -0.0572 | 0.0000 | -0.0560 | 0.0003 | 5 | 5 |
| score_activations_plus_position_minus_score_text_only_baseline | 0.0321 | 0.0000 | 0.0289 | 0.0003 | 0 | 5 |

### Per-Seed CIs

| Run | Seed | Comparison | CI low | CI high | Excludes 0 |
|---|---:|---|---:|---:|---|
| seed_0 | 0 | score_activations_plus_position_minus_score_position_baseline | -0.1191 | -0.0195 | True |
| seed_0 | 0 | score_activations_plus_position_minus_score_text_only_baseline | -0.0023 | 0.0655 | False |
| seed_1 | 1 | score_activations_plus_position_minus_score_position_baseline | -0.1191 | -0.0195 | True |
| seed_1 | 1 | score_activations_plus_position_minus_score_text_only_baseline | -0.0023 | 0.0655 | False |
| seed_2 | 2 | score_activations_plus_position_minus_score_position_baseline | -0.1191 | -0.0195 | True |
| seed_2 | 2 | score_activations_plus_position_minus_score_text_only_baseline | -0.0023 | 0.0655 | False |
| seed_3 | 3 | score_activations_plus_position_minus_score_position_baseline | -0.1191 | -0.0195 | True |
| seed_3 | 3 | score_activations_plus_position_minus_score_text_only_baseline | -0.0023 | 0.0655 | False |
| seed_4 | 4 | score_activations_plus_position_minus_score_position_baseline | -0.1191 | -0.0195 | True |
| seed_4 | 4 | score_activations_plus_position_minus_score_text_only_baseline | -0.0023 | 0.0655 | False |
