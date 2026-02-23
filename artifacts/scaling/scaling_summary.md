# Scaling Matrix Summary

- Shared problems: 20
- Shared splits: train=14, val=3, test=3

| Setting | Best model | Primary metric | Mean primary metric (best) | CI rows | Storage estimate |
|---|---|---|---:|---:|---:|
| deepseek-r1-distill-llama-8b__correct_base_solution | activations_plus_position | pr_auc | 0.1684 | 10 | 31.85 MiB |
| deepseek-r1-distill-llama-8b__incorrect_base_solution | position_baseline | pr_auc | 0.1892 | 10 | 27.71 MiB |
| deepseek-r1-distill-qwen-14b__correct_base_solution | position_baseline | pr_auc | 0.1801 | 10 | 29.17 MiB |
| deepseek-r1-distill-qwen-14b__incorrect_base_solution | mlp_probe | pr_auc | 0.1358 | 10 | 34.84 MiB |
