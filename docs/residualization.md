# Beyond-Position Evaluation via Residualization

When position is a strong confounder, standard probe metrics can overstate how much signal is actually coming from activations. Residualization makes the question explicit: **do activations predict what position cannot?**

## What residualization does

1. Fit a baseline model on confounder features (e.g., position, or position+text) using the **training split only**.
2. Compute residuals: `residual = y - y_hat`.
3. Train an activation-based model to predict the residuals.
4. Evaluate residual metrics on held-out data.

This is correlational “incremental predictiveness,” not a causal claim.

## Target modes

- `anchor_binary` (default): we **do not** residualize the 0/1 anchor labels directly. Instead, we residualize the continuous importance scores and then threshold **within each problem** to form residual anchors.
- `importance_abs`: use absolute counterfactual importance as the continuous target.
- `importance_signed`: use signed counterfactual importance as the continuous target.

## Metrics

Residual evaluation reports:

- Residual Spearman (mean per problem)
- Residual PR-AUC (only when residual anchors are defined)

These are surfaced in aggregate reports under “Beyond-Position Residual Metrics.”

## Configuration

```yaml
labels:
  target_mode: anchor_binary  # or importance_abs / importance_signed
training:
  residualize_against: position  # or position_plus_text / none
```

This keeps the default pipeline unchanged unless explicitly enabled.
