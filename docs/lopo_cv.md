# Leave-One-Problem-Out Cross-Validation (LOPO-CV)

When the number of problems is small (for example, ~20), a single train/val/test split can be fragile. A single held-out set of three problems can swing metrics substantially, and it is easy to over-interpret noise as signal.

LOPO-CV makes each problem its own test fold:

- For each `problem_id`, hold out that problem as the test fold.
- Use the remaining problems for train/validation.
- Pick validation problems deterministically (smallest IDs by sorted order) so results are reproducible without relying on global RNG state.

Why this matters:

- With tiny `N_problems`, per-problem paired deltas are the right unit of evidence.
- We care about whether activations add signal beyond position, so per-fold deltas against the position baseline are the key comparison.

Recommended usage

- Set `split.strategy: lopo_cv` in your config.
- Run `scripts/run_lopo_cv.py` with a small `--max-problems` to sanity-check locally.
- Use the LOPO aggregate outputs to inspect fold-level deltas and bootstrap confidence intervals computed over folds.

Outputs

- Fold artifacts live under `artifacts/runs/<run_name>/fold_<problem_id>/<seed>/...`.
- Aggregation includes mean/std across folds, paired per-problem deltas vs the position baseline, and fold-level bootstrap CIs.

This design makes evaluation decision-grade under tiny `N_problems` without changing the default single-split pipeline.
