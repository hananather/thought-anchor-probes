# High-Confidence Scaling TODOs

## Objective
Prove activation probes add non-trivial signal beyond position/text baselines, with uncertainty and replication across settings.

## Implementation Checklist

- [x] Baselines and controls
- [x] Add `text_only_baseline` (TF-IDF + logistic regression).
- [x] Add `activations_plus_position` model.
- [x] Add optional position-bin diagnostics by relative position.

- [x] Statistical confidence
- [x] Add grouped bootstrap CIs over `problem_id` for PR AUC deltas.
- [x] Report deltas:
- [x] `activations_plus_position - position_baseline`
- [x] `activations_plus_position - text_only_baseline`

- [x] Cache and extraction robustness
- [x] Add extraction cache-reuse guard with provenance comparison.
- [x] Support configurable embedding storage dtype (`float32`/`float16`).
- [x] Store richer extraction provenance in shape payload.

- [x] Experiment matrix
- [x] Add 4 scaling configs:
- [x] llama-8b + correct
- [x] llama-8b + incorrect
- [x] qwen-14b + correct
- [x] qwen-14b + incorrect
- [x] Add matrix runner that builds shared intersection IDs and shared splits.

- [x] RunPod and execution docs
- [x] Add RunPod runbook with `/workspace`, `HF_HOME`, `tmux`, and artifact policy.

- [x] Validation
- [x] Update/add tests for new baselines, CI logic, cache reuse.
- [x] Run `ruff` and `pytest` cleanly.
