# Thought Anchor Probes: AGENTS.md

## Scope
- Use the full four-setting grid as the default scaling run.
- Keep Qwen-14B as the primary paper model.

## Memory Notes
- Primary model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`.
- Optional model: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`.
- Dataset repo: `uzaymacar/math-rollouts`.
- Sampling setup: `temperature_0.6_top_p_0.95`.
- Default split fractions: `0.7 / 0.15 / 0.15`.
- Default scaling runner uses Llama-8B and Qwen-14B settings.

## Run Commands
- Default four-setting run:
```bash
python scripts/run_scaling_grid.py --seeds 0 1 2 3 4 --no-reuse-cache
```
- Qwen-only run:
```bash
python scripts/run_scaling_grid.py \
  --configs \
  configs/scaling_qwen_correct.yaml \
  configs/scaling_qwen_incorrect.yaml \
  --seeds 0 1 2 3 4 --no-reuse-cache
```

## RunPod Memory
- Use at least 48 GB VRAM for Qwen-14B extraction.
- Prefer `RTX 6000 Ada` or `L40S` when available.
