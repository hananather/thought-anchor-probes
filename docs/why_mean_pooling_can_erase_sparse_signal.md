# Why Mean Pooling Can Erase Sparse Signal; Why MultiMax Helps

## Problem
Sentence-level mean pooling assumes evidence is spread across many tokens.  
When only a few tokens are informative (for example one contradiction token in a long sentence), averaging can dilute that evidence toward zero.

If a transformed token feature has one large value and many near-zero values, the pooled value scales roughly like:

\[
\text{mean} \approx \frac{\text{sparse signal}}{\text{sentence length}}
\]

So longer contexts can suppress the exact token-level signal we want to detect.

## Attention Probe (Soft Aggregation)
In the Gemini production probe design, each token activation is transformed with a small MLP:

\[
y_t = \mathrm{MLP}(h_t)
\]

Per head \(k\), compute token attention logits and value projections:

\[
\alpha_{t,k} = \mathrm{softmax}_t(q_k^\top y_t), \quad
v_{t,k} = v_k^\top y_t
\]

Then aggregate across tokens:

\[
z_k = \sum_t \alpha_{t,k} v_{t,k}
\]

The final probe score is a learned combination of head outputs \(z_k\).

## MultiMax (Hard Aggregation)
MultiMax replaces soft averaging with a hard per-head max over tokens:

\[
z_k = \max_t (u_k^\top y_t)
\]

This keeps a strong sparse token from being averaged away by many irrelevant tokens.

## Why This Repo Adds Both
- `attention_probe` gives smooth, trainable token weighting.
- `multimax_probe` gives hard sparse detection behavior that is robust to long-context dilution.
- Both share a lightweight configurable token MLP and multi-head setup.
- Both preserve metric parity with existing probes (`PR-AUC`, `Spearman`, top-k overlap).

## References
- Kramár et al. (2026), *Building Production-Ready Probes For Gemini*: https://arxiv.org/abs/2601.11516
- Neel Nanda interview (2025), practical probe-utility framing: https://80000hours.org/podcast/episodes/neel-nanda-mechanistic-interpretability/
