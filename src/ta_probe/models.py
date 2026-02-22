"""Model builders for baseline and activation probes."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

POSITION_FEATURE_COLUMNS = ["chunk_idx", "num_chunks", "relative_position", "token_count"]


def build_position_features(frame: pd.DataFrame) -> np.ndarray:
    """Create baseline features from sentence position and length."""
    missing = [col for col in POSITION_FEATURE_COLUMNS if col not in frame.columns]
    if missing:
        msg = f"Missing required position feature columns: {missing}"
        raise ValueError(msg)
    return frame[POSITION_FEATURE_COLUMNS].to_numpy(dtype=np.float32)


def make_position_baseline(random_seed: int) -> Pipeline:
    """Logistic regression baseline using only position features."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=random_seed,
                ),
            ),
        ]
    )


def make_linear_probe(random_seed: int) -> Pipeline:
    """Linear probe on mean-pooled sentence embeddings."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=random_seed,
                ),
            ),
        ]
    )


def make_mlp_probe(hidden_dim: int, max_iter: int, random_seed: int) -> Pipeline:
    """Two-layer MLP probe on sentence embeddings."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=(hidden_dim, hidden_dim),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    learning_rate_init=1e-3,
                    max_iter=max_iter,
                    early_stopping=True,
                    n_iter_no_change=20,
                    random_state=random_seed,
                ),
            ),
        ]
    )


def predict_scores(model: Pipeline, features: np.ndarray) -> np.ndarray:
    """Return probability scores from a fitted classifier."""
    probs = model.predict_proba(features)
    return probs[:, 1].astype(np.float32)
