from __future__ import annotations

import numpy as np
import pandas as pd

from ta_probe.models import build_text_features, build_vertical_features, make_text_baseline


def test_build_text_features_returns_object_array() -> None:
    frame = pd.DataFrame({"chunk_text": ["a", "b", "c"]})
    values = build_text_features(frame)
    assert values.dtype == object
    assert values.tolist() == ["a", "b", "c"]


def test_text_baseline_fits_and_predicts() -> None:
    frame = pd.DataFrame(
        {
            "chunk_text": [
                "solve equation by substitution",
                "final answer is boxed",
                "compute derivative carefully",
                "the answer is wrong due to sign",
            ]
        }
    )
    labels = [1, 0, 1, 0]
    model = make_text_baseline(random_seed=0)
    model.fit(build_text_features(frame), labels)
    scores = model.predict_proba(build_text_features(frame))[:, 1]
    assert len(scores) == len(frame)


def test_build_vertical_features_returns_2d_array() -> None:
    frame = pd.DataFrame({"vertical_score": [0.1, 0.2, 0.3]})
    values = build_vertical_features(frame)
    assert values.shape == (3, 1)
    assert values.dtype == np.float32
