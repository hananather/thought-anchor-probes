from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ta_probe import activations


def test_extract_skip_failed_problems_logs_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(activations, "load_model_and_tokenizer", lambda **_: (object(), object()))
    monkeypatch.setattr(activations, "resolve_layer_index", lambda *_args, **_kwargs: 7)
    monkeypatch.setattr(
        activations, "load_problem_metadata", lambda **_: {"chunks_labeled": [{}, {}]}
    )

    def fake_build_problem_vectors(*, problem_id: int, **_kwargs):
        if problem_id == 2:
            raise ValueError("Sequence length exceeds model context")
        vectors = np.ones((2, 3), dtype=np.float32)
        frame = pd.DataFrame(
            [
                {"problem_id": problem_id, "chunk_idx": 0, "anchor": 0},
                {"problem_id": problem_id, "chunk_idx": 1, "anchor": 1},
            ]
        )
        return vectors, frame

    monkeypatch.setattr(activations, "_build_problem_vectors", fake_build_problem_vectors)

    memmap_path = tmp_path / "embeddings.dat"
    shape_path = tmp_path / "shape.json"
    metadata_path = tmp_path / "metadata.parquet"
    failure_log_path = tmp_path / "failures.json"

    result = activations.extract_and_cache_embeddings(
        problem_ids=[1, 2],
        repo_id="repo",
        model_dir="model",
        temp_dir="temp",
        split_dir="split",
        model_name_or_path="model",
        layer_mode="mid",
        layer_index=None,
        counterfactual_field="counterfactual_importance_accuracy",
        anchor_percentile=90.0,
        drop_last_chunk=True,
        device="cpu",
        dtype_name="float32",
        embeddings_memmap_path=memmap_path,
        embeddings_shape_path=shape_path,
        metadata_path=metadata_path,
        skip_failed_problems=True,
        failure_log_path=failure_log_path,
    )

    assert result["processed_problem_ids"] == [1]
    assert result["skipped_problem_ids"] == [2]
    assert "2" in result["skip_reasons"]

    with shape_path.open("r", encoding="utf-8") as handle:
        shape = json.load(handle)
    assert shape["rows"] == 2
    assert shape["hidden_dim"] == 3

    with failure_log_path.open("r", encoding="utf-8") as handle:
        failure_payload = json.load(handle)
    assert failure_payload["skipped_problem_ids"] == [2]


def test_extract_skip_failed_problems_false_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(activations, "load_model_and_tokenizer", lambda **_: (object(), object()))
    monkeypatch.setattr(activations, "resolve_layer_index", lambda *_args, **_kwargs: 7)
    monkeypatch.setattr(
        activations, "load_problem_metadata", lambda **_: {"chunks_labeled": [{}, {}]}
    )

    def always_fail(*_args, **_kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(activations, "_build_problem_vectors", always_fail)

    with pytest.raises(ValueError):
        activations.extract_and_cache_embeddings(
            problem_ids=[1],
            repo_id="repo",
            model_dir="model",
            temp_dir="temp",
            split_dir="split",
            model_name_or_path="model",
            layer_mode="mid",
            layer_index=None,
            counterfactual_field="counterfactual_importance_accuracy",
            anchor_percentile=90.0,
            drop_last_chunk=True,
            device="cpu",
            dtype_name="float32",
            embeddings_memmap_path=tmp_path / "embeddings.dat",
            embeddings_shape_path=tmp_path / "shape.json",
            metadata_path=tmp_path / "metadata.parquet",
            skip_failed_problems=False,
            failure_log_path=None,
        )
