from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ta_probe.train import _validate_embedding_provenance, run_training


def _write_embeddings(
    *,
    path: Path,
    shape_path: Path,
    rows: int,
    hidden_dim: int,
    include_provenance: bool,
) -> None:
    matrix = np.arange(rows * hidden_dim, dtype=np.float32).reshape(rows, hidden_dim)
    memmap = np.memmap(path, dtype=np.float32, mode="w+", shape=(rows, hidden_dim))
    memmap[:] = matrix
    memmap.flush()

    payload: dict[str, object] = {
        "rows": rows,
        "hidden_dim": hidden_dim,
        "dtype": "float32",
        "layer_index": 1,
    }
    if include_provenance:
        payload.update(
            {
                "counterfactual_field": "counterfactual_importance_accuracy",
                "anchor_percentile": 90.0,
                "drop_last_chunk": True,
                "model_name_or_path": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                "compute_dtype": "float32",
            }
        )
    shape_path.write_text(json.dumps(payload), encoding="utf-8")


def _write_metadata(path: Path, problem_ids: list[int], include_vertical: bool = False) -> None:
    rows: list[dict[str, object]] = []
    embedding_row = 0
    for problem_id in problem_ids:
        for chunk_idx in [0, 1]:
            row: dict[str, object] = {
                "problem_id": problem_id,
                "chunk_idx": chunk_idx,
                "num_chunks": 2,
                "chunk_text": f"chunk {chunk_idx} text",
                "relative_position": float(chunk_idx),
                "token_count": 8,
                "embedding_row": embedding_row,
                "anchor": chunk_idx,
                "importance_score": float(chunk_idx),
            }
            if include_vertical:
                row["vertical_score"] = float(chunk_idx) * 0.1 + 0.1
            rows.append(row)
            embedding_row += 1
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_splits(path: Path, train: list[int], val: list[int], test: list[int]) -> None:
    payload = {"train": train, "val": val, "test": test}
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_validate_embedding_provenance_detects_mismatch() -> None:
    shape_payload = {
        "counterfactual_field": "counterfactual_importance_accuracy",
        "anchor_percentile": 90.0,
        "drop_last_chunk": True,
        "model_name_or_path": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    }

    with pytest.raises(ValueError, match="counterfactual_field"):
        _validate_embedding_provenance(
            shape_payload,
            expected_counterfactual_field="different_field",
            expected_anchor_percentile=90.0,
            expected_drop_last_chunk=True,
            expected_model_name_or_path="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            expected_pooling=None,
            expected_layer_mode=None,
            expected_requested_layer_index=None,
            expected_repo_id=None,
            expected_model_dir=None,
            expected_temp_dir=None,
            expected_split_dir=None,
            expected_compute_dtype=None,
        )


def test_run_training_fails_when_shape_lacks_provenance(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.parquet"
    emb_path = tmp_path / "embeddings.dat"
    shape_path = tmp_path / "shape.json"
    splits_path = tmp_path / "splits.json"

    _write_metadata(metadata_path, problem_ids=[1, 2, 3])
    _write_embeddings(
        path=emb_path,
        shape_path=shape_path,
        rows=6,
        hidden_dim=4,
        include_provenance=False,
    )
    _write_splits(splits_path, train=[1], val=[2], test=[3])

    with pytest.raises(ValueError, match="missing provenance key"):
        run_training(
            metadata_path=metadata_path,
            embeddings_memmap_path=emb_path,
            embeddings_shape_path=shape_path,
            splits_path=splits_path,
            metrics_output_path=tmp_path / "metrics.json",
            predictions_output_path=tmp_path / "predictions.parquet",
            random_seed=0,
            k_values=[1],
            mlp_hidden_dim=4,
            mlp_max_iter=10,
            expected_counterfactual_field="counterfactual_importance_accuracy",
            expected_anchor_percentile=90.0,
            expected_drop_last_chunk=True,
            expected_model_name_or_path="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            expected_compute_dtype=None,
            run_tripwires=False,
        )


def test_run_training_fails_when_split_problem_missing_from_metadata(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.parquet"
    emb_path = tmp_path / "embeddings.dat"
    shape_path = tmp_path / "shape.json"
    splits_path = tmp_path / "splits.json"

    _write_metadata(metadata_path, problem_ids=[1, 2])
    _write_embeddings(
        path=emb_path,
        shape_path=shape_path,
        rows=4,
        hidden_dim=4,
        include_provenance=True,
    )
    _write_splits(splits_path, train=[1], val=[2], test=[3])

    with pytest.raises(ValueError, match="missing split problem IDs"):
        run_training(
            metadata_path=metadata_path,
            embeddings_memmap_path=emb_path,
            embeddings_shape_path=shape_path,
            splits_path=splits_path,
            metrics_output_path=tmp_path / "metrics.json",
            predictions_output_path=tmp_path / "predictions.parquet",
            random_seed=0,
            k_values=[1],
            mlp_hidden_dim=4,
            mlp_max_iter=10,
            expected_counterfactual_field="counterfactual_importance_accuracy",
            expected_anchor_percentile=90.0,
            expected_drop_last_chunk=True,
            expected_model_name_or_path="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            expected_compute_dtype=None,
            run_tripwires=False,
        )


def test_run_training_fails_when_compute_dtype_mismatch(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.parquet"
    emb_path = tmp_path / "embeddings.dat"
    shape_path = tmp_path / "shape.json"
    splits_path = tmp_path / "splits.json"

    _write_metadata(metadata_path, problem_ids=[1, 2, 3])
    _write_embeddings(
        path=emb_path,
        shape_path=shape_path,
        rows=6,
        hidden_dim=4,
        include_provenance=True,
    )
    payload = json.loads(shape_path.read_text(encoding="utf-8"))
    payload["compute_dtype"] = "float16"
    shape_path.write_text(json.dumps(payload), encoding="utf-8")
    _write_splits(splits_path, train=[1], val=[2], test=[3])

    with pytest.raises(ValueError, match="compute_dtype"):
        run_training(
            metadata_path=metadata_path,
            embeddings_memmap_path=emb_path,
            embeddings_shape_path=shape_path,
            splits_path=splits_path,
            metrics_output_path=tmp_path / "metrics.json",
            predictions_output_path=tmp_path / "predictions.parquet",
            random_seed=0,
            k_values=[1],
            mlp_hidden_dim=4,
            mlp_max_iter=10,
            expected_counterfactual_field="counterfactual_importance_accuracy",
            expected_anchor_percentile=90.0,
            expected_drop_last_chunk=True,
            expected_model_name_or_path="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            expected_compute_dtype="float32",
            run_tripwires=False,
        )


def test_run_training_adds_vertical_models_when_scores_present(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.parquet"
    emb_path = tmp_path / "embeddings.dat"
    shape_path = tmp_path / "shape.json"
    splits_path = tmp_path / "splits.json"

    _write_metadata(metadata_path, problem_ids=[1, 2, 3], include_vertical=True)
    _write_embeddings(
        path=emb_path,
        shape_path=shape_path,
        rows=6,
        hidden_dim=4,
        include_provenance=True,
    )
    _write_splits(splits_path, train=[1], val=[2], test=[3])

    metrics = run_training(
        metadata_path=metadata_path,
        embeddings_memmap_path=emb_path,
        embeddings_shape_path=shape_path,
        splits_path=splits_path,
        metrics_output_path=tmp_path / "metrics.json",
        predictions_output_path=tmp_path / "predictions.parquet",
        random_seed=0,
        k_values=[1],
        mlp_hidden_dim=4,
        mlp_max_iter=10,
        expected_counterfactual_field="counterfactual_importance_accuracy",
        expected_anchor_percentile=90.0,
        expected_drop_last_chunk=True,
        expected_model_name_or_path="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        run_tripwires=False,
    )

    assert "vertical_attention_baseline" in metrics["test"]
    assert "vertical_attention_plus_position" in metrics["test"]
    assert metrics["has_vertical_scores"] is True
    assert metrics["primary_metric_name"] == "pr_auc"
