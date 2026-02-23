from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ta_probe import activations
from ta_probe.token_probes import AttentionProbe


def test_ragged_token_storage_round_trip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(activations, "load_model_and_tokenizer", lambda **_: (object(), object()))
    monkeypatch.setattr(activations, "resolve_layer_index", lambda *_args, **_kwargs: 5)
    monkeypatch.setattr(
        activations,
        "load_problem_metadata",
        lambda **_: {"chunks_labeled": [{}, {}]},
    )

    hidden = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
        ],
        dtype=np.float32,
    )
    boundaries = [(0, 2), (2, 5)]

    def fake_build_problem_hidden_and_frame(*, problem_id: int, **_kwargs):
        frame = pd.DataFrame(
            [
                {
                    "problem_id": problem_id,
                    "chunk_idx": 0,
                    "num_chunks": 2,
                    "chunk_text": "first",
                    "anchor": 1,
                    "importance_score": 0.9,
                    "importance_signed": 0.9,
                    "token_start": 0,
                    "token_end": 2,
                    "token_count": 2,
                    "relative_position": 0.0,
                    "source_text_len": 10,
                },
                {
                    "problem_id": problem_id,
                    "chunk_idx": 1,
                    "num_chunks": 2,
                    "chunk_text": "second",
                    "anchor": 0,
                    "importance_score": 0.1,
                    "importance_signed": 0.1,
                    "token_start": 2,
                    "token_end": 5,
                    "token_count": 3,
                    "relative_position": 1.0,
                    "source_text_len": 10,
                },
            ]
        )
        return hidden, boundaries, frame, "first second"

    monkeypatch.setattr(
        activations,
        "_build_problem_hidden_and_frame",
        fake_build_problem_hidden_and_frame,
    )

    memmap_path = tmp_path / "token_embeddings.dat"
    shape_path = tmp_path / "shape.json"
    metadata_path = tmp_path / "metadata.parquet"

    result = activations.extract_and_cache_embeddings(
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
        pooling="tokens",
        storage_dtype_name="float32",
        embeddings_memmap_path=memmap_path,
        embeddings_shape_path=shape_path,
        metadata_path=metadata_path,
    )

    metadata = pd.read_parquet(metadata_path)
    assert metadata["token_offset"].tolist() == [0, 2]
    assert metadata["token_length"].tolist() == [2, 3]

    with shape_path.open("r", encoding="utf-8") as handle:
        shape_payload = json.load(handle)
    loaded = np.memmap(
        memmap_path,
        dtype=np.float32,
        mode="r",
        shape=(shape_payload["rows"], shape_payload["hidden_dim"]),
    )

    first_slice = loaded[0:2]
    second_slice = loaded[2:5]
    assert np.allclose(first_slice, hidden[0:2])
    assert np.allclose(second_slice, hidden[2:5])
    assert shape_payload["embedding_layout"] == "token_ragged"
    assert int(shape_payload["total_tokens"]) == 5
    assert int(result["storage_estimate"]["total_tokens"]) == 5


def test_attention_probe_mask_ignores_padding_tokens() -> None:
    torch.manual_seed(0)
    probe = AttentionProbe(input_dim=4, num_heads=2, mlp_width=8, mlp_depth=1)
    probe.eval()

    valid = torch.randn(1, 3, 4)
    padded_a = torch.cat([valid, torch.randn(1, 2, 4)], dim=1)
    padded_b = torch.cat([valid, torch.randn(1, 2, 4) * 10.0], dim=1)
    mask = torch.tensor([[True, True, True, False, False]])

    with torch.no_grad():
        out_a = probe(padded_a, mask)
        out_b = probe(padded_b, mask)

    assert torch.allclose(out_a, out_b, atol=1e-6)
