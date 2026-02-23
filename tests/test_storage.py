from __future__ import annotations

from ta_probe.storage import estimate_embedding_storage, estimate_token_storage


def test_estimate_embedding_storage_sentence_mean_uses_sentence_rows() -> None:
    estimate = estimate_embedding_storage(
        embedding_layout="sentence_mean",
        total_tokens=1000,
        hidden_dim=8,
        storage_dtype_name="float32",
        num_sentences=10,
    )
    assert estimate["embedding_layout"] == "sentence_mean"
    assert estimate["embedding_rows_estimate"] == 10
    assert estimate["embedding_bytes_estimate"] == 10 * 8 * 4
    assert estimate["formula"] == "num_sentences * hidden_dim * bytes_per_float + metadata_overhead"


def test_estimate_embedding_storage_token_ragged_uses_total_tokens() -> None:
    estimate = estimate_embedding_storage(
        embedding_layout="token_ragged",
        total_tokens=1000,
        hidden_dim=8,
        storage_dtype_name="float32",
        num_sentences=10,
    )
    assert estimate["embedding_layout"] == "token_ragged"
    assert estimate["embedding_rows_estimate"] == 1000
    assert estimate["embedding_bytes_estimate"] == 1000 * 8 * 4
    assert estimate["formula"] == "total_tokens * hidden_dim * bytes_per_float + metadata_overhead"


def test_estimate_token_storage_remains_token_layout_wrapper() -> None:
    estimate = estimate_token_storage(
        total_tokens=42,
        hidden_dim=16,
        storage_dtype_name="float16",
        num_sentences=5,
    )
    assert estimate["embedding_layout"] == "token_ragged"
    assert estimate["total_tokens"] == 42
    assert estimate["embedding_rows_estimate"] == 42
