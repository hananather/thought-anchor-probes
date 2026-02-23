"""Disk-usage estimation helpers for activation caches."""

from __future__ import annotations

from typing import Any, Literal


def dtype_num_bytes(dtype_name: str) -> int:
    """Return bytes per scalar for configured storage dtypes."""
    mapping = {
        "float16": 2,
        "float32": 4,
    }
    if dtype_name not in mapping:
        msg = f"Unsupported storage dtype for estimation: {dtype_name}"
        raise ValueError(msg)
    return mapping[dtype_name]


def format_bytes_human(num_bytes: int) -> str:
    """Format bytes using binary units."""
    value = float(max(0, int(num_bytes)))
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    unit_idx = 0
    while value >= 1024.0 and unit_idx < len(units) - 1:
        value /= 1024.0
        unit_idx += 1
    return f"{value:.2f} {units[unit_idx]}"


def _validate_non_negative(name: str, value: int) -> None:
    if int(value) < 0:
        msg = f"{name} must be non-negative"
        raise ValueError(msg)


def estimate_embedding_storage(
    *,
    embedding_layout: Literal["sentence_mean", "token_ragged"],
    hidden_dim: int,
    storage_dtype_name: str,
    num_sentences: int,
    total_tokens: int | None = None,
    metadata_row_overhead_bytes: int = 192,
) -> dict[str, Any]:
    """Estimate storage for sentence-mean or token-ragged activation caches."""
    if hidden_dim <= 0:
        msg = "hidden_dim must be positive"
        raise ValueError(msg)
    _validate_non_negative("num_sentences", num_sentences)
    _validate_non_negative("metadata_row_overhead_bytes", metadata_row_overhead_bytes)
    if total_tokens is not None:
        _validate_non_negative("total_tokens", total_tokens)

    bytes_per_float = dtype_num_bytes(storage_dtype_name)
    if embedding_layout == "sentence_mean":
        embedding_rows = int(num_sentences)
        formula = "num_sentences * hidden_dim * bytes_per_float + metadata_overhead"
    elif embedding_layout == "token_ragged":
        if total_tokens is None:
            msg = "total_tokens is required when embedding_layout='token_ragged'"
            raise ValueError(msg)
        embedding_rows = int(total_tokens)
        formula = "total_tokens * hidden_dim * bytes_per_float + metadata_overhead"
    else:
        msg = f"Unsupported embedding_layout for estimation: {embedding_layout}"
        raise ValueError(msg)

    embedding_bytes = int(embedding_rows) * int(hidden_dim) * int(bytes_per_float)
    metadata_bytes = int(num_sentences) * int(metadata_row_overhead_bytes)
    total_bytes = embedding_bytes + metadata_bytes

    payload: dict[str, Any] = {
        "embedding_layout": embedding_layout,
        "formula": formula,
        "embedding_rows_estimate": int(embedding_rows),
        "hidden_dim": int(hidden_dim),
        "bytes_per_float": int(bytes_per_float),
        "num_sentences": int(num_sentences),
        "metadata_row_overhead_bytes": int(metadata_row_overhead_bytes),
        "embedding_bytes_estimate": int(embedding_bytes),
        "metadata_bytes_estimate": int(metadata_bytes),
        "total_bytes_estimate": int(total_bytes),
        "embedding_human": format_bytes_human(embedding_bytes),
        "metadata_human": format_bytes_human(metadata_bytes),
        "total_human": format_bytes_human(total_bytes),
    }
    if total_tokens is not None:
        payload["total_tokens"] = int(total_tokens)
    return payload


def estimate_token_storage(
    *,
    total_tokens: int,
    hidden_dim: int,
    storage_dtype_name: str,
    num_sentences: int,
    metadata_row_overhead_bytes: int = 192,
) -> dict[str, Any]:
    """Backward-compatible wrapper for ragged token storage estimation."""
    return estimate_embedding_storage(
        embedding_layout="token_ragged",
        total_tokens=total_tokens,
        hidden_dim=hidden_dim,
        storage_dtype_name=storage_dtype_name,
        num_sentences=num_sentences,
        metadata_row_overhead_bytes=metadata_row_overhead_bytes,
    )
