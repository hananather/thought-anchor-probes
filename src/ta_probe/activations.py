"""Activation extraction and sentence embedding cache creation."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from ta_probe.data_loading import load_problem_metadata
from ta_probe.labels import build_problem_label_frame
from ta_probe.spans import get_sentence_token_boundaries, get_whitebox_example_data


def resolve_device(device: str) -> str:
    """Resolve runtime device from config."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(dtype_name: str) -> torch.dtype:
    """Map string dtype names to torch dtypes."""
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        msg = f"Unsupported dtype: {dtype_name}"
        raise ValueError(msg)
    return mapping[dtype_name]


def resolve_numpy_dtype(dtype_name: str) -> np.dtype:
    """Map storage dtype names to numpy dtypes."""
    mapping = {
        "float16": np.float16,
        "float32": np.float32,
    }
    if dtype_name not in mapping:
        msg = f"Unsupported storage dtype: {dtype_name}"
        raise ValueError(msg)
    return np.dtype(mapping[dtype_name])


def _cache_shape_matches(shape_payload: dict[str, Any], expected: dict[str, Any]) -> bool:
    for key, expected_value in expected.items():
        if key not in shape_payload:
            return False
        cached_value = shape_payload.get(key)
        if isinstance(expected_value, float):
            try:
                if not np.isclose(float(cached_value), expected_value, rtol=0.0, atol=1e-9):
                    return False
            except (TypeError, ValueError):
                return False
        else:
            if cached_value != expected_value:
                return False
    return True


def _write_failure_log(
    *,
    failure_log_path: str | Path | None,
    attempted_problem_ids: list[int],
    processed_problem_ids: list[int],
    skipped_problem_ids: list[int],
    skip_reasons: dict[str, str],
    cache_reused: bool,
) -> None:
    if failure_log_path is None:
        return

    failure_file = Path(failure_log_path)
    failure_file.parent.mkdir(parents=True, exist_ok=True)
    with failure_file.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "attempted_problem_ids": [int(problem_id) for problem_id in attempted_problem_ids],
                "processed_problem_ids": [int(problem_id) for problem_id in processed_problem_ids],
                "skipped_problem_ids": [int(problem_id) for problem_id in skipped_problem_ids],
                "skip_reasons": skip_reasons,
                "cache_reused": bool(cache_reused),
            },
            handle,
            indent=2,
            sort_keys=True,
        )


def get_transformer_layers(model: PreTrainedModel):
    """Return the model's block list across common architectures."""
    candidates = [
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("decoder", "layers"),
    ]
    for parent_attr, layers_attr in candidates:
        parent = getattr(model, parent_attr, None)
        if parent is not None and hasattr(parent, layers_attr):
            return getattr(parent, layers_attr)

    msg = "Could not locate transformer layers for hook extraction"
    raise ValueError(msg)


def resolve_layer_index(model: PreTrainedModel, layer_mode: str, layer_index: int | None) -> int:
    """Choose a layer index based on config."""
    layers = get_transformer_layers(model)
    n_layers = len(layers)
    if n_layers <= 0:
        msg = "Model has no transformer layers"
        raise ValueError(msg)

    if layer_mode == "mid":
        return n_layers // 2

    if layer_mode == "index":
        if layer_index is None:
            msg = "layer_index is required when layer_mode='index'"
            raise ValueError(msg)
        if layer_index < 0 or layer_index >= n_layers:
            msg = f"layer_index {layer_index} out of range [0, {n_layers - 1}]"
            raise ValueError(msg)
        return layer_index

    msg = f"Unsupported layer_mode: {layer_mode}"
    raise ValueError(msg)


def load_model_and_tokenizer(
    model_name_or_path: str,
    device: str,
    dtype_name: str,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load model and tokenizer for activation extraction."""
    resolved_device = resolve_device(device)
    torch_dtype = resolve_dtype(dtype_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = None
    primary_error: Exception | None = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
    except TypeError as exc:
        primary_error = exc
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )
            warnings.warn(
                (
                    "Fell back to `torch_dtype` during model load after `dtype` "
                    f"TypeError: {exc}"
                ),
                RuntimeWarning,
                stacklevel=2,
            )
        except Exception as fallback_exc:
            msg = (
                "Failed loading model with both dtype compatibility paths. "
                f"dtype_error={primary_error!r}; torch_dtype_error={fallback_exc!r}"
            )
            raise RuntimeError(msg) from fallback_exc

    if model is None:
        msg = "Model load returned None unexpectedly"
        raise RuntimeError(msg)

    model.eval()
    model.to(resolved_device)
    return model, tokenizer


def extract_layer_hidden_state(
    *,
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    layer_idx: int,
    device: str,
) -> tuple[np.ndarray, list[int]]:
    """Run one forward pass and capture one layer's token activations."""
    resolved_device = resolve_device(device)
    layers = get_transformer_layers(model)

    captured: dict[str, torch.Tensor] = {}

    def hook_fn(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = hidden.detach().to("cpu", dtype=torch.float32)

    hook = layers[layer_idx].register_forward_hook(hook_fn)
    try:
        tokens = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
            return_attention_mask=True,
            return_token_type_ids=False,
        )
        tokens = {key: value.to(resolved_device) for key, value in tokens.items()}
        seq_len = int(tokens["input_ids"].shape[1])
        max_positions = getattr(model.config, "max_position_embeddings", None)
        tokenizer_max = tokenizer.model_max_length
        valid_limits = [limit for limit in [max_positions, tokenizer_max] if isinstance(limit, int)]
        if valid_limits:
            max_len = min(valid_limits)
            if seq_len > max_len:
                msg = (
                    f"Sequence length {seq_len} exceeds model context {max_len}. "
                    "Use a longer-context model or choose shorter traces."
                )
                raise ValueError(msg)

        with torch.no_grad():
            _ = model(**tokens, use_cache=False)
    finally:
        hook.remove()

    if "hidden" not in captured:
        msg = "Failed to capture hidden state from the hooked layer"
        raise RuntimeError(msg)

    hidden = captured["hidden"]
    if hidden.dim() != 3:
        msg = f"Expected hidden state with 3 dims, got {tuple(hidden.shape)}"
        raise RuntimeError(msg)

    hidden_array = hidden[0].numpy()
    if np.isnan(hidden_array).any():
        msg = "Hidden state contains NaNs"
        raise RuntimeError(msg)

    input_ids = tokens["input_ids"][0].detach().to("cpu").tolist()
    return hidden_array, input_ids


def pool_sentence_embeddings(
    hidden_states: np.ndarray,
    token_boundaries: list[tuple[int, int]],
) -> np.ndarray:
    """Mean-pool token activations per sentence span."""
    seq_len, hidden_dim = hidden_states.shape
    if seq_len == 0:
        msg = "Hidden states sequence length must be positive"
        raise ValueError(msg)
    vectors = np.zeros((len(token_boundaries), hidden_dim), dtype=np.float32)

    for idx, (start, end) in enumerate(token_boundaries):
        if start < 0 or start >= seq_len:
            msg = f"Token boundary start {start} out of range for seq_len={seq_len}"
            raise ValueError(msg)
        end = min(end, seq_len)
        if end <= start:
            end = start + 1
        vectors[idx] = hidden_states[start:end].mean(axis=0)

    return vectors


def _build_problem_vectors(
    *,
    problem_id: int,
    problem_data: dict[str, Any],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    layer_idx: int,
    counterfactual_field: str,
    anchor_percentile: float,
    drop_last_chunk: bool,
    device: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Extract per-sentence vectors and metadata rows for one problem."""
    text, sentences, _ = get_whitebox_example_data(problem_data)

    token_boundaries = get_sentence_token_boundaries(text, sentences, tokenizer)
    usable_boundaries = (
        token_boundaries[:-1] if drop_last_chunk and len(token_boundaries) > 1 else token_boundaries
    )

    hidden, _ = extract_layer_hidden_state(
        text=text,
        model=model,
        tokenizer=tokenizer,
        layer_idx=layer_idx,
        device=device,
    )

    if usable_boundaries and usable_boundaries[-1][1] > hidden.shape[0]:
        fallback_text = " ".join(sentences)
        token_boundaries = get_sentence_token_boundaries(fallback_text, sentences, tokenizer)
        usable_boundaries = (
            token_boundaries[:-1]
            if drop_last_chunk and len(token_boundaries) > 1
            else token_boundaries
        )
        hidden, _ = extract_layer_hidden_state(
            text=fallback_text,
            model=model,
            tokenizer=tokenizer,
            layer_idx=layer_idx,
            device=device,
        )
        text = fallback_text

    vectors = pool_sentence_embeddings(hidden, usable_boundaries)

    label_frame = build_problem_label_frame(
        problem_id=problem_id,
        chunks_labeled=problem_data["chunks_labeled"],
        counterfactual_field=counterfactual_field,
        percentile=anchor_percentile,
        drop_last_chunk=drop_last_chunk,
    )

    if len(label_frame) != len(vectors):
        msg = (
            f"Row mismatch for problem {problem_id}: "
            f"labels={len(label_frame)}, vectors={len(vectors)}"
        )
        raise ValueError(msg)

    label_frame["token_start"] = [start for start, _ in usable_boundaries]
    label_frame["token_end"] = [end for _, end in usable_boundaries]
    label_frame["token_count"] = label_frame["token_end"] - label_frame["token_start"]
    denom = np.maximum(label_frame["num_chunks"].to_numpy(dtype=np.float32) - 1, 1)
    label_frame["relative_position"] = label_frame["chunk_idx"].to_numpy(dtype=np.float32) / denom
    label_frame["source_text_len"] = len(text)
    return vectors, label_frame


def extract_and_cache_embeddings(
    *,
    problem_ids: list[int],
    repo_id: str,
    model_dir: str,
    temp_dir: str,
    split_dir: str,
    model_name_or_path: str,
    layer_mode: str,
    layer_index: int | None,
    counterfactual_field: str,
    anchor_percentile: float,
    drop_last_chunk: bool,
    device: str,
    dtype_name: str,
    pooling: str,
    storage_dtype_name: str,
    embeddings_memmap_path: str | Path,
    embeddings_shape_path: str | Path,
    metadata_path: str | Path,
    skip_failed_problems: bool = False,
    failure_log_path: str | Path | None = None,
    reuse_cache_if_valid: bool = False,
) -> dict[str, Any]:
    """Extract sentence vectors and save memmap plus metadata.

    When ``skip_failed_problems`` is true, the function skips problems that fail
    extraction and logs the reason per problem ID.
    """
    if not problem_ids:
        msg = "No problem IDs provided"
        raise ValueError(msg)

    shape_file = Path(embeddings_shape_path)
    metadata_file = Path(metadata_path)
    memmap_file = Path(embeddings_memmap_path)
    expected_cache_payload = {
        "repo_id": repo_id,
        "model_dir": model_dir,
        "temp_dir": temp_dir,
        "split_dir": split_dir,
        "model_name_or_path": model_name_or_path,
        "layer_mode": layer_mode,
        "requested_layer_index": (
            int(layer_index) if layer_index is not None else None
        ),
        "pooling": pooling,
        "counterfactual_field": counterfactual_field,
        "anchor_percentile": float(anchor_percentile),
        "drop_last_chunk": bool(drop_last_chunk),
        "dtype": storage_dtype_name,
        "compute_dtype": dtype_name,
    }
    if (
        reuse_cache_if_valid
        and shape_file.exists()
        and metadata_file.exists()
        and memmap_file.exists()
    ):
        with shape_file.open("r", encoding="utf-8") as handle:
            cached_shape = json.load(handle)
        if _cache_shape_matches(cached_shape, expected_cache_payload):
            metadata_df = pd.read_parquet(metadata_file, columns=["problem_id"])
            available_problem_ids = set(
                metadata_df["problem_id"].to_numpy(dtype=np.int64).tolist()
            )
            attempted_problem_ids = [int(problem_id) for problem_id in problem_ids]
            processed_problem_ids = [
                problem_id
                for problem_id in attempted_problem_ids
                if problem_id in available_problem_ids
            ]
            skipped_problem_ids = [
                problem_id
                for problem_id in attempted_problem_ids
                if problem_id not in available_problem_ids
            ]
            skip_reasons = {
                str(problem_id): "missing from reused cache metadata"
                for problem_id in skipped_problem_ids
            }

            _write_failure_log(
                failure_log_path=failure_log_path,
                attempted_problem_ids=attempted_problem_ids,
                processed_problem_ids=processed_problem_ids,
                skipped_problem_ids=skipped_problem_ids,
                skip_reasons=skip_reasons,
                cache_reused=True,
            )

            if skipped_problem_ids and not skip_failed_problems:
                summary = ", ".join(str(problem_id) for problem_id in skipped_problem_ids)
                msg = (
                    "Reused cache is missing requested problem IDs: "
                    f"{summary}. "
                    "Re-run scripts/extract_embeddings.py without --reuse-cache."
                )
                raise ValueError(msg)

            return {
                "rows": int(cached_shape["rows"]),
                "hidden_dim": int(cached_shape["hidden_dim"]),
                "layer_index": int(cached_shape["layer_index"]),
                "metadata_path": str(metadata_file),
                "embeddings_memmap": str(memmap_file),
                "shape_path": str(shape_file),
                "processed_problem_ids": processed_problem_ids,
                "skipped_problem_ids": skipped_problem_ids,
                "skip_reasons": skip_reasons,
                "cache_reused": True,
            }

    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=model_name_or_path,
        device=device,
        dtype_name=dtype_name,
    )
    layer_idx = resolve_layer_index(model, layer_mode, layer_index)

    processed_problem_ids: list[int] = []
    skipped_problem_ids: list[int] = []
    skip_reasons: dict[str, str] = {}
    extracted_vectors: list[np.ndarray] = []
    extracted_frames: list[pd.DataFrame] = []
    hidden_dim: int | None = None

    for problem_id in problem_ids:
        try:
            payload = load_problem_metadata(
                problem_id=problem_id,
                repo_id=repo_id,
                model_dir=model_dir,
                temp_dir=temp_dir,
                split_dir=split_dir,
            )
            vectors, frame = _build_problem_vectors(
                problem_id=problem_id,
                problem_data=payload,
                model=model,
                tokenizer=tokenizer,
                layer_idx=layer_idx,
                counterfactual_field=counterfactual_field,
                anchor_percentile=anchor_percentile,
                drop_last_chunk=drop_last_chunk,
                device=device,
            )
        except Exception as exc:
            if not skip_failed_problems:
                raise
            skipped_problem_ids.append(problem_id)
            skip_reasons[str(problem_id)] = str(exc)
            continue

        if hidden_dim is None:
            hidden_dim = int(vectors.shape[1])
        elif vectors.shape[1] != hidden_dim:
            msg = (
                f"Hidden dimension mismatch for problem {problem_id}: "
                f"expected {hidden_dim}, got {vectors.shape[1]}"
            )
            raise ValueError(msg)

        processed_problem_ids.append(problem_id)
        extracted_vectors.append(vectors)
        extracted_frames.append(frame)

    if not extracted_frames or hidden_dim is None:
        msg = "No problem embeddings were extracted successfully"
        if skip_reasons:
            msg += f". Reasons: {skip_reasons}"
        raise RuntimeError(msg)

    total_rows = int(sum(len(frame) for frame in extracted_frames))

    memmap_file.parent.mkdir(parents=True, exist_ok=True)

    metadata_file.parent.mkdir(parents=True, exist_ok=True)

    shape_file.parent.mkdir(parents=True, exist_ok=True)

    frame_parts: list[pd.DataFrame] = []
    offset = 0
    storage_dtype = resolve_numpy_dtype(storage_dtype_name)
    memmap = np.memmap(
        memmap_file,
        dtype=storage_dtype,
        mode="w+",
        shape=(total_rows, hidden_dim),
    )

    for vectors, frame in zip(extracted_vectors, extracted_frames, strict=True):
        n_rows = len(frame)
        memmap[offset : offset + n_rows] = vectors.astype(storage_dtype, copy=False)
        frame = frame.copy()
        frame["embedding_row"] = np.arange(offset, offset + n_rows, dtype=np.int64)
        frame_parts.append(frame)
        offset += n_rows

    if offset != total_rows:
        msg = f"Expected {total_rows} rows but wrote {offset}"
        raise RuntimeError(msg)

    memmap.flush()
    metadata_df = pd.concat(frame_parts, ignore_index=True)
    metadata_df.to_parquet(metadata_file, index=False)

    # Store provenance so training can detect stale caches after config changes.
    shape_payload = {
        "rows": int(total_rows),
        "hidden_dim": int(hidden_dim),
        "dtype": storage_dtype_name,
        "layer_index": int(layer_idx),
        "layer_mode": layer_mode,
        "requested_layer_index": (
            int(layer_index) if layer_index is not None else None
        ),
        "pooling": pooling,
        "repo_id": repo_id,
        "model_dir": model_dir,
        "temp_dir": temp_dir,
        "split_dir": split_dir,
        "compute_dtype": dtype_name,
        "counterfactual_field": counterfactual_field,
        "anchor_percentile": float(anchor_percentile),
        "drop_last_chunk": bool(drop_last_chunk),
        "model_name_or_path": model_name_or_path,
    }
    with shape_file.open("w", encoding="utf-8") as handle:
        json.dump(shape_payload, handle, indent=2, sort_keys=True)

    _write_failure_log(
        failure_log_path=failure_log_path,
        attempted_problem_ids=[int(problem_id) for problem_id in problem_ids],
        processed_problem_ids=[int(problem_id) for problem_id in processed_problem_ids],
        skipped_problem_ids=[int(problem_id) for problem_id in skipped_problem_ids],
        skip_reasons=skip_reasons,
        cache_reused=False,
    )

    return {
        "rows": total_rows,
        "hidden_dim": hidden_dim,
        "layer_index": layer_idx,
        "metadata_path": str(metadata_file),
        "embeddings_memmap": str(memmap_file),
        "shape_path": str(shape_file),
        "processed_problem_ids": processed_problem_ids,
        "skipped_problem_ids": skipped_problem_ids,
        "skip_reasons": skip_reasons,
        "cache_reused": False,
    }


def load_cached_embeddings(
    embeddings_memmap_path: str | Path,
    embeddings_shape_path: str | Path,
) -> np.memmap:
    """Load cached memmap embeddings."""
    with Path(embeddings_shape_path).open("r", encoding="utf-8") as handle:
        shape_payload = json.load(handle)
    storage_dtype = resolve_numpy_dtype(shape_payload.get("dtype", "float32"))
    return np.memmap(
        embeddings_memmap_path,
        dtype=storage_dtype,
        mode="r",
        shape=(shape_payload["rows"], shape_payload["hidden_dim"]),
    )
