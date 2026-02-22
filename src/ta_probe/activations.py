"""Activation extraction and sentence embedding cache creation."""

from __future__ import annotations

import json
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

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
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
    vectors = np.zeros((len(token_boundaries), hidden_dim), dtype=np.float32)

    for idx, (start, end) in enumerate(token_boundaries):
        if start < 0 or end > seq_len:
            msg = f"Token boundary ({start}, {end}) out of range for seq_len={seq_len}"
            raise ValueError(msg)
        if end <= start:
            end = min(start + 1, seq_len)
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
    embeddings_memmap_path: str | Path,
    embeddings_shape_path: str | Path,
    metadata_path: str | Path,
) -> dict[str, Any]:
    """Extract sentence vectors and save memmap plus metadata."""
    if not problem_ids:
        msg = "No problem IDs provided"
        raise ValueError(msg)

    metadata_cache: dict[int, dict[str, Any]] = {}
    total_rows = 0
    for problem_id in problem_ids:
        payload = load_problem_metadata(
            problem_id=problem_id,
            repo_id=repo_id,
            model_dir=model_dir,
            temp_dir=temp_dir,
            split_dir=split_dir,
        )
        metadata_cache[problem_id] = payload
        n_chunks = len(payload["chunks_labeled"])
        total_rows += max(0, n_chunks - 1) if drop_last_chunk else n_chunks

    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=model_name_or_path,
        device=device,
        dtype_name=dtype_name,
    )
    layer_idx = resolve_layer_index(model, layer_mode, layer_index)

    memmap_file = Path(embeddings_memmap_path)
    memmap_file.parent.mkdir(parents=True, exist_ok=True)

    metadata_file = Path(metadata_path)
    metadata_file.parent.mkdir(parents=True, exist_ok=True)

    shape_file = Path(embeddings_shape_path)
    shape_file.parent.mkdir(parents=True, exist_ok=True)

    memmap: np.memmap | None = None
    frame_parts: list[pd.DataFrame] = []
    offset = 0
    hidden_dim = 0

    for problem_id in problem_ids:
        vectors, frame = _build_problem_vectors(
            problem_id=problem_id,
            problem_data=metadata_cache[problem_id],
            model=model,
            tokenizer=tokenizer,
            layer_idx=layer_idx,
            counterfactual_field=counterfactual_field,
            anchor_percentile=anchor_percentile,
            drop_last_chunk=drop_last_chunk,
            device=device,
        )

        if memmap is None:
            hidden_dim = vectors.shape[1]
            memmap = np.memmap(
                memmap_file,
                dtype=np.float32,
                mode="w+",
                shape=(total_rows, hidden_dim),
            )

        n_rows = len(frame)
        memmap[offset : offset + n_rows] = vectors
        frame = frame.copy()
        frame["embedding_row"] = np.arange(offset, offset + n_rows, dtype=np.int64)
        frame_parts.append(frame)
        offset += n_rows

    if memmap is None:
        msg = "Failed to create embedding memmap"
        raise RuntimeError(msg)

    if offset != total_rows:
        msg = f"Expected {total_rows} rows but wrote {offset}"
        raise RuntimeError(msg)

    memmap.flush()
    metadata_df = pd.concat(frame_parts, ignore_index=True)
    metadata_df.to_parquet(metadata_file, index=False)

    shape_payload = {
        "rows": int(total_rows),
        "hidden_dim": int(hidden_dim),
        "dtype": "float32",
        "layer_index": int(layer_idx),
    }
    with shape_file.open("w", encoding="utf-8") as handle:
        json.dump(shape_payload, handle, indent=2, sort_keys=True)

    return {
        "rows": total_rows,
        "hidden_dim": hidden_dim,
        "layer_index": layer_idx,
        "metadata_path": str(metadata_file),
        "embeddings_memmap": str(memmap_file),
        "shape_path": str(shape_file),
    }


def load_cached_embeddings(
    embeddings_memmap_path: str | Path,
    embeddings_shape_path: str | Path,
) -> np.memmap:
    """Load cached memmap embeddings."""
    with Path(embeddings_shape_path).open("r", encoding="utf-8") as handle:
        shape_payload = json.load(handle)
    return np.memmap(
        embeddings_memmap_path,
        dtype=np.float32,
        mode="r",
        shape=(shape_payload["rows"], shape_payload["hidden_dim"]),
    )
