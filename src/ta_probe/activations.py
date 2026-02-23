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
from ta_probe.storage import estimate_embedding_storage


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
    tokens = _tokenize_text_for_forward(
        text=text,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )
    layers = get_transformer_layers(model)

    captured: dict[str, torch.Tensor] = {}

    def hook_fn(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = hidden.detach().to("cpu", dtype=torch.float32)

    hook = layers[layer_idx].register_forward_hook(hook_fn)
    try:
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


def _tokenize_text_for_forward(
    *,
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    device: str,
) -> dict[str, torch.Tensor]:
    resolved_device = resolve_device(device)
    tokens = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
        return_attention_mask=True,
        return_token_type_ids=False,
    )
    tokens = {key: value.to(resolved_device) for key, value in tokens.items()}
    _validate_sequence_length_for_model(
        seq_len=int(tokens["input_ids"].shape[1]),
        model=model,
        tokenizer=tokenizer,
    )
    return tokens


def _validate_sequence_length_for_model(
    *,
    seq_len: int,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    max_positions = getattr(model.config, "max_position_embeddings", None)
    tokenizer_max = tokenizer.model_max_length
    valid_limits = [limit for limit in [max_positions, tokenizer_max] if isinstance(limit, int)]
    if not valid_limits:
        return
    max_len = min(valid_limits)
    if seq_len > max_len:
        msg = (
            f"Sequence length {seq_len} exceeds model context {max_len}. "
            "Use a longer-context model or choose shorter traces."
        )
        raise ValueError(msg)


def _capture_layer_attention_for_forward(
    *,
    model: PreTrainedModel,
    layer_idx: int,
    forward_kwargs: dict[str, torch.Tensor | tuple[Any, ...] | Any],
) -> torch.Tensor:
    """Capture one layer's attention tensor without enabling attentions globally."""
    layers = get_transformer_layers(model)
    layer = layers[layer_idx]
    attention_module = getattr(layer, "self_attn", None)
    if attention_module is None or not hasattr(attention_module, "forward"):
        msg = "Could not find self_attn module for vertical attention extraction."
        raise ValueError(msg)

    captured: dict[str, torch.Tensor] = {}
    original_forward = attention_module.forward

    def wrapped_forward(*args, **kwargs):
        kwargs["output_attentions"] = True
        outputs = original_forward(*args, **kwargs)
        attn_weights = outputs[1] if isinstance(outputs, tuple) and len(outputs) > 1 else None
        if isinstance(attn_weights, torch.Tensor):
            captured["attention"] = attn_weights.detach()
        return outputs

    attention_module.forward = wrapped_forward
    try:
        with torch.no_grad():
            _ = model(**forward_kwargs)
    finally:
        attention_module.forward = original_forward

    if "attention" not in captured:
        msg = "Failed to capture attention weights from selected layer."
        raise RuntimeError(msg)
    return captured["attention"]


def _extract_light_attention_for_text(
    *,
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    layer_idx: int,
    device: str,
    last_n_tokens: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract attention from only the last N tokens to all previous tokens."""
    tokens = _tokenize_text_for_forward(
        text=text,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )
    input_ids = tokens["input_ids"]
    attention_mask = tokens.get("attention_mask")
    seq_len = int(input_ids.shape[1])
    if seq_len == 0:
        msg = "Cannot compute light attention for empty token sequence."
        raise ValueError(msg)

    n_query = min(int(last_n_tokens), seq_len)
    prefix_len = seq_len - n_query

    past_key_values = None
    if prefix_len > 0:
        prefix_kwargs: dict[str, Any] = {
            "input_ids": input_ids[:, :prefix_len],
            "use_cache": True,
        }
        if attention_mask is not None:
            prefix_kwargs["attention_mask"] = attention_mask[:, :prefix_len]
        with torch.no_grad():
            prefix_outputs = model(**prefix_kwargs)
        past_key_values = getattr(prefix_outputs, "past_key_values", None)
        if past_key_values is None:
            msg = "Model did not return past_key_values for light attention extraction."
            raise RuntimeError(msg)

    query_kwargs: dict[str, Any] = {
        "input_ids": input_ids[:, prefix_len:],
        "use_cache": False,
    }
    if attention_mask is not None:
        # With cached KV, attention_mask should cover prefix+query tokens.
        query_kwargs["attention_mask"] = attention_mask
    if past_key_values is not None:
        query_kwargs["past_key_values"] = past_key_values

    attention = _capture_layer_attention_for_forward(
        model=model,
        layer_idx=layer_idx,
        forward_kwargs=query_kwargs,
    )
    if attention.dim() != 4:
        msg = f"Expected attention with 4 dims, got {tuple(attention.shape)}"
        raise RuntimeError(msg)

    # [heads, query_tokens, key_tokens]
    attention_np = attention[0].to("cpu", dtype=torch.float32).numpy()
    query_positions = np.arange(prefix_len, seq_len, dtype=np.int64)
    return attention_np, query_positions


def _extract_full_attention_for_text(
    *,
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    layer_idx: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract full-token attention for a single layer."""
    tokens = _tokenize_text_for_forward(
        text=text,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )
    seq_len = int(tokens["input_ids"].shape[1])
    attention = _capture_layer_attention_for_forward(
        model=model,
        layer_idx=layer_idx,
        forward_kwargs={**tokens, "use_cache": False},
    )
    if attention.dim() != 4:
        msg = f"Expected attention with 4 dims, got {tuple(attention.shape)}"
        raise RuntimeError(msg)
    attention_np = attention[0].to("cpu", dtype=torch.float32).numpy()
    query_positions = np.arange(seq_len, dtype=np.int64)
    return attention_np, query_positions


def _sanitize_token_boundaries(
    token_boundaries: list[tuple[int, int]], key_len: int
) -> list[tuple[int, int]]:
    sanitized: list[tuple[int, int]] = []
    for start, end in token_boundaries:
        clamped_start = max(0, min(int(start), key_len))
        clamped_end = max(clamped_start, min(int(end), key_len))
        sanitized.append((clamped_start, clamped_end))
    return sanitized


def _compute_sentence_to_sentence_attention(
    attention: np.ndarray,
    *,
    token_boundaries: list[tuple[int, int]],
    query_positions: np.ndarray,
) -> np.ndarray:
    """Aggregate token attention into a sentence-to-sentence matrix."""
    if attention.ndim != 3:
        msg = f"Expected attention with shape [heads, query, key], got {attention.shape}"
        raise ValueError(msg)
    if query_positions.ndim != 1:
        msg = "query_positions must be a 1D array."
        raise ValueError(msg)

    _heads, num_queries, key_len = attention.shape
    if num_queries != query_positions.shape[0]:
        msg = (
            "query_positions length does not match attention query axis: "
            f"{query_positions.shape[0]} vs {num_queries}."
        )
        raise ValueError(msg)

    boundaries = _sanitize_token_boundaries(token_boundaries, key_len)
    num_sentences = len(boundaries)
    matrix = np.full((num_sentences, num_sentences), np.nan, dtype=np.float32)
    attn_mean = attention.mean(axis=0, dtype=np.float32)

    for query_sentence_idx, (q_start, q_end) in enumerate(boundaries):
        if q_end <= q_start:
            continue
        query_mask = (query_positions >= q_start) & (query_positions < q_end)
        if not query_mask.any():
            continue
        query_rows = np.nonzero(query_mask)[0]
        for key_sentence_idx, (k_start, k_end) in enumerate(boundaries):
            if k_end <= k_start:
                continue
            # Sum over key tokens, then average over query tokens in the sentence.
            key_mass = attn_mean[np.ix_(query_rows, np.arange(k_start, k_end))].sum(axis=1)
            matrix[query_sentence_idx, key_sentence_idx] = float(key_mass.mean())

    return matrix


def _compute_vertical_scores_from_sentence_matrix(
    sentence_attention: np.ndarray,
    *,
    depth_control: bool,
) -> np.ndarray:
    num_sentences = int(sentence_attention.shape[0])
    scores = np.zeros(num_sentences, dtype=np.float32)
    for sentence_idx in range(num_sentences):
        later = sentence_attention[sentence_idx + 1 :, sentence_idx]
        later = later[~np.isnan(later)]
        if later.size == 0:
            continue
        if depth_control:
            scores[sentence_idx] = float(later.mean())
        else:
            scores[sentence_idx] = float(later.sum())
    return scores


def _compute_vertical_scores_from_token_attention(
    attention: np.ndarray,
    *,
    token_boundaries: list[tuple[int, int]],
    query_positions: np.ndarray,
    depth_control: bool,
) -> np.ndarray:
    """Compute per-sentence vertical attention from token-level attention."""
    if attention.ndim != 3:
        msg = f"Expected attention with shape [heads, query, key], got {attention.shape}"
        raise ValueError(msg)

    _heads, _num_queries, key_len = attention.shape
    boundaries = _sanitize_token_boundaries(token_boundaries, key_len)
    attn_mean = attention.mean(axis=0, dtype=np.float32)
    scores = np.zeros(len(boundaries), dtype=np.float32)

    for sentence_idx, (_start, end) in enumerate(boundaries):
        start = boundaries[sentence_idx][0]
        if end <= start:
            continue
        sentence_mass = attn_mean[:, start:end].sum(axis=1)
        eligible = query_positions >= end
        if not eligible.any():
            continue
        values = sentence_mass[eligible]
        if depth_control:
            scores[sentence_idx] = float(values.mean())
        else:
            scores[sentence_idx] = float(values.sum())

    return scores


def _estimate_sequence_length(text: str, tokenizer: PreTrainedTokenizerBase) -> int:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return int(encoded["input_ids"].shape[1])


def compute_vertical_attention_scores(
    *,
    text: str,
    token_boundaries: list[tuple[int, int]],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    layer_idx: int,
    device: str,
    mode: str,
    depth_control: bool,
    light_last_n_tokens: int,
    full_max_seq_len: int,
) -> tuple[np.ndarray | None, str]:
    """Compute per-sentence vertical attention scores using light or full mode."""
    normalized_mode = str(mode).lower()
    if normalized_mode == "off":
        return None, "off"

    if normalized_mode == "light":
        attention, query_positions = _extract_light_attention_for_text(
            text=text,
            model=model,
            tokenizer=tokenizer,
            layer_idx=layer_idx,
            device=device,
            last_n_tokens=light_last_n_tokens,
        )
        return (
            _compute_vertical_scores_from_token_attention(
                attention,
                token_boundaries=token_boundaries,
                query_positions=query_positions,
                depth_control=depth_control,
            ),
            "light",
        )

    if normalized_mode != "full":
        msg = f"Unsupported vertical attention mode: {mode}"
        raise ValueError(msg)

    seq_len = _estimate_sequence_length(text, tokenizer)
    if seq_len <= int(full_max_seq_len):
        attention, query_positions = _extract_full_attention_for_text(
            text=text,
            model=model,
            tokenizer=tokenizer,
            layer_idx=layer_idx,
            device=device,
        )
        sentence_attention = _compute_sentence_to_sentence_attention(
            attention,
            token_boundaries=token_boundaries,
            query_positions=query_positions,
        )
        return (
            _compute_vertical_scores_from_sentence_matrix(
                sentence_attention,
                depth_control=depth_control,
            ),
            "full",
        )

    attention, query_positions = _extract_light_attention_for_text(
        text=text,
        model=model,
        tokenizer=tokenizer,
        layer_idx=layer_idx,
        device=device,
        last_n_tokens=light_last_n_tokens,
    )
    return (
        _compute_vertical_scores_from_token_attention(
            attention,
            token_boundaries=token_boundaries,
            query_positions=query_positions,
            depth_control=depth_control,
        ),
        "light_fallback",
    )


def pool_sentence_embeddings(
    hidden_states: np.ndarray,
    token_boundaries: list[tuple[int, int]],
) -> np.ndarray:
    """Mean-pool token activations per sentence span."""
    spans, _ = collect_sentence_token_spans(hidden_states, token_boundaries)
    if not spans:
        return np.zeros((0, hidden_states.shape[1]), dtype=np.float32)
    return np.stack([span.mean(axis=0) for span in spans], axis=0).astype(np.float32, copy=False)


def collect_sentence_token_spans(
    hidden_states: np.ndarray,
    token_boundaries: list[tuple[int, int]],
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """Return per-sentence token spans plus normalized token boundaries."""
    seq_len, hidden_dim = hidden_states.shape
    if seq_len == 0:
        msg = "Hidden states sequence length must be positive"
        raise ValueError(msg)
    if hidden_dim <= 0:
        msg = "Hidden states hidden_dim must be positive"
        raise ValueError(msg)
    spans: list[np.ndarray] = []
    normalized_boundaries: list[tuple[int, int]] = []

    for start, end in token_boundaries:
        if start < 0 or start >= seq_len:
            msg = f"Token boundary start {start} out of range for seq_len={seq_len}"
            raise ValueError(msg)
        end = min(end, seq_len)
        if end <= start:
            end = start + 1
        spans.append(hidden_states[start:end])
        normalized_boundaries.append((start, end))

    return spans, normalized_boundaries


def _build_problem_hidden_and_frame(
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
) -> tuple[np.ndarray, list[tuple[int, int]], pd.DataFrame, str]:
    """Extract hidden states and sentence metadata rows for one problem."""
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

    _, normalized_boundaries = collect_sentence_token_spans(hidden, usable_boundaries)

    label_frame = build_problem_label_frame(
        problem_id=problem_id,
        chunks_labeled=problem_data["chunks_labeled"],
        counterfactual_field=counterfactual_field,
        percentile=anchor_percentile,
        drop_last_chunk=drop_last_chunk,
    )

    if len(label_frame) != len(normalized_boundaries):
        msg = (
            f"Row mismatch for problem {problem_id}: "
            f"labels={len(label_frame)}, boundaries={len(normalized_boundaries)}"
        )
        raise ValueError(msg)

    label_frame["token_start"] = [start for start, _ in normalized_boundaries]
    label_frame["token_end"] = [end for _, end in normalized_boundaries]
    label_frame["token_count"] = label_frame["token_end"] - label_frame["token_start"]
    denom = np.maximum(label_frame["num_chunks"].to_numpy(dtype=np.float32) - 1, 1)
    label_frame["relative_position"] = label_frame["chunk_idx"].to_numpy(dtype=np.float32) / denom
    label_frame["source_text_len"] = len(text)
    return hidden, normalized_boundaries, label_frame, text


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
    vertical_attention_mode: str = "off",
    vertical_attention_depth_control: bool = True,
    vertical_attention_light_last_n_tokens: int = 1,
    vertical_attention_full_max_seq_len: int = 1024,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Extract mean-pooled sentence vectors and metadata rows for one problem."""
    hidden, usable_boundaries, label_frame, text = _build_problem_hidden_and_frame(
        problem_id=problem_id,
        problem_data=problem_data,
        model=model,
        tokenizer=tokenizer,
        layer_idx=layer_idx,
        counterfactual_field=counterfactual_field,
        anchor_percentile=anchor_percentile,
        drop_last_chunk=drop_last_chunk,
        device=device,
    )
    vectors = pool_sentence_embeddings(hidden, usable_boundaries)
    vertical_scores, mode_used = compute_vertical_attention_scores(
        text=text,
        token_boundaries=usable_boundaries,
        model=model,
        tokenizer=tokenizer,
        layer_idx=layer_idx,
        device=device,
        mode=vertical_attention_mode,
        depth_control=vertical_attention_depth_control,
        light_last_n_tokens=vertical_attention_light_last_n_tokens,
        full_max_seq_len=vertical_attention_full_max_seq_len,
    )
    if vertical_scores is not None:
        if len(vertical_scores) != len(label_frame):
            msg = (
                f"Vertical attention row mismatch for problem {problem_id}: "
                f"scores={len(vertical_scores)}, labels={len(label_frame)}"
            )
            raise ValueError(msg)
        label_frame = label_frame.copy()
        label_frame["vertical_score"] = vertical_scores.astype(np.float32, copy=False)
        label_frame["vertical_attention_mode_used"] = mode_used
    return vectors, label_frame


def _build_storage_estimate(
    *,
    metadata_df: pd.DataFrame,
    hidden_dim: int,
    storage_dtype_name: str,
    memmap_file: Path,
    metadata_file: Path,
    embedding_layout: str | None = None,
) -> dict[str, Any]:
    normalized_layout = (embedding_layout or "").strip().lower()
    if normalized_layout not in {"sentence_mean", "token_ragged"}:
        if "pooling" in metadata_df.columns:
            pooling_values = (
                metadata_df["pooling"]
                .dropna()
                .astype(str)
                .str.lower()
                .unique()
                .tolist()
            )
            if len(pooling_values) == 1 and pooling_values[0] == "tokens":
                normalized_layout = "token_ragged"
            else:
                normalized_layout = "sentence_mean"
        elif "token_length" in metadata_df.columns:
            normalized_layout = "token_ragged"
        else:
            normalized_layout = "sentence_mean"

    token_column = ""
    if "token_length" in metadata_df.columns:
        token_column = "token_length"
    elif "token_count" in metadata_df.columns:
        token_column = "token_count"

    total_tokens: int | None = None
    if token_column:
        total_tokens = int(metadata_df[token_column].to_numpy(dtype=np.int64).sum())
    elif normalized_layout == "token_ragged":
        # Fallback for legacy metadata that omits token counts.
        total_tokens = int(len(metadata_df))

    estimate = estimate_embedding_storage(
        embedding_layout=normalized_layout,  # type: ignore[arg-type]
        hidden_dim=hidden_dim,
        storage_dtype_name=storage_dtype_name,
        num_sentences=int(len(metadata_df)),
        total_tokens=total_tokens,
    )
    estimate["token_count_column"] = token_column or (
        "row_count" if normalized_layout == "token_ragged" else "none"
    )
    estimate["embedding_rows_basis"] = (
        "num_sentences" if normalized_layout == "sentence_mean" else (token_column or "row_count")
    )
    if memmap_file.exists():
        estimate["embedding_bytes_actual"] = int(memmap_file.stat().st_size)
    if metadata_file.exists():
        estimate["metadata_bytes_actual"] = int(metadata_file.stat().st_size)
    if "embedding_bytes_actual" in estimate or "metadata_bytes_actual" in estimate:
        estimate["total_bytes_actual"] = int(
            estimate.get("embedding_bytes_actual", 0) + estimate.get("metadata_bytes_actual", 0)
        )
    return estimate


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
    vertical_attention_mode: str = "off",
    vertical_attention_depth_control: bool = True,
    vertical_attention_light_last_n_tokens: int = 1,
    vertical_attention_full_max_seq_len: int = 1024,
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
    if pooling not in {"mean", "tokens"}:
        msg = f"Unsupported pooling mode: {pooling}"
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
        "requested_layer_index": (int(layer_index) if layer_index is not None else None),
        "pooling": pooling,
        "counterfactual_field": counterfactual_field,
        "anchor_percentile": float(anchor_percentile),
        "drop_last_chunk": bool(drop_last_chunk),
        "dtype": storage_dtype_name,
        "compute_dtype": dtype_name,
        "vertical_attention_mode": vertical_attention_mode,
        "vertical_attention_depth_control": bool(vertical_attention_depth_control),
        "vertical_attention_light_last_n_tokens": int(vertical_attention_light_last_n_tokens),
        "vertical_attention_full_max_seq_len": int(vertical_attention_full_max_seq_len),
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
            metadata_df = pd.read_parquet(metadata_file)
            available_problem_ids = set(metadata_df["problem_id"].to_numpy(dtype=np.int64).tolist())
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

            storage_estimate = _build_storage_estimate(
                metadata_df=metadata_df,
                hidden_dim=int(cached_shape["hidden_dim"]),
                storage_dtype_name=str(cached_shape.get("dtype", storage_dtype_name)),
                memmap_file=memmap_file,
                metadata_file=metadata_file,
                embedding_layout=str(cached_shape.get("embedding_layout", "")),
            )

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
                "storage_estimate": storage_estimate,
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
    extracted_token_blocks: list[np.ndarray] = []
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
            if pooling == "mean":
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
                    vertical_attention_mode=vertical_attention_mode,
                    vertical_attention_depth_control=vertical_attention_depth_control,
                    vertical_attention_light_last_n_tokens=vertical_attention_light_last_n_tokens,
                    vertical_attention_full_max_seq_len=vertical_attention_full_max_seq_len,
                )
                current_hidden_dim = int(vectors.shape[1]) if vectors.size else None
                if current_hidden_dim is not None:
                    if hidden_dim is None:
                        hidden_dim = current_hidden_dim
                    elif current_hidden_dim != hidden_dim:
                        msg = (
                            f"Hidden dimension mismatch for problem {problem_id}: "
                            f"expected {hidden_dim}, got {current_hidden_dim}"
                        )
                        raise ValueError(msg)
                extracted_vectors.append(vectors)
                extracted_frames.append(frame)
            else:
                hidden, boundaries, frame, text = _build_problem_hidden_and_frame(
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
                vertical_scores, mode_used = compute_vertical_attention_scores(
                    text=text,
                    token_boundaries=boundaries,
                    model=model,
                    tokenizer=tokenizer,
                    layer_idx=layer_idx,
                    device=device,
                    mode=vertical_attention_mode,
                    depth_control=vertical_attention_depth_control,
                    light_last_n_tokens=vertical_attention_light_last_n_tokens,
                    full_max_seq_len=vertical_attention_full_max_seq_len,
                )
                if vertical_scores is not None:
                    if len(vertical_scores) != len(frame):
                        msg = (
                            f"Vertical attention row mismatch for problem {problem_id}: "
                            f"scores={len(vertical_scores)}, labels={len(frame)}"
                        )
                        raise ValueError(msg)
                    frame = frame.copy()
                    frame["vertical_score"] = vertical_scores.astype(np.float32, copy=False)
                    frame["vertical_attention_mode_used"] = mode_used
                spans, _ = collect_sentence_token_spans(hidden, boundaries)
                lengths = np.asarray([span.shape[0] for span in spans], dtype=np.int64)
                if spans:
                    token_block = np.concatenate(spans, axis=0).astype(np.float32, copy=False)
                else:
                    token_block = np.zeros((0, hidden.shape[1]), dtype=np.float32)

                current_hidden_dim = int(token_block.shape[1]) if token_block.ndim == 2 else None
                if current_hidden_dim is not None:
                    if hidden_dim is None:
                        hidden_dim = current_hidden_dim
                    elif current_hidden_dim != hidden_dim:
                        msg = (
                            f"Hidden dimension mismatch for problem {problem_id}: "
                            f"expected {hidden_dim}, got {current_hidden_dim}"
                        )
                        raise ValueError(msg)

                frame = frame.copy()
                frame["token_length"] = lengths
                extracted_token_blocks.append(token_block)
                extracted_frames.append(frame)
        except Exception as exc:
            if not skip_failed_problems:
                raise
            skipped_problem_ids.append(problem_id)
            skip_reasons[str(problem_id)] = str(exc)
            continue

        processed_problem_ids.append(problem_id)

    if not extracted_frames or hidden_dim is None:
        msg = "No problem embeddings were extracted successfully"
        if skip_reasons:
            msg += f". Reasons: {skip_reasons}"
        raise RuntimeError(msg)

    memmap_file.parent.mkdir(parents=True, exist_ok=True)
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    shape_file.parent.mkdir(parents=True, exist_ok=True)

    frame_parts: list[pd.DataFrame] = []
    storage_dtype = resolve_numpy_dtype(storage_dtype_name)
    if pooling == "mean":
        total_rows = int(sum(len(frame) for frame in extracted_frames))
    else:
        total_rows = int(sum(block.shape[0] for block in extracted_token_blocks))

    memmap = np.memmap(
        memmap_file,
        dtype=storage_dtype,
        mode="w+",
        shape=(total_rows, hidden_dim),
    )

    offset = 0
    if pooling == "mean":
        for vectors, frame in zip(extracted_vectors, extracted_frames, strict=True):
            n_rows = len(frame)
            memmap[offset : offset + n_rows] = vectors.astype(storage_dtype, copy=False)
            frame = frame.copy()
            frame["embedding_row"] = np.arange(offset, offset + n_rows, dtype=np.int64)
            frame_parts.append(frame)
            offset += n_rows
    else:
        for token_block, frame in zip(extracted_token_blocks, extracted_frames, strict=True):
            frame = frame.copy()
            lengths = frame["token_length"].to_numpy(dtype=np.int64)
            expected_tokens = int(lengths.sum())
            local_offsets = np.zeros(len(lengths), dtype=np.int64)
            if len(lengths) > 1:
                local_offsets[1:] = np.cumsum(lengths[:-1], dtype=np.int64)
            frame["token_offset"] = offset + local_offsets
            frame["token_length"] = lengths
            n_tokens = int(token_block.shape[0])
            if n_tokens != expected_tokens:
                msg = (
                    "Token block length mismatch while writing ragged cache: "
                    f"expected={expected_tokens}, got={n_tokens}"
                )
                raise RuntimeError(msg)
            memmap[offset : offset + n_tokens] = token_block.astype(storage_dtype, copy=False)
            frame_parts.append(frame)
            offset += n_tokens

    if offset != total_rows:
        msg = f"Expected {total_rows} rows but wrote {offset}"
        raise RuntimeError(msg)

    memmap.flush()
    metadata_df = pd.concat(frame_parts, ignore_index=True)
    metadata_df["hidden_dim"] = int(hidden_dim)
    metadata_df["layer_index"] = int(layer_idx)
    metadata_df["storage_dtype"] = storage_dtype_name
    metadata_df["compute_dtype"] = dtype_name
    metadata_df["pooling"] = pooling
    metadata_df["repo_id"] = repo_id
    metadata_df["model_dir"] = model_dir
    metadata_df["temp_dir"] = temp_dir
    metadata_df["split_dir"] = split_dir
    metadata_df["model_name_or_path"] = model_name_or_path
    metadata_df.to_parquet(metadata_file, index=False)

    token_count_column = ""
    if "token_length" in metadata_df.columns:
        token_count_column = "token_length"
    elif "token_count" in metadata_df.columns:
        token_count_column = "token_count"
    if token_count_column:
        total_tokens = int(metadata_df[token_count_column].to_numpy(dtype=np.int64).sum())
    else:
        # Fallback for synthetic/minimal metadata in tests and legacy artifacts.
        total_tokens = int(len(metadata_df))

    # Store provenance so training can detect stale caches after config changes.
    shape_payload = {
        "rows": int(total_rows),
        "hidden_dim": int(hidden_dim),
        "dtype": storage_dtype_name,
        "layer_index": int(layer_idx),
        "layer_mode": layer_mode,
        "requested_layer_index": (int(layer_index) if layer_index is not None else None),
        "pooling": pooling,
        "embedding_layout": "token_ragged" if pooling == "tokens" else "sentence_mean",
        "num_sentences": int(len(metadata_df)),
        "total_tokens": int(total_tokens),
        "repo_id": repo_id,
        "model_dir": model_dir,
        "temp_dir": temp_dir,
        "split_dir": split_dir,
        "compute_dtype": dtype_name,
        "counterfactual_field": counterfactual_field,
        "anchor_percentile": float(anchor_percentile),
        "drop_last_chunk": bool(drop_last_chunk),
        "model_name_or_path": model_name_or_path,
        "vertical_attention_mode": vertical_attention_mode,
        "vertical_attention_depth_control": bool(vertical_attention_depth_control),
        "vertical_attention_light_last_n_tokens": int(vertical_attention_light_last_n_tokens),
        "vertical_attention_full_max_seq_len": int(vertical_attention_full_max_seq_len),
    }
    with shape_file.open("w", encoding="utf-8") as handle:
        json.dump(shape_payload, handle, indent=2, sort_keys=True)

    storage_estimate = _build_storage_estimate(
        metadata_df=metadata_df,
        hidden_dim=int(hidden_dim),
        storage_dtype_name=storage_dtype_name,
        memmap_file=memmap_file,
        metadata_file=metadata_file,
        embedding_layout=("token_ragged" if pooling == "tokens" else "sentence_mean"),
    )

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
        "pooling": pooling,
        "metadata_path": str(metadata_file),
        "embeddings_memmap": str(memmap_file),
        "shape_path": str(shape_file),
        "processed_problem_ids": processed_problem_ids,
        "skipped_problem_ids": skipped_problem_ids,
        "skip_reasons": skip_reasons,
        "storage_estimate": storage_estimate,
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
