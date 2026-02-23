"""Torch token-level probe models and training utilities."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ta_probe.metrics import mean_problem_spearman, precision_recall_auc


def resolve_probe_device(device_name: str) -> torch.device:
    """Resolve probe-training device from config."""
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for deterministic probe training."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class TokenSpanDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Sentence-level view over ragged token activations."""

    def __init__(
        self,
        *,
        frame: pd.DataFrame,
        token_embeddings: np.memmap | np.ndarray,
        target_col: str,
    ) -> None:
        if "token_offset" not in frame.columns or "token_length" not in frame.columns:
            msg = "TokenSpanDataset requires token_offset and token_length columns"
            raise ValueError(msg)

        self.offsets = frame["token_offset"].to_numpy(dtype=np.int64)
        self.lengths = frame["token_length"].to_numpy(dtype=np.int64)
        self.targets = frame[target_col].to_numpy(dtype=np.float32)
        self.token_embeddings = token_embeddings

        if np.any(self.lengths <= 0):
            msg = "token_length must be positive for all rows"
            raise ValueError(msg)

    def __len__(self) -> int:
        return int(self.offsets.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        offset = int(self.offsets[idx])
        length = int(self.lengths[idx])
        span = np.asarray(self.token_embeddings[offset : offset + length], dtype=np.float32)
        tokens = torch.from_numpy(np.ascontiguousarray(span))
        target = torch.tensor(float(self.targets[idx]), dtype=torch.float32)
        return tokens, target


def collate_token_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length token spans and build a boolean mask."""
    if not batch:
        msg = "Cannot collate empty batch"
        raise ValueError(msg)

    token_tensors, targets = zip(*batch, strict=True)
    lengths = [int(tokens.shape[0]) for tokens in token_tensors]
    if any(length <= 0 for length in lengths):
        msg = "All token spans must have positive length"
        raise ValueError(msg)

    batch_size = len(token_tensors)
    max_len = max(lengths)
    hidden_dim = int(token_tensors[0].shape[1])

    padded = torch.zeros((batch_size, max_len, hidden_dim), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    for row_idx, tokens in enumerate(token_tensors):
        length = int(tokens.shape[0])
        padded[row_idx, :length] = tokens
        mask[row_idx, :length] = True

    target_tensor = torch.stack(targets).to(dtype=torch.float32)
    return padded, mask, target_tensor


def _build_token_mlp(
    *,
    input_dim: int,
    width: int,
    depth: int,
) -> tuple[nn.Module, int]:
    if depth <= 0:
        return nn.Identity(), input_dim
    layers: list[nn.Module] = []
    current_dim = input_dim
    for _ in range(depth):
        layers.append(nn.Linear(current_dim, width))
        layers.append(nn.ReLU())
        current_dim = width
    return nn.Sequential(*layers), current_dim


class AttentionProbe(nn.Module):
    """Attention-style token probe with learned query/value projections."""

    def __init__(
        self,
        *,
        input_dim: int,
        num_heads: int,
        mlp_width: int,
        mlp_depth: int,
    ) -> None:
        super().__init__()
        self.token_mlp, proj_dim = _build_token_mlp(
            input_dim=input_dim,
            width=mlp_width,
            depth=mlp_depth,
        )
        self.query = nn.Parameter(torch.empty(num_heads, proj_dim))
        self.value = nn.Parameter(torch.empty(num_heads, proj_dim))
        self.head_mixer = nn.Linear(num_heads, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.query)
        nn.init.xavier_uniform_(self.value)
        nn.init.xavier_uniform_(self.head_mixer.weight)
        nn.init.zeros_(self.head_mixer.bias)

    def forward(self, token_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        token_repr = self.token_mlp(token_states)
        attn_logits = torch.einsum("btd,hd->bth", token_repr, self.query)
        mask_expanded = mask.unsqueeze(-1)
        attn_logits = attn_logits.masked_fill(
            ~mask_expanded, torch.finfo(attn_logits.dtype).min
        )
        attn_weights = torch.softmax(attn_logits, dim=1)
        attn_weights = attn_weights * mask_expanded
        attn_weights = attn_weights / attn_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

        token_values = torch.einsum("btd,hd->bth", token_repr, self.value)
        pooled = (attn_weights * token_values).sum(dim=1)
        logits = self.head_mixer(pooled).squeeze(-1)
        return logits


class MultiMaxProbe(nn.Module):
    """Hard-max multi-head probe over token states."""

    def __init__(
        self,
        *,
        input_dim: int,
        num_heads: int,
        mlp_width: int,
        mlp_depth: int,
    ) -> None:
        super().__init__()
        self.token_mlp, proj_dim = _build_token_mlp(
            input_dim=input_dim,
            width=mlp_width,
            depth=mlp_depth,
        )
        self.head_direction = nn.Parameter(torch.empty(num_heads, proj_dim))
        self.head_mixer = nn.Linear(num_heads, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.head_direction)
        nn.init.xavier_uniform_(self.head_mixer.weight)
        nn.init.zeros_(self.head_mixer.bias)

    def forward(self, token_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        token_repr = self.token_mlp(token_states)
        token_scores = torch.einsum("btd,hd->bth", token_repr, self.head_direction)
        token_scores = token_scores.masked_fill(
            ~mask.unsqueeze(-1), torch.finfo(token_scores.dtype).min
        )
        pooled = token_scores.max(dim=1).values
        logits = self.head_mixer(pooled).squeeze(-1)
        return logits


@dataclass(frozen=True)
class TokenProbeTrainConfig:
    """Training configuration for token-level torch probes."""

    num_heads: int
    mlp_width: int
    mlp_depth: int
    batch_size: int
    max_epochs: int
    patience: int
    learning_rate: float
    weight_decay: float
    continuous_loss: Literal["mse", "huber"]
    device: str


def _build_model(
    *,
    probe_type: Literal["attention_probe", "multimax_probe"],
    input_dim: int,
    config: TokenProbeTrainConfig,
) -> nn.Module:
    if probe_type == "attention_probe":
        return AttentionProbe(
            input_dim=input_dim,
            num_heads=config.num_heads,
            mlp_width=config.mlp_width,
            mlp_depth=config.mlp_depth,
        )
    if probe_type == "multimax_probe":
        return MultiMaxProbe(
            input_dim=input_dim,
            num_heads=config.num_heads,
            mlp_width=config.mlp_width,
            mlp_depth=config.mlp_depth,
        )
    msg = f"Unsupported probe_type: {probe_type}"
    raise ValueError(msg)


def _predict_scores(
    *,
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
    is_continuous: bool,
) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for tokens, mask, _targets in loader:
            tokens = tokens.to(device=device, dtype=torch.float32)
            mask = mask.to(device=device)
            outputs = model(tokens, mask)
            if is_continuous:
                scores = outputs
            else:
                scores = torch.sigmoid(outputs)
            chunks.append(scores.detach().to("cpu").numpy().astype(np.float32, copy=False))
    if not chunks:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32, copy=False)


def train_token_probe(
    *,
    probe_type: Literal["attention_probe", "multimax_probe"],
    token_embeddings: np.memmap | np.ndarray,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    target_col: str,
    is_continuous: bool,
    random_seed: int,
    config: TokenProbeTrainConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Train one token-level probe and return val/test scores."""
    seed_everything(random_seed)
    device = resolve_probe_device(config.device)

    train_dataset = TokenSpanDataset(
        frame=train_frame,
        token_embeddings=token_embeddings,
        target_col=target_col,
    )
    val_dataset = TokenSpanDataset(
        frame=val_frame,
        token_embeddings=token_embeddings,
        target_col=target_col,
    )
    test_dataset = TokenSpanDataset(
        frame=test_frame,
        token_embeddings=token_embeddings,
        target_col=target_col,
    )

    generator = torch.Generator()
    generator.manual_seed(int(random_seed))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_token_batch,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_token_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_token_batch,
    )

    input_dim = int(token_embeddings.shape[1])
    model = _build_model(
        probe_type=probe_type,
        input_dim=input_dim,
        config=config,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    if is_continuous:
        if config.continuous_loss == "mse":
            loss_fn: nn.Module = nn.MSELoss()
        else:
            loss_fn = nn.HuberLoss(delta=1.0)
    else:
        train_targets = train_frame[target_col].to_numpy(dtype=np.float32)
        positives = float(train_targets.sum())
        negatives = float(train_targets.shape[0] - positives)
        pos_weight_value = negatives / max(positives, 1.0)
        pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_metric = float("-inf")
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0

    for _epoch in range(config.max_epochs):
        model.train()
        for tokens, mask, targets in train_loader:
            tokens = tokens.to(device=device, dtype=torch.float32)
            mask = mask.to(device=device)
            targets = targets.to(device=device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(tokens, mask)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

        val_scores = _predict_scores(
            model=model,
            loader=val_loader,
            device=device,
            is_continuous=is_continuous,
        )
        if is_continuous:
            val_metric = mean_problem_spearman(
                val_frame.assign(pred_score=val_scores),
                score_col="pred_score",
                true_importance_col=target_col,
                problem_col="problem_id",
            )
        else:
            val_targets = val_frame[target_col].to_numpy(dtype=np.int64)
            val_metric = precision_recall_auc(val_targets, val_scores)

        if np.isnan(val_metric):
            val_metric = float("-inf")

        if float(val_metric) > best_metric + 1e-8:
            best_metric = float(val_metric)
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= config.patience:
            break

    model.load_state_dict(best_state)

    final_val_scores = _predict_scores(
        model=model,
        loader=val_loader,
        device=device,
        is_continuous=is_continuous,
    )
    final_test_scores = _predict_scores(
        model=model,
        loader=test_loader,
        device=device,
        is_continuous=is_continuous,
    )
    return final_val_scores, final_test_scores
