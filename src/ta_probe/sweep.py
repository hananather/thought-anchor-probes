"""Utilities for large systematic sweep generation and orchestration."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ta_probe.config import ExperimentConfig

DEFAULT_STAGE1_SPLIT_DIRS = [
    "correct_base_solution",
    "incorrect_base_solution",
    "correct_base_solution_forced_answer",
    "incorrect_base_solution_forced_answer",
]
DEFAULT_TARGET_MODES = ["anchor_binary", "importance_abs", "importance_signed"]
DEFAULT_POOLING_MODES = ["mean", "tokens"]
DEFAULT_RESIDUALIZE_AGAINST = ["none", "position", "position_plus_text"]
DEFAULT_ANCHOR_PERCENTILES = [85.0, 90.0, 95.0]
DEFAULT_DROP_LAST_CHUNK = [True, False]
DEFAULT_STAGE2_LIGHT_LAST_N = [1, 4, 8]
DEFAULT_STAGE2_FULL_MAX_SEQ = [1024, 2048]

DEFAULT_STAGE3_MLP_HIDDEN_DIM = [100, 256, 512]
DEFAULT_STAGE3_MLP_MAX_ITER = [300, 800]
DEFAULT_STAGE3_TOKEN_HEADS = [4, 8]
DEFAULT_STAGE3_TOKEN_WIDTH = [128, 256]
DEFAULT_STAGE3_TOKEN_DEPTH = [1, 2]
DEFAULT_STAGE3_TOKEN_BATCH = [16, 32]
DEFAULT_STAGE3_TOKEN_EPOCHS = [50, 100]
DEFAULT_STAGE3_TOKEN_PATIENCE = [5, 10]
DEFAULT_STAGE3_TOKEN_LR = [1e-3, 3e-4]
DEFAULT_STAGE3_TOKEN_WD = [0.0, 1e-2]


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601."""
    return datetime.now(timezone.utc).isoformat()


def layer_quartile_indices(num_layers: int) -> tuple[int, int, int]:
    """Return q1/q2/q3 layer indices for a transformer depth."""
    if num_layers < 4:
        msg = f"num_layers must be >= 4 for quartile selection, got {num_layers}"
        raise ValueError(msg)

    q1 = max(0, num_layers // 4)
    q2 = max(0, num_layers // 2)
    q3 = min(num_layers - 1, (3 * num_layers) // 4)

    if q1 == q2:
        q2 = min(num_layers - 1, q2 + 1)
    if q2 == q3:
        q3 = min(num_layers - 1, q3 + 1)
    return (q1, q2, q3)


def infer_num_layers(model_name_or_path: str, explicit_num_layers: int | None = None) -> int:
    """Infer transformer depth from model config or known model-name fallbacks."""
    if explicit_num_layers is not None:
        if explicit_num_layers <= 0:
            msg = f"explicit_num_layers must be positive, got {explicit_num_layers}"
            raise ValueError(msg)
        return int(explicit_num_layers)

    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        num_hidden_layers = getattr(config, "num_hidden_layers", None)
        if num_hidden_layers is not None:
            value = int(num_hidden_layers)
            if value > 0:
                return value
    except Exception:
        pass

    lowered = model_name_or_path.lower()
    if "qwen-14b" in lowered:
        return 40
    if "llama-8b" in lowered:
        return 32

    msg = "Could not infer num_hidden_layers from model config. Pass --num-layers explicitly."
    raise ValueError(msg)


def forced_split_dir_for(split_dir: str) -> str:
    """Resolve a split directory name to its forced-answer counterpart."""
    if split_dir.endswith("_forced_answer"):
        return split_dir
    if split_dir == "correct_base_solution":
        return "correct_base_solution_forced_answer"
    if split_dir == "incorrect_base_solution":
        return "incorrect_base_solution_forced_answer"
    return f"{split_dir}_forced_answer"


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def make_config_id(stage: str, factor_values: dict[str, Any]) -> str:
    """Build deterministic config ID from stage and factor values."""
    digest = hashlib.sha256(f"{stage}:{_canonical_json(factor_values)}".encode()).hexdigest()
    return f"{stage}_{digest[:12]}"


def build_stage1_factor_values(
    *,
    num_layers: int,
    split_dirs: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Build exhaustive Stage-1 factor grid."""
    q1, q2, q3 = layer_quartile_indices(num_layers)
    layer_options = [
        {"layer_mode": "mid", "layer_index": None},
        {"layer_mode": "index", "layer_index": q1},
        {"layer_mode": "index", "layer_index": q2},
        {"layer_mode": "index", "layer_index": q3},
    ]

    active_split_dirs = (
        list(split_dirs) if split_dirs is not None else list(DEFAULT_STAGE1_SPLIT_DIRS)
    )
    rows: list[dict[str, Any]] = []

    for split_dir, target_mode, pooling, layer_choice, residualize_against in itertools.product(
        active_split_dirs,
        DEFAULT_TARGET_MODES,
        DEFAULT_POOLING_MODES,
        layer_options,
        DEFAULT_RESIDUALIZE_AGAINST,
    ):
        base = {
            "split_dir": split_dir,
            "forced_split_dir": forced_split_dir_for(split_dir),
            "target_mode": target_mode,
            "pooling": pooling,
            "layer_mode": layer_choice["layer_mode"],
            "layer_index": layer_choice["layer_index"],
            "residualize_against": residualize_against,
            "vertical_attention_mode": "off",
            "vertical_attention_depth_control": True,
            "vertical_attention_light_last_n_tokens": 1,
            "vertical_attention_full_max_seq_len": 1024,
        }

        if target_mode == "anchor_binary":
            for percentile, drop_last_chunk in itertools.product(
                DEFAULT_ANCHOR_PERCENTILES,
                DEFAULT_DROP_LAST_CHUNK,
            ):
                row = copy.deepcopy(base)
                row["anchor_percentile"] = float(percentile)
                row["drop_last_chunk"] = bool(drop_last_chunk)
                rows.append(row)
        else:
            row = copy.deepcopy(base)
            row["anchor_percentile"] = 90.0
            row["drop_last_chunk"] = True
            rows.append(row)

    return rows


def build_stage2_factor_expansions() -> list[dict[str, Any]]:
    """Build vertical-attention expansion factors for Stage-2."""
    rows: list[dict[str, Any]] = [
        {
            "vertical_attention_mode": "off",
            "vertical_attention_depth_control": True,
            "vertical_attention_light_last_n_tokens": 1,
            "vertical_attention_full_max_seq_len": 1024,
        }
    ]

    for depth_control, light_last_n in itertools.product(
        [True, False], DEFAULT_STAGE2_LIGHT_LAST_N
    ):
        rows.append(
            {
                "vertical_attention_mode": "light",
                "vertical_attention_depth_control": bool(depth_control),
                "vertical_attention_light_last_n_tokens": int(light_last_n),
                "vertical_attention_full_max_seq_len": 1024,
            }
        )

    for depth_control, full_max_seq_len in itertools.product(
        [True, False], DEFAULT_STAGE2_FULL_MAX_SEQ
    ):
        rows.append(
            {
                "vertical_attention_mode": "full",
                "vertical_attention_depth_control": bool(depth_control),
                "vertical_attention_light_last_n_tokens": 1,
                "vertical_attention_full_max_seq_len": int(full_max_seq_len),
            }
        )

    return rows


def build_stage3_factor_expansions(target_mode: str) -> list[dict[str, Any]]:
    """Build training hyperparameter expansions for Stage-3."""
    continuous_losses = ["huber"] if target_mode == "anchor_binary" else ["huber", "mse"]

    rows: list[dict[str, Any]] = []
    for (
        mlp_hidden_dim,
        mlp_max_iter,
        token_heads,
        token_width,
        token_depth,
        token_batch,
        token_epochs,
        token_patience,
        token_lr,
        token_wd,
        token_continuous_loss,
    ) in itertools.product(
        DEFAULT_STAGE3_MLP_HIDDEN_DIM,
        DEFAULT_STAGE3_MLP_MAX_ITER,
        DEFAULT_STAGE3_TOKEN_HEADS,
        DEFAULT_STAGE3_TOKEN_WIDTH,
        DEFAULT_STAGE3_TOKEN_DEPTH,
        DEFAULT_STAGE3_TOKEN_BATCH,
        DEFAULT_STAGE3_TOKEN_EPOCHS,
        DEFAULT_STAGE3_TOKEN_PATIENCE,
        DEFAULT_STAGE3_TOKEN_LR,
        DEFAULT_STAGE3_TOKEN_WD,
        continuous_losses,
    ):
        rows.append(
            {
                "mlp_hidden_dim": int(mlp_hidden_dim),
                "mlp_max_iter": int(mlp_max_iter),
                "token_probe_heads": int(token_heads),
                "token_probe_mlp_width": int(token_width),
                "token_probe_mlp_depth": int(token_depth),
                "token_probe_batch_size": int(token_batch),
                "token_probe_max_epochs": int(token_epochs),
                "token_probe_patience": int(token_patience),
                "token_probe_learning_rate": float(token_lr),
                "token_probe_weight_decay": float(token_wd),
                "token_probe_continuous_loss": str(token_continuous_loss),
            }
        )
    return rows


def factor_values_from_config(config: ExperimentConfig) -> dict[str, Any]:
    """Extract sweep-relevant factors from an experiment config."""
    return {
        "split_dir": config.dataset.split_dir,
        "forced_split_dir": config.dataset.forced_split_dir,
        "target_mode": config.labels.target_mode,
        "anchor_percentile": float(config.labels.anchor_percentile),
        "drop_last_chunk": bool(config.labels.drop_last_chunk),
        "pooling": config.activations.pooling,
        "layer_mode": config.activations.layer_mode,
        "layer_index": config.activations.layer_index,
        "residualize_against": config.training.residualize_against,
        "vertical_attention_mode": config.activations.vertical_attention.mode,
        "vertical_attention_depth_control": bool(
            config.activations.vertical_attention.depth_control
        ),
        "vertical_attention_light_last_n_tokens": int(
            config.activations.vertical_attention.light_last_n_tokens
        ),
        "vertical_attention_full_max_seq_len": int(
            config.activations.vertical_attention.full_max_seq_len
        ),
        "mlp_hidden_dim": int(config.training.mlp_hidden_dim),
        "mlp_max_iter": int(config.training.mlp_max_iter),
        "token_probe_heads": int(config.training.token_probe_heads),
        "token_probe_mlp_width": int(config.training.token_probe_mlp_width),
        "token_probe_mlp_depth": int(config.training.token_probe_mlp_depth),
        "token_probe_batch_size": int(config.training.token_probe_batch_size),
        "token_probe_max_epochs": int(config.training.token_probe_max_epochs),
        "token_probe_patience": int(config.training.token_probe_patience),
        "token_probe_learning_rate": float(config.training.token_probe_learning_rate),
        "token_probe_weight_decay": float(config.training.token_probe_weight_decay),
        "token_probe_continuous_loss": config.training.token_probe_continuous_loss,
    }


def apply_factor_values(
    *,
    base_config: ExperimentConfig,
    factor_values: dict[str, Any],
    run_root: Path,
) -> ExperimentConfig:
    """Create a derived config with factor values and per-run artifact paths."""
    config = base_config.model_copy(deep=True)

    config.split.strategy = "lopo_cv"

    config.dataset.split_dir = str(factor_values["split_dir"])
    config.dataset.forced_split_dir = str(
        factor_values.get("forced_split_dir", forced_split_dir_for(str(factor_values["split_dir"])))
    )

    config.labels.target_mode = str(factor_values["target_mode"])
    config.labels.anchor_percentile = float(factor_values["anchor_percentile"])
    config.labels.drop_last_chunk = bool(factor_values["drop_last_chunk"])

    config.activations.pooling = str(factor_values["pooling"])
    config.activations.layer_mode = str(factor_values["layer_mode"])
    config.activations.layer_index = (
        None if factor_values.get("layer_index") is None else int(factor_values["layer_index"])
    )
    config.activations.vertical_attention.mode = str(factor_values["vertical_attention_mode"])
    config.activations.vertical_attention.depth_control = bool(
        factor_values["vertical_attention_depth_control"]
    )
    config.activations.vertical_attention.light_last_n_tokens = int(
        factor_values["vertical_attention_light_last_n_tokens"]
    )
    config.activations.vertical_attention.full_max_seq_len = int(
        factor_values["vertical_attention_full_max_seq_len"]
    )

    config.training.residualize_against = str(factor_values["residualize_against"])

    config.training.mlp_hidden_dim = int(
        factor_values.get("mlp_hidden_dim", config.training.mlp_hidden_dim)
    )
    config.training.mlp_max_iter = int(
        factor_values.get("mlp_max_iter", config.training.mlp_max_iter)
    )

    config.training.token_probe_heads = int(
        factor_values.get("token_probe_heads", config.training.token_probe_heads)
    )
    config.training.token_probe_mlp_width = int(
        factor_values.get("token_probe_mlp_width", config.training.token_probe_mlp_width)
    )
    config.training.token_probe_mlp_depth = int(
        factor_values.get("token_probe_mlp_depth", config.training.token_probe_mlp_depth)
    )
    config.training.token_probe_batch_size = int(
        factor_values.get("token_probe_batch_size", config.training.token_probe_batch_size)
    )
    config.training.token_probe_max_epochs = int(
        factor_values.get("token_probe_max_epochs", config.training.token_probe_max_epochs)
    )
    config.training.token_probe_patience = int(
        factor_values.get("token_probe_patience", config.training.token_probe_patience)
    )
    config.training.token_probe_learning_rate = float(
        factor_values.get("token_probe_learning_rate", config.training.token_probe_learning_rate)
    )
    config.training.token_probe_weight_decay = float(
        factor_values.get("token_probe_weight_decay", config.training.token_probe_weight_decay)
    )
    config.training.token_probe_continuous_loss = str(
        factor_values.get(
            "token_probe_continuous_loss",
            config.training.token_probe_continuous_loss,
        )
    )

    run_root.mkdir(parents=True, exist_ok=True)
    config.paths.problem_ids_json = str(run_root / "problem_ids.json")
    config.paths.splits_json = str(run_root / "folds.json")
    config.paths.embeddings_memmap = str(run_root / "sentence_embeddings.dat")
    config.paths.token_embeddings_memmap = str(run_root / "token_embeddings.dat")
    config.paths.embeddings_shape_json = str(run_root / "sentence_embeddings_shape.json")
    config.paths.metadata_parquet = str(run_root / "sentence_metadata.parquet")
    config.paths.metrics_json = str(run_root / "metrics.json")
    config.paths.predictions_parquet = str(run_root / "predictions.parquet")
    config.paths.report_md = str(run_root / "report.md")

    return config


def write_config_yaml(config: ExperimentConfig, output_path: Path) -> None:
    """Write config payload to YAML."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = config.model_dump(mode="python")
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def build_lopo_commands(
    *,
    config_path: Path,
    run_root: Path,
    seeds: list[int],
    reuse_cache: bool,
    skip_failed: bool,
    no_tripwires: bool,
    bootstrap_iterations: int,
    bootstrap_seed: int,
    best_of_k: int,
) -> list[dict[str, Any]]:
    """Build command sequence for one config in the sweep."""
    run_lopo_args: list[str] = [
        "python",
        "scripts/run_lopo_cv.py",
        "--config",
        str(config_path),
        "--run-root",
        str(run_root),
        "--seeds",
        *[str(seed) for seed in seeds],
    ]
    if skip_failed:
        run_lopo_args.append("--skip-failed")
    if reuse_cache:
        run_lopo_args.append("--reuse-cache")
    if no_tripwires:
        run_lopo_args.append("--no-tripwires")

    aggregate_args = [
        "python",
        "scripts/aggregate_runs.py",
        "--run-root",
        str(run_root),
        "--lopo",
        "--best-of-k",
        str(best_of_k),
        "--bootstrap-iterations",
        str(bootstrap_iterations),
        "--bootstrap-seed",
        str(bootstrap_seed),
    ]

    return [
        {"name": "run_lopo_cv", "args": run_lopo_args},
        {"name": "aggregate_runs", "args": aggregate_args},
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL file from list of dict rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one row to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL file into a list of dict rows."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows
