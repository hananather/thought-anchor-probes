"""Pure-function factor expansion for Stage 1/2/3 sweep grids.

Higher-level wrappers around :mod:`ta_probe.sweep` builders that handle
shortlist × expansion cross-products and config YAML generation.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from ta_probe.config import ExperimentConfig, load_config
from ta_probe.sweep import (
    apply_factor_values,
    build_stage1_factor_values,
    build_stage2_factor_expansions,
    build_stage3_factor_expansions,
    make_config_id,
    write_config_yaml,
)


def expand_stage1_grid(
    *,
    num_layers: int,
    split_dirs: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Expand the full Stage-1 factor grid.

    Returns a list of factor-value dicts, each tagged with a deterministic
    ``config_id`` under key ``_config_id``.
    """
    rows = build_stage1_factor_values(num_layers=num_layers, split_dirs=split_dirs)
    for row in rows:
        row["_config_id"] = make_config_id("stage1", row)
    return rows


def expand_stage2_grid(
    shortlist: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Cross-product of *shortlist* configs with Stage-2 vertical-attention expansions.

    Each shortlisted factor dict is merged with every Stage-2 expansion variant.
    Duplicate entries (where the shortlist already matches the expansion) are kept —
    dedup is the caller's responsibility if desired.
    """
    expansions = build_stage2_factor_expansions()
    rows: list[dict[str, Any]] = []
    for base in shortlist:
        for expansion in expansions:
            merged = copy.deepcopy(base)
            merged.update(expansion)
            # Remove any stale _config_id from prior stage
            merged.pop("_config_id", None)
            merged["_config_id"] = make_config_id("stage2", merged)
            rows.append(merged)
    return rows


def expand_stage3_grid(
    finalists: list[dict[str, Any]],
    target_mode: str,
) -> list[dict[str, Any]]:
    """Cross-product of *finalists* with Stage-3 training hyper-grid.

    ``target_mode`` controls whether continuous-loss variants are included
    (passed through to :func:`build_stage3_factor_expansions`).
    """
    expansions = build_stage3_factor_expansions(target_mode)
    rows: list[dict[str, Any]] = []
    for base in finalists:
        for expansion in expansions:
            merged = copy.deepcopy(base)
            merged.update(expansion)
            merged.pop("_config_id", None)
            merged["_config_id"] = make_config_id("stage3", merged)
            rows.append(merged)
    return rows


def generate_config_yaml(
    base_config_path: str | Path,
    factor_overrides: dict[str, Any],
    run_root: Path,
    output_path: Path | None = None,
) -> ExperimentConfig:
    """Load a base config, apply factor overrides, and optionally write YAML.

    Parameters
    ----------
    base_config_path:
        Path to the base experiment YAML (e.g. ``configs/scaling_qwen_correct.yaml``).
    factor_overrides:
        Factor-value dict produced by one of the ``expand_stage*`` functions.
        Internal keys like ``_config_id`` are stripped before application.
    run_root:
        Per-run artifact directory.
    output_path:
        If given, write the derived config YAML here.

    Returns
    -------
    ExperimentConfig
        The fully-resolved config with factors applied and paths set.
    """
    base_config = load_config(base_config_path)

    # Strip internal metadata keys before applying
    clean_factors = {k: v for k, v in factor_overrides.items() if not k.startswith("_")}

    config = apply_factor_values(
        base_config=base_config,
        factor_values=clean_factors,
        run_root=run_root,
    )

    if output_path is not None:
        write_config_yaml(config, output_path)

    return config
