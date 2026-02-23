"""Tests for sweep_factors: config counts, conditional rules, deterministic IDs, validation."""

from __future__ import annotations

from pathlib import Path

from ta_probe.config import ExperimentConfig
from ta_probe.sweep_factors import (
    expand_stage1_grid,
    expand_stage2_grid,
    expand_stage3_grid,
    generate_config_yaml,
)

# --- Stage 1 ---


def test_stage1_config_count_40_layers() -> None:
    rows = expand_stage1_grid(num_layers=40)
    # anchor_binary: 4 split * 2 pooling * 4 layer * 3 residual * 3 percentile * 2 drop = 576
    # continuous (x2): 2 * (4 * 2 * 4 * 3) = 192
    assert len(rows) == 768


def test_stage1_all_rows_have_config_id() -> None:
    rows = expand_stage1_grid(num_layers=40)
    for row in rows:
        assert "_config_id" in row
        assert row["_config_id"].startswith("stage1_")


def test_stage1_config_ids_are_deterministic() -> None:
    rows_a = expand_stage1_grid(num_layers=40)
    rows_b = expand_stage1_grid(num_layers=40)
    ids_a = [r["_config_id"] for r in rows_a]
    ids_b = [r["_config_id"] for r in rows_b]
    assert ids_a == ids_b


def test_stage1_config_ids_are_unique() -> None:
    rows = expand_stage1_grid(num_layers=40)
    ids = [r["_config_id"] for r in rows]
    assert len(ids) == len(set(ids))


def test_stage1_conditional_anchor_binary_variants() -> None:
    rows = expand_stage1_grid(num_layers=40)
    binary = [r for r in rows if r["target_mode"] == "anchor_binary"]
    percentiles = {r["anchor_percentile"] for r in binary}
    drop_values = {r["drop_last_chunk"] for r in binary}
    assert percentiles == {85.0, 90.0, 95.0}
    assert drop_values == {True, False}


def test_stage1_continuous_modes_fixed_labels() -> None:
    rows = expand_stage1_grid(num_layers=40)
    continuous = [r for r in rows if r["target_mode"] != "anchor_binary"]
    assert len(continuous) == 192
    assert {r["anchor_percentile"] for r in continuous} == {90.0}
    assert {r["drop_last_chunk"] for r in continuous} == {True}


def test_stage1_custom_split_dirs() -> None:
    rows = expand_stage1_grid(num_layers=40, split_dirs=["correct_base_solution"])
    # 1 split * (binary: 2*4*3*3*2 + continuous: 2*2*4*3) = 1*(144+48) = 192
    assert len(rows) == 192


# --- Stage 2 ---


def test_stage2_expansion_count() -> None:
    shortlist = [{"split_dir": "correct_base_solution", "pooling": "mean"}]
    rows = expand_stage2_grid(shortlist)
    # 1 off + 6 light (2 depth * 3 last_n) + 4 full (2 depth * 2 seq_len) = 11
    assert len(rows) == 11


def test_stage2_all_have_stage2_config_id() -> None:
    shortlist = [{"split_dir": "correct_base_solution", "pooling": "mean"}]
    rows = expand_stage2_grid(shortlist)
    for row in rows:
        assert row["_config_id"].startswith("stage2_")


def test_stage2_merges_base_factors() -> None:
    shortlist = [
        {"split_dir": "correct_base_solution", "pooling": "mean", "layer_mode": "mid"}
    ]
    rows = expand_stage2_grid(shortlist)
    for row in rows:
        assert row["split_dir"] == "correct_base_solution"
        assert row["pooling"] == "mean"
        assert "vertical_attention_mode" in row


def test_stage2_multiple_shortlist() -> None:
    shortlist = [
        {"split_dir": "correct_base_solution", "pooling": "mean"},
        {"split_dir": "incorrect_base_solution", "pooling": "tokens"},
    ]
    rows = expand_stage2_grid(shortlist)
    assert len(rows) == 2 * 11


# --- Stage 3 ---


def test_stage3_binary_expansion_count() -> None:
    finalists = [{"split_dir": "correct_base_solution"}]
    rows = expand_stage3_grid(finalists, target_mode="anchor_binary")
    # 3 hidden * 2 iter * 2 heads * 2 width * 2 depth * 2 batch * 2 epochs * 2 patience
    # * 2 lr * 2 wd * 1 loss = 3 * 2^9 = 1536
    assert len(rows) == 1536


def test_stage3_continuous_expansion_count() -> None:
    finalists = [{"split_dir": "correct_base_solution"}]
    rows = expand_stage3_grid(finalists, target_mode="importance_abs")
    # Same but 2 loss types = 3 * 2^9 * 2 = 3072
    assert len(rows) == 3072


def test_stage3_all_have_stage3_config_id() -> None:
    finalists = [{"split_dir": "correct_base_solution"}]
    rows = expand_stage3_grid(finalists, target_mode="anchor_binary")
    for row in rows[:10]:  # sample
        assert row["_config_id"].startswith("stage3_")


def test_stage3_merges_finalist_factors() -> None:
    finalists = [{"split_dir": "correct_base_solution", "pooling": "tokens"}]
    rows = expand_stage3_grid(finalists, target_mode="anchor_binary")
    for row in rows[:10]:
        assert row["split_dir"] == "correct_base_solution"
        assert row["pooling"] == "tokens"
        assert "mlp_hidden_dim" in row


# --- generate_config_yaml ---


def test_generate_config_yaml_returns_valid_config(tmp_path: Path) -> None:
    base_path = Path("configs/scaling_qwen_correct.yaml")
    factors = {
        "_config_id": "stage1_test123",
        "split_dir": "incorrect_base_solution",
        "forced_split_dir": "incorrect_base_solution_forced_answer",
        "target_mode": "importance_abs",
        "anchor_percentile": 90.0,
        "drop_last_chunk": True,
        "pooling": "tokens",
        "layer_mode": "index",
        "layer_index": 20,
        "residualize_against": "position",
        "vertical_attention_mode": "off",
        "vertical_attention_depth_control": True,
        "vertical_attention_light_last_n_tokens": 1,
        "vertical_attention_full_max_seq_len": 1024,
    }
    run_root = tmp_path / "run"
    output_path = tmp_path / "config.yaml"

    config = generate_config_yaml(base_path, factors, run_root, output_path=output_path)

    assert isinstance(config, ExperimentConfig)
    assert config.dataset.split_dir == "incorrect_base_solution"
    assert config.labels.target_mode == "importance_abs"
    assert config.activations.pooling == "tokens"
    assert config.activations.layer_index == 20
    assert config.split.strategy == "lopo_cv"
    assert output_path.exists()


def test_generate_config_yaml_strips_internal_keys(tmp_path: Path) -> None:
    base_path = Path("configs/scaling_qwen_correct.yaml")
    factors = {
        "_config_id": "stage1_should_be_stripped",
        "split_dir": "correct_base_solution",
        "forced_split_dir": "correct_base_solution_forced_answer",
        "target_mode": "anchor_binary",
        "anchor_percentile": 90.0,
        "drop_last_chunk": True,
        "pooling": "mean",
        "layer_mode": "mid",
        "layer_index": None,
        "residualize_against": "none",
        "vertical_attention_mode": "off",
        "vertical_attention_depth_control": True,
        "vertical_attention_light_last_n_tokens": 1,
        "vertical_attention_full_max_seq_len": 1024,
    }
    run_root = tmp_path / "run"

    config = generate_config_yaml(base_path, factors, run_root)
    assert isinstance(config, ExperimentConfig)
