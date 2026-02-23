from __future__ import annotations

from pathlib import Path

from ta_probe.config import load_config
from ta_probe.sweep import (
    apply_factor_values,
    build_lopo_commands,
    build_stage1_factor_values,
    layer_quartile_indices,
    load_jsonl,
    make_config_id,
    write_jsonl,
)


def test_layer_quartile_indices_for_qwen_40() -> None:
    assert layer_quartile_indices(40) == (10, 20, 30)


def test_stage1_factor_grid_expected_count() -> None:
    rows = build_stage1_factor_values(num_layers=40)
    # anchor_binary: 4 split * 2 pooling * 4 layer * 3 residual * 3 percentiles * 2 drop = 576
    # two continuous modes: 2 * (4 * 2 * 4 * 3) = 192
    assert len(rows) == 768


def test_stage1_continuous_modes_keep_single_label_setting() -> None:
    rows = build_stage1_factor_values(num_layers=40)
    continuous = [row for row in rows if row["target_mode"] != "anchor_binary"]
    assert continuous
    assert {row["anchor_percentile"] for row in continuous} == {90.0}
    assert {row["drop_last_chunk"] for row in continuous} == {True}


def test_make_config_id_is_deterministic() -> None:
    factors = {
        "split_dir": "correct_base_solution",
        "target_mode": "anchor_binary",
        "pooling": "mean",
        "layer_mode": "mid",
        "layer_index": None,
    }
    config_id_a = make_config_id("stage1", factors)
    config_id_b = make_config_id("stage1", factors)
    assert config_id_a == config_id_b


def test_apply_factor_values_sets_lopo_paths(tmp_path: Path) -> None:
    base = load_config("configs/scaling_qwen_correct.yaml")
    run_root = tmp_path / "run"
    factors = {
        "split_dir": "incorrect_base_solution",
        "forced_split_dir": "incorrect_base_solution_forced_answer",
        "target_mode": "importance_abs",
        "anchor_percentile": 90.0,
        "drop_last_chunk": True,
        "pooling": "tokens",
        "layer_mode": "index",
        "layer_index": 20,
        "residualize_against": "position_plus_text",
        "vertical_attention_mode": "light",
        "vertical_attention_depth_control": False,
        "vertical_attention_light_last_n_tokens": 4,
        "vertical_attention_full_max_seq_len": 1024,
        "mlp_hidden_dim": 256,
        "mlp_max_iter": 800,
        "token_probe_heads": 8,
        "token_probe_mlp_width": 256,
        "token_probe_mlp_depth": 2,
        "token_probe_batch_size": 16,
        "token_probe_max_epochs": 100,
        "token_probe_patience": 10,
        "token_probe_learning_rate": 3e-4,
        "token_probe_weight_decay": 1e-2,
        "token_probe_continuous_loss": "mse",
    }

    updated = apply_factor_values(base_config=base, factor_values=factors, run_root=run_root)

    assert updated.split.strategy == "lopo_cv"
    assert updated.dataset.split_dir == "incorrect_base_solution"
    assert updated.labels.target_mode == "importance_abs"
    assert updated.activations.pooling == "tokens"
    assert updated.activations.layer_mode == "index"
    assert updated.activations.layer_index == 20
    assert updated.training.residualize_against == "position_plus_text"
    assert updated.training.token_probe_heads == 8
    assert updated.paths.embeddings_memmap.endswith("sentence_embeddings.dat")
    assert updated.paths.token_embeddings_memmap.endswith("token_embeddings.dat")


def test_build_lopo_commands_includes_flags(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    run_root = tmp_path / "run"

    commands = build_lopo_commands(
        config_path=config_path,
        run_root=run_root,
        seeds=[0, 1, 2],
        reuse_cache=True,
        skip_failed=True,
        no_tripwires=True,
        bootstrap_iterations=2000,
        bootstrap_seed=3,
        best_of_k=2,
    )

    first_args = commands[0]["args"]
    second_args = commands[1]["args"]

    assert "--reuse-cache" in first_args
    assert "--skip-failed" in first_args
    assert "--no-tripwires" in first_args
    assert second_args[:3] == ["python", "scripts/aggregate_runs.py", "--run-root"]
    assert "--lopo" in second_args


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    rows = [
        {"config_id": "a", "status": "success"},
        {"config_id": "b", "status": "failed"},
    ]
    write_jsonl(path, rows)
    loaded = load_jsonl(path)
    assert loaded == rows
