"""Minimal integration smoke test: generate configs and validate schema compliance."""

from __future__ import annotations

from pathlib import Path

from ta_probe.config import ExperimentConfig
from ta_probe.sweep import make_config_id
from ta_probe.sweep_factors import expand_stage1_grid, generate_config_yaml
from ta_probe.sweep_schema import (
    ManifestEntry,
    RunRegistryEntry,
    RunStatus,
    append_manifest,
    append_registry,
    load_manifest,
    load_registry,
)


def test_smoke_stage1_grid_to_manifest(tmp_path: Path) -> None:
    """Generate a small Stage-1 grid and write it as a valid manifest."""
    rows = expand_stage1_grid(num_layers=40, split_dirs=["correct_base_solution"])
    manifest_path = tmp_path / "manifest.jsonl"

    for row in rows[:5]:
        entry = ManifestEntry(
            sweep_id="smoke_test",
            stage="stage1",
            config_id=row["_config_id"],
            config_path=f"configs/generated/{row['_config_id']}.yaml",
            factor_values={k: v for k, v in row.items() if not k.startswith("_")},
            commands=[],
        )
        append_manifest(manifest_path, entry)

    loaded = load_manifest(manifest_path)
    assert len(loaded) == 5
    for entry in loaded:
        assert entry.sweep_id == "smoke_test"
        assert entry.stage == "stage1"
        assert entry.config_id.startswith("stage1_")
        assert entry.factor_values.get("split_dir") == "correct_base_solution"


def test_smoke_generated_configs_validate(tmp_path: Path) -> None:
    """Configs generated from Stage-1 factors parse as valid ExperimentConfig."""
    rows = expand_stage1_grid(num_layers=40, split_dirs=["correct_base_solution"])
    base_config_path = Path("configs/scaling_qwen_correct.yaml")

    for i, row in enumerate(rows[:3]):
        run_root = tmp_path / f"run_{i}"
        config = generate_config_yaml(base_config_path, row, run_root)
        assert isinstance(config, ExperimentConfig)
        assert config.split.strategy == "lopo_cv"
        assert config.dataset.split_dir == row["split_dir"]
        assert config.labels.target_mode == row["target_mode"]


def test_smoke_registry_lifecycle(tmp_path: Path) -> None:
    """Simulate a config going through pending -> running -> success."""
    registry_path = tmp_path / "registry.jsonl"
    config_id = "stage1_smoke_test"

    # pending
    append_registry(
        registry_path,
        RunRegistryEntry(config_id=config_id, stage="stage1", status=RunStatus.pending),
    )
    # running
    append_registry(
        registry_path,
        RunRegistryEntry(
            config_id=config_id,
            stage="stage1",
            status=RunStatus.running,
            start_time="2026-01-01T00:00:00+00:00",
        ),
    )
    # success
    append_registry(
        registry_path,
        RunRegistryEntry(
            config_id=config_id,
            stage="stage1",
            status=RunStatus.success,
            attempt=1,
            start_time="2026-01-01T00:00:00+00:00",
            end_time="2026-01-01T00:05:00+00:00",
            duration_sec=300.0,
            metrics={"primary_metric": 0.85},
        ),
    )

    loaded = load_registry(registry_path)
    assert len(loaded) == 3
    assert loaded[0].status == RunStatus.pending
    assert loaded[1].status == RunStatus.running
    assert loaded[2].status == RunStatus.success
    assert loaded[2].metrics["primary_metric"] == 0.85


def test_smoke_manifest_config_ids_match_hash() -> None:
    """Config IDs in the grid match what make_config_id would produce."""
    rows = expand_stage1_grid(num_layers=40, split_dirs=["correct_base_solution"])
    for row in rows[:10]:
        clean = {k: v for k, v in row.items() if not k.startswith("_")}
        expected = make_config_id("stage1", clean)
        assert row["_config_id"] == expected
