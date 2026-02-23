"""Tests for sweep_schema: JSONL round-trip, status transitions, hash determinism."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ta_probe.sweep_schema import (
    ManifestEntry,
    RunRegistryEntry,
    RunStatus,
    append_manifest,
    append_registry,
    config_id_hash,
    load_manifest,
    load_registry,
)


def _sample_manifest_entry(**overrides: object) -> ManifestEntry:
    defaults = {
        "sweep_id": "sweep_001",
        "stage": "stage1",
        "config_id": "stage1_abc123abc123",
        "config_path": "configs/generated/s1_001.yaml",
        "factor_values": {"split_dir": "correct_base_solution", "pooling": "mean"},
        "commands": [{"name": "run_lopo_cv", "args": ["python", "run.py"]}],
    }
    defaults.update(overrides)
    return ManifestEntry(**defaults)


def _sample_registry_entry(**overrides: object) -> RunRegistryEntry:
    defaults = {
        "config_id": "stage1_abc123abc123",
        "stage": "stage1",
    }
    defaults.update(overrides)
    return RunRegistryEntry(**defaults)


# --- Round-trip tests ---


def test_manifest_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    entry1 = _sample_manifest_entry(config_id="stage1_aaa")
    entry2 = _sample_manifest_entry(config_id="stage1_bbb")

    append_manifest(path, entry1)
    append_manifest(path, entry2)

    loaded = load_manifest(path)
    assert len(loaded) == 2
    assert loaded[0].config_id == "stage1_aaa"
    assert loaded[1].config_id == "stage1_bbb"
    assert loaded[0].factor_values == entry1.factor_values


def test_registry_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "registry.jsonl"
    entry = _sample_registry_entry(status=RunStatus.success, attempt=2, duration_sec=42.5)

    append_registry(path, entry)

    loaded = load_registry(path)
    assert len(loaded) == 1
    assert loaded[0].status == RunStatus.success
    assert loaded[0].attempt == 2
    assert loaded[0].duration_sec == pytest.approx(42.5)


def test_manifest_empty_file_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    assert load_manifest(path) == []


def test_registry_empty_file_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    assert load_registry(path) == []


# --- Status transitions ---


def test_registry_status_values() -> None:
    for status in ["pending", "running", "success", "failed", "degraded"]:
        entry = _sample_registry_entry(status=status)
        assert entry.status == RunStatus(status)


def test_registry_rejects_invalid_status() -> None:
    with pytest.raises(ValidationError):
        _sample_registry_entry(status="unknown_status")


# --- Hash determinism ---


def test_config_id_hash_deterministic() -> None:
    factors = {"split_dir": "correct_base_solution", "pooling": "mean", "layer_mode": "mid"}
    h1 = config_id_hash("stage1", factors)
    h2 = config_id_hash("stage1", factors)
    assert h1 == h2


def test_config_id_hash_differs_by_stage() -> None:
    factors = {"split_dir": "correct_base_solution", "pooling": "mean"}
    h1 = config_id_hash("stage1", factors)
    h2 = config_id_hash("stage2", factors)
    assert h1 != h2


def test_config_id_hash_differs_by_factors() -> None:
    base = {"split_dir": "correct_base_solution", "pooling": "mean"}
    alt = {"split_dir": "incorrect_base_solution", "pooling": "mean"}
    assert config_id_hash("stage1", base) != config_id_hash("stage1", alt)


def test_config_id_hash_key_order_invariant() -> None:
    factors_a = {"pooling": "mean", "split_dir": "correct_base_solution"}
    factors_b = {"split_dir": "correct_base_solution", "pooling": "mean"}
    assert config_id_hash("stage1", factors_a) == config_id_hash("stage1", factors_b)


# --- Append idempotency ---


def test_append_manifest_is_additive(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    entry = _sample_manifest_entry()

    append_manifest(path, entry)
    append_manifest(path, entry)

    loaded = load_manifest(path)
    assert len(loaded) == 2
    assert loaded[0].config_id == loaded[1].config_id


def test_append_registry_is_additive(tmp_path: Path) -> None:
    path = tmp_path / "registry.jsonl"
    e1 = _sample_registry_entry(status=RunStatus.pending, attempt=1)
    e2 = _sample_registry_entry(status=RunStatus.running, attempt=1)
    e3 = _sample_registry_entry(status=RunStatus.success, attempt=1)

    append_registry(path, e1)
    append_registry(path, e2)
    append_registry(path, e3)

    loaded = load_registry(path)
    assert len(loaded) == 3
    assert [e.status for e in loaded] == [RunStatus.pending, RunStatus.running, RunStatus.success]


# --- Model defaults ---


def test_manifest_entry_has_created_at() -> None:
    entry = _sample_manifest_entry()
    assert entry.created_at  # non-empty ISO timestamp


def test_registry_entry_defaults() -> None:
    entry = _sample_registry_entry()
    assert entry.status == RunStatus.pending
    assert entry.attempt == 1
    assert entry.start_time is None
    assert entry.artifacts == {}
    assert entry.metrics == {}
