"""Pydantic data models for sweep manifest and run registry JSONL contracts."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ta_probe.sweep import _canonical_json, append_jsonl, load_jsonl


class RunStatus(str, Enum):
    """Status of a sweep run entry."""

    pending = "pending"
    running = "running"
    success = "success"
    failed = "failed"
    degraded = "degraded"


class ManifestEntry(BaseModel):
    """One row in the sweep manifest JSONL — describes a generated config."""

    sweep_id: str
    stage: str
    config_id: str
    config_path: str
    factor_values: dict[str, Any]
    commands: list[dict[str, Any]] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class RunRegistryEntry(BaseModel):
    """One row in the run registry JSONL — tracks execution status of a config."""

    config_id: str
    stage: str
    status: RunStatus = RunStatus.pending
    attempt: int = 1
    start_time: str | None = None
    end_time: str | None = None
    duration_sec: float | None = None
    error_type: str | None = None
    error_message: str | None = None
    artifacts: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)


def config_id_hash(stage: str, factor_values: dict[str, Any]) -> str:
    """Deterministic config ID from stage + factor dict. Delegates to sweep.make_config_id."""
    import hashlib

    digest = hashlib.sha256(f"{stage}:{_canonical_json(factor_values)}".encode()).hexdigest()
    return f"{stage}_{digest[:12]}"


def append_manifest(path: Path, entry: ManifestEntry) -> None:
    """Append one ManifestEntry to the manifest JSONL file."""
    append_jsonl(path, entry.model_dump(mode="python"))


def load_manifest(path: Path) -> list[ManifestEntry]:
    """Load all ManifestEntry rows from a manifest JSONL file."""
    return [ManifestEntry.model_validate(row) for row in load_jsonl(path)]


def append_registry(path: Path, entry: RunRegistryEntry) -> None:
    """Append one RunRegistryEntry to the run registry JSONL file."""
    append_jsonl(path, entry.model_dump(mode="python"))


def load_registry(path: Path) -> list[RunRegistryEntry]:
    """Load all RunRegistryEntry rows from a registry JSONL file."""
    return [RunRegistryEntry.model_validate(row) for row in load_jsonl(path)]
