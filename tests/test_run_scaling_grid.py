from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from ta_probe.config import load_config


def _load_scaling_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_scaling_grid.py"
    spec = importlib.util.spec_from_file_location("run_scaling_grid", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load run_scaling_grid module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_discovery_cache_paths_unique_for_scaling_configs() -> None:
    module = _load_scaling_module()
    configs = [
        load_config("configs/scaling_llama_correct.yaml"),
        load_config("configs/scaling_llama_incorrect.yaml"),
        load_config("configs/scaling_qwen_correct.yaml"),
        load_config("configs/scaling_qwen_incorrect.yaml"),
    ]

    cache_paths = module.build_discovery_cache_paths(configs, Path("artifacts/scaling/id_cache"))
    assert len(cache_paths) == len(configs)
    assert len({str(path) for path in cache_paths}) == len(configs)


def test_build_discovery_cache_paths_raises_on_collisions() -> None:
    module = _load_scaling_module()
    config = load_config("configs/scaling_llama_correct.yaml")

    with pytest.raises(ValueError, match="not unique"):
        module.build_discovery_cache_paths([config, config], Path("artifacts/scaling/id_cache"))


def test_validate_scaling_config_parity_raises_on_split_mismatch() -> None:
    module = _load_scaling_module()
    config_a = load_config("configs/scaling_llama_correct.yaml")
    config_b = load_config("configs/scaling_qwen_correct.yaml")
    config_b.training.train_fraction = 0.6

    with pytest.raises(ValueError, match="training.train_fraction"):
        module.validate_scaling_config_parity([config_a, config_b])
