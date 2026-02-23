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


def test_validate_scaling_config_parity_raises_on_token_probe_lr_mismatch() -> None:
    module = _load_scaling_module()
    config_a = load_config("configs/scaling_llama_correct.yaml")
    config_b = load_config("configs/scaling_qwen_correct.yaml")
    config_b.training.token_probe_learning_rate = 2 * config_a.training.token_probe_learning_rate

    with pytest.raises(ValueError, match="training.token_probe_learning_rate"):
        module.validate_scaling_config_parity([config_a, config_b])


def test_parity_fields_include_token_probe_training_settings() -> None:
    module = _load_scaling_module()
    expected_fields = {
        ("training", "token_probe_heads"),
        ("training", "token_probe_mlp_width"),
        ("training", "token_probe_mlp_depth"),
        ("training", "token_probe_batch_size"),
        ("training", "token_probe_max_epochs"),
        ("training", "token_probe_patience"),
        ("training", "token_probe_learning_rate"),
        ("training", "token_probe_weight_decay"),
        ("training", "token_probe_continuous_loss"),
        ("training", "token_probe_device"),
    }
    assert expected_fields.issubset(set(module.PARITY_FIELDS))


def test_extract_best_metric_means_keeps_pr_auc_semantics() -> None:
    module = _load_scaling_module()
    summary_rows = [
        {"model": "linear_probe", "pr_auc_mean": 0.12, "spearman_mean": 0.55},
        {"model": "mlp_probe", "pr_auc_mean": 0.18, "spearman_mean": 0.51},
    ]
    best_primary, best_pr_auc = module._extract_best_metric_means(
        summary_rows=summary_rows,
        best_model="linear_probe",
        primary_metric="spearman",
    )
    assert best_primary == pytest.approx(0.55)
    assert best_pr_auc == pytest.approx(0.12)


def test_extract_best_metric_means_supports_aggregate_summary_schema() -> None:
    module = _load_scaling_module()
    summary_rows = [
        {"model": "linear_probe", "pr_auc_mean": 0.12, "spearman_mean_mean": 0.55},
    ]
    best_primary, best_pr_auc = module._extract_best_metric_means(
        summary_rows=summary_rows,
        best_model="linear_probe",
        primary_metric="spearman",
    )
    assert best_primary == pytest.approx(0.55)
    assert best_pr_auc == pytest.approx(0.12)


def test_resolve_primary_metric_name_prefers_primary_metric_name_field() -> None:
    module = _load_scaling_module()
    payload = {"primary_metric_name": "spearman", "primary_metric": "pr_auc"}
    assert module._resolve_primary_metric_name(payload) == "spearman"
    assert module._resolve_primary_metric_column(payload) == "spearman_mean"


def test_resolve_primary_metric_name_supports_legacy_primary_metric_column() -> None:
    module = _load_scaling_module()
    payload = {"primary_metric": "spearman_mean"}
    assert module._resolve_primary_metric_name(payload) == "spearman"
    assert module._resolve_primary_metric_column(payload) == "spearman_mean"


def test_update_scaling_readme_replaces_marker_block(tmp_path: Path) -> None:
    module = _load_scaling_module()
    readme_path = tmp_path / "README.md"
    summary_path = tmp_path / "scaling_summary.md"
    readme_path.write_text(
        "\n".join(
            [
                "# Title",
                "",
                "## Scaling Grid Results",
                "",
                "<!-- SCALING_RESULTS_START -->",
                "old content",
                "<!-- SCALING_RESULTS_END -->",
                "",
                "tail",
                "",
            ]
        ),
        encoding="utf-8",
    )
    summary_path.write_text("# Scaling Matrix Summary\n\n- Shared problems: 2\n", encoding="utf-8")

    updated_path = module._update_scaling_readme(
        repo_root=tmp_path,
        readme_path="README.md",
        summary_markdown_path=summary_path,
    )
    updated = readme_path.read_text(encoding="utf-8")

    assert updated_path == readme_path
    assert "# Scaling Matrix Summary" in updated
    assert "old content" not in updated


def test_parse_args_supports_no_tripwires_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_scaling_module()
    monkeypatch.setattr(
        "sys.argv",
        ["run_scaling_grid.py", "--configs", "configs/scaling_qwen_correct.yaml", "--no-tripwires"],
    )
    args = module.parse_args()
    assert args.no_tripwires is True
