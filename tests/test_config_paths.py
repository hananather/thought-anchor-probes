from __future__ import annotations

from pathlib import Path

from ta_probe.config import load_config, resolve_embeddings_memmap_path


def test_resolve_embeddings_memmap_path_uses_sibling_token_file_when_default(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        (
            "activations:\n"
            "  pooling: tokens\n"
            "paths:\n"
            "  embeddings_memmap: artifacts/runs/pilot/sentence_embeddings.dat\n"
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    resolved = resolve_embeddings_memmap_path(config)
    assert resolved == "artifacts/runs/pilot/token_embeddings.dat"


def test_resolve_embeddings_memmap_path_uses_explicit_token_path(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        (
            "activations:\n"
            "  pooling: tokens\n"
            "paths:\n"
            "  embeddings_memmap: artifacts/runs/pilot/sentence_embeddings.dat\n"
            "  token_embeddings_memmap: artifacts/custom_tokens.dat\n"
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    resolved = resolve_embeddings_memmap_path(config)
    assert resolved == "artifacts/custom_tokens.dat"
