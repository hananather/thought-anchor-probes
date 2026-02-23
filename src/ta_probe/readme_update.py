"""Helpers for deterministic README results block updates."""

from __future__ import annotations

from pathlib import Path

RESULTS_START = "<!-- EXPERIMENT_RESULTS_START -->"
RESULTS_END = "<!-- EXPERIMENT_RESULTS_END -->"
SCALING_RESULTS_START = "<!-- SCALING_RESULTS_START -->"
SCALING_RESULTS_END = "<!-- SCALING_RESULTS_END -->"


def replace_markdown_block(
    readme_text: str,
    new_block_markdown: str,
    *,
    start_marker: str,
    end_marker: str,
    missing_markers_message: str,
) -> str:
    """Replace one README block between marker comments."""
    if start_marker not in readme_text or end_marker not in readme_text:
        raise ValueError(missing_markers_message)

    before, remainder = readme_text.split(start_marker, maxsplit=1)
    _, after = remainder.split(end_marker, maxsplit=1)

    block = f"{start_marker}\n{new_block_markdown.strip()}\n{end_marker}"
    return f"{before.rstrip()}\n\n{block}\n{after.lstrip()}"


def replace_results_block(readme_text: str, new_block_markdown: str) -> str:
    """Replace the README experiment results block between marker comments."""
    return replace_markdown_block(
        readme_text,
        new_block_markdown,
        start_marker=RESULTS_START,
        end_marker=RESULTS_END,
        missing_markers_message=(
            "README markers not found. Add EXPERIMENT_RESULTS_START and EXPERIMENT_RESULTS_END."
        ),
    )


def update_readme_results(readme_path: str | Path, new_block_markdown: str) -> str:
    """Update README in place and return updated text."""
    path = Path(readme_path)
    original = path.read_text(encoding="utf-8")
    updated = replace_results_block(original, new_block_markdown)
    path.write_text(updated, encoding="utf-8")
    return updated


def replace_scaling_results_block(readme_text: str, new_block_markdown: str) -> str:
    """Replace the README scaling results block between marker comments."""
    return replace_markdown_block(
        readme_text,
        new_block_markdown,
        start_marker=SCALING_RESULTS_START,
        end_marker=SCALING_RESULTS_END,
        missing_markers_message=(
            "README markers not found. Add SCALING_RESULTS_START and SCALING_RESULTS_END."
        ),
    )


def update_readme_scaling_results(readme_path: str | Path, new_block_markdown: str) -> str:
    """Update README scaling results block in place and return updated text."""
    path = Path(readme_path)
    original = path.read_text(encoding="utf-8")
    updated = replace_scaling_results_block(original, new_block_markdown)
    path.write_text(updated, encoding="utf-8")
    return updated
