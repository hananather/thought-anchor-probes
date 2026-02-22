"""Helpers for deterministic README results block updates."""

from __future__ import annotations

from pathlib import Path

RESULTS_START = "<!-- EXPERIMENT_RESULTS_START -->"
RESULTS_END = "<!-- EXPERIMENT_RESULTS_END -->"


def replace_results_block(readme_text: str, new_block_markdown: str) -> str:
    """Replace the README experiment results block between marker comments."""
    if RESULTS_START not in readme_text or RESULTS_END not in readme_text:
        msg = "README markers not found. Add EXPERIMENT_RESULTS_START and EXPERIMENT_RESULTS_END."
        raise ValueError(msg)

    before, remainder = readme_text.split(RESULTS_START, maxsplit=1)
    _, after = remainder.split(RESULTS_END, maxsplit=1)

    block = f"{RESULTS_START}\n{new_block_markdown.strip()}\n{RESULTS_END}"
    return f"{before.rstrip()}\n\n{block}\n{after.lstrip()}"


def update_readme_results(readme_path: str | Path, new_block_markdown: str) -> str:
    """Update README in place and return updated text."""
    path = Path(readme_path)
    original = path.read_text(encoding="utf-8")
    updated = replace_results_block(original, new_block_markdown)
    path.write_text(updated, encoding="utf-8")
    return updated
