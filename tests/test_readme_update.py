from __future__ import annotations

from ta_probe.readme_update import replace_results_block


def test_replace_results_block_is_deterministic() -> None:
    original = (
        "# Title\n\n"
        "<!-- EXPERIMENT_RESULTS_START -->\n"
        "old\n"
        "<!-- EXPERIMENT_RESULTS_END -->\n\n"
        "Body\n"
    )
    updated = replace_results_block(original, "new content")

    assert "new content" in updated
    assert "old" not in updated
    assert updated.count("<!-- EXPERIMENT_RESULTS_START -->") == 1
    assert updated.count("<!-- EXPERIMENT_RESULTS_END -->") == 1
