from __future__ import annotations

import numpy as np

from ta_probe import activations


def test_token_vertical_depth_control_removes_future_token_count_bias() -> None:
    # One head, 4 query tokens, 4 key tokens. Every query attends equally (mass=1)
    # to each available previous key token, creating a depth bias in raw sums.
    attention = np.zeros((1, 4, 4), dtype=np.float32)
    for query_idx in range(4):
        attention[0, query_idx, : query_idx + 1] = 1.0

    token_boundaries = [(0, 1), (1, 2), (2, 3), (3, 4)]
    query_positions = np.arange(4, dtype=np.int64)

    raw_scores = activations._compute_vertical_scores_from_token_attention(
        attention,
        token_boundaries=token_boundaries,
        query_positions=query_positions,
        depth_control=False,
    )
    controlled_scores = activations._compute_vertical_scores_from_token_attention(
        attention,
        token_boundaries=token_boundaries,
        query_positions=query_positions,
        depth_control=True,
    )

    assert raw_scores.tolist()[:3] == [3.0, 2.0, 1.0]
    assert controlled_scores.tolist()[:3] == [1.0, 1.0, 1.0]


def test_full_mode_sentence_matrix_vertical_scores() -> None:
    # Two sentences of two tokens each.
    # Query sentence 1 splits attention equally between sentence 0 and itself.
    attention = np.zeros((1, 4, 4), dtype=np.float32)
    attention[0, 0, 0:2] = 0.5
    attention[0, 1, 0:2] = 0.5
    attention[0, 2, 0:2] = 0.25
    attention[0, 2, 2:4] = 0.25
    attention[0, 3, 0:2] = 0.25
    attention[0, 3, 2:4] = 0.25

    matrix = activations._compute_sentence_to_sentence_attention(
        attention,
        token_boundaries=[(0, 2), (2, 4)],
        query_positions=np.arange(4, dtype=np.int64),
    )
    scores = activations._compute_vertical_scores_from_sentence_matrix(
        matrix, depth_control=True
    )

    assert np.isclose(matrix[1, 0], 0.5)
    assert np.isclose(scores[0], 0.5)
    assert np.isclose(scores[1], 0.0)


def test_compute_vertical_attention_scores_off_mode_short_circuits() -> None:
    scores, mode = activations.compute_vertical_attention_scores(
        text="x",
        token_boundaries=[(0, 1)],
        model=object(),  # type: ignore[arg-type]
        tokenizer=object(),  # type: ignore[arg-type]
        layer_idx=0,
        device="cpu",
        mode="off",
        depth_control=True,
        light_last_n_tokens=1,
        full_max_seq_len=8,
    )
    assert scores is None
    assert mode == "off"
