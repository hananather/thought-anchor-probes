from __future__ import annotations

from ta_probe.spans import (
    get_sentence_char_boundaries,
    get_sentence_token_boundaries,
    split_solution_into_chunks,
)


class FakeTokenizer:
    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
        **_kwargs,
    ):
        tokens = text.split(" ")
        offsets = []
        input_ids = []
        cursor = 0
        for idx, token in enumerate(tokens):
            start = text.find(token, cursor)
            end = start + len(token)
            offsets.append((start, end))
            input_ids.append(idx)
            cursor = end + 1
        payload = {"input_ids": input_ids}
        if return_offsets_mapping:
            payload["offset_mapping"] = offsets
        return payload

    def decode(self, ids, skip_special_tokens: bool = False):
        _ = skip_special_tokens
        return " ".join(f"tok{idx}" for idx in ids)


def test_split_solution_into_chunks_matches_tutorial_cases() -> None:
    text = (
        "<think>First, I understand the problem. Next, I'll solve for x. Finally, I verify!</think>"
    )
    expected = ["First, I understand the problem.", "Next, I'll solve for x.", "Finally, I verify!"]
    assert split_solution_into_chunks(text) == expected


def test_get_sentence_char_boundaries() -> None:
    text = "Alpha beta. Gamma delta."
    sentences = ["Alpha beta.", "Gamma delta."]
    boundaries = get_sentence_char_boundaries(text, sentences)
    assert boundaries == [(0, 11), (12, 24)]


def test_get_sentence_token_boundaries() -> None:
    text = "Alpha beta. Gamma delta."
    sentences = ["Alpha beta.", "Gamma delta."]
    tokenizer = FakeTokenizer()
    boundaries = get_sentence_token_boundaries(text, sentences, tokenizer)
    assert boundaries == [(0, 2), (2, 4)]
