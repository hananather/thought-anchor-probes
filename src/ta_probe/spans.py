"""Sentence chunking and token boundary utilities."""

from __future__ import annotations

import random
import re
from typing import Any

import numpy as np
import pandas as pd
from transformers import PreTrainedTokenizerBase


def split_solution_into_chunks(text: str) -> list[str]:
    """Split solution text into sentence-like chunks.

    This mirrors the ARENA tutorial behavior for thought-anchor chunking.
    """
    if "<think>" in text:
        text = text.split("<think>", maxsplit=1)[1]
    if "</think>" in text:
        text = text.split("</think>", maxsplit=1)[0]
    text = text.strip()

    text = re.sub(r"(\d)\.(\d)", r"\1<DECIMAL>\2", text)
    text = re.sub(r"\n(\d)\.(\s)", r"\n\1<DECIMAL>\2", text)

    sentences = re.split(r"([!?:\n]|(?<!\n\d)\.)", text)
    chunks: list[str] = []
    for idx in range(0, len(sentences) - 1, 2):
        chunks.append((sentences[idx] + sentences[idx + 1]).replace("\n", " "))

    chunks = [re.sub(r"<DECIMAL>", ".", chunk) for chunk in chunks]
    if not chunks:
        return []

    merged = [chunks[0]]
    for chunk in chunks[1:]:
        if len(merged[-1]) < 10:
            merged[-1] += chunk
        else:
            merged.append(chunk)
    return [chunk.strip() for chunk in merged if chunk.strip()]


def get_whitebox_example_data(problem_data: dict[str, Any]) -> tuple[str, list[str], np.ndarray]:
    """Extract text, sentence list, and counterfactual scores from one problem payload."""
    chunks_labeled = problem_data["chunks_labeled"]
    sentences = [chunk.get("chunk", "") for chunk in chunks_labeled]

    base_solution = problem_data.get("base_solution", {})
    problem = problem_data.get("problem", {})

    text_candidates = [
        base_solution.get("full_cot"),
        base_solution.get("solution"),
        problem.get("solution"),
        " ".join(sentences),
    ]
    text = next(
        (candidate for candidate in text_candidates if isinstance(candidate, str) and candidate), ""
    )

    counterfactual_scores = np.array(
        [
            -float(chunk.get("counterfactual_importance_accuracy", 0.0))
            for chunk in chunks_labeled[:-1]
        ],
        dtype=np.float32,
    )
    return text, sentences, counterfactual_scores


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def get_sentence_char_boundaries(text: str, sentences: list[str]) -> list[tuple[int, int]]:
    """Map each sentence to character boundaries in the full text."""
    boundaries: list[tuple[int, int]] = []
    cursor = 0

    for sentence in sentences:
        if not sentence:
            boundaries.append((cursor, cursor))
            continue

        start = text.find(sentence, cursor)
        if start == -1:
            normalized = _normalize_spaces(sentence)
            escaped = re.escape(normalized).replace(r"\ ", r"\s+")
            match = re.search(escaped, text[cursor:])
            if match is None:
                msg = "Could not align sentence to full text"
                raise ValueError(msg)
            start = cursor + match.start()
            end = cursor + match.end()
        else:
            end = start + len(sentence)

        boundaries.append((start, end))
        cursor = end

    return boundaries


def get_sentence_token_boundaries(
    text: str,
    sentences: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> list[tuple[int, int]]:
    """Map each sentence to token boundaries for a fast tokenizer."""
    encoding = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
        return_token_type_ids=False,
    )

    if "offset_mapping" not in encoding:
        msg = "Tokenizer must provide offset mapping. Use a fast tokenizer."
        raise ValueError(msg)

    offsets: list[tuple[int, int]] = list(encoding["offset_mapping"])
    char_boundaries = get_sentence_char_boundaries(text, sentences)

    token_boundaries: list[tuple[int, int]] = []
    for char_start, char_end in char_boundaries:
        token_start = None
        for idx, (_, token_end_char) in enumerate(offsets):
            if token_end_char > char_start:
                token_start = idx
                break

        if token_start is None:
            token_start = len(offsets)

        token_end = len(offsets)
        for idx in range(token_start, len(offsets)):
            token_start_char, _ = offsets[idx]
            if token_start_char >= char_end:
                token_end = idx
                break

        if token_end <= token_start:
            token_end = min(token_start + 1, len(offsets))

        token_boundaries.append((token_start, token_end))

    return token_boundaries


def span_integrity_report(
    text: str,
    sentences: list[str],
    tokenizer: PreTrainedTokenizerBase,
    sample_size: int = 20,
    seed: int = 0,
) -> pd.DataFrame:
    """Spot-check whether token spans decode to the expected sentence text."""
    boundaries = get_sentence_token_boundaries(text, sentences, tokenizer)
    encoding = tokenizer(text, add_special_tokens=False)
    input_ids: list[int] = list(encoding["input_ids"])

    n = len(sentences)
    sample_count = min(sample_size, n)
    rng = random.Random(seed)
    indices = sorted(rng.sample(list(range(n)), sample_count)) if n else []

    rows: list[dict[str, Any]] = []
    for idx in indices:
        token_start, token_end = boundaries[idx]
        decoded = tokenizer.decode(input_ids[token_start:token_end], skip_special_tokens=False)

        expected = _normalize_spaces(sentences[idx]).lower()
        observed = _normalize_spaces(decoded).lower()
        passes = expected in observed or observed in expected

        rows.append(
            {
                "sentence_idx": idx,
                "token_start": token_start,
                "token_end": token_end,
                "expected": sentences[idx],
                "decoded": decoded,
                "pass": bool(passes),
            }
        )

    return pd.DataFrame(rows)
