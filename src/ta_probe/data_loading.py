"""Dataset listing and loading for the Thought Anchors math rollout dataset."""

from __future__ import annotations

import json
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import disable_progress_bars, enable_progress_bars
from tqdm.auto import tqdm

PROBLEM_ID_PATTERN = re.compile(r"problem_(\d+)")


def _problem_root(*, problem_id: int, model_dir: str, temp_dir: str, split_dir: str) -> str:
    return f"{model_dir}/{temp_dir}/{split_dir}/problem_{problem_id}"


def load_single_file(repo_id: str, file_path: str) -> Any:
    """Download one JSON file from the dataset and load it."""
    local_path = hf_hub_download(repo_id=repo_id, filename=file_path, repo_type="dataset")
    with Path(local_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def list_problem_ids(
    *,
    repo_id: str,
    model_dir: str,
    temp_dir: str,
    split_dir: str,
    cache_path: str | Path,
    force_refresh: bool = False,
) -> list[int]:
    """List available problem IDs and cache the result."""
    cache_file = Path(cache_path)
    if cache_file.exists() and not force_refresh:
        with cache_file.open("r", encoding="utf-8") as handle:
            return list(json.load(handle))

    api = HfApi()
    prefix = f"{model_dir}/{temp_dir}/{split_dir}"
    ids: set[int] = set()

    try:
        entries = api.list_repo_tree(
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=prefix,
            recursive=False,
        )
        for entry in entries:
            entry_path = (
                getattr(entry, "path", None)
                or getattr(entry, "rfilename", None)
                or str(entry)
            )
            match = PROBLEM_ID_PATTERN.search(entry_path)
            if match:
                ids.add(int(match.group(1)))
    except Exception:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        for path in files:
            if not path.startswith(f"{prefix}/"):
                continue
            if "/problem_" not in path:
                continue
            match = PROBLEM_ID_PATTERN.search(path)
            if match:
                ids.add(int(match.group(1)))

    sorted_ids = sorted(ids)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with cache_file.open("w", encoding="utf-8") as handle:
        json.dump(sorted_ids, handle, indent=2)
    return sorted_ids


def sample_problem_ids(problem_ids: list[int], num_problems: int, seed: int) -> list[int]:
    """Sample problem IDs with a fixed seed."""
    if num_problems >= len(problem_ids):
        return sorted(problem_ids)
    rng = random.Random(seed)
    sampled = rng.sample(problem_ids, num_problems)
    return sorted(sampled)


def create_splits(
    problem_ids: list[int],
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, list[int]]:
    """Create disjoint train, validation, and test splits by problem ID."""
    if abs((train_fraction + val_fraction + test_fraction) - 1.0) > 1e-6:
        msg = "train_fraction + val_fraction + test_fraction must sum to 1.0"
        raise ValueError(msg)

    rng = random.Random(seed)
    shuffled = list(problem_ids)
    rng.shuffle(shuffled)

    n = len(shuffled)
    if n < 3:
        msg = "Need at least 3 problems to create train, val, and test splits."
        raise ValueError(msg)

    n_train = max(1, int(n * train_fraction))
    n_val = max(1, int(n * val_fraction))

    while n_train + n_val > n - 1:
        if n_train >= n_val and n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            msg = "Invalid split sizes. Increase number of problems or adjust fractions."
            raise ValueError(msg)

    train_ids = sorted(shuffled[:n_train])
    val_ids = sorted(shuffled[n_train : n_train + n_val])
    test_ids = sorted(shuffled[n_train + n_val :])

    if not test_ids:
        msg = "Test split is empty. Increase number of problems."
        raise ValueError(msg)

    return {"train": train_ids, "val": val_ids, "test": test_ids}


def write_json(path: str | Path, payload: dict[str, Any] | list[Any]) -> None:
    """Write JSON with stable formatting."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_problem_metadata(
    *,
    problem_id: int,
    repo_id: str,
    model_dir: str,
    temp_dir: str,
    split_dir: str,
) -> dict[str, Any]:
    """Load only the small metadata files for one problem."""
    root = _problem_root(
        problem_id=problem_id, model_dir=model_dir, temp_dir=temp_dir, split_dir=split_dir
    )
    problem = load_single_file(repo_id, f"{root}/problem.json")
    base_solution = load_single_file(repo_id, f"{root}/base_solution.json")
    chunks_labeled = load_single_file(repo_id, f"{root}/chunks_labeled.json")
    return {
        "problem": problem,
        "base_solution": base_solution,
        "chunks_labeled": chunks_labeled,
    }


def load_problem_full(
    *,
    problem_id: int,
    repo_id: str,
    model_dir: str,
    temp_dir: str,
    split_dir: str,
    forced_split_dir: str,
    load_forced: bool = True,
    max_workers: int = 16,
    verbose: bool = True,
) -> dict[str, Any]:
    """Load metadata plus per-chunk rollout files for verification."""
    disable_progress_bars()

    metadata = load_problem_metadata(
        problem_id=problem_id,
        repo_id=repo_id,
        model_dir=model_dir,
        temp_dir=temp_dir,
        split_dir=split_dir,
    )

    n_chunks = len(metadata["chunks_labeled"])
    chunk_solutions: list[list[dict[str, Any]] | None] = [None] * n_chunks
    chunk_solutions_forced: list[list[dict[str, Any]] | None] = [None] * n_chunks

    root = _problem_root(
        problem_id=problem_id, model_dir=model_dir, temp_dir=temp_dir, split_dir=split_dir
    )
    forced_root = _problem_root(
        problem_id=problem_id,
        model_dir=model_dir,
        temp_dir=temp_dir,
        split_dir=forced_split_dir,
    )

    def load_chunk(chunk_idx: int) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]] | None]:
        regular = load_single_file(repo_id, f"{root}/chunk_{chunk_idx}/solutions.json")
        forced = None
        if load_forced:
            forced = load_single_file(repo_id, f"{forced_root}/chunk_{chunk_idx}/solutions.json")
        return chunk_idx, regular, forced

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(load_chunk, idx) for idx in range(n_chunks)]
        progress = tqdm(
            as_completed(futures),
            total=n_chunks,
            desc="Loading chunk rollouts",
            disable=not verbose,
        )
        for future in progress:
            idx, regular, forced = future.result()
            chunk_solutions[idx] = regular
            if load_forced:
                chunk_solutions_forced[idx] = forced

    enable_progress_bars()

    payload = {
        "problem": metadata["problem"],
        "base_solution": metadata["base_solution"],
        "chunks_labeled": metadata["chunks_labeled"],
        "chunk_solutions": chunk_solutions,
    }
    if load_forced:
        payload["chunk_solutions_forced"] = chunk_solutions_forced
    return payload
