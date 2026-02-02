# src/data/cl_schedule.py
from __future__ import annotations

from typing import List
import numpy as np


def make_cl_batches(
    indices: List[int] | np.ndarray,
    num_batches: int = 7,
    init_ratio: float = 0.37,
    seed: int = 42,
) -> List[List[int]]:
    """
    Returns list of batches (each is a list of indices).
    Batch 1 has ~init_ratio of data, remaining split evenly across num_batches-1.
    """
    rng = np.random.default_rng(seed)
    idx = np.array(indices, dtype=np.int64)
    rng.shuffle(idx)

    if num_batches <= 1:
        return [idx.tolist()]

    init = int(round(init_ratio * len(idx)))
    # ensure remaining can be split into num_batches-1 non-empty chunks
    init = max(1, min(len(idx) - (num_batches - 1), init))

    first = idx[:init]
    rem = idx[init:]

    # split rem into num_batches-1 chunks
    per = len(rem) // (num_batches - 1)
    chunks = [rem[i * per:(i + 1) * per] for i in range(num_batches - 2)]
    chunks.append(rem[(num_batches - 2) * per:])

    batches = [first.tolist()] + [c.tolist() for c in chunks]
    return batches