# src/data/split_clients_dwrl.py
from __future__ import annotations

from typing import List, Literal
import numpy as np


def split_indices_equal(n: int, n_clients: int, seed: int = 42) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    sizes = [n // n_clients] * n_clients
    for i in range(n % n_clients):
        sizes[i] += 1
    out, start = [], 0
    for s in sizes:
        out.append(np.sort(perm[start:start + s]))
        start += s
    return out


def split_indices_dirichlet(
    targets: np.ndarray,
    n_clients: int,
    alpha: float,
    seed: int = 42,
) -> List[np.ndarray]:
    """
    Dirichlet non-IID split (same spirit as your CIFAR code).
    targets: shape [N], integer labels
    """
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)
    n_classes = int(targets.max()) + 1

    class_indices = [np.where(targets == c)[0] for c in range(n_classes)]
    for ci in class_indices:
        rng.shuffle(ci)

    client_lists = [[] for _ in range(n_clients)]

    for c in range(n_classes):
        idx_c = class_indices[c]
        n_c = len(idx_c)
        if n_c == 0:
            continue

        props = rng.dirichlet(np.ones(n_clients) * alpha)
        counts = (props * n_c).astype(int)

        # fix rounding
        while counts.sum() < n_c:
            counts[rng.integers(0, n_clients)] += 1

        start = 0
        for k in range(n_clients):
            take = int(counts[k])
            if take > 0:
                client_lists[k].extend(idx_c[start:start + take].tolist())
                start += take

    return [np.array(sorted(ix), dtype=np.int64) for ix in client_lists]


def make_client_splits(
    n: int,
    targets: np.ndarray | None,
    n_clients: int = 4,
    mode: Literal["equal", "dirichlet"] = "equal",
    alpha: float = 0.2,
    seed: int = 42,
) -> List[np.ndarray]:
    if mode == "equal":
        return split_indices_equal(n, n_clients, seed=seed)
    if mode == "dirichlet":
        if targets is None:
            raise ValueError("targets must be provided for mode='dirichlet'")
        return split_indices_dirichlet(targets, n_clients, alpha=alpha, seed=seed)
    raise ValueError(f"Unknown split mode: {mode}")


def make_cl_schedule(
    client_indices: List[np.ndarray],
    cl_batches: int = 7,
    seed: int = 42,
    init_frac: float = 0.466,
) -> List[List[np.ndarray]]:
    """
    Build continual-learning (CL) schedule per client.
    Returns:
      cl_schedule[cid][r] = np.ndarray of indices for round r (new data only)
.
    Strategy:
      - shuffle each client's indices with (seed + cid)
      - first batch gets ~init_frac of data (like CIFAR script)
      - remaining data split across (cl_batches-1) batches
    """
    if cl_batches < 1:
        raise ValueError("cl_batches must be >= 1")

    cl_schedule: List[List[np.ndarray]] = []

    for cid, idx in enumerate(client_indices):
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            cl_schedule.append([idx.copy() for _ in range(cl_batches)])
            continue

        rng = np.random.default_rng(seed + cid)
        idx = idx.copy()
        rng.shuffle(idx)

        if cl_batches == 1:
            cl_schedule.append([np.sort(idx)])
            continue

        # first batch size ~ init_frac
        init = int(round(init_frac * len(idx)))
        # ensure we can still allocate at least 1 element to each remaining batch
        init = max(1, min(len(idx) - (cl_batches - 1), init))

        first = idx[:init]
        rem = idx[init:]

        # split rem into (cl_batches-1) chunks as evenly as possible
        chunks = np.array_split(rem, cl_batches - 1)

        batches = [np.sort(first)] + [np.sort(c) for c in chunks]
        # ensure exact length
        assert len(batches) == cl_batches
        cl_schedule.append(batches)

    return cl_schedule