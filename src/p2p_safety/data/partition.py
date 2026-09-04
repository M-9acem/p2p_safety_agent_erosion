"""Split a task dataset across agents with a Dirichlet distribution.

Concentration is a config knob (spec section 5): low concentration gives
skewed, non-IID shard sizes; high concentration approaches an even split.
This only controls shard *size* per agent, not label skew, since the task
data (Alpaca/Dolly) is instruction-following without a class label.
"""
from __future__ import annotations

import random


def dirichlet_shard_sizes(n_examples: int, n_agents: int, alpha: float, seed: int = 0) -> list[int]:
    """Return n_agents shard sizes summing to n_examples, drawn from a
    Dirichlet(alpha) proportion split.

    Falls back to numpy if available; otherwise uses a Gamma-sampling
    construction (Dirichlet = normalized independent Gammas) over the
    stdlib `random` module so this has no hard numpy dependency.
    """
    try:
        import numpy as np

        rng = np.random.default_rng(seed)
        proportions = rng.dirichlet([alpha] * n_agents)
    except ImportError:
        rng = random.Random(seed)
        gammas = [rng.gammavariate(alpha, 1.0) for _ in range(n_agents)]
        total = sum(gammas)
        proportions = [g / total for g in gammas]

    sizes = [int(p * n_examples) for p in proportions]
    # Fix rounding: hand out the remainder to the largest shards first so
    # sum(sizes) == n_examples exactly.
    remainder = n_examples - sum(sizes)
    order = sorted(range(n_agents), key=lambda i: proportions[i], reverse=True)
    for i in range(remainder):
        sizes[order[i % n_agents]] += 1
    return sizes


def partition_indices(n_examples: int, n_agents: int, alpha: float, seed: int = 0) -> list[list[int]]:
    """Partition example indices [0, n_examples) into n_agents disjoint,
    contiguous-after-shuffle shards sized per `dirichlet_shard_sizes`."""
    sizes = dirichlet_shard_sizes(n_examples, n_agents, alpha, seed)
    indices = list(range(n_examples))
    random.Random(seed).shuffle(indices)

    shards = []
    cursor = 0
    for size in sizes:
        shards.append(sorted(indices[cursor : cursor + size]))
        cursor += size
    return shards
