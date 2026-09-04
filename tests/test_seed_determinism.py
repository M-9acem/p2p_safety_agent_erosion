"""Spec section 9: seed determinism is one of the required tests."""
import random

import numpy as np

from p2p_safety.data.partition import dirichlet_shard_sizes
from p2p_safety.graph import build_graph, choose_agents_by_placement
from p2p_safety.utils import seed_everything


def test_seed_everything_makes_python_random_reproducible():
    seed_everything(123)
    a = [random.random() for _ in range(5)]
    seed_everything(123)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_seed_everything_makes_numpy_reproducible():
    seed_everything(123)
    a = np.random.rand(5).tolist()
    seed_everything(123)
    b = np.random.rand(5).tolist()
    assert a == b


def test_downstream_graph_and_partition_reproducible_given_seed():
    seed_everything(42)
    g1 = build_graph("random", n_agents=10, seed=42, p=0.3)
    sizes1 = dirichlet_shard_sizes(1000, 10, alpha=0.5, seed=42)
    placement1 = choose_agents_by_placement(g1, k=2, placement="spread", seed=42)

    seed_everything(42)
    g2 = build_graph("random", n_agents=10, seed=42, p=0.3)
    sizes2 = dirichlet_shard_sizes(1000, 10, alpha=0.5, seed=42)
    placement2 = choose_agents_by_placement(g2, k=2, placement="spread", seed=42)

    assert sorted(g1.edges) == sorted(g2.edges)
    assert sizes1 == sizes2
    assert placement1 == placement2
