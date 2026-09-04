import networkx as nx
import pytest

from p2p_safety.graph import (
    build_graph,
    choose_agents_by_placement,
    closed_neighborhood,
    hop_distances,
    neighbors,
)


def test_ring_neighbor_set_correctness():
    g = build_graph("ring", n_agents=6, seed=0)
    for i in range(6):
        expected = {(i - 1) % 6, (i + 1) % 6}
        assert set(neighbors(g, i)) == expected
        assert closed_neighborhood(g, i) == sorted(expected | {i})


def test_complete_neighbor_set_correctness():
    g = build_graph("complete", n_agents=5, seed=0)
    for i in range(5):
        assert set(neighbors(g, i)) == set(range(5)) - {i}
        assert closed_neighborhood(g, i) == list(range(5))


def test_random_graph_is_connected_and_reproducible():
    g1 = build_graph("random", n_agents=10, seed=42, p=0.3)
    g2 = build_graph("random", n_agents=10, seed=42, p=0.3)
    assert nx.is_connected(g1)
    assert sorted(g1.edges) == sorted(g2.edges)


def test_unknown_topology_raises():
    with pytest.raises(ValueError):
        build_graph("mesh", n_agents=4)


def test_hop_distances_zero_at_source():
    g = build_graph("ring", n_agents=6, seed=0)
    dist = hop_distances(g, sources=[0])
    assert dist[0] == 0
    assert dist[1] == 1
    assert dist[3] == 3  # farthest point on a 6-ring


def test_hop_distances_multi_source_takes_min():
    g = build_graph("ring", n_agents=8, seed=0)
    dist = hop_distances(g, sources=[0, 4])
    assert dist[2] == 2  # equidistant from both sources on an 8-ring
    assert dist[0] == 0
    assert dist[4] == 0


@pytest.mark.parametrize("placement", ["random", "well_connected", "peripheral", "clustered", "spread"])
def test_choose_agents_by_placement_returns_k_valid_nodes(placement):
    g = build_graph("random", n_agents=12, seed=1, p=0.3)
    chosen = choose_agents_by_placement(g, k=3, placement=placement, seed=0)
    assert len(chosen) == 3
    assert len(set(chosen)) == 3
    assert set(chosen).issubset(set(g.nodes))


def test_well_connected_picks_highest_degree():
    g = build_graph("random", n_agents=12, seed=1, p=0.3)
    chosen = choose_agents_by_placement(g, k=3, placement="well_connected", seed=0)
    degrees = dict(g.degree)
    top3 = sorted(degrees, key=lambda n: -degrees[n])[:3]
    assert set(chosen) == set(top3)


def test_placement_k_exceeds_n_agents_raises():
    g = build_graph("ring", n_agents=4, seed=0)
    with pytest.raises(ValueError):
        choose_agents_by_placement(g, k=5, placement="random")
