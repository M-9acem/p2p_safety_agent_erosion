"""Graph construction and neighbor lookup for the P2P simulation.

Deliberately plain (spec section 4): ring, Erdos-Renyi random, or complete
graph over N agents. No mixing matrices, no edge weights.
"""
from __future__ import annotations

import networkx as nx


def build_graph(topology: str, n_agents: int, seed: int = 0, p: float = 0.3) -> nx.Graph:
    """Build the agent communication graph.

    Args:
        topology: "ring", "random", or "complete".
        n_agents: number of agents (nodes).
        seed: RNG seed, used only for "random".
        p: edge probability for the Erdos-Renyi "random" graph.

    Returns:
        An undirected, connected networkx.Graph with nodes 0..n_agents-1.
    """
    if n_agents < 2:
        raise ValueError(f"need at least 2 agents, got {n_agents}")

    if topology == "ring":
        g = nx.cycle_graph(n_agents)
    elif topology == "complete":
        g = nx.complete_graph(n_agents)
    elif topology == "random":
        g = _connected_erdos_renyi(n_agents, p, seed)
    else:
        raise ValueError(f"unknown topology: {topology}")

    if not nx.is_connected(g):
        raise RuntimeError(f"generated {topology} graph is disconnected")
    return g


def _connected_erdos_renyi(n_agents: int, p: float, seed: int) -> nx.Graph:
    """Erdos-Renyi graph, resampled until connected.

    A disconnected graph would let a component drift independently, which
    is a confound we don't want. Retry with the same seeded RNG rather than
    silently falling back to a different topology.
    """
    rng = seed
    for attempt in range(1000):
        g = nx.erdos_renyi_graph(n_agents, p, seed=rng)
        if nx.is_connected(g):
            return g
        rng += 1
    raise RuntimeError(
        f"could not sample a connected random graph (n={n_agents}, p={p}) "
        "after 1000 attempts; raise p"
    )


def neighbors(g: nx.Graph, agent_id: int) -> list[int]:
    """Open neighborhood: agents directly connected to `agent_id`."""
    return sorted(g.neighbors(agent_id))


def closed_neighborhood(g: nx.Graph, agent_id: int) -> list[int]:
    """Self plus direct neighbors — exactly what AVERAGE (spec section 4)
    uniformly averages over each round."""
    return sorted(set(neighbors(g, agent_id)) | {agent_id})


def hop_distances(g: nx.Graph, sources: list[int]) -> dict[int, int]:
    """Shortest-path hop distance from each node to the nearest node in
    `sources`. Used for the RQ3 plot: ASR vs. hop distance from the nearest
    safety-holding agent.

    A node in `sources` itself has distance 0. Unreachable nodes (only
    possible if the graph were disconnected, which build_graph disallows)
    are not included.
    """
    if not sources:
        raise ValueError("sources must be non-empty")
    dist: dict[int, int] = {}
    for s in sources:
        for node, d in nx.single_source_shortest_path_length(g, s).items():
            if node not in dist or d < dist[node]:
                dist[node] = d
    return dist


def choose_agents_by_placement(
    g: nx.Graph, k: int, placement: str, seed: int = 0
) -> list[int]:
    """Choose k agents to be safety-holding (RQ3), by placement strategy.

    Args:
        placement: "random", "well_connected" (highest degree), "peripheral"
            (lowest degree), "clustered" (a connected block via BFS from a
            random seed node), or "spread" (greedily maximize minimum
            pairwise hop distance among the chosen set).
    """
    nodes = list(g.nodes)
    if k > len(nodes):
        raise ValueError(f"k={k} exceeds n_agents={len(nodes)}")
    rng = __import__("random").Random(seed)

    if placement == "random":
        return sorted(rng.sample(nodes, k))

    if placement == "well_connected":
        return sorted(sorted(nodes, key=lambda n: (-g.degree[n], n))[:k])

    if placement == "peripheral":
        return sorted(sorted(nodes, key=lambda n: (g.degree[n], n))[:k])

    if placement == "clustered":
        start = rng.choice(nodes)
        order = [start] + [n for n in nx.bfs_tree(g, start) if n != start]
        return sorted(order[:k])

    if placement == "spread":
        chosen = [rng.choice(nodes)]
        while len(chosen) < k:
            dist = hop_distances(g, chosen)
            remaining = [n for n in nodes if n not in chosen]
            best = max(remaining, key=lambda n: (dist.get(n, 0), -n))
            chosen.append(best)
        return sorted(chosen)

    raise ValueError(f"unknown placement: {placement}")
