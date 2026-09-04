from p2p_safety.data.partition import dirichlet_shard_sizes, partition_indices


def test_shard_sizes_sum_to_total():
    sizes = dirichlet_shard_sizes(n_examples=1000, n_agents=8, alpha=0.5, seed=0)
    assert len(sizes) == 8
    assert sum(sizes) == 1000
    assert all(s >= 0 for s in sizes)


def test_shard_sizes_reproducible_with_same_seed():
    a = dirichlet_shard_sizes(1000, 8, alpha=0.5, seed=7)
    b = dirichlet_shard_sizes(1000, 8, alpha=0.5, seed=7)
    assert a == b


def test_high_concentration_is_closer_to_even_than_low_concentration():
    even = dirichlet_shard_sizes(1000, 8, alpha=100.0, seed=0)
    skewed = dirichlet_shard_sizes(1000, 8, alpha=0.05, seed=0)
    assert (max(even) - min(even)) < (max(skewed) - min(skewed))


def test_partition_indices_disjoint_and_covers_all():
    shards = partition_indices(n_examples=100, n_agents=5, alpha=0.5, seed=0)
    all_indices = [i for shard in shards for i in shard]
    assert sorted(all_indices) == list(range(100))
    assert len(set(all_indices)) == 100
