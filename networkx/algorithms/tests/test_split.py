"""Tests for split graph recognition."""

import itertools

import pytest

import networkx as nx


def _brute_force_is_split(G):
    nodes = list(G)
    for r in range(len(nodes) + 1):
        for K in itertools.combinations(nodes, r):
            Kset = set(K)
            I = [v for v in nodes if v not in Kset]
            if all(G.has_edge(u, v) for u, v in itertools.combinations(K, 2)) and (
                not any(G.has_edge(u, v) for u, v in itertools.combinations(I, 2))
            ):
                return True
    return False


@pytest.mark.parametrize(
    "G",
    [
        nx.null_graph(),
        nx.empty_graph(1),
        nx.empty_graph(5),
        nx.complete_graph(5),
        nx.star_graph(4),
        nx.path_graph(3),
        nx.path_graph(4),
    ],
)
def test_split_graphs(G):
    assert nx.is_split_graph(G)


@pytest.mark.parametrize(
    "G",
    [
        nx.cycle_graph(4),
        nx.cycle_graph(5),
        nx.path_graph(5),
        nx.Graph([(0, 1), (2, 3)]),  # 2K2
    ],
)
def test_non_split_graphs(G):
    assert not nx.is_split_graph(G)


def test_clique_with_pendants():
    G = nx.complete_graph(4)
    G.add_edges_from([(0, "a"), (1, "b"), (2, "b"), (3, "c")])
    assert nx.is_split_graph(G)


def test_complement_is_split():
    G = nx.complete_graph(4)
    G.add_edges_from([(0, 10), (1, 11), (1, 12)])
    assert nx.is_split_graph(G)
    assert nx.is_split_graph(nx.complement(G))


@pytest.mark.parametrize("n", range(1, 7))
def test_matches_brute_force_small_graphs(n):
    for G in nx.graph_atlas_g():
        if len(G) == n:
            assert nx.is_split_graph(G) == _brute_force_is_split(G)


@pytest.mark.parametrize("seed", range(10))
def test_matches_brute_force_random(seed):
    G = nx.gnp_random_graph(8, 0.5, seed=seed)
    assert nx.is_split_graph(G) == _brute_force_is_split(G)


@pytest.mark.parametrize("graph_type", [nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph])
def test_not_implemented(graph_type):
    G = graph_type([(0, 1)])
    with pytest.raises(nx.NetworkXNotImplemented):
        nx.is_split_graph(G)


def _assert_valid_split_partition(G, partition):
    clique, independent_set = partition
    assert isinstance(clique, set)
    assert isinstance(independent_set, set)
    assert clique.isdisjoint(independent_set)
    assert clique | independent_set == set(G)
    for u, v in itertools.combinations(clique, 2):
        assert G.has_edge(u, v)
    for u, v in itertools.combinations(independent_set, 2):
        assert not G.has_edge(u, v)


@pytest.mark.parametrize("n", range(7))
def test_certificate_small_graphs(n):
    for G in nx.graph_atlas_g():
        if len(G) != n:
            continue
        result, partition = nx.is_split_graph(G, certificate=True)
        assert result == nx.is_split_graph(G)
        if result:
            _assert_valid_split_partition(G, partition)
        else:
            assert partition is None


@pytest.mark.parametrize("seed", range(10))
def test_certificate_random(seed):
    G = nx.gnp_random_graph(8, 0.5, seed=seed)
    result, partition = nx.is_split_graph(G, certificate=True)
    assert result == _brute_force_is_split(G)
    if result:
        _assert_valid_split_partition(G, partition)
    else:
        assert partition is None


def test_certificate_clique_with_pendants():
    G = nx.complete_graph(4)
    G.add_edges_from([(0, "a"), (1, "b"), (2, "b"), (3, "c")])
    result, partition = nx.is_split_graph(G, certificate=True)
    assert result
    _assert_valid_split_partition(G, partition)


def test_certificate_null_graph():
    assert nx.is_split_graph(nx.null_graph(), certificate=True) == (
        True,
        (set(), set()),
    )


def test_certificate_non_split():
    assert nx.is_split_graph(nx.cycle_graph(4), certificate=True) == (False, None)
    assert nx.is_split_graph(nx.Graph([(0, 1), (2, 3)]), certificate=True) == (
        False,
        None,
    )


def test_certificate_default_returns_bool():
    assert nx.is_split_graph(nx.star_graph(3)) is True
    assert nx.is_split_graph(nx.cycle_graph(4)) is False
