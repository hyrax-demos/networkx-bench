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
