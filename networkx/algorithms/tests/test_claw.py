"""Tests for claw-free graph recognition."""

import itertools

import pytest

import networkx as nx


def _brute_force_is_claw_free(G):
    for v in G:
        nbrs = [u for u in G[v] if u != v]
        for a, b, c in itertools.combinations(nbrs, 3):
            if not (G.has_edge(a, b) or G.has_edge(a, c) or G.has_edge(b, c)):
                return False
    return True


@pytest.mark.parametrize(
    "G",
    [
        nx.star_graph(3),
        nx.star_graph(5),
        nx.complete_bipartite_graph(1, 3),
        nx.complete_bipartite_graph(3, 3),
        nx.petersen_graph(),
        nx.balanced_tree(2, 3),
    ],
)
def test_not_claw_free(G):
    assert not nx.is_claw_free(G)


@pytest.mark.parametrize(
    "G",
    [
        nx.null_graph(),
        nx.empty_graph(1),
        nx.empty_graph(5),
        nx.path_graph(6),
        nx.cycle_graph(7),
        nx.complete_graph(6),
        nx.star_graph(2),
    ],
)
def test_claw_free(G):
    assert nx.is_claw_free(G)


@pytest.mark.parametrize(
    "H",
    [
        nx.star_graph(5),
        nx.complete_graph(5),
        nx.petersen_graph(),
        nx.complete_bipartite_graph(3, 4),
        nx.balanced_tree(3, 2),
        nx.gnp_random_graph(15, 0.3, seed=42),
    ],
)
def test_line_graphs_are_claw_free(H):
    assert nx.is_claw_free(nx.line_graph(H))


def test_claw_with_extra_edge_among_leaves():
    G = nx.star_graph(3)
    G.add_edge(1, 2)
    assert nx.is_claw_free(G)


def test_self_loops_ignored():
    G = nx.path_graph(3)
    G.add_edge(1, 1)
    assert nx.is_claw_free(G)
    G = nx.star_graph(2)
    G.add_edge(0, 0)
    assert nx.is_claw_free(G)


@pytest.mark.parametrize("seed", range(20))
def test_matches_brute_force(seed):
    G = nx.gnp_random_graph(9, 0.5, seed=seed)
    assert nx.is_claw_free(G) == _brute_force_is_claw_free(G)


@pytest.mark.parametrize("graph_type", [nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph])
def test_not_implemented(graph_type):
    G = graph_type([(0, 1), (0, 2), (0, 3)])
    with pytest.raises(nx.NetworkXNotImplemented):
        nx.is_claw_free(G)
