"""Tests for the Freeman centralization indices.

Disconnected-graph behaviour (chosen and documented in the function
docstrings): ``degree_centralization`` and ``betweenness_centralization``
are well defined on disconnected graphs and compute normally, since the star
graph remains the maximum over all simple graphs on ``n`` nodes and the
result therefore stays in [0, 1]. ``closeness_centralization`` raises
:exc:`networkx.NetworkXError` for disconnected graphs with 3 or more nodes,
because its star-based normalization assumes all distances are finite.
Graphs with fewer than 3 nodes always give 0.0, even if disconnected.
"""

import pytest

import networkx as nx

ALL_FUNCS = [
    nx.degree_centralization,
    nx.closeness_centralization,
    nx.betweenness_centralization,
]
DISCONNECTED_OK_FUNCS = [
    nx.degree_centralization,
    nx.betweenness_centralization,
]


@pytest.mark.parametrize("func", ALL_FUNCS)
@pytest.mark.parametrize("k", [2, 3, 5, 10])
def test_star_graph_is_one(func, k):
    assert func(nx.star_graph(k)) == pytest.approx(1.0)


@pytest.mark.parametrize("func", ALL_FUNCS)
@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_complete_graph_is_zero(func, n):
    assert func(nx.complete_graph(n)) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("func", "expected"),
    [
        (nx.degree_centralization, 1 / 6),
        (nx.closeness_centralization, 19 / 45),
        (nx.betweenness_centralization, 5 / 12),
    ],
)
def test_path_graph_5_hand_computed(func, expected):
    assert func(nx.path_graph(5)) == pytest.approx(expected)


@pytest.mark.parametrize("func", ALL_FUNCS)
@pytest.mark.parametrize(
    "G",
    [
        nx.empty_graph(0),
        nx.empty_graph(1),
        nx.path_graph(2),
        nx.empty_graph(2),  # two isolated nodes
    ],
    ids=["0-nodes", "1-node", "2-nodes-edge", "2-isolated-nodes"],
)
def test_small_graphs_are_zero(func, G):
    result = func(G)
    assert isinstance(result, float)
    assert result == 0.0


@pytest.mark.parametrize("func", ALL_FUNCS)
def test_directed_raises(func):
    with pytest.raises(nx.NetworkXNotImplemented):
        func(nx.path_graph(4, create_using=nx.DiGraph))


@pytest.mark.parametrize("func", ALL_FUNCS)
def test_multigraph_raises(func):
    G = nx.MultiGraph([(0, 1), (0, 1), (1, 2), (2, 3)])
    with pytest.raises(nx.NetworkXNotImplemented):
        func(G)


class TestDisconnected:
    @pytest.mark.parametrize("func", DISCONNECTED_OK_FUNCS)
    def test_two_disjoint_edges_is_zero(self, func):
        G = nx.Graph([(0, 1), (2, 3)])
        assert func(G) == pytest.approx(0.0)

    def test_closeness_raises_on_disconnected(self):
        G = nx.Graph([(0, 1), (2, 3)])
        with pytest.raises(nx.NetworkXError):
            nx.closeness_centralization(G)

    @pytest.mark.parametrize("func", DISCONNECTED_OK_FUNCS)
    def test_star_plus_isolated_nodes(self, func):
        G = nx.star_graph(3)
        G.add_nodes_from([10, 11])
        result = func(G)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_closeness_raises_on_star_plus_isolated_nodes(self):
        G = nx.star_graph(3)
        G.add_nodes_from([10, 11])
        with pytest.raises(nx.NetworkXError):
            nx.closeness_centralization(G)


@pytest.mark.parametrize("func", ALL_FUNCS)
@pytest.mark.parametrize(
    ("n", "p", "seed"),
    [(10, 0.3, 42), (15, 0.2, 1), (20, 0.5, 7), (12, 0.1, 3)],
)
def test_random_graphs_float_in_unit_interval(func, n, p, seed):
    G = nx.gnp_random_graph(n, p, seed=seed)
    if func is nx.closeness_centralization and not nx.is_connected(G):
        pytest.skip("closeness_centralization is undefined for disconnected graphs")
    result = func(G)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0
