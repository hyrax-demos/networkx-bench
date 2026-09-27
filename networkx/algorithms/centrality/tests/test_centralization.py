"""Tests for the Freeman centralization indices.

Disconnected-graph behaviour (chosen and documented in the function
docstrings): ``degree_centralization`` and ``betweenness_centralization``
are well defined on disconnected graphs and compute normally, since the star
graph remains the maximum over all simple graphs on ``n`` nodes and the
result therefore stays in [0, 1]. ``closeness_centralization`` raises
:exc:`networkx.NetworkXError` for disconnected graphs with 3 or more nodes,
because its star-based normalization assumes all distances are finite.
``eigenvector_centralization`` likewise raises :exc:`networkx.NetworkXError`
for disconnected graphs with 3 or more nodes, because the leading
eigenvector is not unique there.
Graphs with fewer than 3 nodes always give 0.0, even if disconnected.
"""

import math

import pytest

import networkx as nx

ALL_FUNCS = [
    nx.degree_centralization,
    nx.closeness_centralization,
    nx.betweenness_centralization,
    nx.eigenvector_centralization,
]
CONNECTED_ONLY_FUNCS = [
    nx.closeness_centralization,
    nx.eigenvector_centralization,
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
        # Leading eigenvector of P5 is proportional to (1, sqrt3, 2, sqrt3, 1),
        # so the sum of differences is sqrt3 - 1, divided by sqrt2.
        (nx.eigenvector_centralization, (math.sqrt(3) - 1) / math.sqrt(2)),
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


@pytest.mark.parametrize(
    ("G", "expected"),
    [
        # Self-loop on the path's end node must not change the value.
        (nx.Graph(list(nx.path_graph(5).edges) + [(0, 0)]), 1 / 6),
        # Self-loop on a star leaf must not lower the value below 1.
        (nx.Graph(list(nx.star_graph(4).edges) + [(1, 1)]), 1.0),
        # Self-loop on the star center must not be hidden by the clamp.
        (nx.Graph(list(nx.star_graph(4).edges) + [(0, 0)]), 1.0),
    ],
    ids=["path5-leaf-loop", "star-leaf-loop", "star-center-loop"],
)
def test_degree_centralization_ignores_self_loops(G, expected):
    edges_before = set(G.edges)
    assert nx.degree_centralization(G) == pytest.approx(expected)
    assert set(G.edges) == edges_before


class TestDisconnected:
    @pytest.mark.parametrize("func", DISCONNECTED_OK_FUNCS)
    def test_two_disjoint_edges_is_zero(self, func):
        G = nx.Graph([(0, 1), (2, 3)])
        assert func(G) == pytest.approx(0.0)

    @pytest.mark.parametrize("func", CONNECTED_ONLY_FUNCS)
    def test_connected_only_raise_on_disconnected(self, func):
        G = nx.Graph([(0, 1), (2, 3)])
        with pytest.raises(nx.NetworkXError):
            func(G)

    @pytest.mark.parametrize("func", DISCONNECTED_OK_FUNCS)
    def test_star_plus_isolated_nodes(self, func):
        G = nx.star_graph(3)
        G.add_nodes_from([10, 11])
        result = func(G)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    @pytest.mark.parametrize("func", CONNECTED_ONLY_FUNCS)
    def test_connected_only_raise_on_star_plus_isolated_nodes(self, func):
        G = nx.star_graph(3)
        G.add_nodes_from([10, 11])
        with pytest.raises(nx.NetworkXError):
            func(G)


@pytest.mark.parametrize("func", ALL_FUNCS)
@pytest.mark.parametrize(
    ("n", "p", "seed"),
    [(10, 0.3, 42), (15, 0.2, 1), (20, 0.5, 7), (12, 0.1, 3)],
)
def test_random_graphs_float_in_unit_interval(func, n, p, seed):
    G = nx.gnp_random_graph(n, p, seed=seed)
    if func in CONNECTED_ONLY_FUNCS and not nx.is_connected(G):
        pytest.skip(f"{func.__name__} is undefined for disconnected graphs")
    result = func(G)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


@pytest.mark.parametrize(
    "G",
    [
        nx.Graph(list(nx.star_graph(4).edges) + [(1, 1)]),
        nx.Graph(list(nx.star_graph(4).edges) + [(0, 0)]),
    ],
    ids=["star-leaf-loop", "star-center-loop"],
)
def test_eigenvector_centralization_ignores_self_loops(G):
    edges_before = set(G.edges)
    assert nx.eigenvector_centralization(G) == pytest.approx(1.0)
    assert set(G.edges) == edges_before


MEASURES = ["degree", "closeness", "betweenness", "eigenvector"]
WEIGHTED_MEASURES = ["betweenness"]
UNWEIGHTED_MEASURES = [m for m in MEASURES if m not in WEIGHTED_MEASURES]
MEASURE_FUNCS = dict(zip(MEASURES, ALL_FUNCS))
PATH5_EXPECTED = {
    "degree": 1 / 6,
    "closeness": 19 / 45,
    "betweenness": 5 / 12,
    "eigenvector": (math.sqrt(3) - 1) / math.sqrt(2),
}


def _weighted_star_like_k4():
    """K4 where every shortest weighted path goes through node 0."""
    G = nx.complete_graph(4)
    nx.set_edge_attributes(G, 3, "length")
    for v in (1, 2, 3):
        G[0][v]["length"] = 1
    return G


class TestCentralizationDispatcher:
    def test_default_measure_is_degree(self):
        G = nx.path_graph(5)
        assert nx.centralization(G) == nx.degree_centralization(G)

    @pytest.mark.parametrize("measure", MEASURES)
    @pytest.mark.parametrize("k", [2, 3, 5, 10])
    def test_star_graph_is_one(self, measure, k):
        assert nx.centralization(nx.star_graph(k), measure=measure) == pytest.approx(
            1.0
        )

    @pytest.mark.parametrize("measure", MEASURES)
    @pytest.mark.parametrize("n", [3, 4, 5, 6])
    def test_complete_graph_is_zero(self, measure, n):
        result = nx.centralization(nx.complete_graph(n), measure=measure)
        assert result == pytest.approx(0.0)

    @pytest.mark.parametrize("measure", MEASURES)
    def test_path_graph_5_hand_computed(self, measure):
        result = nx.centralization(nx.path_graph(5), measure=measure)
        assert result == pytest.approx(PATH5_EXPECTED[measure])

    @pytest.mark.parametrize("measure", MEASURES)
    def test_matches_measure_function(self, measure):
        G = nx.gnp_random_graph(12, 0.4, seed=5)
        assert nx.is_connected(G)
        assert nx.centralization(G, measure=measure) == MEASURE_FUNCS[measure](G)

    @pytest.mark.parametrize("measure", ["pagerank", "Degree", "", None, 3])
    def test_unknown_measure_raises(self, measure):
        with pytest.raises(ValueError, match="Unknown centralization measure"):
            nx.centralization(nx.path_graph(5), measure=measure)

    @pytest.mark.parametrize("measure", MEASURES)
    def test_directed_raises(self, measure):
        with pytest.raises(nx.NetworkXNotImplemented):
            nx.centralization(nx.path_graph(4, create_using=nx.DiGraph), measure)


class TestWeighted:
    @pytest.mark.parametrize("measure", MEASURES)
    def test_weight_none_ignores_edge_attributes(self, measure):
        G = nx.path_graph(5)
        nx.set_edge_attributes(G, {(0, 1): 10, (1, 2): 0.5}, "weight")
        result = nx.centralization(G, measure=measure, weight=None)
        assert result == pytest.approx(PATH5_EXPECTED[measure])

    @pytest.mark.parametrize("measure", WEIGHTED_MEASURES)
    def test_weighted_k4_behaves_like_star(self, measure):
        G = _weighted_star_like_k4()
        assert nx.centralization(G, measure=measure) == pytest.approx(0.0)
        result = nx.centralization(G, measure=measure, weight="length")
        assert result == pytest.approx(1.0)
        assert MEASURE_FUNCS[measure](G, weight="length") == pytest.approx(1.0)

    @pytest.mark.parametrize("measure", WEIGHTED_MEASURES)
    def test_uniform_weights_match_unweighted(self, measure):
        G = nx.path_graph(5)
        nx.set_edge_attributes(G, 2.5, "length")
        result = nx.centralization(G, measure=measure, weight="length")
        assert result == pytest.approx(PATH5_EXPECTED[measure])

    @pytest.mark.parametrize("measure", WEIGHTED_MEASURES)
    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_random_weights_in_unit_interval(self, measure, seed):
        G = nx.gnp_random_graph(15, 0.3, seed=seed)
        for i, (u, v) in enumerate(G.edges):
            G[u][v]["length"] = 1 + (i * 7919 + seed) % 13
        result = nx.centralization(G, measure=measure, weight="length")
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    @pytest.mark.parametrize("measure", UNWEIGHTED_MEASURES)
    def test_weight_rejected_by_dispatcher(self, measure):
        G = _weighted_star_like_k4()
        with pytest.raises(ValueError, match="does not support edge weights"):
            nx.centralization(G, measure=measure, weight="length")

    @pytest.mark.parametrize("measure", UNWEIGHTED_MEASURES)
    def test_weight_rejected_by_measure_function(self, measure):
        G = _weighted_star_like_k4()
        with pytest.raises(ValueError, match="does not support edge weights"):
            MEASURE_FUNCS[measure](G, weight="length")

    @pytest.mark.parametrize("measure", UNWEIGHTED_MEASURES)
    def test_weight_rejected_even_for_small_graphs(self, measure):
        with pytest.raises(ValueError, match="does not support edge weights"):
            nx.centralization(nx.path_graph(2), measure=measure, weight="weight")
