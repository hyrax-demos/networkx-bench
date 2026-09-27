"""Unit tests for the :mod:`networkx.algorithms.graph_classes` module."""

import itertools

import pytest

import networkx as nx

RECOGNIZERS = [
    nx.is_path_graph,
    nx.is_cycle_graph,
    nx.is_star_graph,
    nx.is_wheel_graph,
    nx.is_complete_graph,
    nx.is_complete_bipartite_graph,
]


def _add_one_edge(G):
    """Return a copy of G with one non-edge added (None if G is complete)."""
    for u, v in itertools.combinations(G, 2):
        if not G.has_edge(u, v):
            H = G.copy()
            H.add_edge(u, v)
            return H
    return None


def _remove_one_edge(G):
    """Yield copies of G with each single edge removed."""
    for e in G.edges:
        H = G.copy()
        H.remove_edge(*e)
        yield H


@pytest.mark.parametrize("func", RECOGNIZERS)
def test_null_graph_raises(func):
    with pytest.raises(nx.NetworkXPointlessConcept):
        func(nx.null_graph())


@pytest.mark.parametrize("func", RECOGNIZERS)
@pytest.mark.parametrize("graph_type", [nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph])
def test_unsupported_graph_types(func, graph_type):
    with pytest.raises(nx.NetworkXNotImplemented):
        func(nx.path_graph(3, create_using=graph_type))


@pytest.mark.parametrize("func", RECOGNIZERS)
def test_self_loop_rejected(func):
    G = nx.complete_graph(4)
    G.add_edge(0, 0)
    assert not func(G)


@pytest.mark.parametrize("func", RECOGNIZERS)
def test_node_labels_irrelevant(func):
    G = nx.wheel_graph(6)
    H = nx.relabel_nodes(G, {i: f"n{i}" for i in G})
    assert func(G) == func(H)


class TestPathGraph:
    @pytest.mark.parametrize("n", range(1, 9))
    def test_generator(self, n):
        assert nx.is_path_graph(nx.path_graph(n))

    @pytest.mark.parametrize("n", range(3, 9))
    def test_add_edge(self, n):
        assert not nx.is_path_graph(_add_one_edge(nx.path_graph(n)))

    @pytest.mark.parametrize("n", range(2, 9))
    def test_remove_edge(self, n):
        for H in _remove_one_edge(nx.path_graph(n)):
            assert not nx.is_path_graph(H)

    def test_other_trees(self):
        assert not nx.is_path_graph(nx.star_graph(3))
        # disconnected, degrees <= 2 and n - 1 edges: isolated node + triangle
        G = nx.disjoint_union(nx.path_graph(1), nx.cycle_graph(3))
        assert G.number_of_edges() == len(G) - 1
        assert not nx.is_path_graph(G)


class TestCycleGraph:
    @pytest.mark.parametrize("n", range(3, 10))
    def test_generator(self, n):
        assert nx.is_cycle_graph(nx.cycle_graph(n))

    @pytest.mark.parametrize("n", [1, 2])
    def test_degenerate(self, n):
        assert not nx.is_cycle_graph(nx.cycle_graph(n))

    @pytest.mark.parametrize("n", range(4, 10))
    def test_add_edge(self, n):
        assert not nx.is_cycle_graph(_add_one_edge(nx.cycle_graph(n)))

    @pytest.mark.parametrize("n", range(3, 10))
    def test_remove_edge(self, n):
        for H in _remove_one_edge(nx.cycle_graph(n)):
            assert not nx.is_cycle_graph(H)

    def test_disjoint_cycles(self):
        G = nx.disjoint_union(nx.cycle_graph(3), nx.cycle_graph(4))
        assert not nx.is_cycle_graph(G)


class TestStarGraph:
    @pytest.mark.parametrize("n", range(9))
    def test_generator(self, n):
        assert nx.is_star_graph(nx.star_graph(n))

    @pytest.mark.parametrize("n", range(2, 9))
    def test_add_edge(self, n):
        assert not nx.is_star_graph(_add_one_edge(nx.star_graph(n)))

    @pytest.mark.parametrize("n", range(1, 9))
    def test_remove_edge(self, n):
        for H in _remove_one_edge(nx.star_graph(n)):
            assert not nx.is_star_graph(H)

    def test_non_stars(self):
        assert not nx.is_star_graph(nx.path_graph(4))
        assert nx.is_star_graph(nx.path_graph(3))


class TestWheelGraph:
    @pytest.mark.parametrize("n", range(4, 11))
    def test_generator(self, n):
        assert nx.is_wheel_graph(nx.wheel_graph(n))

    @pytest.mark.parametrize("n", range(1, 4))
    def test_degenerate(self, n):
        assert not nx.is_wheel_graph(nx.wheel_graph(n))

    @pytest.mark.parametrize("n", range(5, 11))
    def test_add_edge(self, n):
        assert not nx.is_wheel_graph(_add_one_edge(nx.wheel_graph(n)))

    @pytest.mark.parametrize("n", range(4, 11))
    def test_remove_edge(self, n):
        for H in _remove_one_edge(nx.wheel_graph(n)):
            assert not nx.is_wheel_graph(H)

    def test_hub_with_disconnected_rim(self):
        # hub joined to two disjoint triangles: right edge count, wrong rim
        G = nx.disjoint_union(nx.cycle_graph(3), nx.cycle_graph(3))
        G.add_edges_from(("hub", v) for v in list(G))
        # 7 nodes, 6 + 6 = 12 edges == 2 * (7 - 1)
        assert not nx.is_wheel_graph(G)


class TestCompleteGraph:
    @pytest.mark.parametrize("n", range(1, 9))
    def test_generator(self, n):
        assert nx.is_complete_graph(nx.complete_graph(n))

    @pytest.mark.parametrize("n", range(2, 9))
    def test_remove_edge(self, n):
        for H in _remove_one_edge(nx.complete_graph(n)):
            assert not nx.is_complete_graph(H)

    def test_add_node(self):
        G = nx.complete_graph(4)
        G.add_node(10)
        assert not nx.is_complete_graph(G)

    def test_complete_is_already_saturated(self):
        assert _add_one_edge(nx.complete_graph(5)) is None


class TestCompleteBipartiteGraph:
    @pytest.mark.parametrize("m, n", list(itertools.product(range(5), range(1, 5))))
    def test_generator(self, m, n):
        assert nx.is_complete_bipartite_graph(nx.complete_bipartite_graph(m, n))

    @pytest.mark.parametrize("n", range(7))
    def test_stars_are_complete_bipartite(self, n):
        assert nx.is_complete_bipartite_graph(nx.star_graph(n))

    @pytest.mark.parametrize("m, n", list(itertools.product(range(1, 5), range(1, 5))))
    def test_add_edge(self, m, n):
        H = _add_one_edge(nx.complete_bipartite_graph(m, n))
        if H is not None:  # K_{1,1} is already complete
            assert not nx.is_complete_bipartite_graph(H)

    @pytest.mark.parametrize("m, n", list(itertools.product(range(2, 5), range(2, 5))))
    def test_remove_edge(self, m, n):
        for H in _remove_one_edge(nx.complete_bipartite_graph(m, n)):
            assert not nx.is_complete_bipartite_graph(H)

    def test_remove_edge_from_star(self):
        # removing a star edge leaves an isolated node: not K_{m,n}
        for H in _remove_one_edge(nx.star_graph(3)):
            assert not nx.is_complete_bipartite_graph(H)

    def test_non_bipartite(self):
        assert not nx.is_complete_bipartite_graph(nx.cycle_graph(5))
        assert not nx.is_complete_bipartite_graph(nx.complete_graph(3))
        assert nx.is_complete_bipartite_graph(nx.cycle_graph(4))  # K_{2,2}

    def test_matching_degrees_but_wrong_structure(self):
        # 6-cycle: 2-regular with 6 edges; a node's neighborhood has size 2,
        # so |A|=4, |B|=2 and a*b == 8 != 6; also check C_6 + chords variants
        assert not nx.is_complete_bipartite_graph(nx.cycle_graph(6))
        # Two disjoint K_{2,2}: 8 nodes, 8 edges; |B|=2, a*b=12
        G = nx.disjoint_union(nx.cycle_graph(4), nx.cycle_graph(4))
        assert not nx.is_complete_bipartite_graph(G)
