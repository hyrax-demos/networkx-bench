from functools import partial

import pytest

import networkx as nx


class TestBFS:
    @classmethod
    def setup_class(cls):
        # simple graph
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (1, 3), (2, 4), (3, 4)])
        cls.G = G

    def test_successor(self):
        assert dict(nx.bfs_successors(self.G, source=0)) == {0: [1], 1: [2, 3], 2: [4]}

    def test_predecessor(self):
        with pytest.deprecated_call():
            assert dict(nx.bfs_predecessors(self.G, source=0)) == {
                1: 0,
                2: 1,
                3: 1,
                4: 2,
            }

    def test_bfs_tree(self):
        T = nx.bfs_tree(self.G, source=0)
        assert sorted(T.nodes()) == sorted(self.G.nodes())
        assert sorted(T.edges()) == [(0, 1), (1, 2), (1, 3), (2, 4)]

    def test_bfs_edges(self):
        edges = nx.bfs_edges(self.G, source=0)
        assert list(edges) == [(0, 1), (1, 2), (1, 3), (2, 4)]

    def test_bfs_edges_reverse(self):
        D = nx.DiGraph()
        D.add_edges_from([(0, 1), (1, 2), (1, 3), (2, 4), (3, 4)])
        edges = nx.bfs_edges(D, source=4, reverse=True)
        assert list(edges) == [(4, 2), (4, 3), (2, 1), (1, 0)]

    def test_bfs_edges_sorting(self):
        D = nx.DiGraph()
        D.add_edges_from([(0, 1), (0, 2), (1, 4), (1, 3), (2, 5)])
        sort_desc = partial(sorted, reverse=True)
        edges_asc = nx.bfs_edges(D, source=0, sort_neighbors=sorted)
        edges_desc = nx.bfs_edges(D, source=0, sort_neighbors=sort_desc)
        assert list(edges_asc) == [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5)]
        assert list(edges_desc) == [(0, 2), (0, 1), (2, 5), (1, 4), (1, 3)]

    def test_bfs_tree_isolates(self):
        G = nx.Graph()
        G.add_node(1)
        G.add_node(2)
        T = nx.bfs_tree(G, source=1)
        assert sorted(T.nodes()) == [1]
        assert sorted(T.edges()) == []

    def test_bfs_layers(self):
        expected = {
            0: [0],
            1: [1],
            2: [2, 3],
            3: [4],
        }
        for sources in [0, [0], (i for i in [0]), [0, 0]]:
            assert dict(enumerate(nx.bfs_layers(self.G, sources))) == expected

    def test_bfs_layers_missing_source(self):
        with pytest.raises(nx.NetworkXError):
            next(nx.bfs_layers(self.G, sources="abc"))
        with pytest.raises(nx.NetworkXError):
            next(nx.bfs_layers(self.G, sources=["abc"]))

    def test_descendants_at_distance(self):
        for distance, descendants in enumerate([{0}, {1}, {2, 3}, {4}]):
            assert nx.descendants_at_distance(self.G, 0, distance) == descendants

    def test_descendants_at_distance_missing_source(self):
        with pytest.raises(nx.NetworkXError):
            nx.descendants_at_distance(self.G, "abc", 0)

    def test_bfs_labeled_edges_directed(self):
        D = nx.cycle_graph(5, create_using=nx.DiGraph)
        expected = [
            (0, 1, "tree"),
            (1, 2, "tree"),
            (2, 3, "tree"),
            (3, 4, "tree"),
            (4, 0, "reverse"),
        ]
        answer = list(nx.bfs_labeled_edges(D, 0))
        assert expected == answer

        D.add_edge(4, 4)
        expected.append((4, 4, "level"))
        answer = list(nx.bfs_labeled_edges(D, 0))
        assert expected == answer

        D.add_edge(0, 2)
        D.add_edge(1, 5)
        D.add_edge(2, 5)
        D.remove_edge(4, 4)
        expected = [
            (0, 1, "tree"),
            (0, 2, "tree"),
            (1, 2, "level"),
            (1, 5, "tree"),
            (2, 3, "tree"),
            (2, 5, "forward"),
            (3, 4, "tree"),
            (4, 0, "reverse"),
        ]
        answer = list(nx.bfs_labeled_edges(D, 0))
        assert expected == answer

        G = D.to_undirected()
        G.add_edge(4, 4)
        expected = [
            (0, 1, "tree"),
            (0, 2, "tree"),
            (0, 4, "tree"),
            (1, 2, "level"),
            (1, 5, "tree"),
            (2, 3, "tree"),
            (2, 5, "forward"),
            (4, 3, "forward"),
            (4, 4, "level"),
        ]
        answer = list(nx.bfs_labeled_edges(G, 0))
        assert expected == answer


class TestBreadthLimitedSearch:
    @classmethod
    def setup_class(cls):
        # a tree
        G = nx.Graph()
        nx.add_path(G, [0, 1, 2, 3, 4, 5, 6])
        nx.add_path(G, [2, 7, 8, 9, 10])
        cls.G = G
        # a disconnected graph
        D = nx.Graph()
        D.add_edges_from([(0, 1), (2, 3)])
        nx.add_path(D, [2, 7, 8, 9, 10])
        cls.D = D

    def test_limited_bfs_successor(self):
        assert dict(nx.bfs_successors(self.G, source=1, depth_limit=3)) == {
            1: [0, 2],
            2: [3, 7],
            3: [4],
            7: [8],
        }
        result = {
            n: sorted(s) for n, s in nx.bfs_successors(self.D, source=7, depth_limit=2)
        }
        assert result == {8: [9], 2: [3], 7: [2, 8]}

    def test_limited_bfs_predecessor(self):
        with pytest.deprecated_call():
            assert dict(nx.bfs_predecessors(self.G, source=1, depth_limit=3)) == {
                0: 1,
                2: 1,
                3: 2,
                4: 3,
                7: 2,
                8: 7,
            }
        with pytest.deprecated_call():
            assert dict(nx.bfs_predecessors(self.D, source=7, depth_limit=2)) == {
                2: 7,
                3: 2,
                8: 7,
                9: 8,
            }

    def test_limited_bfs_tree(self):
        T = nx.bfs_tree(self.G, source=3, depth_limit=1)
        assert sorted(T.edges()) == [(3, 2), (3, 4)]

    def test_limited_bfs_edges(self):
        edges = nx.bfs_edges(self.G, source=9, depth_limit=4)
        assert list(edges) == [(9, 8), (9, 10), (8, 7), (7, 2), (2, 1), (2, 3)]

    def test_limited_bfs_layers(self):
        assert dict(enumerate(nx.bfs_layers(self.G, sources=[0]))) == {
            0: [0],
            1: [1],
            2: [2],
            3: [3, 7],
            4: [4, 8],
            5: [5, 9],
            6: [6, 10],
        }
        assert dict(enumerate(nx.bfs_layers(self.D, sources=2))) == {
            0: [2],
            1: [3, 7],
            2: [8],
            3: [9],
            4: [10],
        }

    def test_limited_descendants_at_distance(self):
        for distance, descendants in enumerate(
            [{0}, {1}, {2}, {3, 7}, {4, 8}, {5, 9}, {6, 10}]
        ):
            assert nx.descendants_at_distance(self.G, 0, distance) == descendants
        for distance, descendants in enumerate([{2}, {3, 7}, {8}, {9}, {10}]):
            assert nx.descendants_at_distance(self.D, 2, distance) == descendants


class TestKHopNeighbors:
    def test_path_graph(self):
        G = nx.path_graph(7)
        assert nx.k_hop_neighbors(G, 3, 1) == {2, 4}
        assert nx.k_hop_neighbors(G, 3, 2) == {1, 2, 4, 5}
        assert nx.k_hop_neighbors(G, 0, 2, include_source=True) == {0, 1, 2}
        assert nx.k_hop_neighbors(G, 3, 10) == {0, 1, 2, 4, 5, 6}

    def test_directed_follows_out_edges(self):
        D = nx.DiGraph([(0, 1), (1, 2), (2, 3), (4, 0)])
        assert nx.k_hop_neighbors(D, 0, 2) == {1, 2}
        assert nx.k_hop_neighbors(D, 3, 5) == set()
        assert nx.k_hop_neighbors(D, 4, 1, include_source=True) == {4, 0}

    def test_cycle_does_not_include_source_by_default(self):
        G = nx.cycle_graph(4)
        assert nx.k_hop_neighbors(G, 0, 4) == {1, 2, 3}
        assert nx.k_hop_neighbors(G, 0, 4, include_source=True) == {0, 1, 2, 3}

    def test_k_zero(self):
        G = nx.path_graph(3)
        assert nx.k_hop_neighbors(G, 1, 0) == set()
        assert nx.k_hop_neighbors(G, 1, 0, include_source=True) == {1}

    def test_source_not_in_graph(self):
        G = nx.path_graph(3)
        with pytest.raises(nx.NetworkXError):
            nx.k_hop_neighbors(G, 99, 1)

    def test_negative_k(self):
        G = nx.path_graph(3)
        with pytest.raises(ValueError):
            nx.k_hop_neighbors(G, 0, -1)


class TestKHopNeighborsSources:
    def test_union_of_disjoint_sources(self):
        G = nx.path_graph(7)
        assert nx.k_hop_neighbors(G, k=1, sources=[1, 5]) == {0, 2, 4, 6}

    def test_overlapping_sources(self):
        G = nx.path_graph(7)
        # Each source is within 1 hop of the other, so both appear even
        # though include_source is False.
        assert nx.k_hop_neighbors(G, k=1, sources=[2, 3]) == {1, 2, 3, 4}
        assert nx.k_hop_neighbors(G, k=2, sources=[1, 3]) == {0, 1, 2, 3, 4, 5}
        # Duplicates in sources are harmless.
        assert nx.k_hop_neighbors(G, k=1, sources=[3, 3]) == {2, 4}
        expected = set()
        for s in [0, 2, 3]:
            expected |= nx.k_hop_neighbors(G, s, 2)
        assert nx.k_hop_neighbors(G, k=2, sources=iter([0, 2, 3])) == expected

    def test_include_source(self):
        G = nx.path_graph(7)
        assert nx.k_hop_neighbors(G, k=1, sources=[0, 6], include_source=True) == {
            0,
            1,
            5,
            6,
        }

    def test_directed(self):
        D = nx.DiGraph([(0, 1), (1, 2), (2, 3), (4, 0)])
        assert nx.k_hop_neighbors(D, k=1, sources=[0, 4]) == {0, 1}
        assert nx.k_hop_neighbors(D, k=5, sources=[3]) == set()
        assert nx.k_hop_neighbors(D, k=1, sources=[3, 2]) == {3}

    def test_k_zero(self):
        G = nx.path_graph(4)
        assert nx.k_hop_neighbors(G, k=0, sources=[0, 2]) == set()
        assert nx.k_hop_neighbors(G, k=0, sources=[0, 2], include_source=True) == {
            0,
            2,
        }

    def test_empty_sources(self):
        G = nx.path_graph(3)
        assert nx.k_hop_neighbors(G, k=2, sources=[]) == set()

    def test_invalid_source_in_sources(self):
        G = nx.path_graph(3)
        with pytest.raises(nx.NetworkXError, match="99"):
            nx.k_hop_neighbors(G, k=1, sources=[0, 99])

    def test_source_and_sources_exclusive(self):
        G = nx.path_graph(3)
        with pytest.raises(nx.NetworkXError):
            nx.k_hop_neighbors(G, 0, 1, sources=[1])
        with pytest.raises(nx.NetworkXError):
            nx.k_hop_neighbors(G, k=1)

    def test_negative_k(self):
        G = nx.path_graph(3)
        with pytest.raises(ValueError):
            nx.k_hop_neighbors(G, k=-1, sources=[0])


class TestAllKHopNeighbors:
    def test_matches_single_source(self):
        G = nx.petersen_graph()
        for k in range(4):
            for inc in (False, True):
                result = nx.all_k_hop_neighbors(G, k, include_source=inc)
                assert set(result) == set(G)
                for n in G:
                    assert result[n] == nx.k_hop_neighbors(G, n, k, include_source=inc)

    def test_path_graph(self):
        G = nx.path_graph(4)
        assert nx.all_k_hop_neighbors(G, 1) == {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2}}

    def test_directed(self):
        D = nx.DiGraph([(0, 1), (1, 2), (2, 3), (4, 0)])
        assert nx.all_k_hop_neighbors(D, 2) == {
            0: {1, 2},
            1: {2, 3},
            2: {3},
            3: set(),
            4: {0, 1},
        }

    def test_cycle_excludes_self_by_default(self):
        G = nx.cycle_graph(3)
        assert nx.all_k_hop_neighbors(G, 3) == {0: {1, 2}, 1: {0, 2}, 2: {0, 1}}

    def test_k_zero(self):
        G = nx.path_graph(3)
        assert nx.all_k_hop_neighbors(G, 0) == {0: set(), 1: set(), 2: set()}
        assert nx.all_k_hop_neighbors(G, 0, include_source=True) == {
            0: {0},
            1: {1},
            2: {2},
        }

    def test_empty_graph(self):
        assert nx.all_k_hop_neighbors(nx.Graph(), 2) == {}

    def test_negative_k(self):
        with pytest.raises(ValueError):
            nx.all_k_hop_neighbors(nx.path_graph(3), -1)
