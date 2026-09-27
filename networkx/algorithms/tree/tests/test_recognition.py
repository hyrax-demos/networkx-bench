import pytest

import networkx as nx


class TestTreeRecognition:
    graph = nx.Graph
    multigraph = nx.MultiGraph

    @classmethod
    def setup_class(cls):
        cls.T1 = cls.graph()

        cls.T2 = cls.graph()
        cls.T2.add_node(1)

        cls.T3 = cls.graph()
        cls.T3.add_nodes_from(range(5))
        edges = [(i, i + 1) for i in range(4)]
        cls.T3.add_edges_from(edges)

        cls.T5 = cls.multigraph()
        cls.T5.add_nodes_from(range(5))
        edges = [(i, i + 1) for i in range(4)]
        cls.T5.add_edges_from(edges)

        cls.T6 = cls.graph()
        cls.T6.add_nodes_from([6, 7])
        cls.T6.add_edge(6, 7)

        cls.F1 = nx.compose(cls.T6, cls.T3)

        cls.N4 = cls.graph()
        cls.N4.add_node(1)
        cls.N4.add_edge(1, 1)

        cls.N5 = cls.graph()
        cls.N5.add_nodes_from(range(5))

        cls.N6 = cls.graph()
        cls.N6.add_nodes_from(range(3))
        cls.N6.add_edges_from([(0, 1), (1, 2), (2, 0)])

        cls.NF1 = nx.compose(cls.T6, cls.N6)

    def test_null_tree(self):
        with pytest.raises(nx.NetworkXPointlessConcept):
            nx.is_tree(self.graph())

    def test_null_tree2(self):
        with pytest.raises(nx.NetworkXPointlessConcept):
            nx.is_tree(self.multigraph())

    def test_null_forest(self):
        with pytest.raises(nx.NetworkXPointlessConcept):
            nx.is_forest(self.graph())

    def test_null_forest2(self):
        with pytest.raises(nx.NetworkXPointlessConcept):
            nx.is_forest(self.multigraph())

    def test_is_tree(self):
        assert nx.is_tree(self.T2)
        assert nx.is_tree(self.T3)
        assert nx.is_tree(self.T5)

    def test_is_not_tree(self):
        assert not nx.is_tree(self.N4)
        assert not nx.is_tree(self.N5)
        assert not nx.is_tree(self.N6)

    def test_is_forest(self):
        assert nx.is_forest(self.T2)
        assert nx.is_forest(self.T3)
        assert nx.is_forest(self.T5)
        assert nx.is_forest(self.F1)
        assert nx.is_forest(self.N5)

    def test_is_not_forest(self):
        assert not nx.is_forest(self.N4)
        assert not nx.is_forest(self.N6)
        assert not nx.is_forest(self.NF1)


class TestDirectedTreeRecognition(TestTreeRecognition):
    graph = nx.DiGraph
    multigraph = nx.MultiDiGraph


def test_disconnected_graph():
    # https://github.com/networkx/networkx/issues/1144
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 4)])
    assert not nx.is_tree(G)

    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 4)])
    assert not nx.is_tree(G)


def test_dag_nontree():
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (0, 2), (1, 2)])
    assert not nx.is_tree(G)
    assert nx.is_directed_acyclic_graph(G)


def test_multicycle():
    G = nx.MultiDiGraph()
    G.add_edges_from([(0, 1), (0, 1)])
    assert not nx.is_tree(G)
    assert nx.is_directed_acyclic_graph(G)


def test_emptybranch():
    G = nx.DiGraph()
    G.add_nodes_from(range(10))
    assert nx.is_branching(G)
    assert not nx.is_arborescence(G)


def test_is_branching_empty_graph_raises():
    G = nx.DiGraph()
    with pytest.raises(nx.NetworkXPointlessConcept, match="G has no nodes."):
        nx.is_branching(G)


def test_path():
    G = nx.DiGraph()
    nx.add_path(G, range(5))
    assert nx.is_branching(G)
    assert nx.is_arborescence(G)


def test_notbranching1():
    # Acyclic violation.
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(10))
    G.add_edges_from([(0, 1), (1, 0)])
    assert not nx.is_branching(G)
    assert not nx.is_arborescence(G)


def test_notbranching2():
    # In-degree violation.
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(10))
    G.add_edges_from([(0, 1), (0, 2), (3, 2)])
    assert not nx.is_branching(G)
    assert not nx.is_arborescence(G)


def test_notarborescence1():
    # Not an arborescence due to not spanning.
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(10))
    G.add_edges_from([(0, 1), (0, 2), (1, 3), (5, 6)])
    assert nx.is_branching(G)
    assert not nx.is_arborescence(G)


def test_notarborescence2():
    # Not an arborescence due to in-degree violation.
    G = nx.MultiDiGraph()
    nx.add_path(G, range(5))
    G.add_edge(6, 4)
    assert not nx.is_branching(G)
    assert not nx.is_arborescence(G)


def test_is_arborescense_empty_graph_raises():
    G = nx.DiGraph()
    with pytest.raises(nx.NetworkXPointlessConcept, match="G has no nodes."):
        nx.is_arborescence(G)


def _spider(leg_length, legs=3):
    G = nx.Graph()
    G.add_node(0)
    node = 1
    for _ in range(legs):
        prev = 0
        for _ in range(leg_length):
            G.add_edge(prev, node)
            prev = node
            node += 1
    return G


class TestIsCaterpillar:
    def test_null_graph_raises(self):
        with pytest.raises(nx.NetworkXPointlessConcept, match="G has no nodes."):
            nx.is_caterpillar(nx.Graph())

    def test_directed_raises(self):
        with pytest.raises(nx.NetworkXNotImplemented):
            nx.is_caterpillar(nx.DiGraph([(0, 1)]))

    def test_top_level_namespace(self):
        assert nx.is_caterpillar is nx.algorithms.tree.recognition.is_caterpillar

    def test_single_node(self):
        G = nx.Graph()
        G.add_node(0)
        assert nx.is_caterpillar(G)

    def test_single_edge(self):
        assert nx.is_caterpillar(nx.complete_graph(2))

    @pytest.mark.parametrize("n", range(1, 8))
    def test_path(self, n):
        assert nx.is_caterpillar(nx.path_graph(n))

    @pytest.mark.parametrize("n", range(1, 8))
    def test_star(self, n):
        assert nx.is_caterpillar(nx.star_graph(n))

    def test_hand_built_caterpillar(self):
        G = nx.path_graph(5)  # spine 0-1-2-3-4
        leaf = 5
        for spine_node, count in [(0, 2), (1, 1), (2, 3), (4, 2)]:
            for _ in range(count):
                G.add_edge(spine_node, leaf)
                leaf += 1
        assert nx.is_caterpillar(G)

    def test_spider_legs_length_two(self):
        assert not nx.is_caterpillar(_spider(2))

    def test_spider_legs_length_one_is_star(self):
        assert nx.is_caterpillar(_spider(1))

    def test_cycle(self):
        assert not nx.is_caterpillar(nx.cycle_graph(4))

    def test_disconnected_forest(self):
        G = nx.Graph([(0, 1), (2, 3)])
        assert not nx.is_caterpillar(G)

    def test_multigraph_parallel_edge(self):
        G = nx.MultiGraph([(0, 1), (0, 1)])
        assert not nx.is_caterpillar(G)

    def test_multigraph_tree(self):
        assert nx.is_caterpillar(nx.MultiGraph(nx.path_graph(4)))

    def test_input_not_mutated(self):
        G = _spider(2)
        nodes, edges = set(G), set(G.edges())
        nx.is_caterpillar(G)
        H = nx.star_graph(3)
        nx.is_caterpillar(H)
        assert set(G) == nodes
        assert set(G.edges()) == edges
        assert set(H) == set(range(4))
        assert H.number_of_edges() == 3


class TestIsLobster:
    def test_null_graph_raises(self):
        with pytest.raises(nx.NetworkXPointlessConcept, match="G has no nodes."):
            nx.is_lobster(nx.Graph())

    def test_directed_raises(self):
        with pytest.raises(nx.NetworkXNotImplemented):
            nx.is_lobster(nx.DiGraph([(0, 1)]))

    def test_top_level_namespace(self):
        assert nx.is_lobster is nx.algorithms.tree.recognition.is_lobster

    def test_single_node(self):
        G = nx.Graph()
        G.add_node(0)
        assert nx.is_lobster(G)

    def test_single_edge(self):
        assert nx.is_lobster(nx.complete_graph(2))

    @pytest.mark.parametrize("n", range(1, 8))
    def test_path(self, n):
        assert nx.is_lobster(nx.path_graph(n))

    @pytest.mark.parametrize("n", range(1, 8))
    def test_star(self, n):
        assert nx.is_lobster(nx.star_graph(n))

    def test_caterpillar(self):
        G = nx.path_graph(5)
        G.add_edges_from([(0, 5), (2, 6), (2, 7), (4, 8)])
        assert nx.is_caterpillar(G)
        assert nx.is_lobster(G)

    def test_hand_built_lobster_not_caterpillar(self):
        # Spine 0-1-2-3; node 1 has two pendant paths of length 2 and
        # node 2 has one, plus some extra leaves.
        G = nx.path_graph(4)
        nx.add_path(G, [1, 10, 11])
        nx.add_path(G, [1, 12, 13])
        nx.add_path(G, [2, 14, 15])
        G.add_edges_from([(0, 20), (3, 21), (10, 22)])
        assert not nx.is_caterpillar(G)
        assert nx.is_lobster(G)

    def test_spider_legs_length_two(self):
        assert nx.is_lobster(_spider(2))

    def test_spider_legs_length_three(self):
        assert not nx.is_lobster(_spider(3))

    def test_cycle(self):
        assert not nx.is_lobster(nx.cycle_graph(4))

    def test_disconnected_forest(self):
        G = nx.Graph([(0, 1), (2, 3)])
        assert not nx.is_lobster(G)

    def test_multigraph_parallel_edge(self):
        assert not nx.is_lobster(nx.MultiGraph([(0, 1), (0, 1)]))

    def test_input_not_mutated(self):
        G = _spider(3)
        nodes, edges = set(G), set(G.edges())
        nx.is_lobster(G)
        assert set(G) == nodes
        assert set(G.edges()) == edges

    def test_caterpillar_implies_lobster(self):
        for seed in range(20):
            for n in (1, 2, 5, 10, 20):
                G = nx.random_labeled_tree(n, seed=seed)
                if nx.is_caterpillar(G):
                    assert nx.is_lobster(G)
