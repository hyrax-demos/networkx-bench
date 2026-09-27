"""Unit tests for the :mod:`networkx.algorithms.isolates` module."""

import networkx as nx


def test_is_isolate():
    G = nx.Graph()
    G.add_edge(0, 1)
    G.add_node(2)
    assert not nx.is_isolate(G, 0)
    assert not nx.is_isolate(G, 1)
    assert nx.is_isolate(G, 2)


def test_isolates():
    G = nx.Graph()
    G.add_edge(0, 1)
    G.add_nodes_from([2, 3])
    assert sorted(nx.isolates(G)) == [2, 3]


def test_number_of_isolates():
    G = nx.Graph()
    G.add_edge(0, 1)
    G.add_nodes_from([2, 3])
    assert nx.number_of_isolates(G) == 2


def test_pendant_nodes():
    G = nx.path_graph(4)
    G.add_node(4)
    assert sorted(nx.pendant_nodes(G)) == [0, 3]


def test_pendant_nodes_empty():
    G = nx.Graph()
    assert list(nx.pendant_nodes(G)) == []
    assert nx.number_of_pendant_nodes(G) == 0


def test_pendant_nodes_single_edge():
    G = nx.Graph([(0, 1)])
    assert sorted(nx.pendant_nodes(G)) == [0, 1]


def test_pendant_nodes_self_loop():
    # A self-loop adds 2 to the degree, so node 0 is not pendant.
    G = nx.Graph([(0, 0)])
    assert list(nx.pendant_nodes(G)) == []
    G.add_edge(1, 2)
    G.add_edge(2, 2)
    assert list(nx.pendant_nodes(G)) == [1]


def test_pendant_nodes_directed_total_degree():
    G = nx.DiGraph([(0, 1), (1, 2), (3, 1)])
    # Node 0 has out-degree 1, node 2 in-degree 1, node 3 out-degree 1.
    assert sorted(nx.pendant_nodes(G)) == [0, 2, 3]
    # Reciprocal edges give total degree 2: not pendant.
    H = nx.DiGraph([(0, 1), (1, 0)])
    assert list(nx.pendant_nodes(H)) == []


def test_pendant_nodes_multigraph():
    G = nx.MultiGraph([(0, 1), (0, 1), (1, 2)])
    assert list(nx.pendant_nodes(G)) == [2]


def test_number_of_pendant_nodes():
    G = nx.star_graph(5)
    assert nx.number_of_pendant_nodes(G) == 5
    G = nx.DiGraph([(0, 1), (1, 2), (3, 1)])
    assert nx.number_of_pendant_nodes(G) == 3
