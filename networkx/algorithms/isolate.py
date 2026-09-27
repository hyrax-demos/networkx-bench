"""
Functions for identifying isolate (degree zero) nodes.
"""

import networkx as nx

__all__ = [
    "is_isolate",
    "isolates",
    "number_of_isolates",
    "number_of_pendant_nodes",
    "pendant_nodes",
]


@nx._dispatchable
def is_isolate(G, n):
    """Determines whether a node is an isolate.

    An *isolate* is a node with no neighbors (that is, with degree
    zero). For directed graphs, this means no in-neighbors and no
    out-neighbors.

    Parameters
    ----------
    G : NetworkX graph

    n : node
        A node in `G`.

    Returns
    -------
    is_isolate : bool
       True if and only if `n` has no neighbors.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_edge(1, 2)
    >>> G.add_node(3)
    >>> nx.is_isolate(G, 2)
    False
    >>> nx.is_isolate(G, 3)
    True
    """
    return G.degree(n) == 0


@nx._dispatchable
def isolates(G):
    """Iterator over isolates in the graph.

    An *isolate* is a node with no neighbors (that is, with degree
    zero). For directed graphs, this means no in-neighbors and no
    out-neighbors.

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    iterator
        An iterator over the isolates of `G`.

    Examples
    --------
    To get a list of all isolates of a graph, use the :class:`list`
    constructor:

    >>> G = nx.Graph()
    >>> G.add_edge(1, 2)
    >>> G.add_node(3)
    >>> list(nx.isolates(G))
    [3]

    To remove all isolates in the graph, first create a list of the
    isolates, then use :meth:`Graph.remove_nodes_from`:

    >>> G.remove_nodes_from(list(nx.isolates(G)))
    >>> list(G)
    [1, 2]

    For digraphs, isolates have zero in-degree and zero out_degree:

    >>> G = nx.DiGraph([(0, 1), (1, 2)])
    >>> G.add_node(3)
    >>> list(nx.isolates(G))
    [3]

    """
    return (n for n, d in G.degree() if d == 0)


@nx._dispatchable
def number_of_isolates(G):
    """Returns the number of isolates in the graph.

    An *isolate* is a node with no neighbors (that is, with degree
    zero). For directed graphs, this means no in-neighbors and no
    out-neighbors.

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    int
        The number of degree zero nodes in the graph `G`.

    """
    return sum(1 for v in isolates(G))


@nx._dispatchable
def pendant_nodes(G):
    """Iterator over pendant nodes in the graph.

    A *pendant node* (also called a *leaf*) is a node with degree exactly
    one. For directed graphs, the total degree (in-degree plus out-degree)
    is used, so a pendant node has exactly one incident edge in either
    direction. A self-loop contributes two to the degree of its node, so
    a node whose only edge is a self-loop is not a pendant node.

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    iterator
        An iterator over the pendant nodes of `G`.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> list(nx.pendant_nodes(G))
    [0, 3]

    For digraphs, the total degree is used:

    >>> G = nx.DiGraph([(0, 1), (1, 2), (3, 1)])
    >>> sorted(nx.pendant_nodes(G))
    [0, 2, 3]

    See Also
    --------
    number_of_pendant_nodes
    isolates
    """
    return (n for n, d in G.degree() if d == 1)


@nx._dispatchable
def number_of_pendant_nodes(G):
    """Returns the number of pendant nodes in the graph.

    A *pendant node* (also called a *leaf*) is a node with degree exactly
    one. For directed graphs, the total degree (in-degree plus out-degree)
    is used.

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    int
        The number of degree one nodes in the graph `G`.

    Examples
    --------
    >>> G = nx.star_graph(3)
    >>> nx.number_of_pendant_nodes(G)
    3

    See Also
    --------
    pendant_nodes
    number_of_isolates
    """
    return sum(1 for v in pendant_nodes(G))
