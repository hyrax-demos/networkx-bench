"""Recognizers for small, well-known families of graphs.

Each function in this module answers whether a graph is isomorphic to a
member of a named graph family (paths, cycles, stars, wheels, complete
graphs and complete bipartite graphs). They correspond to the generators
:func:`~networkx.generators.classic.path_graph`,
:func:`~networkx.generators.classic.cycle_graph`,
:func:`~networkx.generators.classic.star_graph`,
:func:`~networkx.generators.classic.wheel_graph`,
:func:`~networkx.generators.classic.complete_graph` and
:func:`~networkx.algorithms.bipartite.generators.complete_bipartite_graph`.

All recognizers work on undirected simple graphs only. Directed graphs and
multigraphs raise :exc:`~networkx.NetworkXNotImplemented`, the null graph
(no nodes) raises :exc:`~networkx.NetworkXPointlessConcept`, and a graph
containing a self-loop is never a member of any of these families.
"""

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = [
    "is_complete_bipartite_graph",
    "is_complete_graph",
    "is_cycle_graph",
    "is_path_graph",
    "is_star_graph",
    "is_wheel_graph",
]


def _check_non_null(G):
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept("Graph has no nodes.")


def _has_selfloop(G):
    return any(u in nbrs for u, nbrs in G.adj.items())


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def is_path_graph(G):
    """Returns True if `G` is a path graph.

    A path graph on $n$ nodes is a connected graph with $n - 1$ edges in
    which every node has degree at most two. The graph with a single node
    is the path on one node.

    Parameters
    ----------
    G : NetworkX graph
        An undirected simple graph.

    Returns
    -------
    bool
        True if `G` is isomorphic to ``nx.path_graph(len(G))``.

    Raises
    ------
    NetworkXPointlessConcept
        If `G` is the null graph.
    NetworkXNotImplemented
        If `G` is directed or a multigraph.

    Examples
    --------
    >>> nx.is_path_graph(nx.path_graph(5))
    True
    >>> nx.is_path_graph(nx.cycle_graph(5))
    False

    See Also
    --------
    path_graph
    is_cycle_graph
    """
    _check_non_null(G)
    n = len(G)
    if G.number_of_edges() != n - 1 or _has_selfloop(G):
        return False
    if any(d > 2 for _, d in G.degree):
        return False
    return nx.is_connected(G)


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def is_cycle_graph(G):
    """Returns True if `G` is a cycle graph.

    A cycle graph on $n \\geq 3$ nodes is a connected graph in which every
    node has degree exactly two. Graphs with fewer than three nodes are not
    simple cycles, so this function returns False for them.

    Parameters
    ----------
    G : NetworkX graph
        An undirected simple graph.

    Returns
    -------
    bool
        True if `G` is isomorphic to ``nx.cycle_graph(len(G))`` and has at
        least three nodes.

    Raises
    ------
    NetworkXPointlessConcept
        If `G` is the null graph.
    NetworkXNotImplemented
        If `G` is directed or a multigraph.

    Examples
    --------
    >>> nx.is_cycle_graph(nx.cycle_graph(6))
    True
    >>> nx.is_cycle_graph(nx.path_graph(6))
    False

    See Also
    --------
    cycle_graph
    is_path_graph
    """
    _check_non_null(G)
    n = len(G)
    if n < 3 or G.number_of_edges() != n or _has_selfloop(G):
        return False
    if any(d != 2 for _, d in G.degree):
        return False
    return nx.is_connected(G)


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def is_star_graph(G):
    """Returns True if `G` is a star graph.

    A star graph on $n$ nodes consists of one center node adjacent to every
    other node and no other edges; it is the complete bipartite graph
    $K_{1, n-1}$. The graph with a single node (``nx.star_graph(0)``) and
    the single edge ``nx.star_graph(1)`` are both stars.

    Parameters
    ----------
    G : NetworkX graph
        An undirected simple graph.

    Returns
    -------
    bool
        True if `G` is isomorphic to ``nx.star_graph(len(G) - 1)``.

    Raises
    ------
    NetworkXPointlessConcept
        If `G` is the null graph.
    NetworkXNotImplemented
        If `G` is directed or a multigraph.

    Examples
    --------
    >>> nx.is_star_graph(nx.star_graph(4))
    True
    >>> nx.is_star_graph(nx.path_graph(4))
    False

    See Also
    --------
    star_graph
    is_complete_bipartite_graph
    """
    _check_non_null(G)
    n = len(G)
    if G.number_of_edges() != n - 1 or _has_selfloop(G):
        return False
    # n - 1 edges and a node adjacent to all others means no other edges.
    return any(d == n - 1 for _, d in G.degree)


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def is_wheel_graph(G):
    """Returns True if `G` is a wheel graph.

    A wheel graph on $n \\geq 4$ nodes consists of a hub node adjacent to
    every other node, with the remaining $n - 1$ nodes forming a cycle.
    The smallest wheel is the complete graph $K_4$. Graphs with fewer than
    four nodes return False.

    Parameters
    ----------
    G : NetworkX graph
        An undirected simple graph.

    Returns
    -------
    bool
        True if `G` is isomorphic to ``nx.wheel_graph(len(G))`` and has at
        least four nodes.

    Raises
    ------
    NetworkXPointlessConcept
        If `G` is the null graph.
    NetworkXNotImplemented
        If `G` is directed or a multigraph.

    Examples
    --------
    >>> nx.is_wheel_graph(nx.wheel_graph(6))
    True
    >>> nx.is_wheel_graph(nx.cycle_graph(6))
    False

    See Also
    --------
    wheel_graph
    is_cycle_graph
    """
    _check_non_null(G)
    n = len(G)
    if n < 4 or G.number_of_edges() != 2 * (n - 1) or _has_selfloop(G):
        return False
    hub = next((v for v, d in G.degree if d == n - 1), None)
    if hub is None:
        return False
    rim = G.subgraph(v for v in G if v != hub)
    return is_cycle_graph(rim)


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def is_complete_graph(G):
    """Returns True if `G` is a complete graph.

    A complete graph has an edge between every pair of distinct nodes.
    The graph with a single node is complete.

    Parameters
    ----------
    G : NetworkX graph
        An undirected simple graph.

    Returns
    -------
    bool
        True if `G` is isomorphic to ``nx.complete_graph(len(G))``.

    Raises
    ------
    NetworkXPointlessConcept
        If `G` is the null graph.
    NetworkXNotImplemented
        If `G` is directed or a multigraph.

    Examples
    --------
    >>> nx.is_complete_graph(nx.complete_graph(5))
    True
    >>> G = nx.complete_graph(5)
    >>> G.remove_edge(0, 1)
    >>> nx.is_complete_graph(G)
    False

    See Also
    --------
    complete_graph
    """
    _check_non_null(G)
    n = len(G)
    if _has_selfloop(G):
        return False
    return G.number_of_edges() == n * (n - 1) // 2


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def is_complete_bipartite_graph(G):
    """Returns True if `G` is a complete bipartite graph.

    A complete bipartite graph $K_{m, n}$ has its nodes partitioned into two
    sets of sizes $m$ and $n$ with an edge between every pair of nodes from
    different sets and no edge inside a set. Stars $K_{1, n}$ are complete
    bipartite. An edgeless graph is $K_{0, n}$ (one side empty), matching
    ``nx.complete_bipartite_graph(0, n)``, and is also accepted.

    Parameters
    ----------
    G : NetworkX graph
        An undirected simple graph.

    Returns
    -------
    bool
        True if `G` is isomorphic to ``nx.complete_bipartite_graph(m, n)``
        for some $m, n \\geq 0$ with $m + n = |V(G)|$.

    Raises
    ------
    NetworkXPointlessConcept
        If `G` is the null graph.
    NetworkXNotImplemented
        If `G` is directed or a multigraph.

    Examples
    --------
    >>> nx.is_complete_bipartite_graph(nx.complete_bipartite_graph(2, 3))
    True
    >>> nx.is_complete_bipartite_graph(nx.star_graph(4))
    True
    >>> nx.is_complete_bipartite_graph(nx.cycle_graph(5))
    False

    See Also
    --------
    complete_bipartite_graph
    is_star_graph
    """
    _check_non_null(G)
    if _has_selfloop(G):
        return False
    # Any node u must lie on one side A; its neighbors are then exactly the
    # other side B.
    u = nx.utils.arbitrary_element(G)
    B = set(G[u])
    a, b = len(G) - len(B), len(B)
    if G.number_of_edges() != a * b:
        return False
    # Every node of A must be adjacent to exactly B, and every node of B must
    # be adjacent to exactly A (i.e. |A| neighbors, none of them in B).
    for v, nbrs in G.adj.items():
        if v in B:
            if len(nbrs) != a or any(w in B for w in nbrs):
                return False
        elif nbrs.keys() != B:
            return False
    return True
