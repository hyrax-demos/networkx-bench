"""
Recognition Tests
=================

A *forest* is an acyclic, undirected graph, and a *tree* is a connected forest.
Depending on the subfield, there are various conventions for generalizing these
definitions to directed graphs.

In one convention, directed variants of forest and tree are defined in an
identical manner, except that the direction of the edges is ignored. In effect,
each directed edge is treated as a single undirected edge. Then, additional
restrictions are imposed to define *branchings* and *arborescences*.

In another convention, directed variants of forest and tree correspond to
the previous convention's branchings and arborescences, respectively. Then two
new terms, *polyforest* and *polytree*, are defined to correspond to the other
convention's forest and tree.

Summarizing::

   +-----------------------------+
   | Convention A | Convention B |
   +=============================+
   | forest       | polyforest   |
   | tree         | polytree     |
   | branching    | forest       |
   | arborescence | tree         |
   +-----------------------------+

Each convention has its reasons. The first convention emphasizes definitional
similarity in that directed forests and trees are only concerned with
acyclicity and do not have an in-degree constraint, just as their undirected
counterparts do not. The second convention emphasizes functional similarity
in the sense that the directed analog of a spanning tree is a spanning
arborescence. That is, take any spanning tree and choose one node as the root.
Then every edge is assigned a direction such there is a directed path from the
root to every other node. The result is a spanning arborescence.

NetworkX follows convention "A". Explicitly, these are:

undirected forest
   An undirected graph with no undirected cycles.

undirected tree
   A connected, undirected forest.

directed forest
   A directed graph with no undirected cycles. Equivalently, the underlying
   graph structure (which ignores edge orientations) is an undirected forest.
   In convention B, this is known as a polyforest.

directed tree
   A weakly connected, directed forest. Equivalently, the underlying graph
   structure (which ignores edge orientations) is an undirected tree. In
   convention B, this is known as a polytree.

branching
   A directed forest with each node having, at most, one parent. So the maximum
   in-degree is equal to 1. In convention B, this is known as a forest.

arborescence
   A directed tree with each node having, at most, one parent. So the maximum
   in-degree is equal to 1. In convention B, this is known as a tree.

For trees and arborescences, the adjective "spanning" may be added to designate
that the graph, when considered as a forest/branching, consists of a single
tree/arborescence that includes all nodes in the graph. It is true, by
definition, that every tree/arborescence is spanning with respect to the nodes
that define the tree/arborescence and so, it might seem redundant to introduce
the notion of "spanning". However, the nodes may represent a subset of
nodes from a larger graph, and it is in this context that the term "spanning"
becomes a useful notion.

"""

import networkx as nx

__all__ = [
    "is_arborescence",
    "is_branching",
    "is_caterpillar",
    "is_forest",
    "is_lobster",
    "is_tree",
]


@nx.utils.not_implemented_for("undirected")
@nx._dispatchable
def is_arborescence(G):
    """
    Returns True if `G` is an arborescence.

    An arborescence is a directed tree with maximum in-degree equal to 1.

    Parameters
    ----------
    G : graph
        The graph to test.

    Returns
    -------
    b : bool
        A boolean that is True if `G` is an arborescence.

    Examples
    --------
    >>> G = nx.DiGraph([(0, 1), (0, 2), (2, 3), (3, 4)])
    >>> nx.is_arborescence(G)
    True
    >>> G.remove_edge(0, 1)
    >>> G.add_edge(1, 2)  # maximum in-degree is 2
    >>> nx.is_arborescence(G)
    False

    Notes
    -----
    In another convention, an arborescence is known as a *tree*.

    See Also
    --------
    is_tree

    """
    return is_tree(G) and max(d for n, d in G.in_degree()) <= 1


@nx.utils.not_implemented_for("undirected")
@nx._dispatchable
def is_branching(G):
    """
    Returns True if `G` is a branching.

    A branching is a directed forest with maximum in-degree equal to 1.

    Parameters
    ----------
    G : directed graph
        The directed graph to test.

    Returns
    -------
    b : bool
        A boolean that is True if `G` is a branching.

    Examples
    --------
    >>> G = nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 4)])
    >>> nx.is_branching(G)
    True
    >>> G.remove_edge(2, 3)
    >>> G.add_edge(3, 1)  # maximum in-degree is 2
    >>> nx.is_branching(G)
    False

    Notes
    -----
    In another convention, a branching is also known as a *forest*.

    See Also
    --------
    is_forest

    """
    return is_forest(G) and max(d for n, d in G.in_degree()) <= 1


@nx._dispatchable
def is_forest(G):
    """
    Returns True if `G` is a forest.

    A forest is a graph with no undirected cycles.

    For directed graphs, `G` is a forest if the underlying graph is a forest.
    The underlying graph is obtained by treating each directed edge as a single
    undirected edge in a multigraph.

    Parameters
    ----------
    G : graph
        The graph to test.

    Returns
    -------
    b : bool
        A boolean that is True if `G` is a forest.

    Raises
    ------
    NetworkXPointlessConcept
        If `G` is empty.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_edges_from([(1, 2), (1, 3), (2, 4), (2, 5)])
    >>> nx.is_forest(G)
    True
    >>> G.add_edge(4, 1)
    >>> nx.is_forest(G)
    False

    Notes
    -----
    In another convention, a directed forest is known as a *polyforest* and
    then *forest* corresponds to a *branching*.

    See Also
    --------
    is_branching

    """
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept("G has no nodes.")

    if G.is_directed():
        components = nx.weakly_connected_components(G)
    else:
        components = nx.connected_components(G)

    return all(len(c) - 1 == sum(1 for _ in G.edges(c)) for c in components)


@nx._dispatchable
def is_tree(G):
    """
    Returns True if `G` is a tree.

    A tree is a connected graph with no undirected cycles.

    For directed graphs, `G` is a tree if the underlying graph is a tree. The
    underlying graph is obtained by treating each directed edge as a single
    undirected edge in a multigraph.

    Parameters
    ----------
    G : graph
        The graph to test.

    Returns
    -------
    b : bool
        A boolean that is True if `G` is a tree.

    Raises
    ------
    NetworkXPointlessConcept
        If `G` is empty.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_edges_from([(1, 2), (1, 3), (2, 4), (2, 5)])
    >>> nx.is_tree(G)  # n-1 edges
    True
    >>> G.add_edge(3, 4)
    >>> nx.is_tree(G)  # n edges
    False

    Notes
    -----
    In another convention, a directed tree is known as a *polytree* and then
    *tree* corresponds to an *arborescence*.

    See Also
    --------
    is_arborescence

    """
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept("G has no nodes.")

    if G.is_directed():
        is_connected = nx.is_weakly_connected
    else:
        is_connected = nx.is_connected

    # A connected graph with no cycles has n-1 edges.
    return len(G) - 1 == G.number_of_edges() and is_connected(G)


def _is_caterpillar_tree(T):
    """Returns True if the tree `T` is a caterpillar.

    `T` must already be known to be an undirected tree, or be empty. The
    graph is not modified. An empty `T` is considered a caterpillar so that
    callers can pass the (possibly empty) remainder of a leaf-stripped tree.
    """
    spine = {n for n, d in T.degree() if d > 1}
    # The remainder of a tree after removing its leaves is again a tree, so
    # it is a path (or empty) exactly when every node has degree <= 2 in it.
    return all(sum(1 for nbr in T[n] if nbr in spine) <= 2 for n in spine)


@nx.utils.not_implemented_for("directed")
@nx._dispatchable
def is_caterpillar(G):
    """
    Returns True if `G` is a caterpillar.

    A caterpillar is a tree in which removing all leaves (degree-1 nodes)
    leaves a path or nothing at all.

    Parameters
    ----------
    G : undirected graph
        The graph to test.

    Returns
    -------
    b : bool
        A boolean that is True if `G` is a caterpillar.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is directed.

    NetworkXPointlessConcept
        If `G` is empty.

    See Also
    --------
    is_tree
    is_lobster

    Notes
    -----
    A single node and a single edge are caterpillars, as are all paths and
    stars. Graphs that are not trees, including multigraphs with parallel
    edges, are not caterpillars.

    Examples
    --------
    >>> nx.is_caterpillar(nx.path_graph(5))
    True
    >>> nx.is_caterpillar(nx.star_graph(4))
    True

    A spider with three legs of length 2 is not a caterpillar, because
    removing its leaves leaves a star with three leaves:

    >>> G = nx.Graph([(0, 1), (1, 2), (0, 3), (3, 4), (0, 5), (5, 6)])
    >>> nx.is_caterpillar(G)
    False
    >>> nx.is_caterpillar(nx.cycle_graph(4))  # not a tree
    False

    """
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept("G has no nodes.")

    return nx.is_tree(G) and _is_caterpillar_tree(G)


@nx.utils.not_implemented_for("directed")
@nx._dispatchable
def is_lobster(G):
    """
    Returns True if `G` is a lobster.

    A lobster is a tree in which removing all leaves (degree-1 nodes)
    leaves a caterpillar or nothing at all.

    Parameters
    ----------
    G : undirected graph
        The graph to test.

    Returns
    -------
    b : bool
        A boolean that is True if `G` is a lobster.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is directed.

    NetworkXPointlessConcept
        If `G` is empty.

    See Also
    --------
    is_tree
    is_caterpillar

    Notes
    -----
    Every caterpillar is a lobster. Graphs that are not trees, including
    multigraphs with parallel edges, are not lobsters.

    Examples
    --------
    A spider with three legs of length 2 is a lobster (but not a
    caterpillar), because removing its leaves leaves a star:

    >>> G = nx.Graph([(0, 1), (1, 2), (0, 3), (3, 4), (0, 5), (5, 6)])
    >>> nx.is_lobster(G)
    True

    A spider with three legs of length 3 is not a lobster:

    >>> G = nx.Graph()
    >>> nx.add_path(G, [0, 1, 2, 3])
    >>> nx.add_path(G, [0, 4, 5, 6])
    >>> nx.add_path(G, [0, 7, 8, 9])
    >>> nx.is_lobster(G)
    False

    """
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept("G has no nodes.")

    if not nx.is_tree(G):
        return False

    # Removing the leaves of a tree leaves a (possibly empty) tree; use a
    # read-only subgraph view so that `G` is not modified.
    remainder = G.subgraph([n for n, d in G.degree() if d > 1])
    return len(remainder) == 0 or _is_caterpillar_tree(remainder)
