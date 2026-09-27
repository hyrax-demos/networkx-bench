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
    "is_forest",
    "is_spider",
    "is_tree",
    "spider_legs",
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


def _spider_center(G):
    """Return ``(is_spider, center)`` for an undirected graph `G`.

    `center` is the unique node of degree three or more, or None when `G` is
    not a spider or is a path (including the single-node graph).
    Raises NetworkXPointlessConcept if `G` is empty (via `is_tree`).
    """
    if not is_tree(G):
        return False, None
    branch_nodes = (n for n, d in G.degree() if d >= 3)
    center = next(branch_nodes, None)
    # A spider has at most one branch node: the second ``next`` must fail.
    if next(branch_nodes, None) is not None:
        return False, None
    return True, center


@nx.utils.not_implemented_for("directed")
@nx._dispatchable
def is_spider(G, center=False):
    """
    Returns True if `G` is a spider.

    A spider (also called a starlike tree or subdivided star) is a tree with
    at most one node of degree three or more. Equivalently, it is a tree that
    is the union of paths (the "legs") sharing a single common endpoint.

    Parameters
    ----------
    G : undirected graph
        The graph to test.

    center : bool, optional (default=False)
        If True, also return the branch node of the spider.

    Returns
    -------
    b : bool
        A boolean that is True if `G` is a spider. Graphs that are not trees,
        including forests with more than one component, give False.

    (b, c) : tuple
        Returned instead of `b` when `center` is True. `c` is the unique node
        of degree three or more, or None if `G` is a path (including the
        single-node graph) or is not a spider.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is directed.

    NetworkXPointlessConcept
        If `G` is empty (has no nodes).

    Examples
    --------
    >>> G = nx.star_graph(3)
    >>> nx.add_path(G, [1, 4, 5])
    >>> nx.is_spider(G)
    True
    >>> nx.is_spider(G, center=True)
    (True, 0)
    >>> nx.is_spider(nx.path_graph(4), center=True)
    (True, None)
    >>> G.add_edges_from([(4, 6), (4, 7)])  # second node of degree >= 3
    >>> nx.is_spider(G)
    False
    >>> nx.is_spider(G, center=True)
    (False, None)

    A forest with more than one tree is not a spider, even if every
    component is:

    >>> nx.is_spider(nx.disjoint_union(nx.path_graph(3), nx.path_graph(2)))
    False

    Notes
    -----
    Every path graph and every star graph is a spider, as is the single-node
    graph.

    See Also
    --------
    is_tree
    spider_legs

    """
    result = _spider_center(G)
    return result if center else result[0]


@nx.utils.not_implemented_for("directed")
@nx._dispatchable
def spider_legs(G):
    """
    Returns the leg lengths of the spider `G`, sorted in descending order.

    A leg is a path from the branch node (the unique node of degree three or
    more) to a leaf; its length is its number of edges. A path graph has no
    branch node and is treated as a single leg measured from one endpoint,
    so a path on `n` nodes gives ``[n - 1]`` and the single-node graph
    gives ``[]``.

    Parameters
    ----------
    G : undirected graph
        A spider.

    Returns
    -------
    legs : list of int
        The leg lengths, sorted in descending order.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is directed.

    NetworkXPointlessConcept
        If `G` is empty (has no nodes).

    NetworkXError
        If `G` is not a spider.

    Examples
    --------
    >>> G = nx.star_graph(3)
    >>> nx.add_path(G, [1, 4, 5])
    >>> nx.add_path(G, [2, 6])
    >>> nx.spider_legs(G)
    [3, 2, 1]
    >>> nx.spider_legs(nx.star_graph(4))
    [1, 1, 1, 1]
    >>> nx.spider_legs(nx.path_graph(5))
    [4]
    >>> nx.spider_legs(nx.cycle_graph(4))
    Traceback (most recent call last):
        ...
    networkx.exception.NetworkXError: G is not a spider.

    See Also
    --------
    is_spider

    """
    spider, center = _spider_center(G)
    if not spider:
        raise nx.NetworkXError("G is not a spider.")
    if center is None:
        return [len(G) - 1] if len(G) > 1 else []
    legs = []
    for nbr in G[center]:
        prev, node, length = center, nbr, 1
        # Walk outward along the leg until reaching its leaf.
        while G.degree(node) == 2:
            prev, node = node, next(n for n in G[node] if n != prev)
            length += 1
        legs.append(length)
    return sorted(legs, reverse=True)
