"""Recognition of split graphs."""

import networkx as nx
from networkx.utils.decorators import not_implemented_for

__all__ = ["is_split_graph"]


@nx._dispatchable
@not_implemented_for("directed")
@not_implemented_for("multigraph")
def is_split_graph(G, *, certificate=False):
    r"""Returns True if `G` is a split graph, else False.

    A graph is a *split graph* if its node set can be partitioned into two
    sets $K$ and $I$ such that $K$ induces a clique and $I$ induces an
    independent set. Either part may be empty.

    Parameters
    ----------
    G : NetworkX graph
        An undirected simple graph.

    certificate : bool, optional (default=False)
        If True, also return a partition of the nodes witnessing that `G`
        is a split graph.

    Returns
    -------
    bool or tuple
        If `certificate` is False, True if `G` is a split graph and False
        otherwise. If `certificate` is True, a pair ``(is_split, partition)``
        where ``partition`` is a tuple ``(clique, independent_set)`` of node
        sets if `G` is a split graph, and ``(False, None)`` otherwise. The
        two sets are disjoint, cover all nodes of `G`, `clique` induces a
        complete subgraph and `independent_set` induces an edgeless one.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is directed or a multigraph.

    Examples
    --------
    >>> G = nx.star_graph(4)
    >>> nx.is_split_graph(G)
    True
    >>> nx.is_split_graph(nx.cycle_graph(4))
    False

    With ``certificate=True`` the partition is returned as well:

    >>> is_split, (clique, independent_set) = nx.is_split_graph(G, certificate=True)
    >>> is_split
    True
    >>> clique  # doctest: +SKIP
    {0, 1}
    >>> sorted(clique | independent_set)
    [0, 1, 2, 3, 4]
    >>> nx.is_split_graph(nx.cycle_graph(4), certificate=True)
    (False, None)

    Notes
    -----
    Uses the degree-sequence characterization of Hammer and Simeone [1]_.
    Let $d_1 \ge d_2 \ge \dots \ge d_n$ be the degree sequence of `G` and let
    $m = \max\{i : d_i \ge i - 1\}$. Then `G` is a split graph if and only if

    .. math::

        \sum_{i=1}^{m} d_i = m(m - 1) + \sum_{i=m+1}^{n} d_i.

    The running time is $O(n \log n + m)$ for sorting the degrees. Self-loops
    are counted in the degree as usual in NetworkX; graphs with self-loops
    are not simple and the result for them is not meaningful.

    The null graph (no nodes) is considered a split graph.

    See Also
    --------
    is_chordal, is_perfect_graph

    References
    ----------
    .. [1] P. L. Hammer and B. Simeone, "The splittance of a graph",
       Combinatorica 1 (1981), 275--284. https://doi.org/10.1007/BF02579333
    """
    partition = _split_partition(G)
    if certificate:
        return (partition is not None, partition)
    return partition is not None


def _split_partition(G):
    """Return ``(clique, independent_set)`` if `G` is split, else None."""
    # Hammer-Simeone (1981) characterization: with degrees sorted
    # d_1 >= ... >= d_n and m = max{i : d_i >= i - 1}, G is split iff
    # sum_{i<=m} d_i == m(m-1) + sum_{i>m} d_i. The first m vertices form
    # the clique K and the rest the independent set I. See P. L. Hammer and
    # B. Simeone, "The splittance of a graph", Combinatorica 1 (1981) 275-284.
    ordered = sorted(G.degree(), key=lambda nd: nd[1], reverse=True)
    degrees = [d for _, d in ordered]
    m = 0
    for i, d in enumerate(degrees, start=1):
        if d >= i - 1:
            m = i
        else:
            break
    if sum(degrees[:m]) != m * (m - 1) + sum(degrees[m:]):
        return None
    clique = {v for v, _ in ordered[:m]}
    independent_set = {v for v, _ in ordered[m:]}
    return clique, independent_set
