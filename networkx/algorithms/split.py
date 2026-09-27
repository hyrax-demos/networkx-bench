"""Recognition of split graphs."""

import networkx as nx
from networkx.utils.decorators import not_implemented_for

__all__ = ["is_split_graph"]


@nx._dispatchable
@not_implemented_for("directed")
@not_implemented_for("multigraph")
def is_split_graph(G):
    r"""Returns True if `G` is a split graph, else False.

    A graph is a *split graph* if its node set can be partitioned into two
    sets $K$ and $I$ such that $K$ induces a clique and $I$ induces an
    independent set. Either part may be empty.

    Parameters
    ----------
    G : NetworkX graph
        An undirected simple graph.

    Returns
    -------
    bool
        True if `G` is a split graph, False otherwise.

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
    degrees = sorted((d for _, d in G.degree()), reverse=True)
    m = 0
    for i, d in enumerate(degrees, start=1):
        if d >= i - 1:
            m = i
        else:
            break
    return sum(degrees[:m]) == m * (m - 1) + sum(degrees[m:])
