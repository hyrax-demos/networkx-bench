"""Freeman centralization indices of a graph.

A centralization index summarizes a node-level centrality measure into a
single graph-level score describing how strongly the graph is organized
around its most central node.
"""

import math

import networkx as nx
from networkx.utils.decorators import not_implemented_for

__all__ = [
    "betweenness_centralization",
    "closeness_centralization",
    "degree_centralization",
    "eigenvector_centralization",
]


def _freeman_centralization(centrality, denom):
    """Return ``sum(max - c) / denom`` clamped to a float in [0, 1].

    Parameters
    ----------
    centrality : dict
        Mapping from nodes to node-level centrality values.
    denom : float
        The theoretical maximum of the sum of differences, attained by the
        star graph on the same number of nodes. Must be positive.
    """
    values = list(centrality.values())
    c_max = max(values)
    total = sum(c_max - c for c in values)
    return float(min(1.0, max(0.0, total / denom)))


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def degree_centralization(G):
    r"""Compute the Freeman degree centralization of a graph.

    Degree centralization [1]_ measures how far the degree centralities of
    the nodes deviate from the most central node, relative to the largest
    possible deviation over all graphs with the same number of nodes:

    .. math::

        C_D(G) = \frac{\sum_{v \in G} (c^* - c(v))}{n - 2}

    where `c(v)` is the degree centrality of `v` as computed by
    :func:`~networkx.algorithms.centrality.degree_centrality` (the degree
    divided by `n - 1`), `c^*` is its maximum over all nodes and `n` is the
    number of nodes. The denominator `n - 2` is the value of the numerator
    for the star graph on `n` nodes, the theoretical maximum.

    Parameters
    ----------
    G : graph
        An undirected NetworkX graph without parallel edges.

    Returns
    -------
    centralization : float
        The degree centralization of `G`, a value in [0, 1]. It is 1.0 for
        a star graph and 0.0 for any regular graph (for example a complete
        graph). Graphs with fewer than 3 nodes have centralization 0.0.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is directed or is a multigraph.

    See Also
    --------
    degree_centrality
    closeness_centralization
    betweenness_centralization
    eigenvector_centralization

    Notes
    -----
    Graphs with fewer than 3 nodes have centralization 0.0 by definition,
    since the normalizing denominator would be zero.

    Disconnected graphs are handled normally: degree centrality is well
    defined for every node, and the star graph remains the maximum over all
    simple graphs on `n` nodes, so the result stays in [0, 1].

    Multigraphs are rejected because parallel edges can push degree
    centrality above 1, which would break the [0, 1] guarantee.

    Self-loops are ignored: a self-loop would add 2 to a node's degree
    without connecting it to any other node, so the result is computed on
    `G` with its self-loops removed (`G` itself is not modified).

    References
    ----------
    .. [1] Freeman, L. C. (1978).
       Centrality in social networks conceptual clarification.
       Social Networks 1(3):215-239.
       https://doi.org/10.1016/0378-8733(78)90021-7

    Examples
    --------
    >>> nx.degree_centralization(nx.star_graph(4))
    1.0
    >>> nx.degree_centralization(nx.complete_graph(5))
    0.0
    >>> round(nx.degree_centralization(nx.path_graph(5)), 4)
    0.1667
    """
    n = G.number_of_nodes()
    if n < 3:
        return 0.0
    if nx.number_of_selfloops(G):
        # A self-loop adds 2 to a node's degree, which can push degree
        # centrality past 1 and invalidate the star-based normalization.
        # Ignore self-loops, as closeness and betweenness already do.
        G = nx.restricted_view(G, [], list(nx.selfloop_edges(G)))
    return _freeman_centralization(nx.degree_centrality(G), n - 2)


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def closeness_centralization(G):
    r"""Compute the Freeman closeness centralization of a graph.

    Closeness centralization [1]_ measures how far the closeness
    centralities of the nodes deviate from the most central node, relative
    to the largest possible deviation over all connected graphs with the
    same number of nodes:

    .. math::

        C_C(G) = \frac{\sum_{v \in G} (c^* - c(v))}{(n - 1)(n - 2) / (2n - 3)}

    where `c(v)` is the closeness centrality of `v` as computed by
    :func:`~networkx.algorithms.centrality.closeness_centrality` with its
    default normalization, `c^*` is its maximum over all nodes and `n` is
    the number of nodes. In the star graph on `n` nodes the center has
    closeness 1 and each leaf has `(n - 1) / (2n - 3)`, so the denominator
    `(n - 1)(n - 2) / (2n - 3)` is the theoretical maximum.

    Parameters
    ----------
    G : graph
        An undirected NetworkX graph without parallel edges.

    Returns
    -------
    centralization : float
        The closeness centralization of `G`, a value in [0, 1]. It is 1.0
        for a star graph and 0.0 for a complete graph. Graphs with fewer
        than 3 nodes have centralization 0.0.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is directed or is a multigraph.

    NetworkXError
        If `G` has at least 3 nodes and is not connected.

    See Also
    --------
    closeness_centrality
    degree_centralization
    betweenness_centralization
    eigenvector_centralization

    Notes
    -----
    Graphs with fewer than 3 nodes have centralization 0.0 by definition,
    since the normalizing denominator would be zero. This holds even if
    such a graph is disconnected.

    Disconnected graphs with 3 or more nodes raise
    :exc:`~networkx.NetworkXError`, because the star-based normalization
    assumes all shortest-path distances are finite.

    Edge weights are ignored; distances are hop counts.

    References
    ----------
    .. [1] Freeman, L. C. (1978).
       Centrality in social networks conceptual clarification.
       Social Networks 1(3):215-239.
       https://doi.org/10.1016/0378-8733(78)90021-7

    Examples
    --------
    >>> nx.closeness_centralization(nx.star_graph(4))
    1.0
    >>> nx.closeness_centralization(nx.complete_graph(5))
    0.0
    >>> round(nx.closeness_centralization(nx.path_graph(5)), 4)
    0.4222
    """
    n = G.number_of_nodes()
    if n < 3:
        return 0.0
    if not nx.is_connected(G):
        raise nx.NetworkXError(
            "closeness_centralization is not defined for disconnected graphs"
        )
    denom = (n - 1) * (n - 2) / (2 * n - 3)
    return _freeman_centralization(nx.closeness_centrality(G), denom)


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def betweenness_centralization(G):
    r"""Compute the Freeman betweenness centralization of a graph.

    Betweenness centralization [1]_ measures how far the betweenness
    centralities of the nodes deviate from the most central node, relative
    to the largest possible deviation over all graphs with the same number
    of nodes:

    .. math::

        C_B(G) = \frac{\sum_{v \in G} (c^* - c(v))}{n - 1}

    where `c(v)` is the normalized betweenness centrality of `v` as computed
    by :func:`~networkx.algorithms.centrality.betweenness_centrality` with
    ``normalized=True`` (scaled by `2 / ((n - 1)(n - 2))`), `c^*` is its
    maximum over all nodes and `n` is the number of nodes. In the star graph
    on `n` nodes the center has betweenness 1 and each leaf has 0, so the
    denominator `n - 1` is the theoretical maximum.

    Parameters
    ----------
    G : graph
        An undirected NetworkX graph without parallel edges.

    Returns
    -------
    centralization : float
        The betweenness centralization of `G`, a value in [0, 1]. It is 1.0
        for a star graph and 0.0 for a complete graph. Graphs with fewer
        than 3 nodes have centralization 0.0.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is directed or is a multigraph.

    See Also
    --------
    betweenness_centrality
    degree_centralization
    closeness_centralization
    eigenvector_centralization

    Notes
    -----
    Graphs with fewer than 3 nodes have centralization 0.0 by definition,
    since the normalizing denominator would be zero.

    Disconnected graphs are handled normally: betweenness centrality only
    counts shortest paths between mutually reachable nodes, and the star
    graph remains the maximum over all simple graphs on `n` nodes, so the
    result stays in [0, 1].

    Edge weights are ignored; paths are counted by hop count.

    References
    ----------
    .. [1] Freeman, L. C. (1978).
       Centrality in social networks conceptual clarification.
       Social Networks 1(3):215-239.
       https://doi.org/10.1016/0378-8733(78)90021-7

    Examples
    --------
    >>> nx.betweenness_centralization(nx.star_graph(4))
    1.0
    >>> nx.betweenness_centralization(nx.complete_graph(5))
    0.0
    >>> round(nx.betweenness_centralization(nx.path_graph(5)), 4)
    0.4167
    """
    n = G.number_of_nodes()
    if n < 3:
        return 0.0
    centrality = nx.betweenness_centrality(G, normalized=True)
    return _freeman_centralization(centrality, n - 1)


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def eigenvector_centralization(G):
    r"""Compute the Freeman eigenvector centralization of a graph.

    Eigenvector centralization [1]_ measures how far the eigenvector
    centralities of the nodes deviate from the most central node, relative
    to the largest possible deviation over all connected graphs with the
    same number of nodes:

    .. math::

        C_E(G) = \frac{\sum_{v \in G} (c^* - c(v))}{((n - 1) - \sqrt{n - 1}) / \sqrt{2}}

    where `c(v)` is the eigenvector centrality of `v` as computed by
    :func:`~networkx.algorithms.centrality.eigenvector_centrality` (scaled to
    unit Euclidean norm), `c^*` is its maximum over all nodes and `n` is the
    number of nodes. In the star graph on `n` nodes the center has
    eigenvector centrality `1 / \sqrt{2}` and each leaf has
    `1 / \sqrt{2(n - 1)}`, so the denominator
    `((n - 1) - \sqrt{n - 1}) / \sqrt{2}` is the theoretical maximum.

    Parameters
    ----------
    G : graph
        An undirected NetworkX graph without parallel edges.

    Returns
    -------
    centralization : float
        The eigenvector centralization of `G`, a value in [0, 1]. It is 1.0
        for a star graph and 0.0 for any regular connected graph (for
        example a complete graph). Graphs with fewer than 3 nodes have
        centralization 0.0.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is directed or is a multigraph.

    NetworkXError
        If `G` has at least 3 nodes and is not connected.

    PowerIterationFailedConvergence
        If the underlying power iteration does not converge.

    See Also
    --------
    eigenvector_centrality
    degree_centralization
    closeness_centralization
    betweenness_centralization

    Notes
    -----
    Graphs with fewer than 3 nodes have centralization 0.0 by definition,
    since the normalizing denominator would be zero. This holds even if
    such a graph is disconnected.

    Disconnected graphs with 3 or more nodes raise
    :exc:`~networkx.NetworkXError`, because the leading eigenvector of a
    disconnected graph is not unique and the star-based normalization
    assumes a connected graph.

    Edge weights are ignored. Self-loops are ignored as well: the result is
    computed on `G` with its self-loops removed (`G` itself is not
    modified), so that the star remains the theoretical maximum.

    The eigenvector centralities are computed with a tighter tolerance than
    the :func:`eigenvector_centrality` default so that the result is
    accurate to well below the default :func:`pytest.approx` tolerance.

    References
    ----------
    .. [1] Freeman, L. C. (1978).
       Centrality in social networks conceptual clarification.
       Social Networks 1(3):215-239.
       https://doi.org/10.1016/0378-8733(78)90021-7

    Examples
    --------
    >>> round(nx.eigenvector_centralization(nx.star_graph(4)), 4)
    1.0
    >>> round(nx.eigenvector_centralization(nx.complete_graph(5)), 4)
    0.0
    >>> round(nx.eigenvector_centralization(nx.path_graph(5)), 4)
    0.5176
    """
    n = G.number_of_nodes()
    if n < 3:
        return 0.0
    if not nx.is_connected(G):
        raise nx.NetworkXError(
            "eigenvector_centralization is not defined for disconnected graphs"
        )
    if nx.number_of_selfloops(G):
        G = nx.restricted_view(G, [], list(nx.selfloop_edges(G)))
    centrality = nx.eigenvector_centrality(G, max_iter=1000, tol=1.0e-10)
    denom = ((n - 1) - math.sqrt(n - 1)) / math.sqrt(2)
    return _freeman_centralization(centrality, denom)
