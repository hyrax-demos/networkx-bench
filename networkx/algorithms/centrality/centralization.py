"""Freeman centralization indices for graphs.

A centralization index summarizes how strongly a graph is organized around a
single node. It is computed from a node-level centrality measure and
normalized by its theoretical maximum, which is attained by the star graph
on the same number of nodes [1]_.

References
----------
.. [1] Freeman, L. C. (1978).
   Centrality in social networks conceptual clarification.
   Social Networks 1(3):215-239.
   https://doi.org/10.1016/0378-8733(78)90021-7
"""

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = [
    "degree_centralization",
    "closeness_centralization",
    "betweenness_centralization",
]


def _freeman_centralization(centrality, denom):
    """Return ``sum(max - c) / denom`` clamped to a float in [0, 1].

    Parameters
    ----------
    centrality : dict
        Node-level centrality values keyed by node.
    denom : float
        The value of ``sum(max - c)`` for the star graph on the same number
        of nodes (the theoretical maximum). Must be positive.
    """
    values = centrality.values()
    c_max = max(values)
    total = sum(c_max - c for c in values)
    return float(min(1.0, max(0.0, total / denom)))


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def degree_centralization(G):
    r"""Compute the Freeman degree centralization of a graph.

    The degree centralization of an undirected graph $G$ with $n$ nodes is

    .. math::

        C_D(G) = \frac{\sum_{v \in G} (c^* - c(v))}{n - 2}

    where $c(v)$ is the degree centrality of $v$ as returned by
    :func:`degree_centrality` (degree divided by $n - 1$) and
    $c^* = \max_v c(v)$. The denominator $n - 2$ is the value of the
    numerator for the star graph on $n$ nodes, which is the theoretical
    maximum over all simple graphs with $n$ nodes.

    Parameters
    ----------
    G : graph
        An undirected, simple NetworkX graph.

    Returns
    -------
    centralization : float
        The degree centralization of `G`, a value in [0, 1]. Graphs with
        fewer than 3 nodes have centralization 0.0.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is directed or is a multigraph.

    See Also
    --------
    degree_centrality
    closeness_centralization
    betweenness_centralization

    Notes
    -----
    Graphs with fewer than 3 nodes return 0.0 because the normalizing
    denominator vanishes.

    Disconnected graphs are handled normally: degree centrality is well
    defined for every node and the star graph remains the maximum over all
    simple graphs on $n$ nodes, so the result stays in [0, 1].

    The result is clamped to [0, 1] to absorb floating-point drift.

    Multigraphs are rejected because parallel edges can push degree
    centrality above 1, which would break the [0, 1] guarantee.

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
    centrality = nx.degree_centrality(G)
    return _freeman_centralization(centrality, n - 2)


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def closeness_centralization(G):
    r"""Compute the Freeman closeness centralization of a graph.

    The closeness centralization of a connected undirected graph $G$ with
    $n$ nodes is

    .. math::

        C_C(G) = \frac{\sum_{v \in G} (c^* - c(v))}{(n - 1)(n - 2) / (2n - 3)}

    where $c(v)$ is the closeness centrality of $v$ as returned by
    :func:`closeness_centrality` with its default normalization and
    $c^* = \max_v c(v)$. In the star graph on $n$ nodes the center has
    closeness 1 and each leaf has closeness $(n - 1)/(2n - 3)$, so the
    numerator for the star, and hence the denominator, is
    $(n - 1)(n - 2)/(2n - 3)$.

    Parameters
    ----------
    G : graph
        An undirected, simple, connected NetworkX graph.

    Returns
    -------
    centralization : float
        The closeness centralization of `G`, a value in [0, 1]. Graphs with
        fewer than 3 nodes have centralization 0.0.

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

    Notes
    -----
    Graphs with fewer than 3 nodes return 0.0 (even if disconnected)
    because the normalizing denominator vanishes.

    Closeness centralization is not defined for disconnected graphs with 3
    or more nodes, because the star-based normalization assumes all
    distances are finite; a :exc:`NetworkXError` is raised in that case.

    The result is clamped to [0, 1] to absorb floating-point drift.

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
    centrality = nx.closeness_centrality(G)
    return _freeman_centralization(centrality, (n - 1) * (n - 2) / (2 * n - 3))


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def betweenness_centralization(G):
    r"""Compute the Freeman betweenness centralization of a graph.

    The betweenness centralization of an undirected graph $G$ with $n$
    nodes is

    .. math::

        C_B(G) = \frac{\sum_{v \in G} (c^* - c(v))}{n - 1}

    where $c(v)$ is the normalized betweenness centrality of $v$ as returned
    by ``betweenness_centrality(G, normalized=True)`` (scaled by
    $2/((n - 1)(n - 2))$) and $c^* = \max_v c(v)$. In the star graph on $n$
    nodes the center has betweenness 1 and every leaf has 0, so the
    denominator is $n - 1$.

    Parameters
    ----------
    G : graph
        An undirected, simple NetworkX graph.

    Returns
    -------
    centralization : float
        The betweenness centralization of `G`, a value in [0, 1]. Graphs with
        fewer than 3 nodes have centralization 0.0.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is directed or is a multigraph.

    See Also
    --------
    betweenness_centrality
    degree_centralization
    closeness_centralization

    Notes
    -----
    Graphs with fewer than 3 nodes return 0.0 because the normalizing
    denominator vanishes.

    Disconnected graphs are handled normally: betweenness centrality only
    counts shortest paths between mutually reachable nodes, and the star
    graph remains the maximum over all simple graphs on $n$ nodes, so the
    result stays in [0, 1].

    The result is clamped to [0, 1] to absorb floating-point drift.

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
