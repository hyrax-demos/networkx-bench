"""Recognition of claw-free graphs."""

from itertools import combinations

import networkx as nx
from networkx.utils.decorators import not_implemented_for

__all__ = ["is_claw_free"]


@nx._dispatchable
@not_implemented_for("directed")
@not_implemented_for("multigraph")
def is_claw_free(G):
    r"""Returns True if `G` is claw-free, else False.

    A *claw* is the complete bipartite graph $K_{1,3}$: a center node joined
    to three leaves that are pairwise non-adjacent. A graph is *claw-free*
    if it has no induced subgraph isomorphic to a claw, i.e. no node has
    three pairwise non-adjacent neighbors.

    Parameters
    ----------
    G : NetworkX graph
        An undirected simple graph.

    Returns
    -------
    bool
        True if `G` is claw-free, False otherwise.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is directed or a multigraph.

    Examples
    --------
    >>> nx.is_claw_free(nx.line_graph(nx.petersen_graph()))
    True
    >>> nx.is_claw_free(nx.star_graph(3))
    False

    Notes
    -----
    For every node, the algorithm searches its neighborhood for an
    independent set of size three. The worst-case running time is
    $O(\sum_v \deg(v)^3)$. Self-loops are ignored.

    Every line graph is claw-free [1]_. The null graph is claw-free.

    See Also
    --------
    line_graph, is_perfect_graph

    References
    ----------
    .. [1] R. Faudree, E. Flandrin, and Z. Ryjáček, "Claw-free graphs — A
       survey", Discrete Mathematics 164 (1997), 87--147.
       https://doi.org/10.1016/S0012-365X(96)00045-3
    """
    adj = G._adj
    for v, nbrdict in adj.items():
        nbrs = [u for u in nbrdict if u != v]
        if len(nbrs) < 3:
            continue
        for a, b in combinations(nbrs, 2):
            if b in adj[a]:
                continue
            adj_a = adj[a]
            adj_b = adj[b]
            for c in nbrs:
                if c != a and c != b and c not in adj_a and c not in adj_b:
                    return False
    return True
