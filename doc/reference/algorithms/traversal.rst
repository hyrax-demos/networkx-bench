.. _traversal:

Traversal
=========

.. toctree::
   :maxdepth: 2



Depth First Search
------------------
.. automodule:: networkx.algorithms.traversal.depth_first_search
.. autosummary::
   :toctree: generated/

   dfs_edges
   dfs_tree
   dfs_predecessors
   dfs_successors
   dfs_preorder_nodes
   dfs_postorder_nodes
   dfs_labeled_edges

Breadth First Search
--------------------
.. automodule:: networkx.algorithms.traversal.breadth_first_search
.. autosummary::
   :toctree: generated/

   bfs_edges
   bfs_layers
   bfs_tree
   bfs_predecessors
   bfs_successors
   descendants_at_distance
   k_hop_neighbors
   all_k_hop_neighbors
   bfs_labeled_edges
   generic_bfs_edges

K-hop neighborhoods
^^^^^^^^^^^^^^^^^^^
:func:`k_hop_neighbors` returns the nodes within ``k`` hops of a single
``source``, or, with the ``sources`` keyword, the union of the ``k``-hop
neighborhoods of several sources (every source is validated first).
:func:`all_k_hop_neighbors` computes the neighborhood of every node with one
bounded BFS per node. For directed graphs both follow out-edges.

.. doctest::

   >>> import networkx as nx
   >>> G = nx.path_graph(7)
   >>> sorted(nx.k_hop_neighbors(G, 3, 2))
   [1, 2, 4, 5]
   >>> sorted(nx.k_hop_neighbors(G, k=1, sources=[1, 5]))
   [0, 2, 4, 6]
   >>> nx.all_k_hop_neighbors(nx.path_graph(3), 1)
   {0: {1}, 1: {0, 2}, 2: {1}}
   >>> nx.all_k_hop_neighbors(nx.path_graph(3), 0, include_source=True)
   {0: {0}, 1: {1}, 2: {2}}

Beam search
-----------
.. automodule:: networkx.algorithms.traversal.beamsearch
.. autosummary::
   :toctree: generated/

   bfs_beam_edges


Depth First Search on Edges
---------------------------
.. automodule:: networkx.algorithms.traversal.edgedfs
.. autosummary::
   :toctree: generated/

   edge_dfs

Breadth First Search on Edges
-----------------------------
.. automodule:: networkx.algorithms.traversal.edgebfs
.. autosummary::
   :toctree: generated/

   edge_bfs
