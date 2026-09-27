.. _tree:

Tree
====

.. toctree::
   :maxdepth: 2

Recognition
-----------
.. automodule:: networkx.algorithms.tree.recognition
.. autosummary::
   :toctree: generated/

   is_tree
   is_forest
   is_arborescence
   is_branching
   is_spider
   spider_legs

A spider is a tree with at most one node of degree three or more.
:func:`is_spider` can also report that branch node, and :func:`spider_legs`
gives the leg lengths in descending order:

>>> import networkx as nx
>>> G = nx.star_graph(3)
>>> nx.add_path(G, [1, 4, 5])
>>> nx.add_path(G, [2, 6])
>>> nx.is_spider(G, center=True)
(True, 0)
>>> nx.is_spider(nx.path_graph(4), center=True)
(True, None)
>>> nx.spider_legs(G)
[3, 2, 1]
>>> nx.spider_legs(nx.cycle_graph(4))
Traceback (most recent call last):
    ...
networkx.exception.NetworkXError: G is not a spider.

Branchings and Spanning Arborescences
-------------------------------------
.. automodule:: networkx.algorithms.tree.branchings
.. autosummary::
   :toctree: generated/

   branching_weight
   greedy_branching
   maximum_branching
   minimum_branching
   maximum_spanning_arborescence
   minimum_spanning_arborescence
   ArborescenceIterator

Distance Measures
-----------------
.. automodule:: networkx.algorithms.tree.distance_measures
.. autosummary::
   :toctree: generated/

   center
   centroid

Encoding and decoding
---------------------
.. automodule:: networkx.algorithms.tree.coding
.. autosummary::
   :toctree: generated/

   from_nested_tuple
   to_nested_tuple
   from_prufer_sequence
   to_prufer_sequence

Operations
----------
.. automodule:: networkx.algorithms.tree.operations
.. autosummary::
   :toctree: generated/

   join_trees

Spanning Trees
--------------
.. automodule:: networkx.algorithms.tree.mst
.. autosummary::
   :toctree: generated/

   minimum_spanning_tree
   maximum_spanning_tree
   random_spanning_tree
   minimum_spanning_edges
   maximum_spanning_edges
   SpanningTreeIterator
   number_of_spanning_trees

Decomposition
-------------
.. automodule:: networkx.algorithms.tree.decomposition
.. autosummary::
   :toctree: generated/

   junction_tree

Exceptions
----------
.. automodule:: networkx.algorithms.tree.coding
   :noindex:
.. autosummary::
   :toctree: generated/

   NotATree
