PandaDock-GNN Overview
======================

PandaDock-GNN is an SE(3)-equivariant Graph Neural Network scoring function
for protein-ligand binding affinity prediction.

Key Features
------------

**SE(3)-Equivariance**
   The model produces identical predictions regardless of the rotation or
   translation of the input complex. This is achieved through E(n)-equivariant
   graph neural network (EGNN) layers.

**Heterogeneous Graph Representation**
   Protein and ligand atoms are represented as separate node types in a
   heterogeneous graph, allowing the model to learn distinct representations
   for each.

**Multi-Task Learning**
   The model jointly predicts:

   - **pEC50**: Binding affinity (regression)
   - **Activity**: Binary classification (active/inactive)

Architecture
------------

.. code-block:: text

   Input: Protein-Ligand Complex
          │
          ├─ Protein Atoms → Node Features (56 dims)
          ├─ Ligand Atoms  → Node Features (56 dims)
          └─ Interactions  → Edge Features (23 dims)
          │
          ├─ Node Encoders (separate for protein/ligand)
          │
          ├─ EGNN Layers × 6 (SE(3)-equivariant message passing)
          │   - Update node features
          │   - Update coordinates (equivariant)
          │
          ├─ Attention Pooling
          │   - Protein graph → embedding
          │   - Ligand graph  → embedding
          │
          └─ Prediction Heads
              ├─ Affinity → pEC50
              └─ Activity → probability

Node Features (56 dimensions)
-----------------------------

* Element type one-hot (10 dims)
* SYBYL atom type one-hot (16 dims)
* Partial charge (1 dim)
* Hybridization one-hot (4 dims)
* Aromaticity flag (1 dim)
* H-bond donor/acceptor (2 dims)
* Ring membership (1 dim)
* Residue type one-hot (20 dims, protein only)
* Backbone flag (1 dim, protein only)

Edge Features (23 dimensions)
-----------------------------

* Distance (1 dim)
* Gaussian RBF distance encoding (16 dims)
* Bond type one-hot (4 dims)
* Interaction type flags (2 dims)

Benchmark Performance
---------------------

.. note::
   Earlier v4.0 figures (R = 0.88 on PDBbind, R = 0.82 on ULVSH) were withdrawn
   in August 2026. All values below reflect full-dataset evaluations with fixed
   prediction pipelines.

**SAIR Released Model (741,706 co-folded complexes, target-disjoint splits):**

+------------------------------------+--------+-----------+
| Evaluation                         | N      | Pearson R |
+====================================+========+===========+
| Held-out SAIR test split           | 90,219 | 0.407     |
+------------------------------------+--------+-----------+
| Experimental crystal structures    | 202    | 0.467     |
+------------------------------------+--------+-----------+
| PDBbind v2020 refined (indep.)     | 4,640  | 0.531     |
+------------------------------------+--------+-----------+

**PDBbind v2020 Refined Set (n = 4,640, native crystal poses):**

+----------------------------------------------+-----------+------------+
| Model                                        | Pearson R | Spearman ρ |
+==============================================+===========+============+
| SAIR (independent)                           | 0.531     | 0.523      |
+----------------------------------------------+-----------+------------+
| PDBbind-only (own held-out test, n = 624)    | 0.690     | 0.674      |
+----------------------------------------------+-----------+------------+
| BindingDB (independent)                      | 0.277     | 0.281      |
+----------------------------------------------+-----------+------------+

.. warning::
   **Do not use PandaDock-GNN to select docked poses.** Benchmarked against
   the empirical scoring function on 50 complexes (20 poses each), GNN
   rescoring degrades median RMSD from 2.09 A to 5.36 A. Use ``pandadock dock``
   to select poses and ``pandadock-gnn predict`` to estimate affinity of a
   chosen pose.

Usage
-----

**Training on SAIR:**

.. code-block:: bash

   pandadock-gnn train-sair --cache sair_cache/ -o models/sair/ \
                            --rank-weight 1.0 --target-grouped

**Prediction:**

.. code-block:: bash

   pandadock-gnn predict -m models/pandadock_gnn_sair.pt \
                         -p protein.pdb -l ligand.mol2 --strip-hydrogens

References
----------

The EGNN architecture is based on:

   Satorras, V. G., Hoogeboom, E., & Welling, M. (2021).
   E(n) Equivariant Graph Neural Networks.
   *International Conference on Machine Learning (ICML)*.

.. note::

   ULVSH affinity labels are censored: inactive compounds take a floor
   value, and 89 of the 95 test-split pEC50 values are exactly 4.0. A
   Pearson correlation over that distribution reflects separation of
   actives from the floor rather than affinity ranking, so ULVSH is
   reported as activity classification (AUC = 0.94). Use PDBbind or
   BindingDB for continuous affinity performance.
