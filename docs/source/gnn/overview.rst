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

**ULVSH Dataset (942 compounds, 10 protein targets):**

+------------------+-------------+
| Metric           | Value       |
+==================+=============+
| Activity AUC     | 0.94        |
+------------------+-------------+
| Spearman ρ       | 0.80        |
+------------------+-------------+
| RMSE             | 0.32        |
+------------------+-------------+
| MAE              | 0.12        |
+------------------+-------------+

**BindingDB Dataset (8,891 protein-ligand complexes):**

+---------------------------+----------------+-----------+
| Training Configuration    | Test Pearson R | Test RMSE |
+===========================+================+===========+
| BindingDB Only            | 0.81           | -         |
+---------------------------+----------------+-----------+
| BindingDB + ULVSH         | 0.79           | 0.96      |
+---------------------------+----------------+-----------+

**PDBbind v2020 (5,316 complexes):**

+------------------+-------------+
| Metric           | Value       |
+==================+=============+
| Pearson R        | 0.88        |
+------------------+-------------+
| Spearman ρ       | 0.88        |
+------------------+-------------+
| RMSE             | 0.93 pK     |
+------------------+-------------+

PandaDock-GNN **outperforms all baseline methods** including:
VM2, MMPBSA, MMGBSA, Gnina, Hyde, DeltaVina, GFN-FF, and PM6.

Usage
-----

**Training on ULVSH:**

.. code-block:: bash

   pandadock gnn train -d ULVSH/ -o models/ --epochs 100

**Training on BindingDB:**

.. code-block:: bash

   python BindingDB_training/train_bindingdb.py \
       --bindingdb BindingDB_training/bindingdb_affinity.tsv \
       --output models/ --epochs 100

**Combined Training (BindingDB + ULVSH):**

.. code-block:: bash

   python BindingDB_training/train_bindingdb.py \
       --bindingdb BindingDB_training/bindingdb_affinity.tsv \
       --ulvsh ULVSH/ --combined \
       --output models/ --epochs 100

**Prediction:**

.. code-block:: bash

   pandadock gnn predict -m model.pt -p protein.mol2 -l ligand.mol2

**Hybrid Docking (Recommended):**

.. code-block:: bash

   pandadock hybrid -r protein.pdb -l ligand.sdf \
                    --center 10 20 30 --box 20 20 20 \
                    -m model.pt

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
