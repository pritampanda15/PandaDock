Algorithms Overview
===================

PandaDock v3.0 provides a streamlined set of docking algorithms optimized
for accuracy and reliability.

Primary Algorithm
-----------------

**Hierarchical Docking (``pandadock``)**

The main docking algorithm uses a multi-stage hierarchical search:

1. **Coarse Sampling**: Fast grid-based sampling at multiple resolutions
2. **Refinement**: Progressive refinement of promising regions
3. **Scoring**: Energy evaluation with Vina-style scoring
4. **Ranking**: Final ranking with detailed scoring

This algorithm achieves:

* Pearson R = 0.12 on ULVSH benchmark
* Best traditional docking performance
* Consistent pose generation

Usage:

.. code-block:: bash

   pandadock dock -r protein.pdb -l ligand.sdf \
                  --center 10 20 30 --box 20 20 20

Scoring Functions
-----------------

**Vina Scoring (Default)**

Implements AutoDock Vina-style scoring with:

* Gaussian steric interactions (gauss1, gauss2)
* Repulsion for steric clashes
* Hydrophobic interactions
* Hydrogen bonding
* Rotatable bond flexibility penalty

.. code-block:: bash

   pandadock dock -r protein.pdb -l ligand.sdf \
                  --scoring vina ...

**Physics-Based Scoring**

Alternative scoring with:

* Van der Waals (Lennard-Jones)
* Electrostatics (Coulomb)
* Hydrogen bonding
* Solvation effects

.. code-block:: bash

   pandadock dock -r protein.pdb -l ligand.sdf \
                  --scoring physics_based ...

ML Scoring (Recommended)
------------------------

**PandaDock-GNN**

For best accuracy, use the SE(3)-equivariant GNN scoring function:

* Pearson R = 0.67 on ULVSH benchmark
* **5x better** than traditional scoring
* Outperforms all baseline methods

Use via hybrid docking:

.. code-block:: bash

   pandadock hybrid -r protein.pdb -l ligand.sdf \
                    --center 10 20 30 --box 20 20 20 \
                    -m model.pt

See :doc:`../gnn/overview` for details.

Specialized Modes
-----------------

**Flexible Docking** (``pandadock-flex``)

Induced-fit docking with receptor flexibility:

.. code-block:: bash

   pandadock-flex -r protein.pdb -l ligand.sdf \
                  --center 10 20 30 --radius 15

**Metal Docking** (``pandadock-metal``)

Specialized for metalloproteins (Zn, Fe, Mg, Ca, etc.):

.. code-block:: bash

   pandadock-metal dock -r metalloprotein.pdb -l ligand.sdf ...

**Tethered Docking** (``pandadock-tethered``)

Constrained docking for validation:

.. code-block:: bash

   pandadock-tethered dock -r protein.pdb -l ligand.sdf \
                           --reference crystal_ligand.pdb

Algorithm Selection Guide
-------------------------

+---------------------------+-----------------------------------+
| Use Case                  | Recommended Approach              |
+===========================+===================================+
| **Best accuracy**         | ``pandadock hybrid`` with GNN     |
+---------------------------+-----------------------------------+
| **Quick docking**         | ``pandadock dock --fast``         |
+---------------------------+-----------------------------------+
| **Virtual screening**     | ``pandadock hybrid`` (batch)      |
+---------------------------+-----------------------------------+
| **Flexible receptor**     | ``pandadock-flex``                |
+---------------------------+-----------------------------------+
| **Metal coordination**    | ``pandadock-metal``               |
+---------------------------+-----------------------------------+
| **Pose validation**       | ``pandadock-tethered``            |
+---------------------------+-----------------------------------+

Performance Comparison
----------------------

On ULVSH benchmark (942 compounds):

+---------------------------+-----------------+
| Method                    | Pearson R       |
+===========================+=================+
| **PandaDock-GNN (Hybrid)**| **0.67** ⭐     |
+---------------------------+-----------------+
| VM2                       | 0.15            |
+---------------------------+-----------------+
| PandaDock (Hierarchical)  | 0.12            |
+---------------------------+-----------------+
| PM6                       | 0.08            |
+---------------------------+-----------------+
| Gnina, MMPBSA, etc.       | < 0.02          |
+---------------------------+-----------------+
