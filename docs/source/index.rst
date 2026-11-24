Welcome to PandaDock Documentation
==================================

.. image:: _static/logo.png
   :width: 400px
   :align: center
   :alt: PandaDock Logo

|

PandaDock is an open-source molecular docking suite combining a flexible-ligand conformational search with an **SE(3)-equivariant Graph Neural Network** affinity scoring function trained on 741,706 co-folded complexes (SAIR) under target-disjoint splits.

.. note::
   **PandaDock v4.1+** provides flexible-ligand Monte Carlo + L-BFGS search with analytic gradients (CPU and experimental CUDA/MPS GPU backends), multi-ligand virtual screening (``pandadock screen``), and PandaDock-GNN affinity prediction (Pearson R = 0.531 independent on PDBbind v2020 refined, n = 4,640; R = 0.690 for a dedicated target-disjoint PDBbind model on its held-out test split).

Quick Start
-----------

Install PandaDock:

.. code-block:: bash

   conda install -c conda-forge rdkit
   git clone https://github.com/pritampanda15/PandaDock.git
   cd PandaDock
   pip install -e ".[gnn]"

Basic usage:

.. code-block:: bash

   # Flexible-ligand docking with Vina-style scoring (CPU or --device cuda|mps)
   pandadock dock -r protein.pdb -l ligand.sdf \
                  --center 10 20 30 --box 20 20 20

   # Multi-ligand virtual screening on GPU
   pandadock screen -r protein.pdb -l library.sdf \
                    --center 10 20 30 --box 20 20 20 --device cuda

   # Estimate binding affinity of a chosen pose with PandaDock-GNN
   pandadock-gnn predict -m models/pandadock_gnn_sair.pt \
                         -p protein.pdb -l pose1.mol2 --strip-hydrogens

Key Features
------------

**Flexible-Ligand Conformational Search**
   - Torsion-tree representation over all rotatable bonds
   - Uniform SO(3) orientation sampling and L-BFGS relaxation with closed-form analytic gradients
   - Precomputed affinity grids (AutoDock Vina and Vinardo presets) with signature-keyed caching
   - Experimental batched GPU search (``--device cuda|mps``) and virtual screening (``pandadock screen``)

**SE(3)-Equivariant GNN Affinity Scoring**
   - Rotation- and translation-invariant predictions via 6-layer heterogeneous EGNN
   - Trained on 741,706 SAIR complexes under target-disjoint splits
   - Separation of concerns: use the empirical function to select poses and the GNN to estimate affinity

**Specialized Docking Modes (Experimental)**
   - **Flexible Docking** (``pandadock-flex``): Induced-fit receptor side-chain refinement
   - **Metal Docking** (``pandadock-metal``): Metalloprotein coordination scoring
   - **Tethered Docking** (``pandadock-tethered``): Flat-bottom centroid restraint near a reference pose
   - **Analysis & Reporting** (``pandadock-report``): HTML reports, plots, and PandaMap 2D diagrams

Performance Benchmarks
-----------------------

.. note::
   Earlier v4.0 documentation reported R = 0.88 on PDBbind and R = 0.82 on ULVSH.
   Those numbers were withdrawn in August 2026 (small-subset label-informed rescaling
   and 96.1% censored ULVSH labels). All figures below are full-dataset evaluations.

**Pose Accuracy (814 Complexes, 14 Target Families):**

+---------------------------------+------------------+----------------+
| Backend                         | Top-1 <= 2 A     | Best-of-N <= 2 |
+=================================+==================+================+
| CPU (auto exhaustiveness)       | 33.7%            | 57.0%          |
+---------------------------------+------------------+----------------+
| CUDA (A10G, 512 chains, 814)    | 30.2%            | 50.5%          |
+---------------------------------+------------------+----------------+
| Paired 131-complex (CPU vs GPU) | 34.4% vs 30.5%   | 56.5% vs 56.5% |
+---------------------------------+------------------+----------------+

**Affinity Prediction (PandaDock-GNN, Native Crystal / Held-out Splits):**

+---------------------------------+------------------+----------------+
| Evaluation                      | N                | Pearson R      |
+=================================+==================+================+
| SAIR held-out test split        | 90,219           | 0.407          |
+---------------------------------+------------------+----------------+
| Independent crystal complexes   | 202              | 0.467          |
+---------------------------------+------------------+----------------+
| PDBbind v2020 refined (indep.)  | 4,640            | 0.531          |
+---------------------------------+------------------+----------------+
| PDBbind-only held-out test      | 624              | 0.690          |
+---------------------------------+------------------+----------------+

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started/installation
   getting_started/quickstart
   getting_started/basic_usage

.. toctree::
   :maxdepth: 2
   :caption: PandaDock-GNN

   gnn/overview
   gnn/dataset_preparation
   gnn/training
   gnn/prediction
   gnn/hybrid_docking

.. toctree::
   :maxdepth: 2
   :caption: Algorithms

   algorithms/overview
   algorithms/hierarchical
   algorithms/scoring_functions

.. toctree::
   :maxdepth: 2
   :caption: Command Line Interface

   cli/pandadock
   cli/pandadock_gnn
   cli/pandadock_flex
   cli/pandadock_metal
   cli/pandadock_tethered
   cli/pandadock_report

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/basic_docking
   tutorials/gnn_training
   tutorials/hybrid_workflow
   tutorials/flexible_docking
   tutorials/metal_coordination
   tutorials/virtual_screening

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guide/best_practices
   guide/performance
   guide/troubleshooting
   guide/faq

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/docking
   api/gnn
   api/scoring
   api/analysis

.. toctree::
   :maxdepth: 1
   :caption: About

   about/citation
   about/contributing
   about/license
   about/changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. note::

   ULVSH affinity labels are censored: inactive compounds take a floor
   value, and 89 of the 95 test-split pEC50 values are exactly 4.0. A
   Pearson correlation over that distribution reflects separation of
   actives from the floor rather than affinity ranking, so ULVSH is
   reported as activity classification (AUC = 0.94). Use PDBbind or
   BindingDB for continuous affinity performance.
