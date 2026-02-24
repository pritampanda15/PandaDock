Welcome to PandaDock Documentation
==================================

.. image:: _static/logo.png
   :width: 400px
   :align: center
   :alt: PandaDock Logo

|

PandaDock is a next-generation molecular docking platform featuring an **SE(3)-equivariant Graph Neural Network** scoring function that achieves **R > 0.8** correlation with experimental binding affinities.

.. note::
   **PandaDock v4.0** features the PandaDock-GNN scoring function with state-of-the-art performance: Pearson R = 0.88 on PDBbind, R = 0.82 on ULVSH test set, and R = 0.68 on novel GABA receptor dataset. Significantly outperforms all baseline methods including Vina, Gnina, and MM-GBSA.

Quick Start
-----------

Install PandaDock using pip:

.. code-block:: bash

   git clone https://github.com/pritampanda15/PandaDock.git
   cd PandaDock
   pip install -e ".[gnn]"

Basic usage:

.. code-block:: bash

   # Traditional docking with Vina-style scoring
   pandadock dock -r protein.pdb -l ligand.sdf \
                  --center 10 20 30 --box 20 20 20

   # Hybrid docking with GNN rescoring (RECOMMENDED)
   pandadock hybrid -r protein.pdb -l ligand.sdf \
                    --center 10 20 30 --box 20 20 20 \
                    --model models/best_model.pt

Key Features
------------

🧠 **SE(3)-Equivariant GNN Scoring (NEW)**
   - Rotation and translation invariant predictions
   - Heterogeneous graph representation (protein + ligand)
   - Multi-task learning: pEC50 regression + activity classification
   - **Pearson R > 0.8** on held-out test sets
   - Outperforms all traditional scoring methods

🔬 **Advanced Docking Algorithm**
   - **Hierarchical Search**: Multi-resolution coarse-to-fine sampling
   - Vina-style empirical scoring function
   - Physics-based scoring with Lennard-Jones + electrostatics

⚡ **Hybrid Workflow (Recommended)**
   - Traditional pose generation + GNN rescoring
   - Combines speed of physics-based docking with GNN accuracy
   - Best of both worlds approach

🎯 **Specialized Docking Modes**
   - **Flexible Docking** (``pandadock-flex``): Induced-fit with receptor flexibility
   - **Metal Docking** (``pandadock-metal``): Specialized for metalloproteins (Zn, Fe, Mg, Ca, etc.)
   - **Tethered Docking** (``pandadock-tethered``): Constrained near reference positions
   - **Analysis & Reporting** (``pandadock-report``): Publication-ready figures

📊 **Benchmark Performance**
   - PandaDock-GNN outperforms 8+ baseline methods on ULVSH dataset
   - Better than VM2, MMGBSA, Gnina, Hyde, and other scoring functions

Performance Benchmarks
-----------------------

**PDBbind Benchmark** (5,316 complexes):

+---------------------------------+-----------------+----------------+
| Method                          | Type            | Pearson R      |
+=================================+=================+================+
| **PandaDock-GNN**               | SE(3)-EGNN      | **0.88** ⭐    |
+---------------------------------+-----------------+----------------+
| OnionNet-2                      | 3D-CNN          | 0.86           |
+---------------------------------+-----------------+----------------+
| RF-Score v3                     | Random Forest   | 0.80           |
+---------------------------------+-----------------+----------------+
| AutoDock Vina                   | Empirical       | 0.60           |
+---------------------------------+-----------------+----------------+

**ULVSH Benchmark** (942 compounds, 10 targets):

+---------------------------------+-----------------+----------------+
| Method                          | Type            | Pearson R      |
+=================================+=================+================+
| **PandaDock-GNN (test set)**    | ML Scoring      | **0.82** ⭐    |
+---------------------------------+-----------------+----------------+
| VM2                             | Free energy     | 0.15           |
+---------------------------------+-----------------+----------------+
| PM6                             | Semi-empirical  | 0.08           |
+---------------------------------+-----------------+----------------+
| Gnina, MMPBSA, MMGBSA, Vina     | Baselines       | < 0.02         |
+---------------------------------+-----------------+----------------+

**BindingDB Benchmark** (8,891 complexes):

+---------------------------------+-----------------+----------------+
| Training Configuration          | Type            | Pearson R      |
+=================================+=================+================+
| **BindingDB Only**              | ML Scoring      | **0.81** ⭐    |
+---------------------------------+-----------------+----------------+
| **BindingDB + ULVSH**           | ML Scoring      | **0.79**       |
+---------------------------------+-----------------+----------------+

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
