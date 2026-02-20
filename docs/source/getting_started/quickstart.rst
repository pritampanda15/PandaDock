Quick Start
===========

This tutorial will get you up and running with PandaDock in minutes.

Installation
------------

.. code-block:: bash

   git clone https://github.com/pritampanda15/PandaDock.git
   cd PandaDock
   pip install -e ".[gnn]"

This installs PandaDock with GNN support (PyTorch, PyTorch Geometric).

Basic Docking Example
---------------------

The simplest way to perform molecular docking:

.. code-block:: bash

   pandadock dock -r protein.pdb -l ligand.sdf \
                  --center 10 20 30 --box 20 20 20 \
                  -o results/

Where:

* ``-r protein.pdb``: Your receptor (protein) structure
* ``-l ligand.sdf``: Your ligand structure
* ``--center X Y Z``: Center coordinates of the binding site (Å)
* ``--box X Y Z``: Size of the docking grid box (Å)
* ``-o results/``: Output directory

Hybrid Docking with GNN (Recommended)
-------------------------------------

For best accuracy, use the hybrid workflow that combines traditional docking
with GNN rescoring:

.. code-block:: bash

   pandadock hybrid -r protein.pdb -l ligand.sdf \
                    --center 10 20 30 --box 20 20 20 \
                    --model models/best_model.pt \
                    -o hybrid_results/

This workflow:

1. Generates diverse poses using hierarchical docking
2. Rescores all poses using the SE(3)-equivariant GNN
3. Ranks poses by GNN-predicted binding affinity (pEC50)

GNN-Only Prediction
-------------------

To predict binding affinity for an existing complex:

.. code-block:: bash

   pandadock gnn predict -m models/best_model.pt \
                         -p protein.mol2 -l ligand.mol2

Training Your Own GNN Model
---------------------------

Train on the ULVSH dataset:

.. code-block:: bash

   pandadock gnn train --dataset ULVSH/ --output models/ --epochs 100

Compare against baselines:

.. code-block:: bash

   pandadock gnn compare --model models/best_model.pt \
                         --dataset ULVSH/ --output results/

Viewing Results
---------------

After docking completes, check the output directory:

* ``complex1.pdb``: Top-ranked protein-ligand complex
* ``pose1.pdb``: Top-ranked ligand pose
* ``hybrid_results.csv``: Complete results with GNN pEC50 values
* ``interaction_analysis.json``: Detailed interactions

Available Commands
------------------

Main docking commands:

* ``pandadock dock``: Traditional docking with Vina-style scoring
* ``pandadock hybrid``: Hybrid docking with GNN rescoring (recommended)
* ``pandadock list-algorithms``: Show available algorithms

GNN commands:

* ``pandadock gnn download-model``: Download pre-trained model
* ``pandadock gnn train``: Train GNN model
* ``pandadock gnn predict``: Predict binding affinity
* ``pandadock gnn rescore``: Rescore poses from any docking tool
* ``pandadock gnn benchmark``: Benchmark model performance
* ``pandadock gnn compare``: Compare against baselines

Other tools:

* ``pandadock-prepare``: Prepare ligands for docking
* ``pandadock-gridbox``: Generate grid box configurations
* ``pandadock-flex``: Flexible/induced-fit docking
* ``pandadock-metal``: Metal coordination docking
* ``pandadock-tethered``: Tethered docking for validation
* ``pandadock-report``: Generate publication figures

Next Steps
----------

* :doc:`../gnn/overview`: Learn about the GNN architecture
* :doc:`../tutorials/gnn_training`: Train your own model
* :doc:`../tutorials/hybrid_workflow`: Master the hybrid workflow
