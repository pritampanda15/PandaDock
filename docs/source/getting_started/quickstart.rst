Quick Start
===========

This tutorial will get you up and running with PandaDock in minutes.

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

High-Accuracy Docking
---------------------

For best accuracy, use the enhanced hierarchical algorithm:

.. code-block:: bash

   pandadock dock -r protein.pdb -l ligand.sdf \
                  --center 10 20 30 --box 20 20 20 \
                  --algorithm enhanced_hierarchical_cpu \
                  --scoring physics_based \
                  -o high_accuracy/

GPU-Accelerated Docking
-----------------------

For 50-100x speedup with GPU:

.. code-block:: bash

   pandadock dock -r protein.pdb -l ligand.sdf \
                  --center 10 20 30 --box 20 20 20 \
                  --algorithm enhanced_hierarchical_gpu \
                  --gpu \
                  -o gpu_results/

Viewing Results
---------------

After docking completes, check the output directory:

* ``complex1.pdb``: Top-ranked protein-ligand complex
* ``pose1.pdb``: Top-ranked ligand pose
* ``docking_results.json``: Complete results with energies
* ``interaction_analysis.json``: Detailed interactions
* ``binding_affinities.png``: Visualization of results
