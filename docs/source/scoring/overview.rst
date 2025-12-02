Scoring Functions Overview
===========================

PandaDock provides 6 advanced scoring functions for evaluating protein-ligand binding energies.

Available Scoring Functions
----------------------------

physics_based
^^^^^^^^^^^^^
Comprehensive force field scoring with van der Waals, electrostatics, desolvation, hydrogen bonding, and torsional penalties. **Recommended for general docking.**

* **Accuracy**: R = 0.85 correlation with experimental data
* **Speed**: 0.01-0.05 seconds per pose
* **Best for**: General purpose docking, balanced accuracy

empirical
^^^^^^^^^
Fast empirical scoring based on statistical potentials from protein-ligand databases.

* **Accuracy**: R = 0.72 correlation
* **Speed**: 0.001-0.005 seconds per pose (fastest)
* **Best for**: Virtual screening, initial filtering

precision_score
^^^^^^^^^^^^^^^
High-precision interaction energy decomposition with per-residue contributions.

* **Accuracy**: R = 0.87 correlation
* **Speed**: 0.05-0.2 seconds per pose
* **Best for**: Detailed interaction analysis, lead optimization

hybrid
^^^^^^
Combined physics-based and machine learning scoring for maximum accuracy.

* **Accuracy**: R = 0.91 correlation (best)
* **Speed**: 0.1-0.3 seconds per pose
* **Best for**: Critical predictions, final ranking

gpu_precision
^^^^^^^^^^^^^
GPU-accelerated precision scoring for large-scale studies.

* **Accuracy**: R = 0.86 correlation
* **Speed**: 0.0001-0.001 seconds per pose (GPU)
* **Best for**: High-throughput screening with GPU

gpu_mmgbsa
^^^^^^^^^^
GPU-accelerated MM-GBSA rescoring for binding free energies.

* **Accuracy**: R = 0.89 correlation
* **Speed**: 0.001-0.01 seconds per pose (GPU)
* **Best for**: Accurate binding affinity predictions

Usage Examples
--------------

Basic Usage
^^^^^^^^^^^

.. code-block:: bash

   pandadock dock -r protein.pdb -l ligand.sdf \
                  --scoring physics_based \
                  --center 10 20 30 --box 20 20 20

High Accuracy
^^^^^^^^^^^^^

.. code-block:: bash

   pandadock dock -r protein.pdb -l ligand.sdf \
                  --scoring hybrid \
                  --center 10 20 30 --box 20 20 20

Fast Screening
^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock dock -r protein.pdb -l library.sdf \
                  --scoring empirical \
                  --center 10 20 30 --box 20 20 20

GPU Accelerated
^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock dock -r protein.pdb -l ligand.sdf \
                  --scoring gpu_precision \
                  --gpu \
                  --center 10 20 30 --box 20 20 20

Selection Guide
---------------

+------------------+-------------+----------+---------------------+
| Use Case         | Scoring     | Speed    | Accuracy            |
+==================+=============+==========+=====================+
| General docking  | physics_    | Medium   | High (R=0.85)       |
|                  | based       |          |                     |
+------------------+-------------+----------+---------------------+
| Virtual          | empirical   | Fast     | Medium (R=0.72)     |
| screening        |             |          |                     |
+------------------+-------------+----------+---------------------+
| Lead             | hybrid      | Slow     | Highest (R=0.91)    |
| optimization     |             |          |                     |
+------------------+-------------+----------+---------------------+
| Detailed         | precision   | Medium-  | High (R=0.87)       |
| analysis         | _score      | Slow     |                     |
+------------------+-------------+----------+---------------------+
| GPU high-        | gpu_        | Very     | High (R=0.86)       |
| throughput       | precision   | Fast     |                     |
+------------------+-------------+----------+---------------------+

See Also
--------

* :doc:`physics_based` - Physics-based scoring details
* :doc:`empirical` - Empirical scoring details
* :doc:`hybrid` - Hybrid ML scoring details
* :doc:`gpu_scoring` - GPU scoring details
