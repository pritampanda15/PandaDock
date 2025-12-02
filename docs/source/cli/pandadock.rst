pandadock - Main Docking Command
=================================

The ``pandadock dock`` command is the primary interface for molecular docking in PandaDock.

Synopsis
--------

.. code-block:: bash

   pandadock dock [OPTIONS]

Description
-----------

Performs molecular docking of a ligand into a protein receptor using specified algorithm and scoring function. Generates docked poses, binding energies, and interaction analysis.

Required Options
----------------

``-r, --receptor PATH``
    Receptor PDB file (protein structure)

``-l, --ligand PATH``
    Ligand file (SDF, MOL2, or PDB format)

``--center X Y Z`` or ``--grid-config PATH``
    Grid box specification. Either provide center coordinates (X Y Z in Angstroms) or a JSON configuration file.

``--box X Y Z``
    Grid box dimensions (X Y Z in Angstroms). Required if ``--center`` is used.

Algorithm Selection
-------------------

``-a, --algorithm ALGORITHM``
    Docking algorithm to use. Default: ``enhanced_hierarchical_cpu``

    CPU Algorithms:
    
    * ``enhanced_hierarchical_cpu`` - High-accuracy hierarchical search (recommended)
    * ``monte_carlo_cpu`` - Fast Monte Carlo sampling
    * ``genetic_algorithm_cpu`` - Evolutionary algorithm
    * ``hierarchical_cpu`` - Balanced hierarchical search
    * ``crystal_guided_cpu`` - Crystal structure-guided docking

    GPU Algorithms (requires CUDA):
    
    * ``enhanced_hierarchical_gpu`` - GPU-accelerated hierarchical (50-100x speedup)
    * ``cuda_monte_carlo`` - GPU Monte Carlo (100-200x speedup)
    * ``cuda_genetic_algorithm`` - GPU genetic algorithm (80-150x speedup)

Scoring Options
---------------

``-s, --scoring FUNCTION``
    Scoring function to use. Default: ``physics_based``

    Available:
    
    * ``physics_based`` - Comprehensive force field scoring (recommended)
    * ``empirical`` - Fast empirical scoring
    * ``precision_score`` - High-precision energy decomposition
    * ``hybrid`` - Combined physics + ML scoring
    * ``gpu_precision`` - GPU-accelerated precision scoring
    * ``gpu_mmgbsa`` - GPU MM-GBSA rescoring

``--rescoring METHOD``
    Rescoring method for top poses. Default: ``none``

    Options: ``none``, ``mmgbsa``

Output Options
--------------

``-o, --output-dir PATH``
    Output directory for results. Default: ``docking_output``

``-n, --num-poses N``
    Number of poses to generate. Default: 20

``--visualize / --no-visualize``
    Generate visualization plots. Default: enabled

Performance Options
-------------------

CPU Options
^^^^^^^^^^^

``--cpuworkers N``
    Number of CPU worker threads for parallel execution. Default: auto-detect

``--fast``
    Enable fast mode with reduced sampling for quick testing

GPU Options
^^^^^^^^^^^

``--gpu``
    Enable GPU acceleration (requires CUDA and compatible algorithm)

``--gpu-batch-size N``
    Batch size for GPU processing. Default: 1000

``--gpu-memory-limit GB``
    GPU memory limit in gigabytes. Default: 4.0

``--gpuid ID``
    GPU device ID to use. Default: 0

Advanced Options
----------------

``--ensemble / --no-ensemble``
    Use Boltzmann ensemble averaging. Default: enabled

``--grid-config PATH``
    JSON file with grid box configuration

Examples
--------

Basic Docking
^^^^^^^^^^^^^

.. code-block:: bash

   pandadock dock -r protein.pdb -l ligand.sdf \
                  --center 10 20 30 --box 20 20 20

High-Accuracy Docking
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock dock -r protein.pdb -l ligand.sdf \
                  --algorithm enhanced_hierarchical_cpu \
                  --scoring physics_based \
                  --center 10 20 30 --box 20 20 20 \
                  --num-poses 50 \
                  -o high_accuracy/

GPU-Accelerated Docking
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock dock -r protein.pdb -l ligand.sdf \
                  --algorithm enhanced_hierarchical_gpu \
                  --gpu \
                  --center 10 20 30 --box 20 20 20

Fast Screening Mode
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock dock -r protein.pdb -l library.sdf \
                  --algorithm monte_carlo_cpu \
                  --fast \
                  --num-poses 5 \
                  -o screening/

With MM-GBSA Rescoring
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock dock -r protein.pdb -l ligand.sdf \
                  --scoring physics_based \
                  --rescoring mmgbsa \
                  --center 10 20 30 --box 20 20 20

Output Files
------------

The command generates the following outputs in the specified directory:

**Structures:**

* ``complex1.pdb, complex2.pdb, ...`` - Protein-ligand complexes (top 10)
* ``pose1.pdb, pose2.pdb, ...`` - Ligand poses only (top 10)

**Analysis:**

* ``docking_results.json`` - Complete results with energies and metadata
* ``interaction_analysis.json`` - Detailed interaction analysis
* ``summary.txt`` - Human-readable summary

**Visualizations:**

* ``binding_affinities.png`` - Affinity distribution plot
* ``interaction_energies.png`` - Energy decomposition (if available)

Exit Status
-----------

Returns 0 on success, non-zero on error.

See Also
--------

* :doc:`pandadock_flex` - Flexible docking command
* :doc:`pandadock_metal` - Metal docking command
* :doc:`../algorithms/cpu_algorithms` - CPU algorithms documentation
* :doc:`../algorithms/gpu_algorithms` - GPU algorithms documentation
* :doc:`../scoring/overview` - Scoring functions overview
