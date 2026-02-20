pandadock - Main Docking Command
=================================

The ``pandadock`` command is the primary interface for molecular docking in PandaDock.

Synopsis
--------

.. code-block:: bash

   pandadock COMMAND [OPTIONS]

Commands
--------

* ``dock`` - Traditional molecular docking with Vina-style scoring
* ``hybrid`` - Hybrid docking: pose generation + GNN rescoring (recommended)
* ``list-algorithms`` - Show available algorithms and scoring functions
* ``gnn`` - GNN subcommands (train, predict, benchmark, compare)

pandadock dock
--------------

Performs molecular docking using hierarchical search with Vina-style scoring.

**Required Options:**

``-r, --receptor PATH``
    Receptor PDB file (protein structure)

``-l, --ligand PATH``
    Ligand file (SDF, MOL2, or PDB format)

``--center X Y Z`` or ``--grid-config PATH``
    Grid box specification. Either provide center coordinates (in Angstroms) or a JSON configuration file.

``--box X Y Z``
    Grid box dimensions (in Angstroms). Required if ``--center`` is used.

**Scoring Options:**

``-s, --scoring FUNCTION``
    Scoring function to use. Default: ``vina``

    Available:

    * ``vina`` - Vina-style empirical scoring (recommended)
    * ``physics_based`` - Physics-based Lennard-Jones + electrostatics

``--rescoring METHOD``
    Rescoring method for top poses. Default: ``none``

    Options: ``none``, ``mmgbsa``

**Output Options:**

``-o, --output-dir PATH``
    Output directory for results. Default: ``docking_output``

``-n, --num-poses N``
    Number of poses to generate. Default: 20

``--visualize / --no-visualize``
    Generate visualization plots. Default: enabled

**Performance Options:**

``--fast``
    Enable fast mode with reduced sampling for quick testing

``--ensemble / --no-ensemble``
    Use Boltzmann ensemble averaging. Default: enabled

**Example:**

.. code-block:: bash

   pandadock dock -r protein.pdb -l ligand.sdf \
                  --center 10 20 30 --box 20 20 20 \
                  -o results/

pandadock hybrid
----------------

**Recommended workflow** for best accuracy. Combines traditional pose generation
with SE(3)-equivariant GNN rescoring.

**Required Options:**

``-r, --receptor PATH``
    Receptor PDB file

``-l, --ligand PATH``
    Ligand file (SDF, MOL2, or PDB format)

``--center X Y Z`` or ``--grid-config PATH``
    Grid box specification

``--box X Y Z``
    Grid box dimensions

``-m, --model PATH``
    Path to trained GNN model checkpoint

**Optional Options:**

``-o, --output-dir PATH``
    Output directory. Default: ``hybrid_output``

``-n, --num-poses N``
    Number of poses to generate for rescoring. Default: 50

``--top-k N``
    Number of top poses to keep after rescoring. Default: 10

``--fast``
    Use fast mode with reduced sampling

**Example:**

.. code-block:: bash

   pandadock hybrid -r protein.pdb -l ligand.sdf \
                    --center 10 20 30 --box 20 20 20 \
                    -m models/best_model.pt \
                    -o hybrid_results/

**Output:**

The hybrid command generates:

* ``hybrid_results.csv`` - Rankings with GNN and Vina scores
* ``pose_1_pec50_X.XX.pdb`` - Top poses with pEC50 in filename
* ``complex_1.pdb``, etc. - Protein-ligand complexes

pandadock gnn
-------------

GNN subcommands for training and prediction. See :doc:`pandadock_gnn` for details.

* ``pandadock gnn train`` - Train GNN model
* ``pandadock gnn predict`` - Predict binding affinity
* ``pandadock gnn benchmark`` - Benchmark model performance
* ``pandadock gnn compare`` - Compare against baselines

pandadock list-algorithms
-------------------------

Show available docking algorithms and scoring functions.

.. code-block:: bash

   pandadock list-algorithms

Output Files
------------

**dock command:**

* ``complex1.pdb, complex2.pdb, ...`` - Protein-ligand complexes (top 10)
* ``pose1.pdb, pose2.pdb, ...`` - Ligand poses only (top 10)
* ``docking_results.json`` - Complete results with energies
* ``interaction_analysis.json`` - Detailed interaction analysis
* ``binding_affinities.png`` - Affinity distribution plot

**hybrid command:**

* ``hybrid_results.csv`` - Rankings with GNN and Vina scores
* ``pose_N_pec50_X.XX.pdb`` - Top pose structures
* ``complex_N.pdb`` - Protein-ligand complexes

Global Options
--------------

``-v, --verbose``
    Enable verbose logging

``--version``
    Show version information

``-h, --help``
    Show help message

Examples
--------

**Basic Docking:**

.. code-block:: bash

   pandadock dock -r protein.pdb -l ligand.sdf \
                  --center 10 20 30 --box 20 20 20

**Hybrid Docking (Recommended):**

.. code-block:: bash

   pandadock hybrid -r protein.pdb -l ligand.sdf \
                    --center 10 20 30 --box 20 20 20 \
                    -m models/best_model.pt

**Fast Screening:**

.. code-block:: bash

   pandadock dock -r protein.pdb -l ligand.sdf \
                  --center 10 20 30 --box 20 20 20 \
                  --fast --num-poses 10

**With MM-GBSA Rescoring:**

.. code-block:: bash

   pandadock dock -r protein.pdb -l ligand.sdf \
                  --center 10 20 30 --box 20 20 20 \
                  --rescoring mmgbsa

See Also
--------

* :doc:`pandadock_gnn` - GNN commands reference
* :doc:`pandadock_flex` - Flexible docking command
* :doc:`pandadock_metal` - Metal docking command
* :doc:`pandadock_tethered` - Tethered docking command
* :doc:`../gnn/overview` - GNN architecture documentation
