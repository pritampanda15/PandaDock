Basic Usage
===========

This guide covers the fundamental usage patterns of PandaDock.

Available Commands
------------------

PandaDock provides several command-line tools:

Core Commands
^^^^^^^^^^^^^

* ``pandadock``: Main docking interface
* ``pandadock-flex``: Flexible/induced-fit docking
* ``pandadock-metal``: Metal coordination docking
* ``pandadock-ml``: ML-enhanced docking
* ``pandadock-tethered``: Tethered/constrained docking

Utility Commands
^^^^^^^^^^^^^^^^

* ``pandadock-prepare``: Prepare ligands (add H, generate 3D)
* ``pandadock-gridbox``: Generate grid box configurations
* ``pandadock-report``: Generate analysis reports

Listing Algorithms
------------------

To see all available algorithms:

.. code-block:: bash

   pandadock list-algorithms

This will show:

* CPU algorithms
* GPU algorithms (if CUDA is available)
* Scoring functions

Command-Line Options
--------------------

Common options for the ``pandadock dock`` command:

**Required:**

* ``-r, --receptor``: Receptor PDB file
* ``-l, --ligand``: Ligand file (SDF/MOL2/PDB)
* ``--center X Y Z`` or ``--grid-config``: Grid box specification

**Algorithm Selection:**

* ``-a, --algorithm``: Docking algorithm (default: enhanced_hierarchical_cpu)
* ``-s, --scoring``: Scoring function (default: physics_based)

**Output:**

* ``-o, --output-dir``: Output directory (default: docking_output)
* ``-n, --num-poses``: Number of poses to generate (default: 20)

**Performance:**

* ``--gpu``: Enable GPU acceleration
* ``--cpuworkers N``: Number of CPU worker threads
* ``--fast``: Fast mode for quick testing

**Advanced:**

* ``--ensemble``: Use Boltzmann ensemble averaging
* ``--rescoring``: Rescoring method (none, mmgbsa)
* ``--visualize``: Generate visualization plots

Example Workflows
-----------------

Virtual Screening
^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock dock -r protein.pdb -l library.sdf \
                  --algorithm monte_carlo_cpu \
                  --fast --num-poses 5 \
                  -o screening/

Lead Optimization
^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock dock -r protein.pdb -l lead.sdf \
                  --algorithm enhanced_hierarchical_cpu \
                  --scoring hybrid \
                  --rescoring mmgbsa \
                  -o optimization/

Getting Help
------------

For detailed help on any command:

.. code-block:: bash

   pandadock --help
   pandadock dock --help
   pandadock-flex --help
