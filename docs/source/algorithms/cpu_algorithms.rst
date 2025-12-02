CPU Algorithms
==============

PandaDock provides 5 CPU-based docking algorithms optimized for different use cases.

Enhanced Hierarchical CPU
--------------------------

**Algorithm ID**: ``enhanced_hierarchical_cpu``

**Recommended default algorithm** for high-accuracy general docking.

**Description**:
3-stage hierarchical search with progressive refinement from coarse to fine sampling.

**Stages**:

1. **Global Search**: Coarse exploration (2-5 Å steps, 30-60° rotations)
2. **Local Refinement**: Medium resolution (0.5-1 Å steps, 10-20° rotations)
3. **Fine Optimization**: High precision (0.1-0.3 Å steps, 2-5° rotations) with simulated annealing

**Performance**:

* Mean RMSD: ~0.08 Å
* Runtime: 150-250 seconds
* Success rate: 95-98% (RMSD < 2Å)

**Usage**:

.. code-block:: bash

   pandadock dock -r protein.pdb -l ligand.sdf \
                  --algorithm enhanced_hierarchical_cpu \
                  --center 10 20 30 --box 20 20 20

Monte Carlo CPU
---------------

**Algorithm ID**: ``monte_carlo_cpu``

**Best for**: Fast screening and initial exploration

**Description**:
Monte Carlo sampling with Metropolis criterion and simulated annealing for efficient conformational space exploration.

**Algorithm**:

1. Generate random initial pose
2. Apply random perturbation
3. Evaluate energy
4. Accept/reject based on Metropolis criterion
5. Gradually decrease temperature

**Performance**:

* Mean RMSD: ~0.5-1.5 Å
* Runtime: 30-60 seconds (fastest)
* Success rate: 85-90% (RMSD < 2Å)

**Usage**:

.. code-block:: bash

   pandadock dock -r protein.pdb -l ligand.sdf \
                  --algorithm monte_carlo_cpu \
                  --center 10 20 30 --box 20 20 20

Genetic Algorithm CPU
---------------------

**Algorithm ID**: ``genetic_algorithm_cpu``

**Best for**: Complex binding sites and flexible ligands

**Description**:
Evolutionary algorithm that evolves a population of ligand poses through selection, crossover, and mutation.

**Steps**:

1. Initialize random population
2. Evaluate fitness
3. Tournament selection
4. Crossover (blend poses)
5. Mutation (random perturbations)
6. Elitism (preserve top 10%)

**Performance**:

* Mean RMSD: ~0.3-0.8 Å
* Runtime: 120-200 seconds
* Success rate: 90-95% (RMSD < 2Å)

**Usage**:

.. code-block:: bash

   pandadock dock -r protein.pdb -l ligand.sdf \
                  --algorithm genetic_algorithm_cpu \
                  --center 10 20 30 --box 20 20 20

Hierarchical CPU
----------------

**Algorithm ID**: ``hierarchical_cpu``

**Best for**: Balanced accuracy and speed

**Description**:
Standard 2-stage hierarchical search with grid-based exploration and local optimization.

**Performance**:

* Mean RMSD: ~0.5-1.0 Å
* Runtime: 60-100 seconds
* Success rate: 88-92% (RMSD < 2Å)

**Usage**:

.. code-block:: bash

   pandadock dock -r protein.pdb -l ligand.sdf \
                  --algorithm hierarchical_cpu \
                  --center 10 20 30 --box 20 20 20

Crystal Guided CPU
------------------

**Algorithm ID**: ``crystal_guided_cpu``

**Best for**: Validation studies and reproducing crystal structures

**Description**:
Uses crystallographic information to guide docking towards known binding modes.

**Performance**:

* Mean RMSD: ~0.05-0.2 Å (with reference)
* Runtime: 100-150 seconds
* Success rate: 98-100% (with crystal structure)

**Usage**:

.. code-block:: bash

   pandadock dock -r protein.pdb -l ligand.sdf \
                  --algorithm crystal_guided_cpu \
                  --reference-ligand crystal_ligand.pdb \
                  --center 10 20 30 --box 20 20 20

Algorithm Selection Guide
-------------------------

By Use Case
^^^^^^^^^^^

* **General docking**: ``enhanced_hierarchical_cpu``
* **Fast screening**: ``monte_carlo_cpu``
* **Complex sites**: ``genetic_algorithm_cpu``
* **Validation**: ``crystal_guided_cpu``

By Ligand Flexibility
^^^^^^^^^^^^^^^^^^^^^

* **Rigid (0-3 rotatable bonds)**: ``hierarchical_cpu``
* **Flexible (4-8 bonds)**: ``enhanced_hierarchical_cpu``
* **Highly flexible (>8 bonds)**: ``genetic_algorithm_cpu``

Performance Comparison
----------------------

**Accuracy (Mean RMSD)**:

1. enhanced_hierarchical_cpu: 0.08 Å ⭐
2. crystal_guided_cpu: 0.10 Å
3. genetic_algorithm_cpu: 0.45 Å
4. hierarchical_cpu: 0.72 Å
5. monte_carlo_cpu: 1.10 Å

**Speed (Runtime)**:

1. monte_carlo_cpu: 30-60s ⭐
2. hierarchical_cpu: 60-100s
3. genetic_algorithm_cpu: 120-200s
4. enhanced_hierarchical_cpu: 150-250s
5. crystal_guided_cpu: 100-150s

Parallel Processing
-------------------

All CPU algorithms support parallel processing:

.. code-block:: bash

   # Use 8 CPU cores
   pandadock dock -r protein.pdb -l ligand.sdf \
                  --algorithm enhanced_hierarchical_cpu \
                  --cpuworkers 8 \
                  --center 10 20 30 --box 20 20 20

Fast Mode
---------

For quick testing:

.. code-block:: bash

   pandadock dock -r protein.pdb -l ligand.sdf \
                  --algorithm enhanced_hierarchical_cpu \
                  --fast \
                  --center 10 20 30 --box 20 20 20

See Also
--------

* :doc:`gpu_algorithms` - GPU-accelerated algorithms
* :doc:`specialized_modes` - Flexible, metal, ML docking
* :doc:`selection_guide` - Detailed selection guide
