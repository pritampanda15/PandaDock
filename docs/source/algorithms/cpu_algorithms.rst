CPU Docking Algorithms
======================

PandaDock uses a single validated flexible-ligand conformational search engine
(``PandaCoreDocker`` / ``pandadock``).

How It Works
------------

* **Articulated Torsion Tree**: Identifies all rotatable acyclic single bonds
  (freezing amides and conjugated N-C bonds), rooted on the largest rigid
  fragment. Optional ``--sample-rings`` enumerates distinct puckers for
  flexible non-aromatic rings.
* **Grid-Accelerated Scoring**: Precomputes AutoDock Vina or Vinardo affinity
  grids via exact 3D blocked neighbor selection (5.6–9.7x faster than dense
  evaluation) with optional cross-ligand ``GridCache`` reuse.
* **Analytic Gradients + L-BFGS**: Translation, SO(3) exponential-map torque
  derivatives, and torsion-axis velocity projections are computed in closed
  form and relaxed with SciPy L-BFGS-B.
* **Monte Carlo Basin Hopping**: Independent runs start uniformly over the
  docking box, SO(3), and torsion angles, with exhaustiveness scaling
  automatically from 8 to 32 runs based on rotatable bond count. Use
  ``--workers N`` to parallelize runs across CPU cores.
* **Symmetry-Corrected Clustering**: Deduplicates returned modes using
  molecular-graph automorphisms (2.0 A heavy-atom RMSD cutoff).

Algorithm Aliases
-----------------

Legacy algorithm names are retained for backwards compatibility and resolve to
``PandaCoreDocker``:

* ``pandadock`` / ``pandacore`` (current)
* ``monte_carlo_cpu`` (deprecated alias)
* ``genetic_algorithm_cpu`` (deprecated alias)
* ``enhanced_hierarchical_cpu`` (deprecated alias)

.. note::
   ``crystal_guided_cpu`` was intentionally removed in v4.1.0 because it
   biased sampling toward a reference pose and invalidated redocking benchmarks.

Example Usage
-------------

.. code-block:: bash

   pandadock dock -r receptor.pdb -l ligand.sdf \
                  --center 10 12 8 --box 22 22 22 \
                  --scoring vina --workers 4 --seed 42 -o results/
