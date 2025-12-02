Algorithm Selection Guide
=========================

Choosing the right docking algorithm is crucial for obtaining accurate results efficiently. This guide helps you select the optimal algorithm based on your specific use case, system characteristics, and performance requirements.

Quick Selection Table
---------------------

+-------------------+----------------------------+-------------+--------------+
| Use Case          | Recommended Algorithm      | Accuracy    | Speed        |
+===================+============================+=============+==============+
| General docking   | enhanced_hierarchical_cpu  | Very High   | Medium       |
+-------------------+----------------------------+-------------+--------------+
| Fast screening    | monte_carlo_cpu            | Medium      | Very Fast    |
+-------------------+----------------------------+-------------+--------------+
| Complex sites     | genetic_algorithm_cpu      | High        | Medium-Slow  |
+-------------------+----------------------------+-------------+--------------+
| Validation        | crystal_guided_cpu         | Excellent   | Medium       |
+-------------------+----------------------------+-------------+--------------+
| GPU available     | enhanced_hierarchical_gpu  | Very High   | Ultra Fast   |
+-------------------+----------------------------+-------------+--------------+
| Large library     | cuda_monte_carlo           | Medium      | Ultra Fast   |
+-------------------+----------------------------+-------------+--------------+

Decision Tree
-------------

**Step 1: GPU Available?**

* **YES** ’ Use GPU algorithms for massive speedup:

  * High accuracy needed ’ ``enhanced_hierarchical_gpu``
  * Fast screening ’ ``cuda_monte_carlo``
  * Complex binding site ’ ``cuda_genetic_algorithm``

* **NO** ’ Continue to Step 2

**Step 2: What's Your Priority?**

* **Maximum Accuracy** ’ ``enhanced_hierarchical_cpu``
* **Maximum Speed** ’ ``monte_carlo_cpu``
* **Balanced** ’ ``hierarchical_cpu``
* **Validation/Reproduction** ’ ``crystal_guided_cpu``

**Step 3: Consider Ligand Properties**

* **Rigid ligand (0-3 rotatable bonds)** ’ ``hierarchical_cpu``
* **Flexible ligand (4-8 bonds)** ’ ``enhanced_hierarchical_cpu``
* **Highly flexible (>8 bonds)** ’ ``genetic_algorithm_cpu``

**Step 4: Special Cases?**

* **Induced fit required** ’ Use ``pandadock-flex`` (flexible docking)
* **Metalloprotein** ’ Use ``pandadock-metal`` (metal docking)
* **ML scoring preferred** ’ Use ``pandadock-ml`` (ML docking)
* **Constrained docking** ’ Use ``pandadock-tethered`` (tethered docking)

By Use Case
-----------

Drug Discovery Projects
^^^^^^^^^^^^^^^^^^^^^^^

**Lead Identification:**

.. code-block:: bash

   pandadock dock -r target.pdb -l library.sdf \\
                  --algorithm cuda_monte_carlo \\
                  --gpu --fast \\
                  --num-poses 5

* Algorithm: ``cuda_monte_carlo`` (GPU) or ``monte_carlo_cpu`` (CPU)
* Rationale: Fast screening of large libraries
* Expected throughput: 1800-7200 ligands/hour (GPU)

**Lead Optimization:**

.. code-block:: bash

   pandadock dock -r target.pdb -l analogs.sdf \\
                  --algorithm enhanced_hierarchical_cpu \\
                  --scoring hybrid \\
                  --num-poses 50

* Algorithm: ``enhanced_hierarchical_cpu``
* Scoring: ``hybrid`` (physics + ML)
* Rationale: High accuracy for ranking close analogs

**Structure Validation:**

.. code-block:: bash

   pandadock dock -r protein.pdb -l ligand.sdf \\
                  --algorithm crystal_guided_cpu \\
                  --reference-ligand crystal_ligand.pdb

* Algorithm: ``crystal_guided_cpu``
* Rationale: Reproduce crystallographic binding modes

Academic Research
^^^^^^^^^^^^^^^^^

**Method Benchmarking:**

.. code-block:: bash

   pandadock dock -r protein.pdb -l ligand.sdf \\
                  --algorithm enhanced_hierarchical_cpu \\
                  --num-poses 100 \\
                  --ensemble

* Algorithm: ``enhanced_hierarchical_cpu``
* Options: Large pose ensemble, Boltzmann averaging
* Rationale: Comprehensive conformational sampling

**Comparative Studies:**

Run multiple algorithms and compare:

.. code-block:: bash

   # High accuracy baseline
   pandadock dock -r protein.pdb -l ligand.sdf \\
                  --algorithm enhanced_hierarchical_cpu \\
                  -o results_enhanced/

   # Fast alternative
   pandadock dock -r protein.pdb -l ligand.sdf \\
                  --algorithm monte_carlo_cpu \\
                  -o results_mc/

   # GPU accelerated
   pandadock dock -r protein.pdb -l ligand.sdf \\
                  --algorithm enhanced_hierarchical_gpu \\
                  --gpu \\
                  -o results_gpu/

By Target Characteristics
--------------------------

Small, Well-Defined Binding Sites
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Algorithm:** ``hierarchical_cpu`` or ``enhanced_hierarchical_cpu``
* **Example:** Trypsin, carbonic anhydrase
* **Rationale:** Grid-based search works well in confined spaces

Large, Shallow Binding Sites
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Algorithm:** ``genetic_algorithm_cpu`` or ``cuda_genetic_algorithm``
* **Example:** Protein-protein interfaces
* **Rationale:** Evolutionary search better explores large conformational spaces

Flexible Binding Sites
^^^^^^^^^^^^^^^^^^^^^^

* **Mode:** Flexible docking (``pandadock-flex``)
* **Example:** Kinases with DFG-in/out conformations
* **Rationale:** Account for induced-fit effects

Metal-Containing Active Sites
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Mode:** Metal docking (``pandadock-metal``)
* **Example:** MMPs, carbonic anhydrase, zinc fingers
* **Rationale:** Explicit metal coordination constraints

By Ligand Characteristics
--------------------------

Small Rigid Ligands (<15 atoms, 0-3 rotatable bonds)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Algorithm:** ``hierarchical_cpu`` or ``monte_carlo_cpu``
* **Rationale:** Limited conformational space ’ faster algorithms sufficient

Medium Flexibility (15-30 atoms, 4-8 rotatable bonds)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Algorithm:** ``enhanced_hierarchical_cpu``
* **Rationale:** Standard drug-like molecules

Large Flexible Ligands (>30 atoms, >8 rotatable bonds)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Algorithm:** ``genetic_algorithm_cpu`` or flexible docking
* **Rationale:** Extensive conformational sampling needed

Peptides and Macrocycles
^^^^^^^^^^^^^^^^^^^^^^^^^

* **Mode:** Flexible docking (``pandadock-flex``)
* **Options:** ``--refine-ligand --num-receptor-conformers 10``
* **Rationale:** Both ligand and receptor flexibility important

Performance Optimization
------------------------

For Maximum Throughput
^^^^^^^^^^^^^^^^^^^^^^

**GPU Setup (Best):**

.. code-block:: bash

   pandadock dock -r target.pdb -l library.sdf \\
                  --algorithm cuda_monte_carlo \\
                  --gpu \\
                  --gpu-batch-size 2000 \\
                  --fast \\
                  --num-poses 5

Expected: 3000-7200 ligands/hour

**CPU Parallel (Good):**

.. code-block:: bash

   pandadock dock -r target.pdb -l library.sdf \\
                  --algorithm monte_carlo_cpu \\
                  --cpuworkers 16 \\
                  --fast \\
                  --num-poses 5

Expected: 60-120 ligands/hour

For Maximum Accuracy
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock dock -r target.pdb -l ligand.sdf \\
                  --algorithm enhanced_hierarchical_cpu \\
                  --scoring hybrid \\
                  --rescoring mmgbsa \\
                  --num-poses 100 \\
                  --ensemble

Expected RMSD: <0.1 Å

For Balanced Performance
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock dock -r target.pdb -l ligand.sdf \\
                  --algorithm enhanced_hierarchical_gpu \\
                  --gpu \\
                  --num-poses 20

Expected: 720-1800 ligands/hour, RMSD ~0.08 Å

Algorithm Comparison Metrics
-----------------------------

Accuracy Ranking
^^^^^^^^^^^^^^^^

1. ``crystal_guided_cpu`` (with reference): 0.05-0.2 Å
2. ``enhanced_hierarchical_cpu/gpu``: 0.08 Å
3. ``genetic_algorithm_cpu/cuda``: 0.3-0.8 Å
4. ``hierarchical_cpu``: 0.5-1.0 Å
5. ``monte_carlo_cpu/cuda``: 0.5-1.5 Å

Speed Ranking (CPU)
^^^^^^^^^^^^^^^^^^^

1. ``monte_carlo_cpu``: 30-60s
2. ``hierarchical_cpu``: 60-100s
3. ``crystal_guided_cpu``: 100-150s
4. ``genetic_algorithm_cpu``: 120-200s
5. ``enhanced_hierarchical_cpu``: 150-250s

Speed Ranking (GPU)
^^^^^^^^^^^^^^^^^^^

1. ``cuda_monte_carlo``: 0.5-2s (100-200x speedup)
2. ``cuda_genetic_algorithm``: 1-3s (80-150x speedup)
3. ``enhanced_hierarchical_gpu``: 2-5s (50-100x speedup)

Success Rate (RMSD < 2Å)
^^^^^^^^^^^^^^^^^^^^^^^^^

1. ``crystal_guided_cpu``: 98-100%
2. ``enhanced_hierarchical_cpu/gpu``: 95-98%
3. ``flexible_docking``: 92-96%
4. ``genetic_algorithm_cpu/cuda``: 90-95%
5. ``hierarchical_cpu``: 88-92%
6. ``monte_carlo_cpu/cuda``: 85-90%

Common Mistakes to Avoid
-------------------------

L Using ``monte_carlo_cpu`` for Critical Predictions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Problem:** Lowest accuracy among algorithms
* **Solution:** Use ``enhanced_hierarchical_cpu`` or ``hybrid`` scoring

L Using ``enhanced_hierarchical_cpu`` for 10,000+ Compound Library
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Problem:** Too slow (weeks to complete)
* **Solution:** Use GPU algorithms or ``monte_carlo_cpu`` with ``--fast``

L Ignoring GPU Acceleration When Available
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Problem:** Missing 50-200x speedup
* **Solution:** Always use GPU algorithms when CUDA is available

L Not Using Specialized Modes When Needed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Problem:** Poor results for metalloproteins, flexible sites
* **Solution:** Use ``pandadock-metal``, ``pandadock-flex`` appropriately

L Using Default Settings for All Cases
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Problem:** Suboptimal performance/accuracy trade-off
* **Solution:** Tune ``--num-poses``, ``--fast``, ``--ensemble`` based on needs

Validation Strategy
-------------------

Recommended Validation Protocol
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Test on Known Structures:**

   .. code-block:: bash

      pandadock dock -r protein.pdb -l crystal_ligand.sdf \\
                     --algorithm enhanced_hierarchical_cpu \\
                     -o validation/

   Success criterion: RMSD < 2.0 Å to crystal pose

2. **Compare Multiple Algorithms:**

   Run 3 different algorithms, check consensus

3. **Visual Inspection:**

   Examine top poses for reasonable interactions

4. **Rescoring:**

   .. code-block:: bash

      pandadock dock -r protein.pdb -l ligand.sdf \\
                     --algorithm enhanced_hierarchical_cpu \\
                     --rescoring mmgbsa

See Also
--------

* :doc:`cpu_algorithms` - Detailed CPU algorithm documentation
* :doc:`gpu_algorithms` - Detailed GPU algorithm documentation
* :doc:`specialized_modes` - Flexible, metal, ML, tethered docking
* :doc:`../scoring/overview` - Scoring function selection
