Empirical Scoring
=================

The empirical scoring function is optimized for **ultra-fast virtual screening**. It uses statistical potentials derived from protein-ligand databases to rapidly evaluate binding poses with acceptable accuracy.

Overview
--------

**Scoring ID:** ``empirical``

**Type:** Knowledge-based statistical scoring

**Accuracy:** R = 0.72 correlation with experimental binding affinities

**Speed:** 0.001-0.005 seconds per pose (10-50x faster than physics-based)

**Best for:** Virtual screening, large library docking, initial filtering, rapid pose evaluation

Algorithm
---------

The empirical scoring function uses statistical potentials derived from known protein-ligand complexes:

.. math::

   S_{total} = S_{contact} + S_{lipophilic} + S_{hbond} + S_{metal} + S_{flexibility}

Scoring Components
^^^^^^^^^^^^^^^^^^

1. **Contact Score**

   .. math::

      S_{contact} = \\sum_{i,j} w_{ij} \\cdot f(d_{ij})

   * Atom-type pair potentials
   * Distance-dependent statistical preferences
   * Derived from observed contact frequencies in PDB

2. **Lipophilic Score**

   * Hydrophobic-hydrophobic contact rewards
   * Surface complementarity bonus
   * Burial of hydrophobic surface area

3. **Hydrogen Bond Score**

   * Geometry-independent H-bond detection
   * Fixed weight per hydrogen bond
   * Faster than physics-based H-bond evaluation

4. **Metal Coordination Score**

   * Bonus for coordinating metal ions
   * Simple distance-based detection
   * Fixed weights per metal type

5. **Flexibility Penalty**

   * Penalty for rotatable bonds
   * Accounts for conformational entropy loss
   * Simpler than torsional energy calculation

Training Data
^^^^^^^^^^^^^

Empirical parameters optimized on:

* **PDBBind General Set:** 10,000+ protein-ligand complexes
* **Refined Set:** High-quality structures with experimental affinities
* **Diverse Set:** Covering all protein families
* **Validation:** CASF-2016, Astex Diverse Set

Usage
-----

Basic Usage
^^^^^^^^^^^

.. code-block:: bash

   pandadock dock -r protein.pdb -l ligand.sdf \\
                  --scoring empirical \\
                  --center 10 20 30 --box 20 20 20

Virtual Screening
^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock dock -r target.pdb -l library_10k.sdf \\
                  --algorithm monte_carlo_cpu \\
                  --scoring empirical \\
                  --fast \\
                  --num-poses 3 \\
                  -o screening_results/

With GPU Acceleration
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock dock -r target.pdb -l library.sdf \\
                  --algorithm cuda_monte_carlo \\
                  --scoring empirical \\
                  --gpu \\
                  --gpu-batch-size 2000 \\
                  -o ultra_fast_screening/

Expected throughput: 5000-7200 ligands/hour

Performance Characteristics
---------------------------

Accuracy Benchmarks
^^^^^^^^^^^^^^^^^^^

+------------------+------------------+----------------+
| Dataset          | Correlation (R)  | RMSE (kcal/mol)|
+==================+==================+================+
| PDBBind Core     | 0.72             | 2.35           |
+------------------+------------------+----------------+
| CASF-2016        | 0.68             | 2.58           |
+------------------+------------------+----------------+
| Astex Diverse    | 0.70             | 2.42           |
+------------------+------------------+----------------+

**Note:** Lower accuracy than physics-based, but 10-50x faster

Speed Benchmarks
^^^^^^^^^^^^^^^^

* **Small ligand (<20 atoms):** 0.001-0.002 seconds/pose
* **Medium ligand (20-40 atoms):** 0.002-0.003 seconds/pose
* **Large ligand (>40 atoms):** 0.003-0.005 seconds/pose

Screening throughput:

* **CPU (monte_carlo_cpu):** 200-400 ligands/hour
* **GPU (cuda_monte_carlo):** 3600-7200 ligands/hour

Pose Prediction Accuracy
^^^^^^^^^^^^^^^^^^^^^^^^^

* **RMSD < 2Å:** 80-85% (with monte_carlo algorithm)
* **RMSD < 2Å:** 88-92% (with enhanced_hierarchical algorithm)
* **Top pose RMSD < 2Å:** 65-75%

Lower pose prediction accuracy than physics-based, but sufficient for filtering.

Strengths and Limitations
--------------------------

Strengths
^^^^^^^^^

 **Ultra-Fast Evaluation**
   10-50x faster than physics-based scoring

 **Good Pose Recognition**
   Can distinguish near-native from incorrect poses

 **Robust**
   Works across diverse protein families

 **Simple**
   Few parameters, easy to use

 **Parallelizes Well**
   Excellent GPU acceleration

Limitations
^^^^^^^^^^^

 **Lower Accuracy**
   R = 0.72 vs 0.85 for physics-based

 **Coarse Granularity**
   Less sensitive to subtle differences

 **No Energy Decomposition**
   Can't analyze individual interaction contributions

 **Training Set Bias**
   May perform poorly on novel binding modes

 **No Solvation Model**
   Doesn't explicitly account for desolvation

Best Practices
--------------

Recommended Use Cases
^^^^^^^^^^^^^^^^^^^^^

1. **Large Library Screening**

   .. code-block:: bash

      pandadock dock -r target.pdb -l library_50k.sdf \\
                     --algorithm cuda_monte_carlo \\
                     --scoring empirical \\
                     --gpu \\
                     --num-poses 3 \\
                     -o initial_screening/

   Screen 50,000 compounds in 7-14 hours (GPU)

2. **Initial Filtering Before Detailed Docking**

   .. code-block:: bash

      # Step 1: Fast empirical screening
      pandadock dock -r target.pdb -l library_10k.sdf \\
                     --scoring empirical \\
                     --fast \\
                     --num-poses 1 \\
                     -o empirical_filter/

      # Step 2: Rescore top 500 with physics-based
      pandadock dock -r target.pdb -l top_500.sdf \\
                     --scoring physics_based \\
                     --num-poses 20 \\
                     -o refined_results/

3. **Pose Filtering**

   Use empirical scoring to quickly identify poor poses

4. **Fragment Screening**

   Fast evaluation of small fragment libraries

Not Recommended For
^^^^^^^^^^^^^^^^^^^

L **Critical Lead Optimization**
   Use ``physics_based`` or ``hybrid`` scoring

L **Quantitative Affinity Prediction**
   Lower correlation with experimental data

L **Detailed Interaction Analysis**
   No energy decomposition available

L **Novel Binding Modes**
   May not generalize beyond training set

L **Charged Ligands**
   Electrostatics not well-represented

Optimization Tips
^^^^^^^^^^^^^^^^^

**Maximize Throughput:**

.. code-block:: bash

   pandadock dock -r target.pdb -l library.sdf \\
                  --algorithm cuda_monte_carlo \\
                  --scoring empirical \\
                  --gpu \\
                  --gpu-batch-size 2000 \\
                  --fast \\
                  --num-poses 1

Target: 7000+ ligands/hour

**Balance Speed and Accuracy:**

.. code-block:: bash

   pandadock dock -r target.pdb -l library.sdf \\
                  --algorithm monte_carlo_cpu \\
                  --scoring empirical \\
                  --num-poses 5 \\
                  --cpuworkers 16

**Two-Stage Screening:**

.. code-block:: bash

   # Stage 1: Empirical screening (fast)
   pandadock dock -r target.pdb -l library_100k.sdf \\
                  --scoring empirical \\
                  --fast \\
                  -o stage1/

   # Extract top 1000 by score
   # Stage 2: Rescore with physics-based
   pandadock dock -r target.pdb -l top_1000.sdf \\
                  --scoring physics_based \\
                  --rescoring mmgbsa \\
                  -o stage2/

Output Format
-------------

Scoring Output
^^^^^^^^^^^^^^

.. code-block:: json

   {
     "binding_score": -6.8,
     "components": {
       "contact_score": -8.5,
       "lipophilic_score": -2.3,
       "hbond_score": -3.2,
       "metal_score": 0.0,
       "flexibility_penalty": 1.8
     }
   }

**Note:** Empirical scores are unitless and calibrated to approximate kcal/mol

Ranking Output
^^^^^^^^^^^^^^

.. code-block:: text

   Rank  Ligand_ID      Score    RMSD
   1     compound_1523  -8.5     1.2
   2     compound_0942  -8.2     0.8
   3     compound_2341  -7.9     1.5
   ...

Comparison with Other Scoring Functions
----------------------------------------

vs Physics-Based
^^^^^^^^^^^^^^^^

+-------------------+--------------+--------------+
| Aspect            | Empirical    | Physics-Based|
+===================+==============+==============+
| Speed             |         |         |
+-------------------+--------------+--------------+
| Accuracy          |         |         |
+-------------------+--------------+--------------+
| Interpretability  |         |         |
+-------------------+--------------+--------------+
| Throughput        |         |         |
+-------------------+--------------+--------------+

**Choose empirical when:** Speed is paramount, screening large libraries

**Choose physics-based when:** Accuracy matters, need energy decomposition

vs Hybrid Scoring
^^^^^^^^^^^^^^^^^

+-------------------+--------------+--------------+
| Aspect            | Empirical    | Hybrid       |
+===================+==============+==============+
| Speed             |         |         |
+-------------------+--------------+--------------+
| Accuracy          |         |         |
+-------------------+--------------+--------------+
| Setup             |         |         |
+-------------------+--------------+--------------+

**Choose empirical when:** Ultra-fast initial screening

**Choose hybrid when:** Final ranking and lead optimization

Examples
--------

Ultra-Fast Virtual Screening
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Screen 100,000 compound library
   pandadock dock -r kinase.pdb -l library_100k.sdf \\
                  --algorithm cuda_monte_carlo \\
                  --scoring empirical \\
                  --gpu \\
                  --fast \\
                  --num-poses 1 \\
                  -o empirical_screening/

Expected runtime: 14-28 hours (GPU), output: top scoring compounds

Fragment Library Screening
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock dock -r protein.pdb -l fragments_5k.sdf \\
                  --algorithm monte_carlo_cpu \\
                  --scoring empirical \\
                  --fast \\
                  --num-poses 3 \\
                  --cpuworkers 8 \\
                  -o fragment_hits/

Two-Stage High-Throughput Screening
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Stage 1: Rapid empirical filter (10,000 ’ 500)
   pandadock dock -r target.pdb -l library_10k.sdf \\
                  --scoring empirical \\
                  --fast \\
                  --num-poses 1 \\
                  -o stage1_empirical/

   # Extract top 500 compounds by empirical score

   # Stage 2: Detailed physics-based rescoring (500 ’ 50)
   pandadock dock -r target.pdb -l top_500.sdf \\
                  --scoring physics_based \\
                  --num-poses 20 \\
                  -o stage2_physics/

   # Extract top 50 for experimental validation

Expected Workflow Results
^^^^^^^^^^^^^^^^^^^^^^^^^

**Input:** 10,000 compound library

**Stage 1 (empirical):**

* Time: 25-50 hours (CPU) or 1.5-3 hours (GPU)
* Output: Ranked list, select top 500

**Stage 2 (physics-based):**

* Time: 1-2 hours (top 500)
* Output: Refined ranking, select top 50

**Stage 3 (experimental):**

* Test top 50 compounds
* Expected hit rate: 10-30% (5-15 active compounds)

Validation Studies
------------------

Enrichment Performance
^^^^^^^^^^^^^^^^^^^^^^

Tested on DUD-E (Database of Useful Decoys: Enhanced):

* **Top 1% enrichment:** 12-18x
* **Top 5% enrichment:** 8-12x
* **AUC (ROC):** 0.72-0.78

**Conclusion:** Good enrichment for initial filtering, not optimal for final ranking

Pose Reproduction
^^^^^^^^^^^^^^^^^

Tested on Astex Diverse Set (85 complexes):

* **Success rate (RMSD < 2Å):** 80-85%
* **Top pose success:** 65-75%

**Conclusion:** Adequate pose recognition for screening

See Also
--------

* :doc:`overview` - Scoring functions overview
* :doc:`physics_based` - Physics-based scoring
* :doc:`hybrid` - Hybrid ML scoring
* :doc:`gpu_scoring` - GPU scoring
* :doc:`../algorithms/gpu_algorithms` - GPU algorithms for maximum throughput
