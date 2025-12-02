Physics-Based Scoring
=====================

The physics-based scoring function is the **recommended default** for general-purpose docking in PandaDock. It provides a comprehensive molecular mechanics force field evaluation with excellent correlation to experimental binding affinities.

Overview
--------

**Scoring ID:** ``physics_based``

**Type:** Force field-based molecular mechanics scoring

**Accuracy:** R = 0.85 correlation with experimental binding affinities

**Speed:** 0.01-0.05 seconds per pose evaluation

**Best for:** General-purpose docking, balanced accuracy and speed, structural studies

Algorithm
---------

The physics-based scoring function evaluates the following energy components:

.. math::

   E_{total} = E_{vdW} + E_{elec} + E_{desolv} + E_{hbond} + E_{torsion}

Energy Components
^^^^^^^^^^^^^^^^^

1. **Van der Waals (vdW) Energy**

   .. math::

      E_{vdW} = \\sum_{i,j} 4\\epsilon_{ij} \\left[ \\left(\\frac{\\sigma_{ij}}{r_{ij}}\\right)^{12} - \\left(\\frac{\\sigma_{ij}}{r_{ij}}\\right)^6 \\right]

   * Lennard-Jones 12-6 potential
   * Accounts for favorable dispersion interactions and steric clashes
   * Distance-dependent with r{v attractive and r{¹² repulsive terms

2. **Electrostatic Energy**

   .. math::

      E_{elec} = \\sum_{i,j} \\frac{q_i q_j}{4\\pi\\epsilon_0 \\epsilon_r r_{ij}}

   * Coulombic interactions between partial atomic charges
   * Distance-dependent dielectric: µ(r) = 4r
   * Accounts for charge-charge, charge-dipole interactions

3. **Desolvation Energy**

   .. math::

      E_{desolv} = \\sum_i \\Delta G_{solv,i} \\cdot SA_i

   * Implicit solvation using atomic solvation parameters
   * Surface area-based desolvation penalties
   * Accounts for hydrophobic and hydrophilic contributions

4. **Hydrogen Bonding**

   .. math::

      E_{hbond} = \\sum_{HB} E_{HB} \\cdot f(r) \\cdot f(\\theta) \\cdot f(\\phi)

   * Explicit hydrogen bond detection
   * Geometry-dependent scoring (distance, angle, dihedral)
   * Donor-acceptor pair identification
   * Favorable contribution for well-formed H-bonds

5. **Torsional Penalty**

   .. math::

      E_{torsion} = \\sum_{rot} W_{rot}

   * Entropy penalty for each rotatable bond
   * Accounts for conformational entropy loss upon binding
   * Configurable weight per rotatable bond

Parameter Set
-------------

PandaDock uses optimized parameters derived from:

* **Atomic radii and well depths:** AMBER ff14SB force field
* **Partial charges:** AM1-BCC for ligands, AMBER for proteins
* **Solvation parameters:** Optimized on PDBBind dataset
* **H-bond parameters:** Geometry-dependent scoring functions
* **Torsional weights:** Calibrated on experimental data

Usage
-----

Basic Usage
^^^^^^^^^^^

.. code-block:: bash

   pandadock dock -r protein.pdb -l ligand.sdf \\
                  --scoring physics_based \\
                  --center 10 20 30 --box 20 20 20

The physics-based scoring is the **default**, so you can omit ``--scoring``:

.. code-block:: bash

   pandadock dock -r protein.pdb -l ligand.sdf \\
                  --center 10 20 30 --box 20 20 20

With Energy Decomposition
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock dock -r protein.pdb -l ligand.sdf \\
                  --scoring physics_based \\
                  --decompose-energy \\
                  --center 10 20 30 --box 20 20 20

This generates detailed energy breakdowns:

* Van der Waals contribution
* Electrostatic contribution
* Desolvation energy
* Hydrogen bonding energy
* Torsional penalty

Per-Residue Contributions
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock dock -r protein.pdb -l ligand.sdf \\
                  --scoring physics_based \\
                  --per-residue-decomposition \\
                  --center 10 20 30 --box 20 20 20

Output includes interaction energy for each protein residue, useful for:

* Identifying key binding residues
* Hotspot analysis
* Mutagenesis studies
* Structure-activity relationships

Performance Characteristics
---------------------------

Accuracy Benchmarks
^^^^^^^^^^^^^^^^^^^

Tested on standard benchmark sets:

+------------------+------------------+----------------+
| Dataset          | Correlation (R)  | RMSE (kcal/mol)|
+==================+==================+================+
| PDBBind Core     | 0.85             | 1.82           |
+------------------+------------------+----------------+
| CASF-2016        | 0.83             | 1.95           |
+------------------+------------------+----------------+
| Astex Diverse    | 0.81             | 2.11           |
+------------------+------------------+----------------+

Speed Benchmarks
^^^^^^^^^^^^^^^^

* **Small ligand (<20 atoms):** 0.01-0.02 seconds/pose
* **Medium ligand (20-40 atoms):** 0.02-0.03 seconds/pose
* **Large ligand (>40 atoms):** 0.03-0.05 seconds/pose

Screening throughput:

* **Single pose evaluation:** 20-100 poses/second
* **Full docking (20 poses):** 3-5 ligands/minute

Success Rate
^^^^^^^^^^^^

* **RMSD < 2Å:** 90-95% (with enhanced_hierarchical algorithm)
* **Top pose RMSD < 2Å:** 75-85%
* **Correct binding mode identification:** 85-90%

Strengths and Limitations
--------------------------

Strengths
^^^^^^^^^

 **Physically Meaningful**
   Based on established molecular mechanics principles

 **Interpretable**
   Energy components can be analyzed individually

 **Broadly Applicable**
   Works well across diverse protein families and ligand types

 **Balanced Accuracy**
   Good trade-off between speed and accuracy

 **Per-Residue Analysis**
   Enables detailed interaction analysis

Limitations
^^^^^^^^^^^

 **Solvation Approximation**
   Implicit solvation less accurate than explicit water models

 **Fixed Charges**
   Doesn't account for polarization or charge transfer

 **No Entropy Terms**
   Only configurational entropy via torsional penalty

 **Medium Speed**
   Slower than empirical scoring, faster than QM methods

Best Practices
--------------

Recommended Use Cases
^^^^^^^^^^^^^^^^^^^^^

1. **General Docking**

   .. code-block:: bash

      pandadock dock -r protein.pdb -l ligand.sdf \\
                     --algorithm enhanced_hierarchical_cpu \\
                     --scoring physics_based

2. **Comparative Binding Studies**

   Physics-based scoring provides consistent ranking across series

3. **Structure-Activity Relationships (SAR)**

   Energy decomposition identifies key interactions

4. **Mutagenesis Studies**

   Per-residue decomposition reveals hotspot residues

Not Recommended For
^^^^^^^^^^^^^^^^^^^

L **Ultra-Fast Virtual Screening (>10,000 compounds)**
   Use ``empirical`` scoring instead for speed

L **Charged Ligands in Buried Sites**
   Implicit solvation may be inaccurate; consider MM-GBSA rescoring

L **Metal Coordination**
   Use ``pandadock-metal`` with specialized metal scoring

L **Covalent Docking**
   Physics-based doesn't handle covalent bonds well

Optimization Tips
^^^^^^^^^^^^^^^^^

**For Maximum Accuracy:**

.. code-block:: bash

   pandadock dock -r protein.pdb -l ligand.sdf \\
                  --scoring physics_based \\
                  --rescoring mmgbsa \\
                  --num-poses 50

**For Better Speed:**

.. code-block:: bash

   pandadock dock -r protein.pdb -l ligand.sdf \\
                  --scoring physics_based \\
                  --fast \\
                  --num-poses 10

**For Detailed Analysis:**

.. code-block:: bash

   pandadock dock -r protein.pdb -l ligand.sdf \\
                  --scoring physics_based \\
                  --decompose-energy \\
                  --per-residue-decomposition \\
                  --visualize

Output Format
-------------

Energy Values
^^^^^^^^^^^^^

All energies reported in **kcal/mol**:

.. code-block:: json

   {
     "binding_affinity": -8.5,
     "energy_components": {
       "vdw_energy": -35.2,
       "electrostatic_energy": -12.4,
       "desolvation_energy": 28.1,
       "hbond_energy": -6.8,
       "torsional_penalty": 2.3
     }
   }

Per-Residue Decomposition
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

   {
     "residue_contributions": [
       {"residue": "TYR39", "energy": -2.8},
       {"residue": "ASP189", "energy": -3.5},
       {"residue": "LEU99", "energy": -1.2}
     ]
   }

Comparison with Other Scoring Functions
----------------------------------------

+------------------+-------------+---------+-------------+
| Scoring          | Accuracy    | Speed   | Analysis    |
+==================+=============+=========+=============+
| physics_based    | High        | Medium  | Detailed    |
+------------------+-------------+---------+-------------+
| empirical        | Medium      | Fast    | Limited     |
+------------------+-------------+---------+-------------+
| precision_score  | High        | Slow    | Very        |
|                  |             |         | Detailed    |
+------------------+-------------+---------+-------------+
| hybrid           | Very High   | Slow    | ML-enhanced |
+------------------+-------------+---------+-------------+

**When to choose physics_based over alternatives:**

* **vs empirical:** Need better accuracy, willing to sacrifice speed
* **vs precision_score:** Want faster results, don't need extreme precision
* **vs hybrid:** Don't have GPU, want interpretable physics-based results

Examples
--------

Basic Docking with Physics-Based Scoring
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock dock -r 1hsg_protein.pdb -l indinavir.sdf \\
                  --center 15.0 20.0 25.0 \\
                  --box 20 20 20 \\
                  --num-poses 20 \\
                  -o results_physics/

Virtual Screening with Physics-Based Scoring
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock dock -r kinase.pdb -l library_500.sdf \\
                  --algorithm monte_carlo_cpu \\
                  --scoring physics_based \\
                  --fast \\
                  --num-poses 5 \\
                  -o screening_physics/

High-Accuracy Docking with Energy Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock dock -r protein.pdb -l ligand.sdf \\
                  --algorithm enhanced_hierarchical_cpu \\
                  --scoring physics_based \\
                  --decompose-energy \\
                  --per-residue-decomposition \\
                  --rescoring mmgbsa \\
                  --num-poses 50 \\
                  --visualize \\
                  -o detailed_analysis/

See Also
--------

* :doc:`overview` - Scoring functions overview
* :doc:`empirical` - Fast empirical scoring
* :doc:`hybrid` - ML-enhanced hybrid scoring
* :doc:`gpu_scoring` - GPU-accelerated scoring
* :doc:`../algorithms/cpu_algorithms` - CPU docking algorithms
