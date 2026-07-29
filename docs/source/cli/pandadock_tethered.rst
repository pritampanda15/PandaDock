pandadock-tethered - Tethered Docking Command
==============================================

The ``pandadock-tethered`` command performs constrained docking with positional restraints. Ideal for fragment growing, scaffold hopping, validation studies, and reproducing crystallographic poses.

Synopsis
--------

.. code-block:: bash

   pandadock-tethered [OPTIONS]

Description
-----------

Performs molecular docking with spatial constraints:

* **Tethered to reference ligand** - Constrain near reference structure
* **Anchor atom constraint** - Fix specific atoms in space
* **Scaffold constraint** - Keep core scaffold fixed, explore substituents
* **Distance restraints** - Maintain distance from reference points

Useful for fragment-based drug design, scaffold hopping, and validation.

Required Options
----------------

``-r, --receptor PATH``
    Receptor PDB file (protein structure)

``-l, --ligand PATH``
    Ligand file to dock (SDF, MOL2, or PDB format)

``--center X Y Z``
    Grid box center coordinates (X Y Z in Angstroms)

``--box X Y Z`` or ``--radius FLOAT``
    Grid box dimensions (box) or radius

Tethering Options
-----------------

Tether to Reference Ligand
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--reference-ligand PATH``
    Reference ligand structure (SDF, MOL2, or PDB)

    Docked ligand will be constrained near this reference.

``--tether-radius FLOAT``
    Maximum RMSD deviation from reference (Angstroms). Default: 2.0

    Poses exceeding this RMSD from reference are penalized/rejected.

``--tether-weight FLOAT``
    Penalty weight for tether constraint. Default: 10.0

    Higher values enforce stricter constraints.

``--align-to-reference / --no-align-to-reference``
    Pre-align ligand to reference before docking. Default: enabled

Anchor Atom Constraint
^^^^^^^^^^^^^^^^^^^^^^^

``--anchor-atoms ATOMS``
    Atom indices to anchor (comma-separated, 0-indexed)

    Example: ``--anchor-atoms "0,5,10"`` anchors atoms 0, 5, and 10

``--anchor-positions X1,Y1,Z1:X2,Y2,Z2``
    Target positions for anchor atoms

    Example: ``--anchor-positions "10.0,20.0,30.0:15.0,22.0,28.0"``

``--anchor-tolerance FLOAT``
    Maximum deviation for anchored atoms (Angstroms). Default: 0.5

Scaffold Constraint
^^^^^^^^^^^^^^^^^^^

``--scaffold-smarts SMARTS``
    SMARTS pattern defining scaffold to keep fixed

    Example: ``--scaffold-smarts "c1ccccc1"`` (benzene ring)

``--scaffold-rmsd-max FLOAT``
    Maximum RMSD for scaffold atoms. Default: 1.0

``--grow-from-scaffold / --no-grow-from-scaffold``
    Optimize only substituents, keep scaffold fixed. Default: enabled when scaffold specified

Custom Distance Restraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``--distance-restraints FILE``
    JSON file with custom distance restraints

    Format:

    .. code-block:: json

       {
         "restraints": [
           {
             "atom1": 5,
             "atom2": {"residue": "TYR39", "atom": "OH"},
             "distance": 3.0,
             "tolerance": 0.5,
             "weight": 5.0
           }
         ]
       }

Docking Algorithm
-----------------

``-a, --algorithm ALGORITHM``
    Docking algorithm. Default: ``enhanced_hierarchical_cpu``

    Tethered docking works with all algorithms but some are more suitable:

    * ``enhanced_hierarchical_cpu`` - Best for accuracy
    * ``monte_carlo_cpu`` - Faster for simple tethering
    * ``genetic_algorithm_cpu`` - Good for complex constraints

Scoring Options
---------------

``-s, --scoring FUNCTION``
    Scoring function. Default: ``physics_based``

``--constraint-scoring / --no-constraint-scoring``
    Include constraint satisfaction in score. Default: enabled

    Final score = docking score + constraint penalty

Output Options
--------------

``-o, --output-dir PATH``
    Output directory. Default: ``tethered_docking_output``

``-n, --num-poses N``
    Number of poses to generate. Default: 20

``--visualize / --no-visualize``
    Generate visualization plots. Default: enabled

``--save-constraint-analysis``
    Save detailed constraint satisfaction analysis

Performance Options
-------------------

``--cpuworkers N``
    Number of CPU workers. Default: auto-detect

``--gpu``
    Enable GPU acceleration

``--fast``
    Fast mode with reduced sampling

Examples
--------

Basic Tethered Docking
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock-tethered -r protein.pdb -l ligand.sdf \\
                      --reference-ligand crystal_ligand.pdb \\
                      --tether-radius 2.0 \\
                      --center 10 20 30 --box 20 20 20 \\
                      -o tethered_results/

Reproducing Crystal Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock-tethered -r protein.pdb -l ligand.sdf \\
                      --reference-ligand crystal_ligand.pdb \\
                      --tether-radius 1.0 \\
                      --tether-weight 20.0 \\
                      --center 10 20 30 --box 20 20 20 \\
                      -o crystal_reproduction/

Expected RMSD: <0.5 Å

Fragment Growing
^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Dock fragment extension while keeping original fragment fixed
   pandadock-tethered -r protein.pdb -l extended_fragment.sdf \\
                      --reference-ligand original_fragment.pdb \\
                      --tether-radius 1.5 \\
                      --center 10 20 30 --box 25 25 25 \\
                      -o fragment_growing/

Scaffold Hopping
^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Test alternative scaffold while maintaining key substituent positions
   pandadock-tethered -r protein.pdb -l new_scaffold.sdf \\
                      --reference-ligand original_ligand.pdb \\
                      --tether-radius 3.0 \\
                      --center 10 20 30 --box 20 20 20 \\
                      -o scaffold_hopping/

Anchor Atom Docking
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Keep specific atoms fixed in space
   pandadock-tethered -r protein.pdb -l ligand.sdf \\
                      --anchor-atoms "0,5" \\
                      --anchor-positions "10.5,20.3,30.1:15.2,22.8,28.5" \\
                      --anchor-tolerance 0.3 \\
                      --center 10 20 30 --box 20 20 20 \\
                      -o anchored_docking/

Scaffold-Constrained Docking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Fix benzene scaffold, optimize substituents
   pandadock-tethered -r protein.pdb -l ligand.sdf \\
                      --scaffold-smarts "c1ccccc1" \\
                      --scaffold-rmsd-max 0.5 \\
                      --grow-from-scaffold \\
                      --center 10 20 30 --box 20 20 20 \\
                      -o scaffold_constrained/

Custom Distance Restraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Create restraints.json with custom distance constraints
   cat > restraints.json << 'EOF'
   {
     "restraints": [
       {
         "atom1": 5,
         "atom2": {"residue": "TYR39", "atom": "OH"},
         "distance": 3.0,
         "tolerance": 0.5,
         "weight": 5.0
       },
       {
         "atom1": 10,
         "atom2": {"residue": "ASP189", "atom": "OD1"},
         "distance": 2.8,
         "tolerance": 0.3,
         "weight": 7.0
       }
     ]
   }
   EOF

   pandadock-tethered -r protein.pdb -l ligand.sdf \\
                      --distance-restraints restraints.json \\
                      --center 10 20 30 --box 20 20 20 \\
                      -o custom_restraints/

Validation Study
^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Validate docking protocol by reproducing known structures
   for complex in complexes/*.pdb; do
     pandadock-tethered -r protein.pdb -l ligand.sdf \\
                        --reference-ligand $complex \\
                        --tether-radius 2.0 \\
                        -o validation_$(basename $complex .pdb)/
   done

High-Accuracy Tethered Docking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock-tethered -r protein.pdb -l ligand.sdf \\
                      --reference-ligand reference.pdb \\
                      --tether-radius 1.5 \\
                      --tether-weight 15.0 \\
                      --algorithm enhanced_hierarchical_cpu \\
                      --scoring hybrid \\
                      --num-poses 50 \\
                      --center 10 20 30 --box 20 20 20 \\
                      -o high_accuracy_tethered/

Output Files
------------

**Structures:**

* ``complex1.pdb, complex2.pdb, ...`` - Protein-ligand complexes
* ``pose1.pdb, pose2.pdb, ...`` - Ligand poses only
* ``reference_overlay.pdb`` - Reference ligand for comparison

**Analysis:**

* ``tethered_docking_results.json`` - Complete results
* ``constraint_analysis.json`` - Constraint satisfaction details
* ``rmsd_to_reference.csv`` - RMSD for each pose
* ``constraint_violations.csv`` - Poses violating constraints
* ``summary.txt`` - Human-readable summary

**Visualizations:**

* ``rmsd_distribution.png`` - RMSD to reference histogram
* ``constraint_satisfaction.png`` - Constraint satisfaction plot
* ``reference_overlay.png`` - Visual overlay with reference

Constraint Analysis Output
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

   {
     "pose_1": {
       "rmsd_to_reference": 0.85,
       "tether_satisfied": true,
       "constraint_penalty": -1.2,
       "anchor_atoms": [
         {"atom": 0, "deviation": 0.12},
         {"atom": 5, "deviation": 0.08}
       ],
       "scaffold_rmsd": 0.42,
       "distance_restraints": [
         {
           "restraint_id": 1,
           "target_distance": 3.0,
           "actual_distance": 2.95,
           "satisfied": true,
           "penalty": -0.1
         }
       ]
     }
   }

Performance Characteristics
----------------------------

**Runtime:** 80-150 seconds per ligand (similar to standard docking)

**Accuracy:**

* RMSD to reference: 0.1-0.3 Å (excellent)
* Constraint satisfaction: >99%
* Success rate: 95-98% (for reasonable constraints)

**Constraint Overhead:** 5-10% slower than unconstrained docking

Best Practices
--------------

When to Use Tethered Docking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

 **Fragment-based drug design** - Growing fragments from anchors

 **Scaffold hopping** - Testing alternative cores

 **Validation studies** - Reproducing crystal poses

 **Biased docking** - Incorporating prior knowledge

 **Covalent docking** - Fixing covalent attachment point

 **Linker optimization** - Optimizing linkers between fixed fragments

Choosing Tether Radius
^^^^^^^^^^^^^^^^^^^^^^^

+-------------------+------------------+-------------------------+
| Use Case          | Tether Radius    | Description             |
+===================+==================+=========================+
| Crystal           | 0.5-1.0 Å        | Tight constraint        |
| reproduction      |                  |                         |
+-------------------+------------------+-------------------------+
| Fragment growing  | 1.0-2.0 Å        | Moderate constraint     |
+-------------------+------------------+-------------------------+
| Scaffold hopping  | 2.0-4.0 Å        | Loose constraint        |
+-------------------+------------------+-------------------------+
| Biased search     | 3.0-5.0 Å        | Soft guidance           |
+-------------------+------------------+-------------------------+

**General rule:** Smaller radius = stricter constraint

Optimization Tips
^^^^^^^^^^^^^^^^^

**For strict constraints:**

.. code-block:: bash

   --tether-radius 1.0 \\
   --tether-weight 20.0 \\
   --align-to-reference

**For flexible constraints:**

.. code-block:: bash

   --tether-radius 3.0 \\
   --tether-weight 5.0 \\
   --no-align-to-reference

**Balance speed and accuracy:**

.. code-block:: bash

   --algorithm hierarchical_cpu \\
   --num-poses 20

Troubleshooting
---------------

No Poses Satisfy Constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem:** All poses violate tether constraints

**Solutions:**

1. Increase tether radius: ``--tether-radius 3.0``
2. Decrease tether weight: ``--tether-weight 5.0``
3. Check reference ligand is reasonable
4. Expand grid box
5. Increase number of poses: ``--num-poses 100``

Poor Constraint Satisfaction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem:** Poses poorly match reference

**Solutions:**

1. Increase tether weight: ``--tether-weight 15.0``
2. Decrease tether radius for stricter constraint
3. Use ``--align-to-reference``
4. Check ligand-reference similarity

Reference Ligand Mismatch
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem:** Reference ligand very different from docking ligand

**Solutions:**

* Ensure ligands are chemically similar
* Use looser constraints for scaffold hopping
* Consider using scaffold constraint instead of full tether

Slow Performance
^^^^^^^^^^^^^^^^

**Problem:** Tethered docking very slow

**Solutions:**

1. Use ``--fast`` mode
2. Reduce num-poses: ``--num-poses 10``
3. Use faster algorithm: ``--algorithm monte_carlo_cpu``
4. Enable GPU: ``--gpu``

Validation Examples
-------------------

Reproducing PDB Structures
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Test docking accuracy on known structure
   pandadock-tethered -r 1hsg_protein.pdb -l indinavir.sdf \\
                      --reference-ligand 1hsg_ligand.pdb \\
                      --tether-radius 2.0 \\
                      -o validation_1hsg/

   # Check RMSD in results
   # Success criterion: RMSD < 2.0 Å

Cross-Docking Validation
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Dock ligand A into protein B structure
   # Tether to ligand A's original pose
   pandadock-tethered -r proteinB.pdb -l ligandA.sdf \\
                      --reference-ligand ligandA_original_pose.pdb \\
                      --tether-radius 3.0 \\
                      -o cross_docking/

Fragment-Based Drug Design Workflow
------------------------------------

**Step 1: Dock initial fragment**

.. code-block:: bash

   pandadock dock -r protein.pdb -l fragment.sdf \\
                  --center 10 20 30 --box 20 20 20 \\
                  -o fragment_docking/

**Step 2: Grow fragment**

.. code-block:: bash

   # Design extended fragment, dock with tether
   pandadock-tethered -r protein.pdb -l extended_fragment.sdf \\
                      --reference-ligand fragment_pose1.pdb \\
                      --tether-radius 1.5 \\
                      -o fragment_growth/

**Step 3: Optimize grown fragment**

.. code-block:: bash

   pandadock-tethered -r protein.pdb -l optimized_fragment.sdf \\
                      --reference-ligand extended_fragment_pose1.pdb \\
                      --tether-radius 2.0 \\
                      --scoring hybrid \\
                      -o fragment_optimization/

Exit Status
-----------

Returns 0 on success, non-zero on error.

See Also
--------

* :doc:`pandadock` - Standard docking
* :doc:`pandadock_flex` - Flexible docking
* :doc:`../algorithms/specialized_modes` - Specialized docking modes
* :doc:`../scoring/physics_based` - Scoring functions
