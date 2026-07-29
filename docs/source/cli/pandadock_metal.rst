pandadock-metal - Metal Docking Command
========================================

The ``pandadock-metal`` command performs specialized docking for metalloproteins with explicit metal coordination geometry constraints. Essential for accurately docking ligands to metal-containing active sites.

Synopsis
--------

.. code-block:: bash

   pandadock-metal [OPTIONS]

Description
-----------

Performs molecular docking with metal coordination constraints:

* Explicit metal coordination geometry (tetrahedral, octahedral, square planar)
* Donor atom preferences (N, O, S)
* Bond length and angle restraints
* Charge-transfer interactions
* Chelation effects

Supports Zn²z, Fe²z/³z, Mg²z, Ca²z, Mn²z, Cu²z, Ni²z, and Co²z.

Required Options
----------------

``-r, --receptor PATH``
    Receptor PDB file (metalloprotein structure)

``-l, --ligand PATH``
    Ligand file (SDF, MOL2, or PDB format)

``--metal-type METAL``
    Metal element symbol

    Supported: ``ZN``, ``FE``, ``MG``, ``CA``, ``MN``, ``CU``, ``NI``, ``CO``

``--metal-residue RESIDUE_ID``
    Metal residue ID (e.g., "A:201" for chain A, residue 201)

``--center X Y Z``
    Grid box center coordinates (X Y Z in Angstroms)

``--box X Y Z``
    Grid box dimensions (X Y Z in Angstroms)

Metal Coordination Options
---------------------------

``--coordination-geometry GEOMETRY``
    Coordination geometry. Default: auto-detect based on metal type

    Options:

    * ``tetrahedral`` - 4 coordination sites (109.5° angles)
    * ``octahedral`` - 6 coordination sites (90° angles)
    * ``square_planar`` - 4 coordination sites (90°/180° angles)
    * ``trigonal_bipyramidal`` - 5 coordination sites
    * ``irregular`` - No strict geometry (Ca²z, flexible metals)

``--coordination-number N``
    Number of coordination sites. Default: auto from geometry

``--donor-atoms ATOMS``
    Allowed donor atom types (comma-separated). Default: ``N,O,S``

    Examples: ``N,O`` (exclude sulfur), ``N,O,S,CL`` (include chloride)

``--max-coord-bonds N``
    Maximum ligand-metal coordination bonds. Default: 3

``--coord-bond-length-min FLOAT``
    Minimum metal-donor distance (Å). Default: 1.8

``--coord-bond-length-max FLOAT``
    Maximum metal-donor distance (Å). Default: 2.8

``--coord-angle-tolerance FLOAT``
    Coordination angle tolerance (degrees). Default: 20.0

Metal-Specific Parameters
--------------------------

**Zinc (Zn²z):**

* Default geometry: ``tetrahedral``
* Bond length range: 1.9-2.5 Å
* Preferred donors: N (His), S (Cys), O (Asp, Glu, water)

**Iron (Fe²z/Fe³z):**

* Default geometry: ``octahedral``
* Bond length range: 1.9-2.4 Å
* Preferred donors: N (His), O (Asp, Glu, water), S (Cys)

**Magnesium (Mg²z):**

* Default geometry: ``octahedral``
* Bond length range: 2.0-2.3 Å
* Preferred donors: O (phosphate, carboxylate, water)

**Calcium (Ca²z):**

* Default geometry: ``irregular``
* Bond length range: 2.2-2.8 Å
* Coordination number: 6-8 (flexible)

**Copper (Cu²z):**

* Default geometry: ``square_planar``
* Bond length range: 1.9-2.3 Å
* Preferred donors: N (His), S (Cys/Met), O

Docking Algorithm
-----------------

``-a, --algorithm ALGORITHM``
    Docking algorithm. Default: ``enhanced_hierarchical_cpu``

    Recommended for metal docking:

    * ``enhanced_hierarchical_cpu`` - Best accuracy
    * ``genetic_algorithm_cpu`` - Good for complex coordination
    * ``enhanced_hierarchical_gpu`` - GPU acceleration

Scoring Options
---------------

``-s, --scoring FUNCTION``
    Scoring function. Default: ``physics_based``

    Metal-compatible scoring: ``physics_based``, ``precision_score``, ``hybrid``

``--metal-coordination-weight FLOAT``
    Weight for metal coordination energy. Default: 5.0

    Higher values enforce stricter coordination geometry.

``--chelate-bonus FLOAT``
    Energy bonus for bidentate/chelating interactions. Default: -2.0 kcal/mol

Output Options
--------------

``-o, --output-dir PATH``
    Output directory. Default: ``metal_docking_output``

``-n, --num-poses N``
    Number of poses to generate. Default: 20

``--visualize / --no-visualize``
    Generate visualization plots. Default: enabled

``--save-coordination-analysis``
    Save detailed metal coordination analysis

Performance Options
-------------------

``--cpuworkers N``
    Number of CPU workers. Default: auto-detect

``--gpu``
    Enable GPU acceleration

Examples
--------

Basic Zinc Metalloprotein Docking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock-metal -r mmp.pdb -l inhibitor.sdf \\
                   --metal-type ZN \\
                   --metal-residue "A:201" \\
                   --center 10 20 30 --box 20 20 20 \\
                   -o mmp_docking/

Carbonic Anhydrase (Zinc)
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock-metal -r carbonic_anhydrase.pdb -l ligand.sdf \\
                   --metal-type ZN \\
                   --metal-residue "A:263" \\
                   --coordination-geometry tetrahedral \\
                   --donor-atoms N,O,S \\
                   --center 15 22 18 --box 18 18 18 \\
                   -o ca_docking/

Iron-Containing Enzyme
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock-metal -r cytochrome.pdb -l substrate.sdf \\
                   --metal-type FE \\
                   --metal-residue "A:150" \\
                   --coordination-geometry octahedral \\
                   --coord-bond-length-max 2.4 \\
                   --center 12 18 25 --box 20 20 20 \\
                   -o fe_docking/

Calcium-Binding Protein
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock-metal -r calmodulin.pdb -l ligand.sdf \\
                   --metal-type CA \\
                   --metal-residue "A:100" \\
                   --coordination-geometry irregular \\
                   --coordination-number 7 \\
                   --donor-atoms O \\
                   --center 20 15 22 --box 22 22 22 \\
                   -o ca_binding/

Bidentate Chelator Docking
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock-metal -r metalloprotein.pdb -l chelator.sdf \\
                   --metal-type ZN \\
                   --metal-residue "A:201" \\
                   --max-coord-bonds 2 \\
                   --chelate-bonus -3.0 \\
                   --center 10 20 30 --box 20 20 20 \\
                   -o chelator_docking/

High-Accuracy Metal Docking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock-metal -r protein.pdb -l ligand.sdf \\
                   --metal-type ZN \\
                   --metal-residue "A:201" \\
                   --algorithm enhanced_hierarchical_cpu \\
                   --scoring hybrid \\
                   --metal-coordination-weight 7.0 \\
                   --coord-angle-tolerance 15.0 \\
                   --num-poses 50 \\
                   --center 10 20 30 --box 20 20 20 \\
                   -o high_accuracy_metal/

GPU-Accelerated Metal Docking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock-metal -r metalloprotein.pdb -l library.sdf \\
                   --metal-type ZN \\
                   --metal-residue "A:201" \\
                   --algorithm enhanced_hierarchical_gpu \\
                   --gpu \\
                   --center 10 20 30 --box 20 20 20 \\
                   -o gpu_metal_docking/

Multi-Metal Site Docking
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Dock to dinuclear zinc site
   pandadock-metal -r protein.pdb -l ligand.sdf \\
                   --metal-type ZN \\
                   --metal-residue "A:201,A:202" \\
                   --max-coord-bonds 4 \\
                   --center 10 20 30 --box 22 22 22 \\
                   -o dinuclear_zn/

Output Files
------------

**Structures:**

* ``complex1.pdb, complex2.pdb, ...`` - Protein-ligand complexes
* ``pose1.pdb, pose2.pdb, ...`` - Ligand poses only

**Analysis:**

* ``metal_docking_results.json`` - Complete results with coordination analysis
* ``coordination_analysis.json`` - Metal coordination details
* ``coordination_geometry.csv`` - Bond lengths, angles for each pose
* ``summary.txt`` - Human-readable summary

**Visualizations:**

* ``binding_affinities.png`` - Affinity distribution
* ``coordination_histogram.png`` - Coordination bond distribution
* ``metal_ligand_distances.png`` - Metal-donor distance analysis

Coordination Analysis Output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

   {
     "pose_1": {
       "metal_coordination": {
         "metal": "ZN",
         "geometry": "tetrahedral",
         "ligand_donors": [
           {"atom": "N1", "distance": 2.05, "angle": 109.2},
           {"atom": "O2", "distance": 2.12, "angle": 108.8}
         ],
         "protein_donors": [
           {"residue": "HIS96", "atom": "NE2", "distance": 2.08},
           {"residue": "HIS119", "atom": "NE2", "distance": 2.10}
         ],
         "coordination_score": -8.5,
         "geometry_rmsd": 0.15
       }
     }
   }

Performance Characteristics
----------------------------

**Runtime:** 200-400 seconds per ligand (similar to standard docking)

**Accuracy:**

* RMSD: 0.15-0.4 Å for metal-coordinating ligands
* Success rate: 95-98% for known metalloproteins
* Coordination geometry accuracy: >95%

**Supported Targets:**

* Matrix metalloproteinases (MMPs)
* Carbonic anhydrase
* Zinc-finger proteins
* Cytochromes and iron-sulfur proteins
* Calcium-binding proteins
* Copper oxidases

Best Practices
--------------

Input Preparation
^^^^^^^^^^^^^^^^^

1. **Verify metal present in PDB:**

   .. code-block:: bash

      grep "^HETATM" protein.pdb | grep ZN

2. **Check metal residue ID:**

   Note the chain and residue number (e.g., "A:201")

3. **Prepare ligand with potential donors:**

   Ensure ligand has N, O, or S atoms capable of coordination

Coordination Geometry Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+--------+-----------------+-------------+-----------------+
| Metal  | Geometry        | Coord #     | Examples        |
+========+=================+=============+=================+
| Zn²z   | Tetrahedral     | 4           | MMPs, CA        |
+--------+-----------------+-------------+-----------------+
| Fe²z/³z| Octahedral      | 6           | Heme proteins   |
+--------+-----------------+-------------+-----------------+
| Mg²z   | Octahedral      | 6           | Kinases         |
+--------+-----------------+-------------+-----------------+
| Cu²z   | Square planar   | 4           | Oxidases        |
+--------+-----------------+-------------+-----------------+
| Ca²z   | Irregular       | 6-8         | Calmodulin      |
+--------+-----------------+-------------+-----------------+

Parameter Tuning
^^^^^^^^^^^^^^^^

**For strict coordination:**

.. code-block:: bash

   --metal-coordination-weight 10.0 \\
   --coord-angle-tolerance 10.0

**For flexible coordination:**

.. code-block:: bash

   --metal-coordination-weight 3.0 \\
   --coord-angle-tolerance 25.0

Troubleshooting
---------------

No Metal Coordination Found
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem:** Ligand doesn't coordinate to metal

**Solutions:**

1. Increase ``--coord-bond-length-max``
2. Increase ``--coord-angle-tolerance``
3. Verify ligand has donor atoms (``--donor-atoms``)
4. Check metal residue ID is correct
5. Expand grid box to include metal

Poor Coordination Geometry
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem:** Unrealistic metal coordination

**Solutions:**

1. Increase ``--metal-coordination-weight``
2. Decrease ``--coord-angle-tolerance``
3. Specify correct ``--coordination-geometry``
4. Use ``--scoring hybrid`` for better accuracy

Metal Site Not in Grid Box
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem:** Metal ion outside docking grid

**Solutions:**

1. Recalculate ``--center`` to include metal
2. Increase ``--box`` dimensions
3. Verify metal coordinates in PDB file

Validation
----------

**Validate metal docking results:**

1. Visual inspection of coordination geometry
2. Check metal-donor distances (1.8-2.8 Å typical)
3. Verify coordination angles match expected geometry
4. Compare to known crystal structures if available

**Quality metrics:**

* Coordination bond lengths within expected range
* Coordination angles within tolerance
* No steric clashes with protein
* Reasonable binding score

Common Metalloprotein Targets
------------------------------

Matrix Metalloproteinases (MMPs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock-metal -r mmp.pdb -l inhibitor.sdf \\
                   --metal-type ZN \\
                   --coordination-geometry tetrahedral \\
                   --donor-atoms N,O,S

Carbonic Anhydrase
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock-metal -r ca.pdb -l inhibitor.sdf \\
                   --metal-type ZN \\
                   --coordination-geometry tetrahedral \\
                   --donor-atoms N,O

Kinases (Mg²z-ATP Binding Site)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock-metal -r kinase.pdb -l atp_analog.sdf \\
                   --metal-type MG \\
                   --coordination-geometry octahedral \\
                   --donor-atoms O

Exit Status
-----------

Returns 0 on success, non-zero on error.

See Also
--------

* :doc:`pandadock` - Standard docking
* :doc:`pandadock_flex` - Flexible docking
* :doc:`../algorithms/specialized_modes` - Specialized docking modes
* :doc:`../scoring/physics_based` - Physics-based scoring
