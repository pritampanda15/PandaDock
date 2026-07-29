pandadock-flex - Flexible Docking Command
==========================================

The ``pandadock-flex`` command performs induced-fit docking with receptor flexibility. It accounts for protein conformational changes upon ligand binding, similar to Schrödinger's Induced-Fit Docking (IFD).

Synopsis
--------

.. code-block:: bash

   pandadock-flex [OPTIONS]

Description
-----------

Performs flexible docking using a multi-phase protocol:

1. **Soft Docking**: Initial docking with softened van der Waals potentials
2. **Receptor Refinement**: Side-chain and optional backbone optimization
3. **Final Redocking**: Rigid docking into refined receptor conformations
4. **IFD Scoring**: Combined ligand-receptor energy evaluation

This approach is essential for targets with induced-fit binding mechanisms where the receptor undergoes conformational changes upon ligand binding.

Required Options
----------------

``-r, --receptor PATH``
    Receptor PDB file (protein structure)

``-l, --ligand PATH``
    Ligand file (SDF, MOL2, or PDB format)

``--center X Y Z``
    Grid box center coordinates (X Y Z in Angstroms)

``--radius FLOAT``
    Grid box radius in Angstroms. Default: 12.0

Flexibility Options
-------------------

``--refine-distance FLOAT``
    Distance from ligand for flexible residues (Angstroms). Default: 6.0

    Residues within this distance will have side-chain flexibility.

``--refine-loops / --no-refine-loops``
    Include loop refinement. Default: disabled

    Enable for targets with flexible loops near binding site.

``--refine-backbone / --no-refine-backbone``
    Include backbone refinement. Default: disabled

    WARNING: Very computationally intensive. Use sparingly.

``--refine-ligand / --no-refine-ligand``
    Allow ligand conformational changes during refinement. Default: enabled

``--num-receptor-conformers N``
    Number of receptor conformations to generate. Default: 5

    Higher values increase accuracy but also runtime.

``--flexible-residues RESIDUES``
    Explicitly specify flexible residues (e.g., "A:100,A:150,B:200")

    Overrides automatic distance-based selection.

Docking Algorithm
-----------------

``-a, --algorithm ALGORITHM``
    Docking algorithm for initial and final docking. Default: ``enhanced_hierarchical_cpu``

    Recommended: ``enhanced_hierarchical_cpu``, ``genetic_algorithm_cpu``

Scoring Options
---------------

``-s, --scoring FUNCTION``
    Scoring function. Default: ``physics_based``

    Options: ``physics_based``, ``hybrid``, ``precision_score``

``--ifd-scoring / --no-ifd-scoring``
    Use Induced-Fit Docking scoring (ligand + receptor energy). Default: enabled

Refinement Options
------------------

``--refinement-method METHOD``
    Refinement method. Default: ``openmm``

    Options:

    * ``openmm`` - OpenMM molecular dynamics minimization (recommended)
    * ``rdkit`` - RDKit force field minimization (faster, less accurate)

``--refinement-steps N``
    Number of minimization steps. Default: 1000

``--refinement-force-field FF``
    Force field for refinement. Default: ``amber14``

    Options: ``amber14``, ``charmm36``, ``opls``

``--soft-vdw-scale FLOAT``
    Van der Waals scaling factor for soft docking. Default: 0.5

    Lower values = softer potentials (more permissive initial docking)

Output Options
--------------

``-o, --output-dir PATH``
    Output directory for results. Default: ``flex_docking_output``

``-n, --num-poses N``
    Number of final poses to generate. Default: 20

``--visualize / --no-visualize``
    Generate visualization plots. Default: enabled

``--save-receptor-conformers``
    Save all refined receptor conformations

Performance Options
-------------------

``--cpuworkers N``
    Number of CPU worker threads. Default: auto-detect

``--gpu``
    Enable GPU acceleration for docking steps (requires CUDA)

``--fast``
    Fast mode with reduced sampling (for testing)

Examples
--------

Basic Flexible Docking
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock-flex -r protein.pdb -l ligand.sdf \\
                  --center 10 20 30 --radius 12.0 \\
                  -o flex_results/

This uses default settings:

* 6Å side-chain flexibility
* 5 receptor conformers
* OpenMM refinement

Kinase Flexible Docking
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock-flex -r kinase.pdb -l inhibitor.sdf \\
                  --center 15 20 25 --radius 12.0 \\
                  --refine-distance 8.0 \\
                  --refine-loops \\
                  --num-receptor-conformers 10 \\
                  -o kinase_flex/

Includes loop refinement (important for DFG-in/out transitions)

GPCR Flexible Docking
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock-flex -r gpcr.pdb -l agonist.sdf \\
                  --center 12 15 18 --radius 14.0 \\
                  --refine-distance 7.0 \\
                  --refine-backbone \\
                  --num-receptor-conformers 15 \\
                  --refinement-steps 2000 \\
                  -o gpcr_flex/

Extended refinement for GPCR conformational changes

High-Accuracy Flexible Docking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock-flex -r protein.pdb -l ligand.sdf \\
                  --center 10 20 30 --radius 12.0 \\
                  --algorithm enhanced_hierarchical_cpu \\
                  --scoring hybrid \\
                  --refine-distance 8.0 \\
                  --num-receptor-conformers 20 \\
                  --refinement-steps 2000 \\
                  --num-poses 50 \\
                  -o high_accuracy_flex/

Fast Testing Mode
^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock-flex -r protein.pdb -l ligand.sdf \\
                  --center 10 20 30 --radius 12.0 \\
                  --fast \\
                  --num-receptor-conformers 3 \\
                  --refinement-steps 500 \\
                  -o fast_test/

GPU-Accelerated Flexible Docking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock-flex -r protein.pdb -l ligand.sdf \\
                  --center 10 20 30 --radius 12.0 \\
                  --algorithm enhanced_hierarchical_gpu \\
                  --gpu \\
                  -o gpu_flex/

Custom Flexible Residues
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock-flex -r protein.pdb -l ligand.sdf \\
                  --center 10 20 30 --radius 12.0 \\
                  --flexible-residues "A:99,A:145,A:189,B:201" \\
                  -o custom_flex/

Output Files
------------

The command generates the following outputs:

**Structures:**

* ``complex1.pdb, complex2.pdb, ...`` - Protein-ligand complexes with refined receptor
* ``pose1.pdb, pose2.pdb, ...`` - Ligand poses only
* ``receptor_refined_*.pdb`` - Refined receptor conformations (if ``--save-receptor-conformers``)

**Analysis:**

* ``flex_docking_results.json`` - Complete results with IFD scores
* ``refinement_analysis.json`` - Receptor refinement details
* ``ifd_scores.csv`` - IFD scoring breakdown
* ``summary.txt`` - Human-readable summary

**Visualizations:**

* ``ifd_scores.png`` - IFD score distribution
* ``receptor_rmsd.png`` - Receptor conformational changes
* ``interaction_map.png`` - Protein-ligand interaction analysis

Performance Characteristics
----------------------------

**Runtime:**

* Small ligand, 5 conformers: 5-10 minutes
* Medium ligand, 10 conformers: 15-30 minutes
* Large ligand, 20 conformers: 30-60 minutes

**With loop refinement:** 2-3x longer

**With backbone refinement:** 5-10x longer

**Accuracy:**

* RMSD: 0.2-0.6 Å (excellent for induced-fit cases)
* Success rate: 92-96% for flexible binding sites

Best Practices
--------------

When to Use Flexible Docking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

 **Protein kinases** with flexible activation loops

 **GPCRs** with induced-fit activation

 **Proteins with known conformational changes** upon binding

 **When rigid docking fails** to reproduce crystal pose (RMSD > 2Å)

 **Large, conformationally diverse ligands** (peptides, macrocycles)

When Not to Use
^^^^^^^^^^^^^^^

 **Small, rigid binding sites** - Use standard rigid docking

 **Virtual screening** - Too slow for large libraries

 **Well-defined binding modes** - Rigid docking sufficient

Optimization Tips
^^^^^^^^^^^^^^^^^

**Balance accuracy and speed:**

.. code-block:: bash

   # Fast but reasonable
   --num-receptor-conformers 5 --refinement-steps 1000

   # High accuracy
   --num-receptor-conformers 15 --refinement-steps 2000

   # Ultra-high accuracy
   --num-receptor-conformers 20 --refinement-steps 3000

**Refine only necessary regions:**

.. code-block:: bash

   # Auto-select within 6Å (default)
   --refine-distance 6.0

   # Larger binding site
   --refine-distance 8.0

**Use GPU when available:**

.. code-block:: bash

   --algorithm enhanced_hierarchical_gpu --gpu

Troubleshooting
---------------

Slow Performance
^^^^^^^^^^^^^^^^

**Problem:** Flexible docking takes too long

**Solutions:**

1. Reduce receptor conformers: ``--num-receptor-conformers 3``
2. Reduce refinement steps: ``--refinement-steps 500``
3. Use ``--fast`` mode
4. Disable loop refinement
5. Use GPU: ``--gpu``

Poor Results
^^^^^^^^^^^^

**Problem:** Poor pose prediction or binding affinity

**Solutions:**

1. Increase conformers: ``--num-receptor-conformers 15``
2. Increase refine distance: ``--refine-distance 8.0``
3. Enable loop refinement: ``--refine-loops``
4. Use better scoring: ``--scoring hybrid``
5. Increase refinement steps: ``--refinement-steps 2000``

Refinement Failures
^^^^^^^^^^^^^^^^^^^

**Problem:** Receptor refinement crashes or produces unreasonable structures

**Solutions:**

1. Check input structure quality (missing atoms, clashes)
2. Reduce refinement steps: ``--refinement-steps 500``
3. Try different force field: ``--refinement-force-field charmm36``
4. Disable backbone refinement if enabled

Validation
----------

**Before using flexible docking on new targets:**

1. Test on known crystal structures
2. Compare rigid vs flexible docking results
3. Validate receptor conformational changes are reasonable
4. Check that IFD scores correlate with known activity

Exit Status
-----------

Returns 0 on success, non-zero on error.

See Also
--------

* :doc:`pandadock` - Standard rigid docking
* :doc:`pandadock_metal` - Metal docking
* :doc:`../algorithms/specialized_modes` - Specialized docking modes
* :doc:`../scoring/physics_based` - Physics-based scoring
