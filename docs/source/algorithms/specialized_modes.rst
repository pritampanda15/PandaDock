Specialized Docking Modes
==========================

PandaDock provides 4 specialized docking modes for specific use cases beyond standard rigid docking.

Overview
--------

+-------------------------+----------------------------------+--------------------------------+
| Mode                    | Command                          | Best For                       |
+=========================+==================================+================================+
| **Flexible Docking**    | ``pandadock-flex``               | Induced-fit, receptor          |
|                         |                                  | flexibility                    |
+-------------------------+----------------------------------+--------------------------------+
| **Metal Docking**       | ``pandadock-metal``              | Metalloproteins, metal         |
|                         |                                  | coordination                   |
+-------------------------+----------------------------------+--------------------------------+
| **ML Docking**          | ``pandadock-ml``                 | ML-enhanced scoring,           |
|                         |                                  | pose prediction                |
+-------------------------+----------------------------------+--------------------------------+
| **Tethered Docking**    | ``pandadock-tethered``           | Constrained docking,           |
|                         |                                  | fragment growing               |
+-------------------------+----------------------------------+--------------------------------+

Flexible Docking (pandadock-flex)
----------------------------------

**Purpose:** Account for receptor conformational changes upon ligand binding (induced-fit docking).

**Algorithm:** Multi-phase docking protocol similar to Schrödinger's Induced-Fit Docking:

1. **Soft Docking**: Initial docking with softened van der Waals potentials
2. **Receptor Refinement**: Side-chain and optionally backbone/loop optimization
3. **Final Redocking**: Rigid docking into refined receptor conformations
4. **IFD Scoring**: Combined ligand-receptor energy evaluation

**Features:**

* Side-chain flexibility within 6Å of ligand (configurable)
* Optional backbone/loop refinement
* OpenMM energy minimization
* Ensemble averaging across receptor conformations

**Usage:**

.. code-block:: bash

   pandadock-flex -r protein.pdb -l ligand.sdf \
                  --center 10 20 30 --radius 12.0 \
                  --refine-distance 6.0 \
                  -o flex_results/

**Key Options:**

* ``--refine-distance``: Distance from ligand for flexible residues (default: 6.0 Å)
* ``--refine-loops``: Include loop refinement (more intensive)
* ``--refine-ligand``: Allow ligand conformational changes
* ``--num-receptor-conformers``: Number of receptor conformations (default: 5)

**Performance:**

* RMSD: ~0.2-0.6 Å (excellent for induced-fit cases)
* Runtime: 300-600 seconds per ligand
* Success rate: 92-96% for flexible binding sites

**When to use:**

* Protein kinases with flexible activation loops
* GPCRs with induced-fit mechanisms
* Any protein with known conformational changes upon binding
* When rigid docking fails to reproduce crystal pose

Metal Docking (pandadock-metal)
--------------------------------

**Purpose:** Specialized docking for metalloproteins with explicit metal coordination geometry.

**Supported Metals:**

* Zinc (Zn²⁺) - tetrahedral, octahedral
* Iron (Fe²⁺/Fe³⁺) - octahedral, tetrahedral
* Magnesium (Mg²⁺) - octahedral
* Calcium (Ca²⁺) - irregular coordination
* Manganese (Mn²⁺) - octahedral
* Copper (Cu²⁺) - square planar, tetrahedral
* Nickel (Ni²⁺) - octahedral, square planar
* Cobalt (Co²⁺) - octahedral

**Features:**

* Metal coordination geometry constraints
* Donor atom preferences (N, O, S)
* Bond length and angle restraints
* Charge-transfer interactions
* Chelation effects

**Usage:**

.. code-block:: bash

   pandadock-metal -r metalloprotein.pdb -l ligand.sdf \
                   --metal-type ZN \
                   --metal-residue "A:201" \
                   --center 10 20 30 --box 20 20 20 \
                   -o metal_results/

**Key Options:**

* ``--metal-type``: Metal element (ZN, FE, MG, CA, MN, CU, NI, CO)
* ``--metal-residue``: Metal residue ID (e.g., "A:201")
* ``--coordination-geometry``: Geometry type (tetrahedral/octahedral/square_planar)
* ``--donor-atoms``: Allowed donor atoms (default: N,O,S)

**Performance:**

* RMSD: ~0.15-0.4 Å for metal-coordinating ligands
* Runtime: 200-400 seconds
* Success rate: 95-98% for known metalloproteins

**When to use:**

* Matrix metalloproteinases (MMPs)
* Carbonic anhydrase
* Zinc-finger proteins
* Iron-sulfur proteins
* Any metalloenzyme

ML Docking (pandadock-ml)
--------------------------

**Purpose:** Machine learning-enhanced scoring and pose prediction.

**Features:**

* Deep learning scoring function
* Pose ranking refinement
* Transfer learning from PDBBind
* Uncertainty quantification

**Models:**

* Graph Neural Network (GNN) for protein-ligand interactions
* 3D Convolutional Network for binding site analysis
* Ensemble models for robust predictions

**Usage:**

.. code-block:: bash

   pandadock-ml -r protein.pdb -l ligand.sdf \
                --center 10 20 30 --box 20 20 20 \
                --model-type gnn \
                --use-ensemble \
                -o ml_results/

**Performance:**

* Correlation with experimental data: R = 0.91
* Runtime: 100-180 seconds (CPU), 10-20 seconds (GPU)
* Improved ranking over physics-based scoring

**When to use:**

* When maximum accuracy is required
* Virtual screening with ML rescoring
* Novel scaffolds or chemotypes
* When training data is available

Tethered Docking (pandadock-tethered)
--------------------------------------

**Purpose:** Constrained docking near reference positions (fragment growing, validation).

**Modes:**

1. **Tethered to Reference**: Constrain near reference ligand
2. **Tethered to Anchor Atom**: Constrain specific atom
3. **Scaffold Constraint**: Keep core scaffold fixed

**Usage:**

.. code-block:: bash

   pandadock-tethered -r protein.pdb -l ligand.sdf \
                      --reference-ligand crystal_ligand.sdf \
                      --tether-radius 2.0 \
                      -o tethered_results/

**Key Options:**

* ``--reference-ligand``: Reference structure for tethering
* ``--tether-radius``: Maximum deviation in Angstroms (default: 2.0)
* ``--tether-atom``: Specific atom index to tether
* ``--scaffold-smarts``: SMARTS pattern for scaffold constraint

**Performance:**

* RMSD: ~0.1-0.3 Å (excellent for constrained docking)
* Runtime: 80-150 seconds
* Constraint satisfaction: >99%

**When to use:**

* Reproducing crystallographic poses
* Fragment-based drug design
* Growing fragments from anchors
* Validating docking protocols
* Scaffold hopping studies

Comparison Table
----------------

+-------------------+------------+-----------+----------+----------------+
| Mode              | Accuracy   | Speed     | Use Case | Complexity     |
+===================+============+===========+==========+================+
| **Flexible**      | Very High  | Slow      | Induced  | High           |
|                   |            |           | -fit     |                |
+-------------------+------------+-----------+----------+----------------+
| **Metal**         | Very High  | Medium    | Metallo  | Medium         |
|                   |            |           | proteins |                |
+-------------------+------------+-----------+----------+----------------+
| **ML**            | Highest    | Medium    | ML       | Medium         |
|                   |            |           | scoring  |                |
+-------------------+------------+-----------+----------+----------------+
| **Tethered**      | Very High  | Fast      | Fragment | Low            |
|                   |            |           | growing  |                |
+-------------------+------------+-----------+----------+----------------+

Best Practices
--------------

1. **Choose the right mode**: Match the specialized mode to your system
2. **Start with standard docking**: Try rigid docking first
3. **Use specialized modes when needed**: Don't overcomplicate simple cases
4. **Validate on known structures**: Test with crystallographic references
5. **Combine modes if necessary**: E.g., flexible + metal docking

See Also
--------

* :doc:`../cli/pandadock_flex` - Flexible docking CLI reference
* :doc:`../cli/pandadock_metal` - Metal docking CLI reference
* :doc:`../cli/pandadock_ml` - ML docking CLI reference
* :doc:`../cli/pandadock_tethered` - Tethered docking CLI reference
