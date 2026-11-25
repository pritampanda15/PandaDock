Best Practices
==============

Follow these practical guidelines to get reliable, physically realistic docking results in **PandaDock v5.0**.

1. Always Prepare Structures Before Docking
-------------------------------------------

* Add missing hydrogens and assign realistic protonation states at physiological pH (``7.4``) using ``pandadock-prepare`` or Step 1 of **PandaDock Studio**.
* Always minimize 2D/SMILES-derived ligands with MMFF94 (``--minimize-ligand``) before docking to avoid artificial internal strain.
* If your ligand has ionizable amines/carboxylates or ambiguous tautomers near pH 7.4, enable ``--enumerate-states --ph 7.4``.

2. Remove Co-Crystal Ligands Before Redocking
---------------------------------------------

* If you start from an X-ray protein–ligand complex PDB, use the X-ray ligand to define the grid box — then **remove** that X-ray ligand from the receptor PDB before docking so it does not sterically block your new compound.
* In **PandaDock Studio**, clicking the detected Co-Crystal Ligand chip in Step 2 does both automatically while preserving protein atoms, catalytic metals, and waters.

3. Size the Grid Box Appropriately
----------------------------------

* Use **5–6 Å padding** around a reference ligand (typically ``20 × 20 × 20 Å`` to ``24 × 24 × 24 Å``).
* Boxes that are too tight clip large substituents; boxes that are too huge waste search budget on irrelevant surface grooves.

4. Separate Pose Selection from Affinity Prediction
---------------------------------------------------

* Use empirical physics scoring (``pandadock dock --scoring vina`` or ``vinardo``) to select the 3D binding pose.
* Use **PandaDock-GNN** (``pandadock gnn predict`` / ``rescore``) or target-calibrated weights (``pandadock calibrate``) to estimate potency and rank different ligands against the same protein.

5. Always Inspect PoseBusters & 3D/2D Interactions
--------------------------------------------------

* Check ``posebusters_audit.json`` (or the ``PB ✓`` badges in Studio) to verify zero steric clashes, preserved chirality, and low MMFF94 strain.
* Inspect hydrogen bonds (with explicit H), metal coordination, and water bridges in **PandaDock Studio** (PandaLens 3D + 2D View) before promoting a compound to MD (``pandadock to-md``) or synthesis.
