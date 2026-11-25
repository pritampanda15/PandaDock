Quick Start
===========

Run your first end-to-end molecular docking workflow in **PandaDock v5.0** in five minutes — either from the command line or visually in **PandaDock Studio**.

Option A — Visual Workflow in PandaDock Studio (Recommended for Beginners)
--------------------------------------------------------------------------

Launch the local web workspace in your project folder:

.. code-block:: bash

   pandadock-gui --port 8765 --workspace .

1. **Step 1 (Inputs & Preparation):** Drop your receptor PDB and ligand SDF/MOL2/SMILES. Expand **Prepare Protein & Ligand** to add hydrogens, assign pH 7.4 protonation, minimize the ligand in 3D, or strip crystal waters.
2. **Step 2 (Binding Site):** If you uploaded a protein–ligand complex, click the detected **Co-Crystal Ligand** chip to center the box on that pocket and **automatically exclude** the old X-ray ligand from the docking grid (keeping protein, catalytic metals, and waters intact). You can also run **Auto-Detect Pockets**, center on a **Metal Ion**, or toggle **Pick** to click residues directly on the 3D protein.
3. **Step 3–4 (Options & Launch):** Keep clean defaults (or configure waters, restraints, MM-GBSA, and MD handoff) and click **Run Docking & Reports**.
4. **Inspect Results:** Explore explicit-H 3D binding pockets and 18 **PandaLens** interaction classes, open the interactive **2D View** (2D interaction map or RDKit 2D structure), browse 300-DPI plots, and read the embedded A4 vector ``report.pdf`` and ``report.html``.

Option B — Command-Line Workflow
--------------------------------

1. Prepare Your Protein and Ligand
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add missing hydrogens, assign pH 7.4 protonation states, and minimize the ligand in 3D:

.. code-block:: bash

   pandadock-prepare -r raw_protein.pdb -l raw_ligand.sdf --ph 7.4 -o prepared/

2. Define the Binding Site Grid Box
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Center the box on a known X-ray ligand, automatic cavity search, or active-site residues:

.. code-block:: bash

   # From a reference X-ray ligand (5 Å padding)
   pandadock-gridbox -r prepared/receptor.pdb -m similarity \
                     --reference-ligand xray_ligand.sdf --padding 5.0 -o box.json

   # Or blind automatic cavity detection
   pandadock-gridbox -r prepared/receptor.pdb -m cavities --max-sites 3 -o box.json

3. Dock, Audit & Generate Reports
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pandadock dock -r prepared/receptor.pdb -l prepared/ligand.sdf -g box.json \
                  --scoring vina --waters displace --workers 4 --seed 42 -o results/

Every run automatically produces:

* ``poses.sdf`` — Ranked 3D poses with exact bond orders
* ``complex1.pdb`` … — Receptor + docked ligand complexes
* ``posebusters_audit.json`` — 11-point PoseBusters chemistry & geometry audit
* ``report.pdf`` & ``report.html`` — Multi-page A4 vector brochure & editorial HTML report
* 300-DPI diagnostic plots & PandaMap 2D/3D interaction diagrams

v5.0 Medicinal-Chemistry Shortcuts
----------------------------------

.. code-block:: bash

   # Congeneric analog series tethered to a crystal scaffold + SAR Brochure PDF
   pandadock series -r receptor.pdb -l analogs.smi --reference-pose parent.sdf \
                    --center 10 20 30 --box 22 22 22 -o series_out/

   # Covalent warhead docking (8 built-in warheads, e.g. SARS-CoV-2 Mpro Cys145)
   pandadock covalent -r 6lu7.pdb -l nirmatrelvir.smi --residue A:CYS:145 \
                      --warhead auto -o covalent_out/

   # Estimate binding affinity of a physics-docked pose with PandaDock-GNN
   pandadock gnn download-model -o models/
   pandadock gnn predict -m models/pandadock_gnn.pt -p receptor.pdb -l results/pose1.pdb

   # One-command Dock-to-MD handoff (OpenMM + GROMACS + PandaInsight)
   pandadock to-md -r receptor.pdb -p results/poses.sdf --engine both --duration-ns 50 -o md_pkg/

.. important::
   **Separation of Concerns:** Use empirical physics scoring (``pandadock dock``) to select 3D poses, and use **PandaDock-GNN** (``pandadock gnn predict`` / ``rescore``) to estimate affinity and rank different ligands against the same target.

Next Steps
----------

* :doc:`../guide/studio_gui` — Full walkthrough of PandaDock Studio and PandaLens 3D/2D views
* :doc:`../guide/medchem_v5` — Waters, restraints, pH states, covalent warheads, series SAR, PoseBusters & MD handoff
* :doc:`../cli/pandadock` — Complete command-line reference
