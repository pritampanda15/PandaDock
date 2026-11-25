Basic Docking Tutorial
======================

This step-by-step tutorial walks through preparing a protein–ligand pair, defining the binding site, running flexible docking, and inspecting your 3D/2D results in **PandaDock v5.0**.

Step 1 — Prepare Your Structures
--------------------------------

.. code-block:: bash

   pandadock-prepare -r raw_protein.pdb -l raw_ligand.sdf --ph 7.4 -o prepared/

This adds missing hydrogens, sets protonation at physiological pH 7.4, assigns partial charges, and MMFF94-minimizes the ligand in 3D.

Step 2 — Define the Binding Site Grid Box
-----------------------------------------

.. code-block:: bash

   pandadock-gridbox -r prepared/receptor.pdb -m similarity \
                     --reference-ligand xray_ligand.sdf --padding 5.0 -o box.json

Step 3 — Run Flexible Docking
-----------------------------

.. code-block:: bash

   pandadock dock -r prepared/receptor.pdb -l prepared/ligand.sdf -g box.json \
                  --scoring vina --waters displace --workers 4 --seed 42 -o basic_run/

Step 4 — Inspect in 3D, 2D & PDF
--------------------------------

Launch **PandaDock Studio** to inspect explicit hydrogens, 18 PandaLens interaction classes, the live 2D interaction map, 300-DPI plots, and ``report.pdf``:

.. code-block:: bash

   pandadock-gui --workspace basic_run/
