PandaDock Studio (GUI) & PandaLens 3D/2D Viewer
===============================================

**PandaDock Studio** is a local-first, Apple-designed browser workspace (``pandadock-gui`` or ``pandadock gui``) that brings together structure preparation, visual binding-site selection, docking execution, **PandaLens** 3D explicit-H and interactive 2D interaction views, 300-DPI diagnostic plots, and embedded A4 vector PDF/HTML reports in one interface.

Launching Studio
----------------

.. code-block:: bash

   pip install "pandadock[gui]"
   pandadock-gui --port 8765 --workspace .

Guided 4-Step Docking Wizard
----------------------------

1. **Step 1 — Inputs & Preparation:**
   Choose **Single Ligand**, **Ligand Series**, or **Covalent**, then drag-and-drop your protein PDB and ligand SDF/MOL2/SMILES (or load a built-in example complex). Expand **Prepare Protein & Ligand** to add hydrogens, set physiological pH (7.4), run MMFF94 3D minimization, optionally enumerate pH protonation/tautomer states, and optionally strip crystallographic waters.

2. **Step 2 — Define Binding Site (Grid Box):**
   * **Co-Crystal X-Ray Ligand (Schrödinger Maestro–Style):** If your uploaded PDB is a protein–ligand complex, Studio automatically detects the bound X-ray ligand(s) and displays clickable chips (e.g. ``AZM A:701``). Clicking that chip centers the grid box on the pocket **and automatically strips only that co-crystal ligand** from the receptor grid before docking starts — keeping all protein atoms, catalytic metals, and structural waters intact.
   * **Auto-Detect Pockets:** Scans the protein for geometric cavities and lists ranked clickable pocket chips with volumes.
   * **Protein Center / Metal Ion:** One-click centering for blind docking or metalloprotein active sites.
   * **Interactive 3D Residue Picking:** Toggle **Pick** in the 3D toolbar and click residues directly on the protein to center the blue wireframe box around them.

3. **Step 3 — Docking Options:**
   Progressive Apple disclosure panels for Scoring (``vina``, ``vinardo``, ``physics_based``), Exhaustiveness, Compute Device (CPU/CUDA/MPS), Structural Waters (``strip`` / ``keep`` / ``displace``), Differentiable Restraints, MM-GBSA Rescoring, Calibrated Weights, and Dock-to-MD Handoff.

4. **Step 4 — Review & Launch:**
   Click **Run Docking & Reports** to stream live logs via Server-Sent Events (SSE). When complete, Studio automatically loads the 3D poses, plots, and PDF/HTML reports.

Separate 3D/2D Ligand Window & Hide Toggles
-------------------------------------------

When you upload a receptor and an un-docked ligand (often sitting at different coordinates), Studio keeps the **main 3D canvas** strictly for the **Protein** (+ Grid Box + clickable residues) and opens your uploaded **Ligand** in a sleek floating **Ligand Preview** window:

* **``3D | 2D`` Switch:** Toggle instantly between an independent interactive 3Dmol WebGL stage and a crisp **RDKit 2D chemical structure SVG** (during setup) or **PandaLens interactive 2D protein–ligand interaction map** (during docked pose inspection).
* **One-Click Visibility Toggles:** Use **Protein**, **Ligand**, **Lig 3D**, and **2D View** in the bottom toolbar to hide/show structures or pop out the 3D/2D ligand window anytime, with **⤢** expand/compact, **⟲** reset, and **SVG** download.

PandaLens Binding-Site Inspection (Explicit H & 18 Interaction Classes)
-----------------------------------------------------------------------

Opening any completed docking run displays Pose #1..#N in **Binding Pocket** mode powered by **PandaLens**:

* **Explicit 3D Hydrogens (``H``):** Displays white explicit hydrogens on both binding-site protein residues/waters and the docked ligand, evaluating real **D–H···A** hydrogen-bond geometry (H···A ≤ 2.7 Å, D–H···A ≥ 120°).
* **18 Interaction Classes (``Contacts``):** Hydrogen bonds, 2-leg Water bridges, Salt bridges, Attractive/Repulsive charges, Metal coordination, Covalent warheads, Disulfides, π–π stacking, T-shaped π, Cation–π, Amide–π, Donor–π, Alkyl–π, Sulfur–π, Halogen bonds, Hydrophobic contacts, and Steric clashes.
* **Interactive 2D → 3D Highlighting & Draggable Labels:** Click **List (N)** or open **2D View** to hover-highlight individual interactions or atoms live in 3D, click class pills to filter interaction types, and drag residue labels clear of sticks with automatic SVG leader lines.
