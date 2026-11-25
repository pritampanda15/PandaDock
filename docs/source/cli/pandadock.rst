pandadock - Main Docking & v5.0 Suite CLI
=========================================

The ``pandadock`` command is the main hub for **PandaDock v5.0**, providing **18 commands** across flexible docking, virtual screening, congeneric series SAR, covalent warhead docking, PoseBusters validation, target weight calibration, MD handoff, Studio GUI, and SE(3)-equivariant GNN scoring.

Synopsis
--------

.. code-block:: bash

   pandadock COMMAND [OPTIONS]

Command Summary
---------------

* ``pandadock dock`` — Flexible-ligand docking with CPU/GPU search, waters, restraints, pH states, MM-GBSA & PDF/HTML reports
* ``pandadock screen`` — Batched GPU/CPU virtual screening of ligand libraries with shared affinity grids
* ``pandadock series`` *(v5.0)* — Congeneric series / MCS core-tethered docking & A4 SAR brochure PDF/HTML
* ``pandadock covalent`` *(v5.0)* — Covalent warhead docking (8 medicinal-chemistry warheads) + PandaMap covalent maps
* ``pandadock validate-poses`` *(v5.0)* — Built-in 11-point PoseBusters chemistry & geometry validity audit
* ``pandadock calibrate`` *(v5.0)* — Few-shot per-target Vina weight calibration via Ridge LOOCV
* ``pandadock to-md`` *(v5.0)* — One-command Dock-to-MD handoff (OpenMM / GROMACS / PandaInsight)
* ``pandadock gui`` *(v5.0)* — Launch PandaDock Studio Apple-designed web GUI
* ``pandadock hybrid`` — Docking followed by GNN rescoring
* ``pandadock list-algorithms`` — Display available search algorithms and scoring functions
* ``pandadock gnn <subcommand>`` — 8 GNN commands (``download-model``, ``predict``, ``rescore``, ``train``, ``train-sair``, ``benchmark``, ``compare``, ``list-algorithms``); see :doc:`pandadock_gnn`

pandadock dock
--------------

Flexible-ligand Monte Carlo + analytic-gradient L-BFGS docking with Vina, Vinardo, or physics-based scoring.

.. list-table::
   :header-rows: 1
   :widths: 26 14 60

   * - Option
     - Default
     - Description
   * - ``-r, --receptor PATH``
     - Required
     - Prepared receptor PDB file.
   * - ``-l, --ligand PATH``
     - Required
     - Ligand file (SDF, MOL2, or PDB).
   * - ``-g, --grid-config PATH``
     - —
     - Grid box JSON from ``pandadock-gridbox``.
   * - ``--center X Y Z`` / ``--box DX DY DZ``
     - —
     - Grid box center and dimensions in Å (alternative to ``-g``).
   * - ``-s, --scoring``
     - ``vina``
     - ``vina``, ``vinardo``, or ``physics_based``.
   * - ``-n, --num-poses INT``
     - ``20``
     - Maximum ranked binding modes returned.
   * - ``-e, --exhaustiveness INT``
     - auto (8–32)
     - Independent Monte Carlo search runs; higher = deeper search.
   * - ``-j, --workers INT``
     - ``1``
     - Parallel CPU worker threads across search runs.
   * - ``--device TEXT``
     - ``cpu``
     - ``cpu``, ``cuda`` (NVIDIA GPU), or ``mps`` (Apple Silicon GPU).
   * - ``--n-chains INT``
     - ``512``
     - Parallel batched search chains on GPU/MPS.
   * - ``--grid-spacing FLOAT``
     - ``0.375``
     - Precomputed affinity grid spacing in Å.
   * - ``--rigid-ligand`` / ``--sample-rings``
     - off / off
     - Freeze torsions or enumerate non-aromatic ring puckers (ETKDGv3).
   * - ``--waters CHOICE``
     - ``strip``
     - Pocket water protocol: ``strip``, ``keep``, or ``displace``.
   * - ``--restraint SPEC``
     - —
     - Repeatable differentiable distance restraint (e.g. ``A:ZN:301:ZN<2.3:donor``).
   * - ``--restraints-file PATH``
     - —
     - JSON file of multiple distance / pharmacophore restraints.
   * - ``--enumerate-states`` / ``--ph FLOAT``
     - off / ``7.4``
     - Enumerate protonation, tautomer, and stereo states at target pH.
   * - ``--rescoring CHOICE``
     - ``none``
     - ``none`` or ``mmgbsa`` (differential 12 Å pocket MM-GBSA).
   * - ``--weights-file PATH``
     - —
     - Target-calibrated scoring weights JSON from ``pandadock calibrate``.
   * - ``--export-md CHOICE``
     - ``none``
     - Export top pose as MD package: ``openmm``, ``gromacs``, or ``both``.
   * - ``--ensemble`` / ``--visualize``
     - on / on
     - Boltzmann ensemble averaging and automatic 300-DPI plots + PDF/HTML reports.
   * - ``--seed INT`` / ``--fast`` / ``-o PATH``
     - random / off / ``docking_output``
     - Reproducible seed, quick smoke-test mode, and output directory.

pandadock screen
----------------

Batched virtual screening of a ligand library (multi-SDF or folder) against one receptor site, reusing precomputed affinity grids across all ligands.

.. code-block:: bash

   pandadock screen -r receptor.pdb -l library.sdf -g box.json \
                    --device cuda --n-chains 128 --n-steps 8 --max-batch 64 \
                    --top 100 -o screening_output/

Key options: ``-r/--receptor``, ``-l/--ligands``, ``-g/--grid-config`` (or ``--center``/``--box``), ``--device``, ``--n-chains`` (128), ``--n-steps`` (8), ``--max-batch`` (64), ``--grid-spacing`` (0.375), ``--rigid-ligand``, ``--seed``, ``--top``, ``-o/--output-dir``.

pandadock series
----------------

Congeneric analog series docking with Maximum Common Substructure (MCS) flat-bottom core tethering, multi-score consensus ranking, and an A4 vector **SAR Brochure PDF** (``series_report.pdf`` + ``series_report.html``).

.. code-block:: bash

   pandadock series -r receptor.pdb -l analogs.smi \
                    --reference-pose parent_xray.sdf --core-tolerance 0.75 \
                    --center 126.5 119.2 146.1 --box 22 22 22 -o series_output/

Key options: ``-r/--receptor``, ``-l/--ligands`` (SDF/SMILES/dir), ``--center``, ``--box``, ``--reference-pose``, ``--core-smarts``, ``--core-tolerance`` (0.75 Å), ``-s/--scoring``, ``-e/--exhaustiveness`` (8), ``--device``, ``-j/--workers``, ``--seed`` (42), ``--weights-file``, ``--gnn-model``, ``-o/--output-dir``.

pandadock covalent
------------------

Covalent warhead docking anchored to a reactive protein nucleophile (``Cys``, ``Ser``, ``Thr``, ``Lys``, ``Tyr``, ``His``). Automatically detects or applies any of **8 built-in warheads**: ``michael_acceptor``, ``chloroacetamide``, ``nitrile``, ``epoxide``, ``boronic_acid``, ``aldehyde``, ``sulfonyl_fluoride``, and ``disulfide`` (default: ``auto``). Generates ``covalent_poses.sdf``, ``covalent_complex1.pdb``, PandaMap 2D/3D covalent interaction maps, ``report.html``, and ``report.pdf``.

.. code-block:: bash

   pandadock covalent -r 6lu7_mpro.pdb -l nirmatrelvir.smi \
                      --residue A:CYS:145 --warhead auto \
                      --box-size 22.0 --num-poses 9 --exhaustiveness 12 -o covalent_out/

pandadock validate-poses
------------------------

Runs a built-in **11-point PoseBusters chemistry & geometry validity audit** (RDKit sanitization, connectivity, bond lengths, bond angles, internal clashes, aromatic flatness, double-bond flatness, chirality preservation, MMFF94/UFF strain, protein steric clashes, and protein volume overlap) on any docked SDF.

.. code-block:: bash

   pandadock validate-poses -p results/poses.sdf -r receptor.pdb \
                            --reference input_lig.sdf --max-strain 15.0 -o audit.json

pandadock calibrate
-------------------

Per-target few-shot Vina weight calibration using Ridge Leave-One-Out Cross-Validation (LOOCV) on 5–25 known ligands with experimental ΔG, pIC50, or pKi measurements.

.. code-block:: bash

   pandadock calibrate -r receptor.pdb -p known_poses.sdf -y labels.csv \
                       --preset vina --higher-is-better -o calibrated_weights.json

pandadock to-md
---------------

Exports any docked pose into a self-contained **OpenMM**, **GROMACS**, and **PandaInsight** molecular dynamics starter package.

.. code-block:: bash

   pandadock to-md -r receptor.pdb -p results/poses.sdf \
                   --pose-index 0 --engine both --duration-ns 50 --temperature 300 -o md_pkg/

pandadock gui
-------------

Launches **PandaDock Studio** (identical to ``pandadock-gui``). See :doc:`pandadock_gui` and :doc:`../guide/studio_gui`.

.. code-block:: bash

   pandadock gui --host 127.0.0.1 --port 8765 --workspace .

pandadock hybrid
----------------

Runs docking (``-n 50``) followed by GNN rescoring and keeps ``--top-k 10`` poses.

.. warning::
   **Separation of Concerns:** Benchmarked across 814 complexes, the affinity-trained GNN selects worse 3D poses than empirical Vina/Vinardo scoring (median RMSD 5.36 Å vs 2.09 Å). Prefer ``pandadock dock`` to select poses and ``pandadock gnn predict`` / ``rescore`` to estimate affinity.

pandadock list-algorithms
-------------------------

Displays all available search algorithms and scoring functions in the current environment.
