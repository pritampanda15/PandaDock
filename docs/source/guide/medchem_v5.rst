PandaDock v5.0 Medicinal-Chemistry Suite
========================================

PandaDock v5.0 integrates an end-to-end medicinal-chemistry workflow directly into the analytic-gradient docking engine and Panda-family reporting kit.

1. Structural Pocket Waters (``--waters``)
------------------------------------------

Crystallographic waters can either block a ligand artificially or mediate essential hydrogen-bond bridges:

* ``--waters strip`` *(default)*: Removes all solvent ``HOH`` molecules prior to grid construction. Best for general screening.
* ``--waters keep``: Retains pocket waters rigidly as bridging H-bond partners.
* ``--waters displace`` *(v5.0)*: Evaluates bridging vs displaced pocket waters with desolvation penalties and reports which waters remained bridging vs displaced.

2. Differentiable Geometric, H-Bond & Metal Restraints (``--restraint``)
------------------------------------------------------------------------

Enforce known structural biology priors smoothly during both CPU and GPU/MPS L-BFGS optimization using flat-bottom harmonic restraints (zero penalty inside the target distance):

.. code-block:: bash

   # Enforce zinc coordination (< 2.3 Å) and an H-bond to Asp128
   pandadock dock -r 3hs4.pdb -l azm.sdf --center -5.3 -0.5 14.8 --box 20 20 20 \
                  --restraint "A:ZN:301:ZN<2.3:donor" \
                  --restraint "A:ASP:128:OD1<3.2:hbond_donor" -o restrained/

3. pH-Aware Protonation, Tautomer & Stereo Enumeration (``--enumerate-states``)
-------------------------------------------------------------------------------

Automatically enumerates plausible protonation microspecies, tautomers, and unassigned chiral centers at physiological ``--ph 7.4``, docks each state, applies Henderson–Hasselbalch state penalties, and reports the biologically dominant bound state.

4. Built-In 11-Point PoseBusters Validity Audit (``pandadock validate-poses``)
------------------------------------------------------------------------------

Every docking run automatically audits returned poses across 11 chemistry & geometry checks (RDKit sanitization, connectivity, bond lengths, bond angles, internal clashes, aromatic flatness, sp² double-bond flatness, chirality preservation, MMFF94 internal strain, protein steric clashes, and volume overlap). You can also audit poses from Vina, Glide, or GOLD:

.. code-block:: bash

   pandadock validate-poses -p external_poses.sdf -r receptor.pdb -o audit.json

5. Congeneric Series SAR & MCS Core Tethering (``pandadock series``)
--------------------------------------------------------------------

Aligns an entire medicinal-chemistry analog series onto a reference scaffold using Maximum Common Substructure (MCS) flat-bottom tethering (``--core-tolerance 0.75`` Å), scores every analog with consensus metrics, and builds a publication A4 **SAR Brochure PDF** (``series_report.pdf`` + ``series_report.html``).

6. Covalent Warhead Docking (``pandadock covalent``)
----------------------------------------------------

Supports **8 built-in medicinal-chemistry warheads** (``michael_acceptor``, ``chloroacetamide``, ``nitrile``, ``epoxide``, ``boronic_acid``, ``aldehyde``, ``sulfonyl_fluoride``, ``disulfide``, or ``auto``), constructs the reacted adduct anchored to the target nucleophile (``--residue A:CYS:145``), samples the remaining flexible torsions, and produces PandaMap 2D/3D covalent interaction diagrams + PDF/HTML reports.

7. Few-Shot Per-Target Weight Calibration (``pandadock calibrate``)
-------------------------------------------------------------------

Fits target-specific Vina scoring weights from 5–25 known IC50/Ki/ΔG compounds using Ridge Leave-One-Out Cross-Validation (LOOCV), outputting ``calibrated_weights.json`` for ``pandadock dock --weights-file`` or ``pandadock series --weights-file``.

8. One-Command Dock-to-MD Handoff (``pandadock to-md`` / ``--export-md``)
-------------------------------------------------------------------------

Packages your best docked complex with clean coordinates, **OpenMM** Python launch scripts, **GROMACS** topology and ``em/nvt/npt/md.mdp`` inputs, and a **PandaInsight** post-MD trajectory analysis manifest.
