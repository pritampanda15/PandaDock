pandadock-prepare - Structure Preparation
=========================================

Prepares raw protein (PDB/mmCIF) and ligand (SDF/MOL2/PDB) files for molecular docking by adding missing hydrogens, assigning pH-dependent protonation states and partial charges, generating relaxed 3D ligand geometry, and optionally enumerating protonation/tautomer/stereoisomer microspecies.

Synopsis
--------

.. code-block:: bash

   pandadock-prepare -r RECEPTOR.pdb -l LIGAND.sdf [OPTIONS]

Options
-------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Option
     - Default
     - Description
   * - ``-r, --receptor PATH``
     - Required
     - Input receptor structure in PDB or mmCIF format.
   * - ``-l, --ligand PATH``
     - Required
     - Input ligand file in SDF, MOL2, or PDB format.
   * - ``-o, --output-dir PATH``
     - ``pandadock_output``
     - Output directory for prepared receptor, ligand(s), and state report.
   * - ``--ph FLOAT``
     - ``7.0``
     - Target pH for protonation calculation (use ``7.4`` for physiological conditions).
   * - ``--add-hydrogens / --no-add-hydrogens``
     - ``True``
     - Add explicit 3D hydrogen atoms to receptor and ligand.
   * - ``--add-charges / --no-add-charges``
     - ``True``
     - Assign Gasteiger partial charges.
   * - ``--minimize-ligand / --no-minimize-ligand``
     - ``True``
     - Energy-minimize ligand 3D conformer using MMFF94.
   * - ``--fix-missing-residues``
     - ``False``
     - Rebuild missing protein residues/atoms (leave off unless missing atoms are inside the binding site).
   * - ``--enumerate-states`` *(v5.0)*
     - ``False``
     - Enumerate pH-aware protonation states, tautomers, and unassigned chiral stereoisomers with Henderson–Hasselbalch state penalties.

Examples
--------

.. code-block:: bash

   # Standard preparation at physiological pH 7.4
   pandadock-prepare -r raw_protein.pdb -l ligand.sdf --ph 7.4 -o prepared/

   # Prepare + enumerate pH 7.4 microspecies and stereoisomers
   pandadock-prepare -r protein.pdb -l ketamine.sdf --ph 7.4 --enumerate-states -o prepared_states/
