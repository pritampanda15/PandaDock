pandadock-gridbox - Binding Site Definition
===========================================

Defines the 3D docking search box (``box.json``) around your protein's binding pocket using a known reference X-ray ligand, automatic blind cavity detection, named active-site residues, pharmacophore features, or manual coordinates.

Synopsis
--------

.. code-block:: bash

   pandadock-gridbox -r RECEPTOR.pdb -m MODE [OPTIONS]

Modes (``-m / --mode``)
-----------------------

* ``similarity`` — Center and size the box around a known 3D X-ray/reference ligand (``--reference-ligand``) plus ``--padding``.
* ``cavities`` — Automatic geometric cavity detection to find the top druggable pockets on the protein surface.
* ``residues`` — Wrap the box tightly around specific active-site amino acids (e.g. ``--residues A:41,A:145,A:163``).
* ``manual`` — Specify exact ``-c X Y Z`` center and ``-b DX DY DZ`` dimensions directly.
* ``interactive`` — Guided terminal prompt.

Options
-------

.. list-table::
   :header-rows: 1
   :widths: 28 15 57

   * - Option
     - Default
     - Description
   * - ``-r, --receptor PATH``
     - Required
     - Prepared receptor PDB file.
   * - ``-m, --mode TEXT``
     - ``interactive``
     - ``similarity``, ``cavities``, ``residues``, ``manual``, or ``interactive``.
   * - ``-c, --center X Y Z``
     - —
     - Manual grid center in Å (used with ``-m manual``).
   * - ``-b, --box DX DY DZ``
     - —
     - Manual grid dimensions in Å (typically ``20 20 20`` to ``24 24 24``).
   * - ``-s, --spacing FLOAT``
     - ``0.375``
     - Affinity grid point spacing in Å.
   * - ``--reference-ligand PATH``
     - —
     - Reference 3D ligand file for ``-m similarity``.
   * - ``--residues TEXT``
     - —
     - Comma-separated residue list for ``-m residues`` (e.g. ``A:123,B:67``).
   * - ``--padding FLOAT``
     - ``5.0``
     - Extra breathing margin (Å) added around the reference ligand or residues.
   * - ``--pharmacophore TEXT``
     - —
     - Comma-separated features (``hydrophobic,donor,acceptor,aromatic,positive,negative``).
   * - ``--max-sites INT``
     - ``3``
     - Maximum top cavities to return.
   * - ``--min-volume FLOAT``
     - ``50.0``
     - Minimum cavity volume (Å³) for cavity detection.
   * - ``--confidence-threshold FLOAT``
     - ``0.7``
     - Minimum prediction confidence (0–1).
   * - ``-o, --output PATH`` / ``--output-dir PATH``
     - — / ``gridbox_output``
     - Output JSON path (pass directly to ``pandadock dock -g box.json``) or output directory.
   * - ``--visualize`` / ``-v, --verbose``
     - ``False``
     - Export box visualization files and verbose cavity diagnostics.

Examples
--------

.. code-block:: bash

   # Center on a known X-ray ligand with 5 Å padding
   pandadock-gridbox -r receptor.pdb -m similarity --reference-ligand xray.sdf --padding 5.0 -o box.json

   # Automatic blind pocket detection
   pandadock-gridbox -r receptor.pdb -m cavities --max-sites 3 --min-volume 80 -o box.json

   # Center on active-site residues
   pandadock-gridbox -r receptor.pdb -m residues --residues A:41,A:145,A:163 --padding 6.0 -o box.json
