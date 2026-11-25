Flexible / Induced-Fit Docking Tutorial
=======================================

Use ``pandadock-flex`` when pocket side chains must move to accommodate a bulky inhibitor.

4-Phase Induced-Fit Protocol
----------------------------

1. Initial soft-potential docking (``--soft-potential-scale 0.7``)
2. Receptor side-chain χ₁/χ₂ refinement (``--refine-distance 6.0``)
3. High-precision redocking at full VDW scale (``1.0``)
4. Composite IFD ranking

Example
-------

.. code-block:: bash

   pandadock-flex -r receptor.pdb -l bulky_ligand.sdf \
                  --center 10.0 20.0 30.0 --radius 12.0 \
                  --flexible-residues "A:123,A:145" \
                  --final-poses-to-retain 10 -o ifd_run

See :doc:`../cli/pandadock_flex` for all options.
