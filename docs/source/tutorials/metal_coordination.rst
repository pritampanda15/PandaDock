Metal Coordination Docking Tutorial
===================================

Metalloproteins (Zn²⁺, Fe²⁺/³⁺, Mg²⁺, Cu²⁺, Mn²⁺) require specialized coordination geometry handling. PandaDock v5.0 provides two complementary workflows.

Option 1 — Restrained Docking with Displaceable Waters (``pandadock dock``)
---------------------------------------------------------------------------

.. code-block:: bash

   pandadock dock -r 3hs4_zn_hoh.pdb -l azm.sdf \
                  --center -5.3 -0.5 14.8 --box 20 20 20 \
                  --waters displace --restraint "A:ZN:301:ZN<2.3:donor" \
                  -o ca2_zn_run/

Option 2 — Dedicated Metalloprotein Suite (``pandadock-metal``)
---------------------------------------------------------------

.. code-block:: bash

   pandadock-metal analyze-protein -r zinc_enzyme.pdb -o metal_site.json
   pandadock-metal prepare-ligand -i chelator.sdf -o chelator_prep.sdf
   pandadock-metal dock -r zinc_enzyme.pdb -l chelator_prep.sdf \
                        --center -5.3 -0.5 14.8 --box 20 20 20 -o metal_out/

See :doc:`../cli/pandadock_metal` for all 5 subcommands and 31 supported metals.
