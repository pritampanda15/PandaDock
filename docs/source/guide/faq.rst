Frequently Asked Questions (FAQ)
================================

**Should I strip crystallographic waters or keep them?**
   Start with ``--waters strip`` (default) for general screening. If your pocket contains conserved bridging waters (e.g., carbonic anhydrase, kinases, HIV protease), use ``--waters displace`` so PandaDock keeps bridging waters while displacing clashing solvent molecules.

**Which scoring function should I choose?**
   Use ``vina`` as your general-purpose default, ``vinardo`` for buried/hydrophobic pockets, ``--rescoring mmgbsa`` when comparing charged ligands, and ``pandadock calibrate`` / ``pandadock gnn predict`` when ranking potency across a congeneric series.

**When should I use CPU vs Apple Silicon MPS vs NVIDIA CUDA?**
   For a single small drug, ``--device cpu --workers 4`` finishes in seconds. For flexible ligands (10+ rotatable bonds) or virtual screening (``pandadock screen``), ``--device cuda`` or ``--device mps --n-chains 512`` provides 8×–35× speedups.

**How do I inspect binding-site hydrogens and 2D/3D interactions?**
   Run ``pandadock-gui --workspace <results_dir>`` to open **PandaDock Studio**. It displays the binding pocket with explicit hydrogens on both protein residues and ligand, all 18 **PandaLens** interaction classes in 3D, and a live interactive **2D View** with SVG export.

**Where can I download a printable PDF manual?**
   Download the complete 24-page plain-language manual directly here:
   :download:`PandaDock v5.0 Official User Manual (PDF) <../_static/PandaDock_v5.0_User_Manual.pdf>`.
