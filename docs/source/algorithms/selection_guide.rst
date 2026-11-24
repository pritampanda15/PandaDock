Algorithm Selection Guide
=========================

Choosing the right workflow in PandaDock v4.1+:

+--------------------------------+--------------------------------------------+
| Task                           | Recommended Command                        |
+================================+============================================+
| Standard single-ligand docking | ``pandadock dock --scoring vina``          |
+--------------------------------+--------------------------------------------+
| Alternative empirical scoring  | ``pandadock dock --scoring vinardo``       |
+--------------------------------+--------------------------------------------+
| Fast GPU single-ligand docking | ``pandadock dock --device cuda``           |
+--------------------------------+--------------------------------------------+
| Virtual screening (library)    | ``pandadock screen --device cuda``         |
+--------------------------------+--------------------------------------------+
| Affinity estimation (GNN)      | ``pandadock-gnn predict`` (after ``dock``) |
+--------------------------------+--------------------------------------------+
| Induced-fit side chains (exp.) | ``pandadock-flex``                         |
+--------------------------------+--------------------------------------------+
| Metalloproteins (exp.)         | ``pandadock-metal dock``                   |
+--------------------------------+--------------------------------------------+
| Tethered restraint (exp.)      | ``pandadock-tethered dock``                |
+--------------------------------+--------------------------------------------+

Key Practical Guidance
----------------------

1. **Snug box beats extra sampling:** Use 5 A padding around a known ligand
   (``pandadock-gridbox -m similarity``) rather than an oversized box.
2. **Do not use the affinity GNN for pose selection:** Select poses with
   ``pandadock dock`` (Vina/Vinardo), then estimate affinity with
   ``pandadock-gnn predict``.
3. **Reproducibility:** Always pass ``--seed`` for deterministic runs.
4. **Removed algorithms:** ``crystal_guided_cpu`` and the legacy v3.0
   ``*_gpu`` algorithm names were removed; use ``pandadock dock --device cuda``
   for GPU acceleration.
