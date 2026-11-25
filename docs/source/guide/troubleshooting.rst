Troubleshooting
===============

Common issues and quick fixes when running **PandaDock v5.0**.

Ligand Docks Outside the Expected Pocket
----------------------------------------

* **Cause:** Grid box center coordinates do not match the receptor coordinate frame, or the old X-ray ligand was left inside the receptor PDB and physically blocked the pocket.
* **Fix:** Preview the receptor and box in **PandaDock Studio** (``pandadock-gui``) with **Box** enabled, or click the detected Co-Crystal Ligand chip so Studio centers the box and excludes the old ligand automatically.

PoseBusters Flags High Internal Strain or Clashes
-------------------------------------------------

* **Cause:** Starting ligand had unminimized 2D/strained geometry, or the pocket is tight around a bulky substituent.
* **Fix:** Run ``pandadock-prepare --minimize-ligand``, increase ``--exhaustiveness 16``, enable ``--sample-rings`` for saturated rings, or use ``pandadock-flex`` for induced-fit side-chain relaxation.

Catalytic Metal Ion (Zn/Fe/Mg) Not Coordinating
-----------------------------------------------

* **Cause:** Generic empirical grids underestimate directional metal chelation.
* **Fix:** Add a smooth restraint in ``pandadock dock`` (e.g. ``--restraint "A:ZN:301:ZN<2.3:donor"``) or use the dedicated ``pandadock-metal dock`` suite.

Port 8765 Already in Use When Launching Studio
----------------------------------------------

* **Fix:** Specify any open port, e.g. ``pandadock-gui --port 8785 --workspace .``.

GPU Out-of-Memory During Virtual Screening
------------------------------------------

* **Fix:** Lower ``--max-batch 32`` or ``--n-chains 128`` in ``pandadock screen``.
