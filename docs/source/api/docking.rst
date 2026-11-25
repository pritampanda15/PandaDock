Docking Python API
==================

Core docking, covalent docking, congeneric series, and PoseBusters validation modules in **PandaDock v5.0**:

* ``pandadock.docking.core`` — ``Pose``, ``DockingResult``, ``BaseDockingAlgorithm``
* ``pandadock.docking.algorithms.flexible_search`` — Analytic-gradient MC + L-BFGS CPU/GPU search
* ``pandadock.docking.scoring.objective`` — Differentiable Vina/Vinardo objective & flat-bottom restraints
* ``pandadock.docking.validation.posebusters`` — Built-in 11-point PoseBusters audit
* ``pandadock.covalent`` — Covalent warhead reaction templates & rooted covalent docker
* ``pandadock.series`` — Congeneric series MCS core-tethered docking & SAR brochure builder
* ``pandadock.preprocessing.states`` — pH-aware protonation, tautomer & stereo enumerator
* ``pandadock.md_handoff`` — OpenMM, GROMACS & PandaInsight MD package generator
