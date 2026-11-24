Changelog
=========

Version 4.1.1+ (2026)
---------------------

**Major Improvements:**

* **Real Flexible-Ligand Conformational Search**: Replaced legacy rigid/biased search with torsion-tree + Vina/Vinardo affinity grids + analytic-gradient MC/L-BFGS search (CPU and experimental PyTorch CUDA/MPS GPU backends).
* **Virtual Screening**: Added ``pandadock screen`` for multi-ligand GPU batching with automatic size bucketing and signature-keyed grid caching.
* **SAIR GNN Training & Scientific Audit**: Trained on 741,706 co-folded SAIR complexes under target-disjoint splits (R = 0.531 independent on PDBbind v2020 refined, n = 4,640; PDBbind-only model R = 0.690 on held-out test, n = 624). Withdrew earlier v4.0 R = 0.88 / R = 0.82 claims and documented pose-rescoring limitations.
* **Scoring, GPU & Flex Fixes**: Corrected Vina receptor/ligand hydrophobic and H-bond atom typing, eliminated redundant L-BFGS closure calls on GPU, added CPU multi-worker MC, implemented real chi-angle side-chain rotation and strain/RMSD calculations in ``pandadock-flex``, and unified checkpoint metadata across ``pandadock gnn`` and ``pandadock-gnn``.

Version 4.0.0 (2026)
--------------------

**Major Features:**

* **PandaDock-GNN Release**: Initial SE(3)-equivariant GNN architecture, ``pandadock gnn rescore``, ``pandadock gnn download-model``, and multi-dataset training pipeline. *(Note: original v4.0 benchmark figures were subsequently revised downward in v4.1.1 after full-dataset re-evaluation; see README.md.)*

* **Universal GNN Rescorer**: New ``pandadock gnn rescore`` command

  - Rescore poses from ANY docking software (Vina, Glide, GOLD, etc.)
  - Input: Standard SDF file with multiple conformers
  - Output: CSV with GNN scores + annotated SDF

* **Model Download**: New ``pandadock gnn download-model`` command

  - Download pre-trained model from GitHub releases
  - No training required for quick start

* **Combined Training**: Train on merged datasets

  - Support for PDBbind + ULVSH + BindingDB combined training
  - BindingDB training script with 8,891 protein-ligand complexes
  - Unified pK scale across different affinity types
  - BindingDB + ULVSH combined achieves R = 0.79 test correlation

**CLI Additions:**

* ``pandadock gnn rescore`` - Universal pose rescoring
* ``pandadock gnn download-model`` - Pre-trained model download

**Documentation:**

* Complete publication manuscript with algorithms and benchmarks
* Updated benchmark tables with improved performance metrics

Version 3.0.0 (2024)
--------------------

**Major Features:**

* **PandaDock-GNN**: New SE(3)-equivariant Graph Neural Network scoring function

  - Achieves Pearson R = 0.67 on ULVSH benchmark
  - Outperforms all 8 baseline scoring methods
  - Multi-task learning: pEC50 regression + activity classification
  - Heterogeneous graph representation (protein + ligand nodes)

* **Hybrid Docking Workflow**: New ``pandadock hybrid`` command

  - Combines traditional pose generation with GNN rescoring
  - Recommended approach for best accuracy
  - Generates diverse poses, rescores with GNN, ranks by pEC50

* **Vina-Style Scoring**: New default scoring function

  - Implements AutoDock Vina empirical weights
  - Gaussian steric interactions, hydrophobic, hydrogen bonding
  - Rotatable bond flexibility penalty

**CLI Changes:**

* New commands:

  - ``pandadock hybrid`` - Hybrid docking with GNN rescoring
  - ``pandadock gnn train`` - Train GNN model
  - ``pandadock gnn predict`` - Predict binding affinity
  - ``pandadock gnn benchmark`` - Benchmark model performance
  - ``pandadock gnn compare`` - Compare against baselines

* Simplified algorithm selection:

  - Single ``pandadock`` algorithm (hierarchical search)
  - Removed broken GPU algorithms
  - Default scoring changed to ``vina``

**Architecture Changes:**

* Simplified algorithm module: only HierarchicalDocker retained
* Added ``VinaScoring`` class with standard Vina weights
* GNN module structure: data/, models/, training/, scoring.py
* Multi-format molecule parsing: MOL2, PDB, SDF

**Bug Fixes:**

* Fixed edge dimension mismatch in GNN (23 dims, not 21)
* Fixed target tensor squeeze issue in training
* Fixed checkpoint saving (ModelConfig vs TrainingConfig)
* Fixed scatter dtype mismatch on CUDA with AMP

**Dependencies:**

* Added optional ``[gnn]`` extras: torch, torch-geometric, torch-scatter, torch-sparse
* Updated version to 3.0.0

Version 2.x
-----------

Previous versions focused on traditional docking algorithms.

Version 1.x
-----------

Initial release with basic docking functionality.

.. note::

   ULVSH affinity labels are censored: inactive compounds take a floor
   value, and 89 of the 95 test-split pEC50 values are exactly 4.0. A
   Pearson correlation over that distribution reflects separation of
   actives from the floor rather than affinity ranking, so ULVSH is
   reported as activity classification (AUC = 0.94). Use PDBbind or
   BindingDB for continuous affinity performance.
