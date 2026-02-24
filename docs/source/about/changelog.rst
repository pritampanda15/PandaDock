Changelog
=========

Version 4.0.0 (2026)
--------------------

**Major Features:**

* **State-of-the-Art Performance**: Improved GNN training achieves:

  - Pearson R = 0.88 on PDBbind (5,316 complexes)
  - Pearson R = 0.82 on ULVSH held-out test set
  - Pearson R = 0.81 on BindingDB (8,891 complexes)
  - Pearson R = 0.68 on novel GABA receptor dataset (30 compounds)

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
