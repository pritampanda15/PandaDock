PandaDock-GNN Python API
========================

SE(3)-equivariant heterogeneous Graph Neural Network scoring modules (v5 dual-head architecture):

* ``pandadock.gnn.models.egnn`` — 6-layer SE(3)-equivariant EGNN with optional pose-RMSD head (``predict_pose_rmsd=True``)
* ``pandadock.gnn.data`` — Protein–ligand heterogeneous graph builder & SAIR shard loader
* ``pandadock.gnn.training.trainer`` — Target-disjoint training loop with within-target ranking loss
* ``pandadock.gnn.scoring`` — Universal pose affinity predictor & rescorer
* ``pandadock.gnn.cli`` — Dedicated ``pandadock-gnn`` / ``pandadock gnn`` CLI
