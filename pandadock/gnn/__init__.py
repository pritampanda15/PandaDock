"""
PandaDock-GNN: SE(3)-Equivariant Graph Neural Network for Molecular Docking

A novel deep learning scoring function that uses heterogeneous graph neural networks
with SE(3)-equivariant message passing to predict protein-ligand binding affinity.

Key Features:
- SE(3)-equivariant architecture for rotation/translation invariance
- Heterogeneous graph representation (protein + ligand nodes)
- Multi-task learning (affinity + activity prediction)
- Attention-based interpretability

Usage:
    from pandadock.gnn import PandaDockGNN, ULVSHDataset, GNNTrainer

    # Load dataset
    dataset = ULVSHDataset(root='path/to/ULVSH')

    # Create model
    model = PandaDockGNN(hidden_dim=256, num_layers=6)

    # Train
    trainer = GNNTrainer(model)
    trainer.train(dataset)

Reference:
    Satorras et al. "E(n) Equivariant Graph Neural Networks" (2021)
"""

__version__ = "1.0.0"

# Lazy imports to avoid dependency issues
def __getattr__(name):
    if name == "PandaDockGNN":
        from .models.pandadock_gnn import PandaDockGNN
        return PandaDockGNN
    elif name == "ULVSHDataset":
        from .data.dataset import ULVSHDataset
        return ULVSHDataset
    elif name == "GNNTrainer":
        from .training.trainer import GNNTrainer
        return GNNTrainer
    elif name == "GNNScoring":
        from .scoring import GNNScoring
        return GNNScoring
    elif name == "HeterogeneousGraphBuilder":
        from .data.graph_builder import HeterogeneousGraphBuilder
        return HeterogeneousGraphBuilder
    elif name == "MOL2Parser":
        from .data.mol2_parser import MOL2Parser
        return MOL2Parser
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "PandaDockGNN",
    "ULVSHDataset",
    "GNNTrainer",
    "GNNScoring",
    "HeterogeneousGraphBuilder",
    "MOL2Parser",
]
