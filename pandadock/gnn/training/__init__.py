"""
Training pipeline for PandaDock-GNN.

Modules:
- losses: Multi-task loss functions
- metrics: Evaluation metrics (Pearson R, AUC, etc.)
- trainer: Training loop with early stopping
"""

from .losses import MultiTaskLoss, AffinityLoss, ActivityLoss
from .metrics import compute_metrics, PearsonR, SpearmanRho
from .trainer import GNNTrainer, TrainingConfig

__all__ = [
    "MultiTaskLoss",
    "AffinityLoss",
    "ActivityLoss",
    "compute_metrics",
    "PearsonR",
    "SpearmanRho",
    "GNNTrainer",
    "TrainingConfig",
]
