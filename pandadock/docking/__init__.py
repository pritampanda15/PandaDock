"""
PandaDock Molecular Docking System

A modern molecular docking platform featuring:

Main Algorithm:
- HierarchicalDocker: Multi-stage hierarchical search (traditional docking)

Scoring Functions:
- VinaScoring: Vina-style empirical scoring (default)
- PhysicsBasedScoring: Physics-based vdW + electrostatics
- EmpiricalScoring: Hydrophobic, desolvation, entropy terms

ML Scoring (via pandadock gnn):
- SE(3)-Equivariant GNN: Achieves R > 0.8 correlation with experiment

Advanced Features:
- Hybrid docking (traditional + GNN rescoring)
- MM-GBSA rescoring
- Boltzmann ensemble averaging
- Professional visualization and reporting
"""

__version__ = "3.0.0"
__author__ = "Pritam Kumar Panda @ Stanford University"

from .core import DockingEngine, DockingResult
from .algorithms import HierarchicalDocker
from .scoring import (
    VinaScoring,
    PhysicsBasedScoring,
    EmpiricalScoring,
    BoltzmannEnsemble
)

# Optional imports
try:
    from .refinement import (
        SimulatedAnnealingRefiner,
        MMGBSARescoring,
        EnsembleRefiner
    )
except ImportError:
    pass

try:
    from .visualization import (
        DockingVisualizer,
        AffinityPlotter,
        PoseAnalyzer
    )
except ImportError:
    pass

__all__ = [
    'DockingEngine',
    'DockingResult',
    'HierarchicalDocker',
    'VinaScoring',
    'PhysicsBasedScoring',
    'EmpiricalScoring',
    'BoltzmannEnsemble',
]
