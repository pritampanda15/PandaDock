"""
PandaDock Molecular Docking System

A modern, GPU-accelerated molecular docking platform with multiple algorithms:

CPU Algorithms:
- Monte Carlo (MC) docking with simulated annealing
- Genetic Algorithm (GA) docking with ensemble refinement
- Hierarchical docking inspired by Glide

GPU Algorithms:
- CUDA-accelerated Monte Carlo sampling
- GPU-parallel genetic algorithm
- Batch pose evaluation and scoring

Scoring Functions:
- Physics-based scoring (vdW, electrostatics, H-bonds)
- Empirical scoring terms (hydrophobic, desolvation, entropy)
- Glide-inspired modular scoring
- Boltzmann ensemble averaging

Advanced Features:
- MM-GBSA rescoring
- Ensemble binding free energy calculation
- Professional visualization and reporting
"""

__version__ = "1.0.0"
__author__ = "PandaDock Development Team"

from .core import DockingEngine, DockingResult
from .algorithms import (
    MonteCarloDocker,
    GeneticAlgorithmDocker,
    HierarchicalDocker,
    CUDAMonteCarloDocker,
    CUDAGeneticAlgorithmDocker
)
from .scoring import (
    PhysicsBasedScoring,
    EmpiricalScoring,
    PrecisionScoring,
    BoltzmannEnsemble
)
from .refinement import (
    SimulatedAnnealingRefiner,
    MMGBSARescoring,
    EnsembleRefiner
)
from .visualization import (
    DockingVisualizer,
    AffinityPlotter,
    PoseAnalyzer
)

__all__ = [
    'DockingEngine',
    'DockingResult',
    'MonteCarloDocker',
    'GeneticAlgorithmDocker',
    'HierarchicalDocker',
    'CUDAMonteCarloDocker',
    'CUDAGeneticAlgorithmDocker',
    'PhysicsBasedScoring',
    'EmpiricalScoring',
    'GlideScoring',
    'BoltzmannEnsemble',
    'SimulatedAnnealing',
    'MMGBSARescoring',
    'EnsembleRefinement',
    'DockingVisualizer',
    'AffinityPlotter',
    'PoseAnalyzer'
]