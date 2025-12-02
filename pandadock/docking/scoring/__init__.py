"""
Modular Scoring Function System for PandaDock

Physics-Based Scoring:
- Van der Waals interactions (Lennard-Jones)
- Electrostatic interactions (Coulomb)
- Hydrogen bonding
- Solvation/desolvation

Empirical Scoring:
- Hydrophobic interactions
- Entropy penalties
- Metal coordination
- Special interaction rewards

Precision Scoring:
- Complete precision scoring function implementation
- Hierarchical energy terms
- Specialized interaction detection

Ensemble Methods:
- Boltzmann averaging
- Free energy estimation
- Confidence scoring
"""

from .physics_based import PhysicsBasedScoring
from .empirical import EmpiricalScoring
from .precision_score import PrecisionScoring
from .hybrid import HybridScoring
from .ensemble import BoltzmannEnsemble

# GPU scoring functions (optional)
try:
    from .gpu_scoring import GPUPrecisionScoring, GPUMMGBSAScoring
    GPU_SCORING_AVAILABLE = True
except ImportError:
    GPUPrecisionScoring = None
    GPUMMGBSAScoring = None
    GPU_SCORING_AVAILABLE = False

__all__ = [
    'PhysicsBasedScoring',
    'EmpiricalScoring',
    'PrecisionScoring',
    'HybridScoring',
    'BoltzmannEnsemble',
    'GPUPrecisionScoring',
    'GPUMMGBSAScoring',
    'GPU_SCORING_AVAILABLE'
]