"""
PandaDock: Python Molecular Docking Tool Package
"""

# Core protein-ligand handling
from .protein import Protein
from .ligand import Ligand

# Unified scoring functions
from .unified_scoring import (
    ScoringFunction,
    CompositeScoringFunction,
    EnhancedScoringFunction,
    GPUScoringFunction,
    EnhancedGPUScoringFunction,
    TetheredScoringFunction,
)

# Physics-based modules
from .physics import (
    MMFFMinimization,
    MonteCarloSampling,
    PhysicsBasedScoring,
    GeneralizedBornSolvation
)

# Search algorithms
from .search import RandomSearch, GeneticAlgorithm

# Utilities
from .utils import (
    save_docking_results,
    calculate_rmsd
)

# Molecule preparation
from .preparation import (
    prepare_protein,
    prepare_ligand
)

# Validation
from .validation import (
    validate_docking,
    calculate_ensemble_rmsd
)

# Batch screening
from .batch_screening import batch_screening

# Package version
__version__ = '2.1.0'

# Logging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
