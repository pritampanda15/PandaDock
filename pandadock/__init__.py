# __init__.py
# Define version
__version__ = '1.3.1'

# Simple imports first to avoid circular dependencies
from .protein import Protein
from .ligand import Ligand
from .scoring import (
    ScoringFunction, 
    CompositeScoringFunction, 
    EnhancedScoringFunction
)
from .search import RandomSearch, GeneticAlgorithm
from .utils import save_docking_results, calculate_rmsd
from .preparation import prepare_protein, prepare_ligand
from .validation import validate_docking, calculate_ensemble_rmsd
