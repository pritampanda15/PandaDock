# __init__.py
# Import key modules to make them available when importing the package

from .protein import Protein
from .ligand import Ligand
from .scoring import ScoringFunction, CompositeScoringFunction, EnhancedScoringFunction
from .search import RandomSearch, GeneticAlgorithm
from .hybrid_manager import HybridDockingManager
from .main_integration import (
    add_hardware_options, 
    configure_hardware,
    setup_hardware_acceleration
)
from .utils import save_docking_results, calculate_rmsd
from .preparation import prepare_protein, prepare_ligand
from .validation import validate_docking, calculate_ensemble_rmsd

# Import the batch_screening module
from . import batch_screening

# Define version
__version__ = '1.3'
