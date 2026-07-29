"""
PandaDock conformational search.

Components:
- TorsionTree: articulated ligand model exposing rotatable bonds as search DOF
- AffinityGrids: precomputed receptor interaction grids for fast pose scoring
- DockingObjective: energy and gradient over the DOF vector
- MonteCarloSearch: iterated local search over translation, orientation, torsions
"""

from .grid_maps import AffinityGrids, LigandTyping
from .monte_carlo import MonteCarloSearch, SearchConfig, SearchResult, cluster_poses
from .objective import DockingObjective
from .torsion_tree import Torsion, TorsionTree

__all__ = [
    "AffinityGrids",
    "LigandTyping",
    "DockingObjective",
    "MonteCarloSearch",
    "SearchConfig",
    "SearchResult",
    "cluster_poses",
    "Torsion",
    "TorsionTree",
]
