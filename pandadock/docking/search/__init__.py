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
from .restraints import (
    DistanceRestraint,
    RestraintSet,
    build_mcs_restraints,
    load_restraints_json,
    parse_restraint_spec,
)
from .torsion_tree import Torsion, TorsionTree
from .waters import PocketWater, dock_with_water_mode, find_pocket_waters

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
    "DistanceRestraint",
    "RestraintSet",
    "build_mcs_restraints",
    "load_restraints_json",
    "parse_restraint_spec",
    "PocketWater",
    "find_pocket_waters",
    "dock_with_water_mode",
]
