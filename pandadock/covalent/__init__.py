"""
PandaDock Covalent Docking Engine (v5.0).

Supports reaction-SMARTS covalent warhead transformation for 8 medicinal-chemistry
warhead classes anchored to nucleophilic receptor residues (Cys, Ser, Thr, Lys, Tyr, His)
with rooted torsion-tree optimization inside receptor affinity grids.
"""

from .warheads import WARHEAD_LIBRARY, WarheadReaction, detect_warhead, react_ligand_with_warhead
from .covalent_docker import CovalentDocker, CovalentDockingConfig

__all__ = [
    "WARHEAD_LIBRARY",
    "WarheadReaction",
    "detect_warhead",
    "react_ligand_with_warhead",
    "CovalentDocker",
    "CovalentDockingConfig",
]
