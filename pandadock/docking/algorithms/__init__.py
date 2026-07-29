"""
PandaDock docking algorithms.

Production algorithm:
- PandaCoreDocker: flexible-ligand Monte Carlo search with local optimization over
  translation, orientation and ligand torsions, scored against precomputed
  affinity grids. Registered as ``pandadock``.

`HierarchicalDocker` is the name under which `PandaCoreDocker` is registered for
the CLI. `MonteCarloDocker`, `GeneticAlgorithmDocker` and
`EnhancedHierarchicalDocker` are deprecated aliases kept so existing imports keep
working; each module's docstring records what was wrong with the implementation it
replaced. `CrystalGuidedDocker` raises on construction, because it biased poses
toward a reference structure.
"""

from .hierarchical_cpu import HierarchicalDocker
from .pandacore import PandaCoreDocker

# Deprecated aliases -- all resolve to PandaCoreDocker.
from .enhanced_hierarchical_cpu import EnhancedHierarchicalDocker
from .genetic_algorithm_cpu import GeneticAlgorithmDocker
from .monte_carlo_cpu import MonteCarloDocker

PandaDockAlgorithm = HierarchicalDocker

__all__ = [
    "PandaCoreDocker",
    "HierarchicalDocker",
    "PandaDockAlgorithm",
    "MonteCarloDocker",
    "GeneticAlgorithmDocker",
    "EnhancedHierarchicalDocker",
]
