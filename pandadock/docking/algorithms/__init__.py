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

# There is no GPU search path. The conformational search runs on CPU, and the
# legacy CUDA modules were never wired into the engine: their chromosome encoded
# only translation and orientation, so they could not vary ligand conformation.
# The flag is kept because `pandadock-ml` imports it; that import previously
# raised ImportError, which made the whole command unusable.
GPU_AVAILABLE = False

__all__ = [
    "GPU_AVAILABLE",
    "PandaCoreDocker",
    "HierarchicalDocker",
    "PandaDockAlgorithm",
    "MonteCarloDocker",
    "GeneticAlgorithmDocker",
    "EnhancedHierarchicalDocker",
]
