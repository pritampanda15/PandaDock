"""
PandaDock docking algorithms.

Production algorithm:
- PandaCoreDocker: flexible-ligand Monte Carlo search with local optimization over
  translation, orientation and ligand torsions, scored against precomputed
  affinity grids. Registered as ``pandadock``. Supports CPU (default) and
  batched PyTorch GPU search (``--device cuda`` or ``--device mps``) via
  ``pandadock.docking.gpu``.

`HierarchicalDocker` is the name under which `PandaCoreDocker` is registered for
the CLI. `MonteCarloDocker`, `GeneticAlgorithmDocker` and
`EnhancedHierarchicalDocker` are deprecated aliases kept so existing imports keep
working; each module's docstring records what was wrong with the implementation it
replaced. `CrystalGuidedDocker` raises on construction, because it biased poses
toward a reference structure.
"""

import importlib.util

from .hierarchical_cpu import HierarchicalDocker
from .pandacore import PandaCoreDocker

# Deprecated aliases -- all resolve to PandaCoreDocker.
from .enhanced_hierarchical_cpu import EnhancedHierarchicalDocker
from .genetic_algorithm_cpu import GeneticAlgorithmDocker
from .monte_carlo_cpu import MonteCarloDocker

PandaDockAlgorithm = HierarchicalDocker


def _detect_gpu_available() -> bool:
    """Return True if PyTorch is installed with a CUDA or Apple MPS device."""
    if importlib.util.find_spec("torch") is None:
        return False
    try:
        import torch

        if torch.cuda.is_available():
            return True
        mps = getattr(torch.backends, "mps", None)
        return bool(mps is not None and mps.is_available())
    except Exception:
        return False


def __getattr__(name: str):
    # Resolved lazily (PEP 562) so CPU docking workers never pay the RAM and
    # startup cost of importing PyTorch unless a caller explicitly asks for
    # `GPU_AVAILABLE` (e.g. `pandadock-ml` or capability checks).
    if name == "GPU_AVAILABLE":
        value = _detect_gpu_available()
        globals()["GPU_AVAILABLE"] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "GPU_AVAILABLE",
    "PandaCoreDocker",
    "HierarchicalDocker",
    "PandaDockAlgorithm",
    "MonteCarloDocker",
    "GeneticAlgorithmDocker",
    "EnhancedHierarchicalDocker",
]
