"""
Batched GPU evaluation for the docking search.

This subpackage mirrors the validated CPU search rather than reimplementing it.
Every component here is expected to reproduce its CPU counterpart to within
floating-point tolerance, and the tests assert exactly that -- a GPU path that
is fast but scores differently is not an optimisation, it is a second engine
with unmeasured accuracy.

Nothing here is imported by the CPU docking path, and torch is an optional
extra, so `import pandadock` continues to work without it.
"""

from .grids import TorchAffinityGrids, torch_available

__all__ = ["TorchAffinityGrids", "torch_available"]
