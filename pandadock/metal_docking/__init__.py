"""
PandaDock Metal Docking Module

Specialized module for handling metal-containing ligands and metalloproteins
following Glide and AutoDock protocols.
"""

from .metal_core import MetalDockingEngine, MetalCoordinationGeometry
from .metal_parameters import MetalParameterManager
from .metal_scoring import MetalAwareScoring
from .metal_preparation import MetalLigandPreparator, MetalloproteinPreparator

__all__ = [
    'MetalDockingEngine',
    'MetalCoordinationGeometry',
    'MetalParameterManager',
    'MetalAwareScoring',
    'MetalLigandPreparator',
    'MetalloproteinPreparator'
]