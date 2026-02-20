"""
PandaDock Docking Algorithms Module

Primary Algorithm:
- HierarchicalDocker: Multi-stage hierarchical search with coarse-to-fine refinement
  (Best performing traditional algorithm, Pearson R ~0.12 on ULVSH dataset)

For highest accuracy, use PandaDock-GNN (pandadock gnn) which achieves R > 0.8
"""

from .hierarchical_cpu import HierarchicalDocker

# For backwards compatibility, also expose under alternate names
PandaDockAlgorithm = HierarchicalDocker

__all__ = [
    'HierarchicalDocker',
    'PandaDockAlgorithm',
]
