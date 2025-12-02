"""
Docking Algorithms Module

CPU Algorithms:
- MonteCarloDocker: Basic Monte Carlo sampling with simulated annealing
- GeneticAlgorithmDocker: Genetic algorithm with ensemble refinement
- HierarchicalDocker: Original hierarchical search
- EnhancedHierarchicalDocker: Enhanced 3-stage hierarchical search (high accuracy)
- CrystalGuidedDocker: Crystal-guided sampling with clash avoidance (crystal reproduction)

GPU Algorithms:
- CUDAMonteCarloDocker: CUDA-accelerated Monte Carlo
- CUDAGeneticAlgorithmDocker: GPU-parallel genetic algorithm
- EnhancedHierarchicalGPUDocker: GPU-accelerated hierarchical search
"""

from .monte_carlo_cpu import MonteCarloDocker
from .genetic_algorithm_cpu import GeneticAlgorithmDocker
from .hierarchical_cpu import HierarchicalDocker
from .enhanced_hierarchical_cpu import EnhancedHierarchicalDocker
from .crystal_guided_cpu import CrystalGuidedDocker

# GPU algorithms (will check for CUDA availability)
try:
    from .monte_carlo_gpu import CUDAMonteCarloDocker
    from .genetic_algorithm_gpu import CUDAGeneticAlgorithmDocker
    from .enhanced_hierarchical_gpu import EnhancedHierarchicalGPUDocker
    GPU_AVAILABLE = True
except ImportError:
    CUDAMonteCarloDocker = None
    CUDAGeneticAlgorithmDocker = None
    EnhancedHierarchicalGPUDocker = None
    GPU_AVAILABLE = False

__all__ = [
    'MonteCarloDocker',
    'GeneticAlgorithmDocker',
    'HierarchicalDocker',
    'EnhancedHierarchicalDocker',
    'CrystalGuidedDocker',
    'CUDAMonteCarloDocker',
    'CUDAGeneticAlgorithmDocker',
    'EnhancedHierarchicalGPUDocker',
    'GPU_AVAILABLE'
]