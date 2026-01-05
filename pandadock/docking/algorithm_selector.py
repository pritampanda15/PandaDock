#!/usr/bin/env python3
"""
Intelligent Algorithm Selector for PandaDock

Automatically selects the optimal docking algorithm based on:
- Number of ligands to dock
- Conformer library size
- GPU availability
- User preferences for speed vs accuracy

Provides runtime warnings for problematic algorithms.
"""

import logging
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class AlgorithmSelector:
    """Intelligent algorithm selection based on workload characteristics"""

    # Algorithm performance characteristics from benchmark data
    ALGORITHM_PROFILES = {
        'enhanced_hierarchical_cpu': {
            'speed': 0.30,  # seconds per ligand
            'accuracy': 0.014,  # mean RMSD (Å)
            'success_rate': 100.0,  # percentage
            'best_for': 'single_ligand',
            'gpu_required': False,
            'reliable': True,
        },
        'genetic_algorithm_cpu': {
            'speed': 4.84,
            'accuracy': 2.246,
            'success_rate': 89.3,
            'best_for': 'complex_sites',
            'gpu_required': False,
            'reliable': True,
        },
        'monte_carlo_cpu': {
            'speed': 86.86,
            'accuracy': 2.207,
            'success_rate': 95.3,
            'best_for': 'exploration',
            'gpu_required': False,
            'reliable': True,
        },
        'hierarchical_cpu': {
            'speed': 17.90,
            'accuracy': 2.278,
            'success_rate': 94.7,
            'best_for': 'legacy',
            'gpu_required': False,
            'reliable': True,
        },
        'crystal_guided_cpu': {
            'speed': 3.91,
            'accuracy': 2.298,
            'success_rate': 100.0,
            'best_for': 'validation',
            'gpu_required': False,
            'reliable': True,
        },
        'enhanced_hierarchical_gpu': {
            'speed': 0.82,
            'accuracy': 0.015,
            'success_rate': 91.3,
            'best_for': 'batch_processing',
            'gpu_required': True,
            'reliable': True,
        },
        'cuda_genetic_algorithm': {
            'speed': 35.24,
            'accuracy': 0.014,
            'success_rate': 100.0,
            'best_for': 'large_conformers',
            'gpu_required': True,
            'reliable': True,
        },
        'cuda_monte_carlo': {
            'speed': 414.20,
            'accuracy': 0.0,
            'success_rate': 48.7,  # WARNING: Low success rate
            'best_for': 'experimental',
            'gpu_required': True,
            'reliable': False,  # Marked as unreliable
        },
    }

    @classmethod
    def auto_select(cls,
                   num_ligands: int = 1,
                   conformers_per_ligand: int = 10,
                   gpu_available: bool = False,
                   user_preference: str = 'balanced',
                   batch_mode: bool = False) -> str:
        """
        Automatically select the best algorithm for the given workload.

        Args:
            num_ligands: Number of ligands to dock
            conformers_per_ligand: Average conformers per ligand
            gpu_available: Whether GPU is available
            user_preference: 'speed', 'accuracy', or 'balanced'
            batch_mode: Whether running in batch processing mode

        Returns:
            Algorithm name (e.g., 'enhanced_hierarchical_cpu')
        """

        # Case 1: Single ligand, standard conformers (90% of use cases)
        if num_ligands == 1 and conformers_per_ligand < 100:
            if user_preference == 'speed':
                return 'genetic_algorithm_cpu'  # 4.84s, good accuracy
            else:
                return 'enhanced_hierarchical_cpu'  # 0.30s, best accuracy

        # Case 2: Large-scale virtual screening (1000+ ligands)
        if num_ligands >= 1000:
            if gpu_available:
                logger.info(
                    f"Large-scale screening detected ({num_ligands} ligands). "
                    "Using GPU batch processing for optimal throughput."
                )
                return 'enhanced_hierarchical_gpu'
            else:
                logger.info(
                    f"Large-scale screening detected ({num_ligands} ligands). "
                    "Consider using GPU acceleration for better throughput."
                )
                return 'enhanced_hierarchical_cpu'

        # Case 3: Large conformer libraries (100+ conformers per ligand)
        if conformers_per_ligand >= 100:
            if gpu_available:
                logger.info(
                    f"Large conformer library detected ({conformers_per_ligand} conformers/ligand). "
                    "Using GPU genetic algorithm for conformer parallelization."
                )
                return 'cuda_genetic_algorithm'
            else:
                logger.warning(
                    f"Large conformer library ({conformers_per_ligand} conformers/ligand) "
                    "detected. GPU acceleration recommended for better performance."
                )
                return 'enhanced_hierarchical_cpu'

        # Case 4: Batch processing mode (10-1000 ligands)
        if batch_mode or (10 <= num_ligands < 1000):
            if gpu_available:
                return 'enhanced_hierarchical_gpu'
            else:
                return 'enhanced_hierarchical_cpu'

        # Default: Best all-around algorithm
        return 'enhanced_hierarchical_cpu'

    @classmethod
    def get_algorithm_warning(cls, algorithm: str) -> Optional[str]:
        """
        Get warning message for problematic algorithms.

        Args:
            algorithm: Algorithm name

        Returns:
            Warning message or None
        """
        if algorithm not in cls.ALGORITHM_PROFILES:
            return f"Unknown algorithm: {algorithm}"

        profile = cls.ALGORITHM_PROFILES[algorithm]

        # Check for unreliable algorithms
        if not profile['reliable']:
            return (
                f"⚠️  WARNING: {algorithm} has {profile['success_rate']:.1f}% success rate in benchmarks.\n"
                f"   Recommended alternatives:\n"
                f"   - enhanced_hierarchical_cpu (100% success, 0.30s)\n"
                f"   - cuda_genetic_algorithm (100% success, GPU)"
            )

        # Check for superseded algorithms
        if algorithm == 'hierarchical_cpu':
            return (
                f"ℹ️  Note: {algorithm} is superseded by enhanced_hierarchical_cpu.\n"
                f"   enhanced_hierarchical_cpu: 0.30s vs {profile['speed']:.2f}s (60x faster)\n"
                f"   Consider using enhanced_hierarchical_cpu for better performance."
            )

        # Check for GPU without availability
        if profile['gpu_required']:
            return (
                f"ℹ️  GPU algorithm selected: {algorithm}\n"
                f"   Ensure CUDA is properly configured.\n"
                f"   Single ligand docking: CPU may be faster (0.30s vs {profile['speed']:.2f}s)"
            )

        return None

    @classmethod
    def get_recommendation(cls,
                          algorithm: str,
                          num_ligands: int = 1,
                          gpu_available: bool = False) -> str:
        """
        Get recommendation message for the selected algorithm.

        Args:
            algorithm: Selected algorithm
            num_ligands: Number of ligands
            gpu_available: GPU availability

        Returns:
            Recommendation message
        """
        profile = cls.ALGORITHM_PROFILES.get(algorithm)
        if not profile:
            return f"Using algorithm: {algorithm}"

        msg = f"🔧 Algorithm: {algorithm}\n"
        msg += f"   Expected time: ~{profile['speed']:.2f}s per ligand\n"
        msg += f"   Expected accuracy: {profile['accuracy']:.3f}Å RMSD\n"
        msg += f"   Success rate: {profile['success_rate']:.1f}%\n"

        # Add optimization suggestions
        if algorithm == 'enhanced_hierarchical_gpu' and num_ligands == 1:
            msg += "\n💡 Tip: For single ligand docking, enhanced_hierarchical_cpu is faster (0.30s vs 0.82s)"

        if algorithm == 'enhanced_hierarchical_cpu' and num_ligands >= 1000:
            if gpu_available:
                msg += "\n💡 Tip: For large-scale screening, consider enhanced_hierarchical_gpu for batch processing"

        return msg

    @classmethod
    def explain_selection(cls,
                         algorithm: str,
                         num_ligands: int,
                         conformers_per_ligand: int,
                         gpu_available: bool) -> str:
        """
        Explain why this algorithm was auto-selected.

        Args:
            algorithm: Selected algorithm
            num_ligands: Number of ligands
            conformers_per_ligand: Conformers per ligand
            gpu_available: GPU availability

        Returns:
            Explanation message
        """
        reasons = []

        if algorithm == 'enhanced_hierarchical_cpu':
            reasons.append("Best accuracy and speed for single/small-scale docking")
            reasons.append("0.30s per ligand, 0.014Å RMSD, 100% success rate")

        elif algorithm == 'enhanced_hierarchical_gpu':
            if num_ligands >= 1000:
                reasons.append(f"Optimized for large-scale screening ({num_ligands} ligands)")
            reasons.append("GPU batch processing reduces total wall-clock time")

        elif algorithm == 'cuda_genetic_algorithm':
            reasons.append(f"Large conformer library detected ({conformers_per_ligand} conformers/ligand)")
            reasons.append("GPU parallelizes conformer evaluation efficiently")

        elif algorithm == 'genetic_algorithm_cpu':
            reasons.append("Good balance of speed (4.84s) and accuracy")
            reasons.append("Suitable for complex binding sites")

        if reasons:
            return "🤖 Auto-selected because:\n   " + "\n   ".join(reasons)
        else:
            return f"Using algorithm: {algorithm}"


def detect_batch_mode(ligand_path: str) -> tuple[int, bool]:
    """
    Detect if running in batch mode by analyzing ligand input.

    Args:
        ligand_path: Path to ligand file(s)

    Returns:
        Tuple of (num_ligands, is_batch_mode)
    """
    path = Path(ligand_path)

    # Check if it's a directory
    if path.is_dir():
        ligand_files = list(path.glob('*.sdf')) + list(path.glob('*.mol2')) + list(path.glob('*.pdb'))
        return len(ligand_files), True

    # Check if it's a multi-molecule SDF
    if path.suffix.lower() == '.sdf':
        try:
            from rdkit import Chem
            suppl = Chem.SDMolSupplier(str(path))
            count = len([mol for mol in suppl if mol is not None])
            return count, count > 1
        except:
            pass

    # Single ligand
    return 1, False
