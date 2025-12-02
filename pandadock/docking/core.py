"""
Core docking infrastructure for PandaDock

Provides base classes and data structures for molecular docking algorithms.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import json
import time
from Bio.PDB import PDBParser, PDBIO, Structure
from rdkit import Chem
from rdkit.Chem import AllChem


class DockingAlgorithm(Enum):
    """Available docking algorithms"""
    MONTE_CARLO = "monte_carlo"
    GENETIC_ALGORITHM = "genetic_algorithm"
    HIERARCHICAL = "hierarchical"
    CUDA_MONTE_CARLO = "cuda_monte_carlo"
    CUDA_GENETIC_ALGORITHM = "cuda_genetic_algorithm"


class ScoringFunction(Enum):
    """Available scoring functions"""
    PHYSICS_BASED = "physics_based"
    EMPIRICAL = "empirical"
    GLIDE_SCORE = "glide_score"
    HYBRID = "hybrid"


@dataclass
class Pose:
    """Represents a single ligand pose"""
    coordinates: np.ndarray  # 3D coordinates of ligand atoms
    center: np.ndarray  # Center of mass
    rotation: np.ndarray  # Rotation quaternion
    conformer_id: int  # Conformer index
    energy: float = 0.0  # Total binding energy
    energy_components: Dict[str, float] = field(default_factory=dict)
    internal_strain: float = 0.0  # Ligand internal strain energy
    confidence: float = 0.0  # Confidence score
    rmsd: float = 0.0  # RMSD from reference (if available)

    def to_dict(self) -> Dict[str, Any]:
        """Convert pose to dictionary for serialization"""
        # Use original coordinates for ultra-fast poses to prevent corruption
        coords_to_save = self.coordinates
        if hasattr(self, '_ultra_fast_mode') and self._ultra_fast_mode and hasattr(self, '_original_coords'):
            coords_to_save = self._original_coords

        return {
            'coordinates': coords_to_save.tolist(),
            'center': self.center.tolist(),
            'rotation': self.rotation.tolist(),
            'conformer_id': self.conformer_id,
            'energy': self.energy,
            'energy_components': self.energy_components,
            'internal_strain': self.internal_strain,
            'confidence': self.confidence,
            'rmsd': self.rmsd
        }


@dataclass
class DockingResult:
    """Results from molecular docking"""
    ligand_name: str
    receptor_file: str
    grid_center: np.ndarray
    grid_dimensions: np.ndarray
    algorithm_used: str
    scoring_function: str
    poses: List[Pose] = field(default_factory=list)
    ensemble_binding_energy: float = 0.0
    ensemble_confidence: float = 0.0
    boltzmann_weights: List[float] = field(default_factory=list)
    runtime_seconds: float = 0.0
    parameters: Dict[str, Any] = field(default_factory=dict)

    def get_best_pose(self) -> Optional[Pose]:
        """Get the best (lowest energy) pose"""
        if not self.poses:
            return None
        return min(self.poses, key=lambda p: p.energy)

    def get_top_poses(self, n: int = 10) -> List[Pose]:
        """Get top N poses sorted by energy"""
        return sorted(self.poses, key=lambda p: p.energy)[:n]

    def _convert_to_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj

    def save_results(self, output_dir: Path) -> None:
        """Save docking results to files"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON summary
        summary = {
            'ligand_name': self.ligand_name,
            'receptor_file': str(self.receptor_file),
            'grid_center': self.grid_center.tolist(),
            'grid_dimensions': self.grid_dimensions.tolist(),
            'algorithm_used': self.algorithm_used,
            'scoring_function': self.scoring_function,
            'ensemble_binding_energy': self.ensemble_binding_energy,
            'ensemble_confidence': self.ensemble_confidence,
            'num_poses': len(self.poses),
            'runtime_seconds': self.runtime_seconds,
            'parameters': self._convert_to_json_serializable(self.parameters),
            'best_pose_energy': self.get_best_pose().energy if self.poses else None
        }

        with open(output_dir / f"{self.ligand_name}_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        # Save detailed poses
        poses_data = {
            'poses': [pose.to_dict() for pose in self.poses],
            'boltzmann_weights': self.boltzmann_weights
        }

        with open(output_dir / f"{self.ligand_name}_poses.json", 'w') as f:
            json.dump(poses_data, f, indent=2)


class BaseDockingAlgorithm:
    """Base class for all docking algorithms"""

    def __init__(self, name: str, supports_gpu: bool = False):
        self.name = name
        self.supports_gpu = supports_gpu
        self.logger = logging.getLogger(f"pandadock.docking.{name}")
        self._scoring_function = None
        self._parameters = {}

    def set_scoring_function(self, scoring_func):
        """Set the scoring function to use"""
        self._scoring_function = scoring_func

    def set_parameters(self, **params):
        """Set algorithm parameters"""
        self._parameters.update(params)

    def dock(self, receptor_file: str, ligand_mol: Chem.Mol,
             grid_center: np.ndarray, grid_dimensions: np.ndarray,
             **kwargs) -> DockingResult:
        """
        Perform molecular docking

        Args:
            receptor_file: Path to receptor PDB file
            ligand_mol: RDKit molecule object
            grid_center: Center of docking grid
            grid_dimensions: Dimensions of docking grid
            **kwargs: Additional parameters

        Returns:
            DockingResult object containing poses and energies
        """
        raise NotImplementedError("Subclasses must implement dock method")

    def generate_conformers(self, mol: Chem.Mol, num_conformers: int = 50) -> Chem.Mol:
        """Generate ligand conformers using RDKit"""
        # Add hydrogens if not present
        mol = Chem.AddHs(mol)

        # Generate conformers
        cids = AllChem.EmbedMultipleConfs(
            mol,
            numConfs=num_conformers,
            randomSeed=42,
            useExpTorsionAnglePrefs=True,
            useBasicKnowledge=True
        )

        # Optimize conformers
        for cid in cids:
            try:
                AllChem.MMFFOptimizeMolecule(mol, confId=cid, maxIters=500)
            except:
                continue

        return mol

    def _validate_inputs(self, receptor_file: str, ligand_mol: Chem.Mol,
                        grid_center: np.ndarray, grid_dimensions: np.ndarray):
        """Validate docking inputs"""
        if not Path(receptor_file).exists():
            raise FileNotFoundError(f"Receptor file not found: {receptor_file}")

        if ligand_mol is None:
            raise ValueError("Ligand molecule is None")

        if ligand_mol.GetNumAtoms() == 0:
            raise ValueError("Ligand has no atoms")

        if len(grid_center) != 3:
            raise ValueError("Grid center must be 3D coordinates")

        if len(grid_dimensions) != 3:
            raise ValueError("Grid dimensions must be 3D")

        if np.any(grid_dimensions <= 0):
            raise ValueError("Grid dimensions must be positive")


class DockingEngine:
    """Main docking engine that coordinates algorithms and scoring functions"""

    def __init__(self):
        self.logger = logging.getLogger("pandadock.docking.engine")
        self._algorithms: Dict[str, BaseDockingAlgorithm] = {}
        self._scoring_functions = {}

    def register_algorithm(self, algorithm: BaseDockingAlgorithm):
        """Register a docking algorithm"""
        self._algorithms[algorithm.name] = algorithm
        self.logger.debug(f"Registered docking algorithm: {algorithm.name}")

    def register_scoring_function(self, name: str, scoring_func):
        """Register a scoring function"""
        self._scoring_functions[name] = scoring_func
        self.logger.debug(f"Registered scoring function: {name}")

    def list_algorithms(self, gpu_only: bool = False) -> List[str]:
        """List available algorithms"""
        if gpu_only:
            return [name for name, alg in self._algorithms.items() if alg.supports_gpu]
        return list(self._algorithms.keys())

    def list_scoring_functions(self) -> List[str]:
        """List available scoring functions"""
        return list(self._scoring_functions.keys())

    def dock_ligand(self, receptor_file: str, ligand_mol: Chem.Mol,
                   grid_center: np.ndarray, grid_dimensions: np.ndarray,
                   algorithm: str = "monte_carlo",
                   scoring_function: str = "physics_based",
                   **kwargs) -> DockingResult:
        """
        Dock a single ligand

        Args:
            receptor_file: Path to receptor PDB file
            ligand_mol: RDKit molecule object
            grid_center: Center of docking grid
            grid_dimensions: Dimensions of docking grid
            algorithm: Name of docking algorithm to use
            scoring_function: Name of scoring function to use
            **kwargs: Additional parameters for algorithm

        Returns:
            DockingResult object
        """
        # Validate inputs
        if algorithm not in self._algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        if scoring_function not in self._scoring_functions:
            raise ValueError(f"Unknown scoring function: {scoring_function}")

        # Get algorithm and scoring function
        docking_alg = self._algorithms[algorithm]
        scoring_func = self._scoring_functions[scoring_function]

        # Set up algorithm
        docking_alg.set_scoring_function(scoring_func)

        # Record start time
        start_time = time.time()

        # Perform docking
        self.logger.info(f"Starting docking with {algorithm} algorithm and {scoring_function} scoring")
        result = docking_alg.dock(receptor_file, ligand_mol, grid_center, grid_dimensions, **kwargs)

        # Record runtime
        result.runtime_seconds = time.time() - start_time
        result.algorithm_used = algorithm
        result.scoring_function = scoring_function

        self.logger.info(f"Docking completed in {result.runtime_seconds:.2f} seconds")
        self.logger.info(f"Generated {len(result.poses)} poses")

        return result