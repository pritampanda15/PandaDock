"""
Core classes for flexible docking results and data structures
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ..docking.core import Pose


@dataclass
class FlexibleResidue:
    """Represents a flexible residue for induced-fit docking"""
    chain_id: str
    residue_number: int
    residue_name: str
    original_conformation: np.ndarray  # Original side-chain coordinates
    refined_conformations: List[np.ndarray]  # Refined conformations
    energy_change: float  # Energy cost of conformational change


@dataclass
class RefinedComplex:
    """Represents a receptor-ligand complex after refinement"""
    receptor: str  # Path to refined receptor PDB
    ligand_pose: Pose
    flexible_residues: List[FlexibleResidue]
    energy_cost: float  # Total energy cost of receptor refinement
    rmsd_from_original: float  # RMSD of refined structure from original


class FlexiblePose(Pose):
    """Extended pose class for flexible docking results"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ifd_score: float = 0.0
        self.binding_energy: float = 0.0
        self.refinement_cost: float = 0.0
        self.receptor_rmsd: float = 0.0
        self.ligand_strain_energy: float = 0.0
        self.interaction_fingerprint: Dict[str, float] = {}
        self.flexible_residue_contacts: List[str] = []


class FlexibleDockingResult:
    """Comprehensive results from flexible/induced-fit docking"""

    def __init__(self, initial_poses: List[Pose], refined_complexes: List[RefinedComplex],
                 final_poses: List[FlexiblePose], flexible_residues: List[str],
                 total_runtime: float, parameters: Dict[str, Any]):
        self.initial_poses = initial_poses
        self.refined_complexes = refined_complexes
        self.final_poses = final_poses
        self.flexible_residues = flexible_residues
        self.total_runtime = total_runtime
        self.parameters = parameters

        # Statistics
        self.statistics = self._calculate_statistics()

    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive statistics for the flexible docking run"""
        stats = {
            'initial_poses_count': len(self.initial_poses),
            'refined_complexes_count': len(self.refined_complexes),
            'final_poses_count': len(self.final_poses),
            'flexible_residues_count': len(self.flexible_residues),
            'total_runtime_seconds': self.total_runtime,
            'runtime_per_pose': self.total_runtime / max(1, len(self.final_poses))
        }

        if self.final_poses:
            ifd_scores = [pose.ifd_score for pose in self.final_poses]
            binding_energies = [pose.binding_energy for pose in self.final_poses]
            refinement_costs = [pose.refinement_cost for pose in self.final_poses]

            stats.update({
                'best_ifd_score': min(ifd_scores),
                'worst_ifd_score': max(ifd_scores),
                'mean_ifd_score': np.mean(ifd_scores),
                'std_ifd_score': np.std(ifd_scores),
                'best_binding_energy': min(binding_energies),
                'mean_binding_energy': np.mean(binding_energies),
                'mean_refinement_cost': np.mean(refinement_costs),
                'max_refinement_cost': max(refinement_costs)
            })

        return stats

    def get_top_poses(self, n: int = 10) -> List[FlexiblePose]:
        """Get top N poses ranked by IFD score"""
        return sorted(self.final_poses, key=lambda p: p.ifd_score)[:n]

    def get_poses_by_energy_range(self, min_energy: float, max_energy: float) -> List[FlexiblePose]:
        """Get poses within specified IFD score range"""
        return [p for p in self.final_poses
                if min_energy <= p.ifd_score <= max_energy]

    def save_results(self, output_dir: Path) -> None:
        """Save comprehensive flexible docking results"""
        output_dir.mkdir(exist_ok=True)

        # Save summary statistics
        with open(output_dir / "flex_docking_summary.json", 'w') as f:
            json.dump({
                'statistics': self.statistics,
                'parameters': self.parameters,
                'flexible_residues': self.flexible_residues,
                'runtime_breakdown': {
                    'total_runtime': self.total_runtime,
                    'poses_generated': len(self.final_poses),
                    'refinement_cycles': len(self.refined_complexes)
                }
            }, f, indent=2, default=str)

        # Save detailed pose information
        pose_data = []
        for i, pose in enumerate(self.final_poses):
            pose_info = {
                'pose_id': i + 1,
                'ifd_score': pose.ifd_score,
                'binding_energy': pose.binding_energy,
                'refinement_cost': pose.refinement_cost,
                'receptor_rmsd': pose.receptor_rmsd,
                'ligand_strain_energy': pose.ligand_strain_energy,
                'center': pose.center.tolist(),
                # Coordinates are included so the poses can be reconstructed from
                # this file alone. Without them the JSON recorded scores for poses
                # that could only be recovered by re-parsing the PDB output.
                'coordinates': (
                    pose.coordinates.tolist() if pose.coordinates is not None else None
                ),
                'energy': pose.energy,
                'confidence': pose.confidence,
                'flexible_residue_contacts': pose.flexible_residue_contacts,
                'interaction_fingerprint': pose.interaction_fingerprint
            }
            pose_data.append(pose_info)

        with open(output_dir / "flex_poses_detailed.json", 'w') as f:
            json.dump(pose_data, f, indent=2)

        # Save refined complex information
        complex_data = []
        for i, complex_info in enumerate(self.refined_complexes):
            complex_info_dict = {
                'complex_id': i + 1,
                'energy_cost': complex_info.energy_cost,
                'rmsd_from_original': complex_info.rmsd_from_original,
                'flexible_residues_count': len(complex_info.flexible_residues),
                'receptor_pdb_path': complex_info.receptor
            }
            complex_data.append(complex_info_dict)

        with open(output_dir / "refined_complexes.json", 'w') as f:
            json.dump(complex_data, f, indent=2)

        print(f"✓ Flexible docking results saved to {output_dir}")


class FlexDockingConfiguration:
    """Configuration class for flexible docking parameters"""

    def __init__(self):
        # Phase 1: Soft docking parameters
        self.soft_potential_scale = 0.7
        self.initial_poses_to_retain = 20
        self.initial_energy_cutoff = 50.0

        # Phase 2: Refinement parameters
        self.refinement_distance = 6.0
        self.refinement_method = 'minimization'
        self.max_refinement_cycles = 3
        self.refine_loops = False
        self.refine_ligand = True
        self.side_chain_sampling = True

        # Phase 3: Final docking parameters
        self.final_potential_scale = 1.0
        self.final_poses_to_retain = 10

        # Phase 4: Scoring parameters
        self.scoring_function = 'ifd_composite'
        self.rmsd_clustering_threshold = 2.0
        self.energy_penalty_weight = 0.3

        # Computational parameters
        self.use_gpu = False
        self.gpu_id = 0
        self.cpu_workers = None
        self.max_memory_gb = 8.0

        # Quality control
        self.max_clashes_allowed = 3
        self.min_interaction_count = 5
        self.convergence_tolerance = 0.1

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            attr: getattr(self, attr)
            for attr in dir(self)
            if not attr.startswith('_') and not callable(getattr(self, attr))
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FlexDockingConfiguration':
        """Create configuration from dictionary"""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config