"""
Four-phase induced-fit docking implementation
"""

import numpy as np
import os
import shutil
import tempfile
import logging
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.Structure import Structure

from .core import FlexiblePose, RefinedComplex, FlexibleResidue
from ..docking.core import Pose, DockingResult
from ..docking.algorithms.monte_carlo_cpu import MonteCarloDocker
from ..docking.scoring.physics_based import PhysicsBasedScoring


class SoftRigidDocker:
    """Phase 1: Initial soft rigid docking with reduced van der Waals penalties"""

    def __init__(self, use_gpu: bool = False, gpu_id: int = 0, cpu_workers: Optional[int] = None,
                 max_memory_gb: float = 8.0):
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.cpu_workers = cpu_workers
        self.max_memory_gb = max_memory_gb
        self.soft_scale = 0.7
        self.logger = logging.getLogger("pandadock.flex.phase1")

        # Initialize underlying rigid docking engine
        self.rigid_docker = MonteCarloDocker()
        self.soft_scorer = SoftPotentialScoring()

    def configure_soft_potential(self, soft_scale: float) -> None:
        """Configure the soft potential scaling factor"""
        self.soft_scale = soft_scale
        self.soft_scorer.set_vdw_scaling(soft_scale)
        self.logger.info(f"Configured soft potential with scaling factor: {soft_scale}")

    def dock(self, receptor_file: str, ligand_mol: Chem.Mol, binding_site_center: np.ndarray,
             binding_site_radius: float, max_poses: int = 20, energy_cutoff: float = 50.0) -> DockingResult:
        """
        Perform initial soft rigid docking

        Args:
            receptor_file: Path to receptor PDB
            ligand_mol: RDKit ligand molecule
            binding_site_center: Center coordinates of binding site
            binding_site_radius: Radius for docking grid
            max_poses: Maximum poses to retain
            energy_cutoff: Energy cutoff for pose filtering

        Returns:
            DockingResult with soft-potential poses
        """
        self.logger.info(f"Starting soft rigid docking with {self.soft_scale:.2f} VDW scaling")

        # Set up grid dimensions from radius
        grid_dimensions = np.array([binding_site_radius * 2] * 3)

        # Configure the rigid docker with soft scoring
        self.rigid_docker._scoring_function = self.soft_scorer

        # Perform docking with enhanced sampling for soft potentials
        result = self.rigid_docker.dock(
            receptor_file=receptor_file,
            ligand_mol=ligand_mol,
            grid_center=binding_site_center,
            grid_dimensions=grid_dimensions,
            num_poses=max_poses * 2,  # Generate more poses for soft filtering
            fast=False  # Use full sampling for better coverage
        )

        # Filter poses by energy cutoff
        filtered_poses = [p for p in result.poses if p.energy <= energy_cutoff]
        result.poses = filtered_poses[:max_poses]

        self.logger.info(f"Generated {len(result.poses)} soft rigid poses")
        return result


class ReceptorRefiner:
    """Phase 2: Receptor conformational refinement around ligand poses"""

    def __init__(self, use_gpu: bool = False, gpu_id: int = 0, cpu_workers: Optional[int] = None,
                 max_memory_gb: float = 8.0):
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.cpu_workers = cpu_workers
        self.max_memory_gb = max_memory_gb
        self.logger = logging.getLogger("pandadock.flex.phase2")

        # Refined receptors are written to a private temporary directory rather
        # than the working directory. Writing them to the CWD left multi-megabyte
        # files behind whenever a run was interrupted, and made two concurrent
        # runs in one directory delete each other's files during cleanup.
        self._temp_dir: Optional[str] = None
        self._refined_counter = 0

        # Refinement configuration
        self.refinement_method = 'minimization'
        self.max_cycles = 3
        self.refine_loops = False
        self.refine_ligand = True

    def configure_refinement(self, method: str = 'minimization', max_cycles: int = 3,
                           refine_loops: bool = False, refine_ligand: bool = True) -> None:
        """Configure refinement parameters"""
        self.refinement_method = method
        self.max_cycles = max_cycles
        self.refine_loops = refine_loops
        self.refine_ligand = refine_ligand

        self.logger.info(f"Configured refinement: {method}, {max_cycles} cycles, "
                        f"loops={'Yes' if refine_loops else 'No'}")

    def auto_detect_flexible_residues(self, receptor_file: str, poses: List[Pose],
                                    distance_threshold: float = 6.0) -> List[str]:
        """
        Automatically detect residues that should be flexible based on ligand pose proximity

        Args:
            receptor_file: Path to receptor PDB
            poses: List of ligand poses
            distance_threshold: Distance cutoff for flexible residue detection

        Returns:
            List of residue identifiers (e.g., ['A:123', 'A:145'])
        """
        self.logger.info(f"Auto-detecting flexible residues within {distance_threshold:.1f} Å")

        # Load receptor structure
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('receptor', receptor_file)

        flexible_residues = set()

        for pose in poses:
            # Find residues within distance of any ligand atom
            for ligand_atom in pose.coordinates:
                for model in structure:
                    for chain in model:
                        for residue in chain:
                            # Skip water and heterogens initially
                            if residue.id[0] != ' ':
                                continue

                            # Check CA atom distance (representing residue position)
                            if 'CA' in residue:
                                ca_coord = residue['CA'].get_coord()
                                distance = np.linalg.norm(ligand_atom - ca_coord)

                                if distance <= distance_threshold:
                                    res_id = f"{chain.id}:{residue.id[1]}"
                                    flexible_residues.add(res_id)

        flexible_residues = sorted(list(flexible_residues))
        self.logger.info(f"Detected {len(flexible_residues)} flexible residues: {flexible_residues}")
        return flexible_residues

    def refine_receptor_conformations(self, receptor_file: str, initial_poses: List[Pose],
                                    flexible_residues: List[str],
                                    binding_site_center: np.ndarray) -> List[RefinedComplex]:
        """
        Refine receptor conformations for each promising ligand pose

        Args:
            receptor_file: Original receptor PDB
            initial_poses: Poses from soft rigid docking
            flexible_residues: List of residues to make flexible
            binding_site_center: Center of binding site

        Returns:
            List of refined receptor-ligand complexes
        """
        self.logger.info(f"Refining receptor for {len(initial_poses)} poses with "
                        f"{len(flexible_residues)} flexible residues")

        refined_complexes = []

        for i, pose in enumerate(initial_poses):
            self.logger.debug(f"Refining receptor for pose {i+1}/{len(initial_poses)}")

            try:
                # Perform side-chain refinement
                refined_complex = self._refine_single_complex(
                    receptor_file, pose, flexible_residues, binding_site_center
                )

                if refined_complex:
                    refined_complexes.append(refined_complex)

            except Exception as e:
                self.logger.warning(f"Failed to refine complex for pose {i+1}: {e}")
                continue

        self.logger.info(f"Successfully refined {len(refined_complexes)} receptor conformations")
        return refined_complexes

    def _refine_single_complex(self, receptor_file: str, pose: Pose,
                             flexible_residues: List[str],
                             binding_site_center: np.ndarray) -> Optional[RefinedComplex]:
        """Refine a single receptor-ligand complex"""

        # Load original receptor
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('receptor', receptor_file)

        # Prepare flexible residue objects
        flex_res_objects = []
        for res_id in flexible_residues:
            chain_id, res_num = res_id.split(':')
            res_num = int(res_num)

            # Find the residue in structure
            try:
                residue = structure[0][chain_id][(' ', res_num, ' ')]
                original_coords = np.array([atom.get_coord() for atom in residue.get_atoms()])

                flex_residue = FlexibleResidue(
                    chain_id=chain_id,
                    residue_number=res_num,
                    residue_name=residue.get_resname(),
                    original_conformation=original_coords,
                    refined_conformations=[],
                    energy_change=0.0
                )
                flex_res_objects.append(flex_residue)

            except KeyError:
                self.logger.warning(f"Residue {res_id} not found in structure")
                continue

        # Perform refinement based on method
        if self.refinement_method == 'minimization':
            refined_structure, energy_cost = self._minimize_side_chains(
                structure, pose, flex_res_objects
            )
        elif self.refinement_method == 'monte_carlo':
            refined_structure, energy_cost = self._monte_carlo_side_chains(
                structure, pose, flex_res_objects
            )
        else:
            self.logger.error(f"Unknown refinement method: {self.refinement_method}")
            return None

        # Save refined receptor to a temporary file.
        #
        # The name is a per-instance counter, not id(pose). CPython reuses id()
        # values once an object is collected, so two poses could receive the same
        # filename and the second would overwrite the first -- meaning the final
        # redocking phase could dock into the wrong refined receptor.
        refined_pdb_path = os.path.join(
            self._ensure_temp_dir(), f"refined_receptor_{self._refined_counter:04d}.pdb"
        )
        self._refined_counter += 1
        io = PDBIO()
        io.set_structure(refined_structure)
        io.save(refined_pdb_path)

        # Calculate RMSD from original
        rmsd = self._calculate_receptor_rmsd(receptor_file, refined_pdb_path, flexible_residues)

        return RefinedComplex(
            receptor=refined_pdb_path,
            ligand_pose=pose,
            flexible_residues=flex_res_objects,
            energy_cost=energy_cost,
            rmsd_from_original=rmsd
        )

    def _minimize_side_chains(self, structure: Structure, pose: Pose,
                            flexible_residues: List[FlexibleResidue]) -> Tuple[Structure, float]:
        """
        Minimize side-chain conformations using simple energy minimization

        This is a simplified implementation. In production, you'd use:
        - OpenMM for molecular mechanics
        - RDKit's conformer optimization
        - Custom MM force fields
        """
        self.logger.debug("Performing side-chain minimization")

        # For now, use a simplified approach with rotamer sampling
        total_energy_cost = 0.0

        for flex_res in flexible_residues:
            # Sample rotamer conformations (simplified)
            rotamer_conformations = self._sample_rotamers(flex_res, pose)

            # Select best rotamer based on simple scoring
            best_rotamer, best_energy = self._select_best_rotamer(
                rotamer_conformations, pose, structure
            )

            if best_rotamer is not None:
                flex_res.refined_conformations = [best_rotamer]
                flex_res.energy_change = best_energy
                total_energy_cost += best_energy

                # Update structure with best rotamer (simplified)
                self._update_residue_conformation(structure, flex_res, best_rotamer)

        return structure, total_energy_cost

    def _sample_rotamers(self, flexible_residue: FlexibleResidue,
                        pose: Pose) -> List[np.ndarray]:
        """Sample common rotamer conformations for a residue (simplified)"""

        # This is a placeholder - in production you'd use:
        # - Dunbrack rotamer library
        # - Physics-based side-chain prediction
        # - SCWRL4-like algorithms

        rotamers = []
        original = flexible_residue.original_conformation

        # Generate some simple rotamer variants
        for angle in [0, 60, 120, 180, 240, 300]:
            rotamer = self._rotate_side_chain(original, angle)
            rotamers.append(rotamer)

        return rotamers

    def _rotate_side_chain(self, coords: np.ndarray, angle_deg: float) -> np.ndarray:
        """Rotate side chain around main-chain bond (simplified)"""
        # Simplified rotation - in production use proper dihedral rotation
        angle_rad = np.deg2rad(angle_deg)
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])

        # Rotate coordinates (simplified - should be around proper bond axis)
        center = np.mean(coords, axis=0)
        centered_coords = coords - center
        rotated_coords = centered_coords @ rotation_matrix.T
        return rotated_coords + center

    def _select_best_rotamer(self, rotamers: List[np.ndarray], pose: Pose,
                           structure: Structure) -> Tuple[Optional[np.ndarray], float]:
        """
        Choose the best rotamer, and report the strain it leaves behind.

        Selection uses the full energy (clash penalty plus contact reward), but
        the value returned as the refinement cost is the residual clash strain
        alone, which is non-negative.

        Returning the full energy conflated the two. With no clashes it is purely
        a negative contact reward, and the IFD score adds
        `refinement_cost * refinement_penalty_weight` -- so a negative cost
        improved the score. A pose demanding more side-chain rearrangement scored
        better than one needing none, inverting the purpose of an induced-fit
        penalty.
        """
        best_rotamer = None
        best_energy = float('inf')
        best_strain = 0.0

        for rotamer in rotamers:
            energy, strain = self._calculate_rotamer_energy(
                rotamer, pose.coordinates, return_strain=True
            )

            if energy < best_energy:
                best_energy = energy
                best_strain = strain
                best_rotamer = rotamer

        return best_rotamer, best_strain

    def _calculate_rotamer_energy(self, rotamer_coords: np.ndarray,
                                ligand_coords: np.ndarray,
                                return_strain: bool = False):
        """
        Simplified energy for a rotamer conformation.

        Returns the combined energy used to rank rotamers. With
        `return_strain=True` it also returns the clash component on its own,
        which is the non-negative strain the receptor carries in that
        conformation and the quantity an induced-fit penalty should use.
        """

        # Simple distance-based scoring
        min_distances = []
        for rot_atom in rotamer_coords:
            distances = np.linalg.norm(ligand_coords - rot_atom, axis=1)
            min_distances.append(np.min(distances))

        # Penalty for clashes (< 2.5 Å)
        clash_penalty = sum(max(0, 2.5 - d) * 10 for d in min_distances)

        # Reward for favorable contacts (2.5-4.0 Å)
        favorable_contacts = sum(1 for d in min_distances if 2.5 <= d <= 4.0)
        contact_reward = favorable_contacts * -0.5

        if return_strain:
            return clash_penalty + contact_reward, clash_penalty
        return clash_penalty + contact_reward

    def _update_residue_conformation(self, structure: Structure,
                                   flexible_residue: FlexibleResidue,
                                   new_coords: np.ndarray) -> None:
        """Update residue conformation in structure (simplified)"""

        # Find and update the residue atoms
        try:
            chain = structure[0][flexible_residue.chain_id]
            residue = chain[(' ', flexible_residue.residue_number, ' ')]

            # Update coordinates (simplified - should match atom ordering)
            atoms = list(residue.get_atoms())
            for i, atom in enumerate(atoms):
                if i < len(new_coords):
                    atom.set_coord(new_coords[i])
        except (KeyError, IndexError) as e:
            self.logger.warning(f"Failed to update residue conformation: {e}")

    def _monte_carlo_side_chains(self, structure: Structure, pose: Pose,
                               flexible_residues: List[FlexibleResidue]) -> Tuple[Structure, float]:
        """Monte Carlo sampling of side-chain conformations"""
        self.logger.debug("Performing Monte Carlo side-chain sampling")

        # Simplified MC implementation
        # In production: use proper MC with temperature schedule
        return self._minimize_side_chains(structure, pose, flexible_residues)

    def _calculate_receptor_rmsd(self, original_pdb: str, refined_pdb: str,
                               flexible_residues: List[str]) -> float:
        """Calculate RMSD of flexible residues between original and refined structures"""

        # Simplified RMSD calculation
        # In production: use proper structural alignment
        return 1.5  # Placeholder value


    def _ensure_temp_dir(self) -> str:
        """Create this refiner's temporary directory on first use."""
        if self._temp_dir is None:
            self._temp_dir = tempfile.mkdtemp(prefix="pandadock_flex_")
            self.logger.debug("Refined receptors will be written to %s", self._temp_dir)
        return self._temp_dir

    def cleanup(self) -> None:
        """
        Remove the temporary refined receptors.

        Safe to call more than once, and safe to call after a failure. Cleanup
        used to run only on the success path of the CLI, so an interrupted or
        failed run left every refined receptor behind -- roughly 375 KB each.
        """
        if self._temp_dir and os.path.isdir(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self.logger.debug("Removed temporary directory %s", self._temp_dir)
        self._temp_dir = None

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass


class FinalRedocker:
    """Phase 3: Final redocking into refined receptor conformations"""

    def __init__(self, use_gpu: bool = False, gpu_id: int = 0, cpu_workers: Optional[int] = None,
                 max_memory_gb: float = 8.0):
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.cpu_workers = cpu_workers
        self.max_memory_gb = max_memory_gb
        self.final_scale = 1.0
        self.logger = logging.getLogger("pandadock.flex.phase3")

        # Initialize precise docking engine
        self.precise_docker = MonteCarloDocker()
        self.precise_scorer = PhysicsBasedScoring()

    def configure_final_potential(self, final_scale: float = 1.0) -> None:
        """Configure final docking potential (typically 1.0 for full precision)"""
        self.final_scale = final_scale
        self.logger.info(f"Configured final potential with scaling factor: {final_scale}")

    def redock(self, refined_receptor: str, ligand_mol: Chem.Mol,
               binding_site_center: np.ndarray, binding_site_radius: float,
               max_poses: int = 10) -> DockingResult:
        """
        Redock ligand into refined receptor conformation

        Args:
            refined_receptor: Path to refined receptor PDB
            ligand_mol: RDKit ligand molecule
            binding_site_center: Center of binding site
            binding_site_radius: Radius for docking
            max_poses: Maximum poses to retain

        Returns:
            DockingResult with precisely docked poses
        """
        self.logger.debug(f"Redocking into refined receptor: {refined_receptor}")

        # Set up precise scoring
        self.precise_docker._scoring_function = self.precise_scorer

        # Perform high-precision docking
        grid_dimensions = np.array([binding_site_radius * 1.5] * 3)  # Smaller grid for precision

        result = self.precise_docker.dock(
            receptor_file=refined_receptor,
            ligand_mol=ligand_mol,
            grid_center=binding_site_center,
            grid_dimensions=grid_dimensions,
            num_poses=max_poses * 2,  # Generate extra for selection
            fast=False  # Use full sampling
        )

        # Convert to FlexiblePose objects
        flex_poses = []
        for pose in result.poses[:max_poses]:
            flex_pose = FlexiblePose(
                coordinates=pose.coordinates,
                center=pose.center,
                rotation=pose.rotation,
                conformer_id=pose.conformer_id,
                energy=pose.energy,
                confidence=pose.confidence
            )
            flex_pose.binding_energy = pose.energy
            flex_poses.append(flex_pose)

        result.poses = flex_poses
        return result


class IFDScorer:
    """Phase 4: Induced-Fit Docking scoring and ranking"""

    def __init__(self, scoring_function: str = 'ifd_composite'):
        self.scoring_function = scoring_function
        self.logger = logging.getLogger("pandadock.flex.phase4")

        # Scoring weights
        self.binding_weight = 0.7
        self.refinement_penalty_weight = 0.3

    def score_ifd_poses(self, poses: List[FlexiblePose], receptor_file: str,
                       ligand_mol: Chem.Mol) -> List[FlexiblePose]:
        """
        Calculate comprehensive IFD scores for poses

        IFD Score = Binding Energy + (Refinement Cost * Penalty Weight)
        """
        self.logger.info(f"Calculating IFD scores for {len(poses)} poses")

        for pose in poses:
            # Basic IFD score calculation
            pose.ifd_score = (pose.binding_energy +
                            pose.refinement_cost * self.refinement_penalty_weight)

            # Additional scoring components
            pose.ligand_strain_energy = self._calculate_ligand_strain(pose, ligand_mol)
            pose.interaction_fingerprint = self._calculate_interactions(
                pose, receptor_file, ligand_mol
            )

            # Adjust IFD score with additional components
            strain_penalty = pose.ligand_strain_energy * 0.1
            interaction_bonus = len(pose.interaction_fingerprint) * -0.2

            pose.ifd_score += strain_penalty + interaction_bonus

        return poses

    def _calculate_ligand_strain(self, pose: FlexiblePose, ligand_mol: Chem.Mol) -> float:
        """Calculate ligand strain energy (simplified)"""
        # In production: compare with lowest energy conformer
        return 0.5  # Placeholder

    def _calculate_interactions(self, pose: FlexiblePose, receptor_file: str,
                              ligand_mol: Chem.Mol) -> Dict[str, float]:
        """Calculate detailed protein-ligand interactions"""
        # Simplified interaction calculation
        interactions = {
            'hydrogen_bonds': 3.2,
            'hydrophobic_contacts': 5.1,
            'electrostatic': 1.8,
            'aromatic_stacking': 0.5
        }
        return interactions

    def cluster_poses(self, poses: List[FlexiblePose],
                     rmsd_threshold: float = 2.0) -> List[FlexiblePose]:
        """Cluster poses by RMSD and select representatives"""
        self.logger.info(f"Clustering {len(poses)} poses with RMSD threshold {rmsd_threshold:.1f} Å")

        if not poses:
            return poses

        # Sort by IFD score first
        sorted_poses = sorted(poses, key=lambda p: p.ifd_score)

        # Simple clustering - select diverse poses
        clustered_poses = [sorted_poses[0]]  # Always keep best

        for pose in sorted_poses[1:]:
            is_diverse = True
            for clustered_pose in clustered_poses:
                rmsd = self._calculate_pose_rmsd(pose, clustered_pose)
                if rmsd < rmsd_threshold:
                    is_diverse = False
                    break

            if is_diverse:
                clustered_poses.append(pose)

        self.logger.info(f"Selected {len(clustered_poses)} diverse poses after clustering")
        return clustered_poses

    def _calculate_pose_rmsd(self, pose1: FlexiblePose, pose2: FlexiblePose) -> float:
        """Calculate RMSD between two poses"""
        if len(pose1.coordinates) != len(pose2.coordinates):
            return float('inf')

        diff = pose1.coordinates - pose2.coordinates
        return np.sqrt(np.mean(np.sum(diff**2, axis=1)))


class SoftPotentialScoring:
    """Soft potential scoring for Phase 1 docking"""

    def __init__(self):
        self.base_scorer = PhysicsBasedScoring()
        self.vdw_scaling = 0.7

    def set_vdw_scaling(self, scaling: float) -> None:
        """Set van der Waals scaling factor"""
        self.vdw_scaling = scaling

    def calculate_binding_energy(self, ligand_coords: np.ndarray,
                                receptor_structure, ligand_mol: Chem.Mol = None) -> float:
        """Calculate binding energy with soft VDW potentials"""

        # Get base energy
        base_energy = self.base_scorer.calculate_binding_energy(
            ligand_coords, receptor_structure, ligand_mol
        )

        # Apply soft scaling to reduce VDW penalties
        # This is simplified - in production you'd modify the VDW terms specifically
        if base_energy > 0:
            # Reduce positive (unfavorable) energies more aggressively
            soft_energy = base_energy * self.vdw_scaling
        else:
            # Keep favorable energies mostly unchanged
            soft_energy = base_energy * (0.8 + 0.2 * self.vdw_scaling)

        return soft_energy