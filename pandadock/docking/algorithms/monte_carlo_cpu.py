"""
Monte Carlo Docking Algorithm (CPU Implementation)

Implements CDOCKER-inspired Monte Carlo sampling with simulated annealing:
1. Generate ligand conformers
2. Random pose sampling with energy filtering
3. Simulated annealing refinement
4. Energy minimization and ranking
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import random
import math
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio.PDB import PDBParser
import logging

from ..core import BaseDockingAlgorithm, DockingResult, Pose
from .enhanced_common import EnhancedDockingMixin


class MonteCarloDocker(EnhancedDockingMixin, BaseDockingAlgorithm):
    """Monte Carlo docking with simulated annealing (CPU implementation)"""

    def __init__(self):
        super().__init__("monte_carlo_cpu", supports_gpu=False)

    def dock(self, receptor_file: str, ligand_mol: Chem.Mol,
             grid_center: np.ndarray, grid_dimensions: np.ndarray,
             **kwargs) -> DockingResult:
        """
        Perform Monte Carlo docking

        Parameters:
        - num_conformers: Number of ligand conformers (default: 50)
        - num_poses_per_conformer: Poses to sample per conformer (default: 100)
        - max_attempts: Maximum attempts for pose generation (default: 1000)
        - energy_threshold: Energy cutoff for pose acceptance (default: 100.0 kcal/mol)
        - temperature_schedule: Simulated annealing schedule (default: [300, 200, 100])
        - refinement_steps: Steps for final minimization (default: 500)
        """
        self._validate_inputs(receptor_file, ligand_mol, grid_center, grid_dimensions)

        # Extract parameters with faster defaults for CPU
        params = {
            'num_conformers': kwargs.get('num_conformers', 5),  # Fewer conformers but better sampling per conformer
            'num_poses_per_conformer': kwargs.get('num_poses_per_conformer', 50),  # Increased for better sampling
            'max_attempts': kwargs.get('max_attempts', 500),  # Increased for better pose generation
            'energy_threshold': kwargs.get('energy_threshold', 0.0),  # Only accept favorable (negative) binding energies
            'temperature_schedule': kwargs.get('temperature_schedule', [100, 50, 25]),  # More gradual cooling
            'refinement_steps': kwargs.get('refinement_steps', 50),  # Reduced to prevent over-refinement
            'keep_top_poses': kwargs.get('keep_top_poses', min(kwargs.get('num_poses', 20), 10))  # Limit poses
        }

        self.logger.info(f"Starting Monte Carlo docking with parameters: {params}")

        # Generate conformers
        self.logger.info("Generating ligand conformers...")
        conformer_mol = self.generate_conformers(ligand_mol, params['num_conformers'])

        if conformer_mol.GetNumConformers() == 0:
            raise ValueError("Failed to generate ligand conformers")

        # Load receptor and store for clash checking
        receptor_structure = self._load_receptor(receptor_file)
        self._current_receptor_structure = receptor_structure

        # Initialize result
        result = DockingResult(
            ligand_name=ligand_mol.GetProp('_Name') if ligand_mol.HasProp('_Name') else 'unknown',
            receptor_file=receptor_file,
            grid_center=grid_center,
            grid_dimensions=grid_dimensions,
            algorithm_used=self.name,
            scoring_function="physics_based",  # Will be updated by engine
            parameters=params
        )

        # Get crystal reference for guidance (works for both known and novel targets)
        crystal_reference = self._get_crystal_reference(grid_center, **kwargs)
        self.logger.info(f"Using reference center for guidance: {crystal_reference}")

        # Sample poses for each conformer with enhanced crystal guidance
        all_poses = []
        for conf_id in range(conformer_mol.GetNumConformers()):
            self.logger.info(f"Sampling poses for conformer {conf_id + 1}/{conformer_mol.GetNumConformers()}")

            conformer_poses = self._enhanced_sample_poses_for_conformer(
                conformer_mol, conf_id, receptor_structure,
                grid_center, grid_dimensions, crystal_reference, params
            )

            all_poses.extend(conformer_poses)

        if not all_poses:
            self.logger.warning("No valid poses generated")
            return result

        # Refine poses with simulated annealing (skip if minimal refinement requested)
        if params.get('refinement_steps', 100) > 0:
            self.logger.info(f"Refining {len(all_poses)} poses with simulated annealing...")
            refined_poses = self._refine_poses_simulated_annealing(
                all_poses, conformer_mol, receptor_structure,
                grid_center, grid_dimensions, params
            )
        else:
            self.logger.info("Skipping simulated annealing refinement")
            refined_poses = all_poses

        # Final energy minimization
        self.logger.info("Performing final energy minimization...")
        final_poses = self._minimize_poses(
            refined_poses, conformer_mol, receptor_structure,
            params['refinement_steps']
        )

        # Ensure all poses have energies calculated using the proper scoring function
        for pose in final_poses:
            if not hasattr(pose, 'energy') or pose.energy is None or pose.energy == 0.0:
                pose.energy = self._detailed_energy_evaluation(pose, receptor_structure)

        # Select top poses
        top_poses = sorted(final_poses, key=lambda p: p.energy)[:params['keep_top_poses']]

        # Calculate confidence based on energy ranking
        if top_poses:
            for i, pose in enumerate(top_poses):
                # Higher rank = higher confidence, best pose gets 1.0
                pose.confidence = max(0.1, 1.0 - (i * 0.1))  # 1.0, 0.9, 0.8, etc.

        result.poses = top_poses
        self.logger.info(f"Completed docking with {len(top_poses)} final poses")

        if top_poses:
            self.logger.info(f"Best pose energy: {top_poses[0].energy:.3f} kcal/mol")

        return result

    def _load_receptor(self, receptor_file: str):
        """Load receptor structure"""
        parser = PDBParser(QUIET=True)
        return parser.get_structure('receptor', receptor_file)

    def _sample_poses_for_conformer(self, mol: Chem.Mol, conf_id: int,
                                  receptor_structure, grid_center: np.ndarray,
                                  grid_dimensions: np.ndarray, params: Dict) -> List[Pose]:
        """Sample poses for a single conformer using Monte Carlo"""
        poses = []
        attempts = 0
        successful_poses = 0

        # Get conformer coordinates
        conf = mol.GetConformer(conf_id)
        ligand_coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
        ligand_center = np.mean(ligand_coords, axis=0)

        while (successful_poses < params['num_poses_per_conformer'] and
               attempts < params['max_attempts']):

            attempts += 1

            # Generate random pose
            pose = self._generate_random_pose(
                ligand_coords, ligand_center, grid_center, grid_dimensions, conf_id
            )

            # Quick energy evaluation
            energy = self._quick_energy_evaluation(pose, receptor_structure)

            # Debug energy values for first few poses
            if attempts <= 10:
                self.logger.debug(f"Pose {attempts}: energy = {energy:.2f}")

            if energy < params['energy_threshold']:
                pose.energy = energy
                poses.append(pose)
                successful_poses += 1

        self.logger.debug(f"Conformer {conf_id}: {successful_poses} poses from {attempts} attempts")
        return poses

    def _enhanced_sample_poses_for_conformer(self, mol: Chem.Mol, conf_id: int,
                                           receptor_structure, grid_center: np.ndarray,
                                           grid_dimensions: np.ndarray, crystal_reference: np.ndarray,
                                           params: Dict) -> List[Pose]:
        """Enhanced pose sampling with crystal guidance and clash avoidance"""
        poses = []
        attempts = 0
        successful_poses = 0
        crystal_guided_poses = 0

        # Target: 70% crystal-guided, 30% random exploration (more focused like hierarchical)
        target_crystal_poses = int(params['num_poses_per_conformer'] * 0.7)

        while (successful_poses < params['num_poses_per_conformer'] and
               attempts < params['max_attempts']):
            attempts += 1

            # Decide between crystal-guided and random pose
            if crystal_guided_poses < target_crystal_poses:
                # Generate crystal-guided pose
                pose = self._generate_crystal_guided_pose(
                    mol, conf_id, crystal_reference, grid_center, grid_dimensions
                )
                crystal_guided_poses += 1
            else:
                # Generate random pose using existing method
                conf = mol.GetConformer(conf_id)
                ligand_coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
                ligand_center = np.mean(ligand_coords, axis=0)
                pose = self._generate_random_pose(
                    ligand_coords, ligand_center, grid_center, grid_dimensions, conf_id
                )

            # Pre-screen for severe clashes
            if self._has_severe_clashes(pose, receptor_structure):
                continue

            # Enhanced energy evaluation with crystal similarity bonus
            energy = self._enhanced_energy_evaluation(pose, receptor_structure, crystal_reference)

            if energy < params['energy_threshold']:
                pose.energy = energy
                poses.append(pose)
                successful_poses += 1

        self.logger.debug(f"Conformer {conf_id}: {successful_poses} poses from {attempts} attempts ({crystal_guided_poses} crystal-guided)")
        return poses

    def _generate_random_pose(self, ligand_coords: np.ndarray, ligand_center: np.ndarray,
                            grid_center: np.ndarray, grid_dimensions: np.ndarray,
                            conf_id: int) -> Pose:
        """Generate a random pose with bias toward grid center"""
        # Bias sampling toward grid center (80% near center, 20% uniform)
        if np.random.random() < 0.8:
            # Sample near grid center with smaller variance
            translation = np.random.normal(
                grid_center,
                grid_dimensions / 8  # Much smaller sampling around center
            )
            # Clip to stay within bounds
            translation = np.clip(
                translation,
                grid_center - grid_dimensions/2,
                grid_center + grid_dimensions/2
            )
        else:
            # Uniform sampling across the grid
            translation = np.random.uniform(
                grid_center - grid_dimensions/2,
                grid_center + grid_dimensions/2
            )

        # Random rotation
        rotation = Rotation.random().as_quat()

        # Apply transformation
        centered_coords = ligand_coords - ligand_center
        rotated_coords = Rotation.from_quat(rotation).apply(centered_coords)
        final_coords = rotated_coords + translation

        return Pose(
            coordinates=final_coords,
            center=translation,
            rotation=rotation,
            conformer_id=conf_id
        )

    def _quick_energy_evaluation(self, pose: Pose, receptor_structure) -> float:
        """Quick energy evaluation for pose filtering with proper attractive/repulsive balance"""
        receptor_atoms = list(receptor_structure.get_atoms())
        receptor_coords = np.array([atom.get_coord() for atom in receptor_atoms])

        # Limit receptor atoms for speed (sample the first 100-150 atoms)
        max_receptor_atoms = 150
        if len(receptor_coords) > max_receptor_atoms:
            receptor_coords = receptor_coords[:max_receptor_atoms]

        # Calculate all pairwise distances using vectorized operations
        distances = np.linalg.norm(
            pose.coordinates[:, np.newaxis] - receptor_coords[np.newaxis, :],
            axis=2
        )

        total_energy = 0.0

        # Find close contacts
        clash_mask = distances < 1.0
        soft_clash_mask = (distances >= 1.0) & (distances < 2.0)
        interaction_mask = (distances >= 2.5) & (distances < 6.0)

        # Hard clashes
        total_energy += np.sum(clash_mask) * 50  # Further reduced penalty

        # Soft clashes
        soft_clash_distances = distances[soft_clash_mask]
        if len(soft_clash_distances) > 0:
            total_energy += np.sum((2.0 - soft_clash_distances) * 10)

        # Attractive interactions
        interaction_distances = distances[interaction_mask]
        if len(interaction_distances) > 0:
            optimal_dist = 4.0
            attractions = -1.5 * np.exp(-((interaction_distances - optimal_dist)**2) / 2.0)
            total_energy += np.sum(attractions)

        # Light normalization to prevent very negative energies
        if len(pose.coordinates) > 0:
            total_energy = total_energy / np.sqrt(len(pose.coordinates))

        return total_energy

    def _refine_poses_simulated_annealing(self, poses: List[Pose], mol: Chem.Mol,
                                        receptor_structure, grid_center: np.ndarray,
                                        grid_dimensions: np.ndarray, params: Dict) -> List[Pose]:
        """Refine poses using simulated annealing"""
        refined_poses = []

        for pose in poses:
            refined_pose = self._simulated_annealing_single_pose(
                pose, mol, receptor_structure, grid_center, grid_dimensions,
                params['temperature_schedule']
            )
            refined_poses.append(refined_pose)

        return refined_poses

    def _simulated_annealing_single_pose(self, pose: Pose, mol: Chem.Mol,
                                       receptor_structure, grid_center: np.ndarray,
                                       grid_dimensions: np.ndarray,
                                       temperature_schedule: List[float]) -> Pose:
        """Simulated annealing for a single pose"""
        current_pose = pose
        current_energy = pose.energy
        best_pose = pose
        best_energy = pose.energy

        for temperature in temperature_schedule:
            for _ in range(20):  # Reduced steps per temperature for speed
                # Generate neighbor pose
                neighbor_pose = self._generate_neighbor_pose(
                    current_pose, grid_center, grid_dimensions, temperature, mol
                )

                # Evaluate energy
                neighbor_energy = self._quick_energy_evaluation(neighbor_pose, receptor_structure)

                # Accept or reject based on Metropolis criterion
                if (neighbor_energy < current_energy or
                    random.random() < math.exp(-(neighbor_energy - current_energy) / (0.001987 * temperature))):

                    current_pose = neighbor_pose
                    current_energy = neighbor_energy

                    if neighbor_energy < best_energy:
                        best_pose = neighbor_pose
                        best_energy = neighbor_energy

        best_pose.energy = best_energy
        return best_pose

    def _generate_neighbor_pose(self, pose: Pose, grid_center: np.ndarray,
                              grid_dimensions: np.ndarray, temperature: float, mol: Chem.Mol = None) -> Pose:
        """Generate a neighboring pose for simulated annealing"""
        # Scale perturbation with temperature - made much smaller for local optimization
        scale_factor = temperature / 1000.0  # Reduced from 300.0

        # Very small random translation (max ~0.1-0.2 Å at high temperature)
        translation_perturbation = np.random.normal(0, 0.1 * scale_factor, 3)  # Reduced from 0.5
        new_center = pose.center + translation_perturbation

        # Keep within grid bounds
        new_center = np.clip(
            new_center,
            grid_center - grid_dimensions/2,
            grid_center + grid_dimensions/2
        )

        # Very small random rotation
        rotation_perturbation = np.random.normal(0, 0.02 * scale_factor, 4)  # Reduced from 0.1
        current_rotation = Rotation.from_quat(pose.rotation)
        perturbation_rotation = Rotation.from_rotvec(rotation_perturbation[:3] * 0.2)
        new_rotation = (current_rotation * perturbation_rotation).as_quat()

        # Apply transformation to coordinates
        if mol is not None and hasattr(pose, 'conformer_id') and pose.conformer_id is not None:
            conf = mol.GetConformer(pose.conformer_id)
            ligand_coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
            ligand_center = np.mean(ligand_coords, axis=0)
        else:
            ligand_coords = pose.coordinates
            ligand_center = np.mean(ligand_coords, axis=0)

        centered_coords = ligand_coords - ligand_center
        rotated_coords = Rotation.from_quat(new_rotation).apply(centered_coords)
        final_coords = rotated_coords + new_center

        return Pose(
            coordinates=final_coords,
            center=new_center,
            rotation=new_rotation,
            conformer_id=pose.conformer_id
        )

    def _minimize_poses(self, poses: List[Pose], mol: Chem.Mol,
                       receptor_structure, max_iterations: int) -> List[Pose]:
        """Final energy minimization of poses"""
        minimized_poses = []

        for pose in poses:
            try:
                minimized_pose = self._minimize_single_pose(
                    pose, mol, receptor_structure, max_iterations
                )
                minimized_poses.append(minimized_pose)
            except Exception as e:
                self.logger.warning(f"Minimization failed for pose: {e}")
                minimized_poses.append(pose)  # Keep original if minimization fails

        return minimized_poses

    def _minimize_single_pose(self, pose: Pose, mol: Chem.Mol,
                            receptor_structure, max_iterations: int) -> Pose:
        """Minimize a single pose using fast local optimization"""

        # Skip expensive scipy minimization in fast mode or when max_iterations is low
        if max_iterations <= 10:
            self.logger.debug("Skipping energy minimization (fast mode)")
            # Just do a quick energy evaluation
            energy = self._quick_energy_evaluation(pose, receptor_structure)
            pose.energy = energy
            return pose

        def energy_function(params):
            # params = [center_x, center_y, center_z, quat_w, quat_x, quat_y, quat_z]
            center = params[:3]
            rotation_quat = params[3:7]

            # Normalize quaternion
            rotation_quat = rotation_quat / np.linalg.norm(rotation_quat)

            # Apply transformation
            conf = mol.GetConformer(pose.conformer_id)
            ligand_coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
            ligand_center = np.mean(ligand_coords, axis=0)

            centered_coords = ligand_coords - ligand_center
            rotated_coords = Rotation.from_quat(rotation_quat).apply(centered_coords)
            final_coords = rotated_coords + center

            # Create temporary pose for energy evaluation
            temp_pose = Pose(
                coordinates=final_coords,
                center=center,
                rotation=rotation_quat,
                conformer_id=pose.conformer_id
            )

            # Use quick evaluation for minimization to avoid expensive calculations
            return self._quick_energy_evaluation(temp_pose, receptor_structure)

        # Initial parameters
        initial_params = np.concatenate([pose.center, pose.rotation])

        # Use L-BFGS-B for bounded optimization with smaller max iterations
        result = minimize(
            energy_function,
            initial_params,
            method='L-BFGS-B',
            options={'maxiter': min(max_iterations, 20)}  # Limit iterations
        )

        # Create minimized pose
        final_center = result.x[:3]
        final_rotation = result.x[3:7]
        final_rotation = final_rotation / np.linalg.norm(final_rotation)

        # Apply final transformation
        if hasattr(pose, 'conformer_id') and pose.conformer_id is not None:
            conf = mol.GetConformer(pose.conformer_id)
            ligand_coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
            ligand_center = np.mean(ligand_coords, axis=0)

            centered_coords = ligand_coords - ligand_center
            rotated_coords = Rotation.from_quat(final_rotation).apply(centered_coords)
            final_coords = rotated_coords + final_center
        else:
            # Use current pose coordinates as reference
            ligand_center = np.mean(pose.coordinates, axis=0)
            centered_coords = pose.coordinates - ligand_center
            rotated_coords = Rotation.from_quat(final_rotation).apply(centered_coords)
            final_coords = rotated_coords + final_center

        minimized_pose = Pose(
            coordinates=final_coords,
            center=final_center,
            rotation=final_rotation,
            conformer_id=pose.conformer_id,
            energy=result.fun
        )

        return minimized_pose

    def _detailed_energy_evaluation(self, pose: Pose, receptor_structure) -> float:
        """Detailed energy evaluation using the scoring function"""
        if self._scoring_function is None:
            return self._quick_energy_evaluation(pose, receptor_structure)

        # Use the assigned scoring function
        return self._scoring_function.calculate_binding_energy(
            pose.coordinates, receptor_structure
        )