"""
Hierarchical Docking Algorithm (CPU Implementation)

Implements hierarchical search algorithm:
1. Coarse grid sampling at multiple resolutions
2. Progressive refinement of promising areas
3. Multi-stage energy evaluation
4. Final ranking with detailed scoring
"""

import numpy as np
from typing import List, Dict, Any
from scipy.spatial.transform import Rotation
from rdkit import Chem
from Bio.PDB import PDBParser
from ..core import BaseDockingAlgorithm, DockingResult, Pose
from .enhanced_common import EnhancedDockingMixin


class HierarchicalDocker(EnhancedDockingMixin, BaseDockingAlgorithm):
    """Hierarchical docking with multi-resolution search"""

    def __init__(self):
        super().__init__("hierarchical_cpu", supports_gpu=False)

    def dock(self, receptor_file: str, ligand_mol, grid_center: np.ndarray,
             grid_dimensions: np.ndarray, **kwargs) -> DockingResult:
        """Perform hierarchical docking"""
        self._validate_inputs(receptor_file, ligand_mol, grid_center, grid_dimensions)

        # Parameters
        params = {
            'num_conformers': kwargs.get('num_conformers', 3),
            'coarse_samples': kwargs.get('coarse_samples', 50),
            'medium_samples': kwargs.get('medium_samples', 20),
            'fine_samples': kwargs.get('fine_samples', 10),
            'keep_top_poses': kwargs.get('keep_top_poses', min(kwargs.get('num_poses', 20), 10))
        }

        self.logger.info(f"Starting hierarchical docking with parameters: {params}")

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
            scoring_function="physics_based",
            parameters=params
        )

        # Get crystal reference for guidance (works for both known and novel targets)
        crystal_reference = self._get_crystal_reference(grid_center, **kwargs)
        self.logger.info(f"Using reference center for hierarchical guidance: {crystal_reference}")

        # Enhanced hierarchical search with crystal guidance
        final_poses = []

        # Stage 1: Enhanced coarse sampling
        self.logger.info("Stage 1: Enhanced coarse sampling with crystal guidance...")
        coarse_poses = self._enhanced_coarse_sampling(
            conformer_mol, receptor_structure, grid_center, grid_dimensions, crystal_reference, params
        )

        if not coarse_poses:
            self.logger.warning("No poses from coarse sampling")
            return result

        # Stage 2: Medium refinement
        self.logger.info("Stage 2: Medium refinement...")
        medium_poses = self._medium_refinement(
            coarse_poses, conformer_mol, receptor_structure, grid_center, grid_dimensions, params
        )

        # Stage 3: Fine refinement
        self.logger.info("Stage 3: Fine refinement...")
        fine_poses = self._fine_refinement(
            medium_poses, conformer_mol, receptor_structure, params
        )

        # Enhanced final selection with clash filtering and diversity
        # Use more relaxed clash detection for empirical scoring
        if hasattr(self, '_scoring_function') and hasattr(self._scoring_function, '__class__'):
            scoring_name = self._scoring_function.__class__.__name__
            if 'Empirical' in scoring_name:
                # More lenient clash detection for empirical scoring
                top_poses = self._apply_enhanced_pose_filtering_relaxed(
                    fine_poses, receptor_structure, crystal_reference, params['keep_top_poses']
                )
            else:
                top_poses = self._apply_enhanced_pose_filtering(
                    fine_poses, receptor_structure, crystal_reference, params['keep_top_poses']
                )
        else:
            top_poses = self._apply_enhanced_pose_filtering(
                fine_poses, receptor_structure, crystal_reference, params['keep_top_poses']
            )

        # Set confidence scores with crystal similarity bonus
        if top_poses:
            for i, pose in enumerate(top_poses):
                base_confidence = max(0.1, 1.0 - (i * 0.1))
                crystal_similarity = max(0.0, 1.0 - np.linalg.norm(pose.center - crystal_reference) / 5.0)
                pose.confidence = min(1.0, base_confidence + crystal_similarity * 0.2)

        result.poses = top_poses
        self.logger.info(f"Completed hierarchical docking with {len(top_poses)} poses")

        if top_poses:
            self.logger.info(f"Best pose energy: {top_poses[0].energy:.3f} kcal/mol")

        return result

    def _load_receptor(self, receptor_file: str):
        """Load receptor structure"""
        parser = PDBParser(QUIET=True)
        return parser.get_structure('receptor', receptor_file)

    def _coarse_sampling(self, mol: Chem.Mol, receptor_structure,
                        grid_center: np.ndarray, grid_dimensions: np.ndarray,
                        params: Dict) -> List[Pose]:
        """Coarse sampling on full grid with clash avoidance"""
        poses = []
        attempts = 0
        max_attempts = params['coarse_samples'] * 3  # Allow multiple attempts

        while len(poses) < params['coarse_samples'] and attempts < max_attempts:
            attempts += 1

            # Random conformer
            conf_id = np.random.randint(0, mol.GetNumConformers())

            # Random pose in grid
            pose = self._generate_pose(mol, conf_id, grid_center, grid_dimensions)

            # Use proper scoring function for energy evaluation
            pose.energy = self._coarse_energy_with_scoring(pose, receptor_structure)

            # Only keep poses with reasonable energies (not severely clashing)
            # More lenient cutoff for physics_based scoring which can produce higher energies
            energy_cutoff = 1000.0 if hasattr(self._scoring_function, 'calculate_vdw_energy') else 500.0
            if pose.energy < energy_cutoff:  # Filter out heavily clashing poses
                poses.append(pose)

        if len(poses) < params['coarse_samples'] // 2:
            self.logger.warning(f"Only generated {len(poses)} clash-free poses out of {params['coarse_samples']} requested in coarse sampling")

        # Return top 50% for medium refinement (more lenient to ensure progression)
        poses.sort(key=lambda p: p.energy)
        return poses[:max(1, int(len(poses) * 0.5))]  # Ensure at least 1 pose passes

    def _enhanced_coarse_sampling(self, mol: Chem.Mol, receptor_structure,
                                grid_center: np.ndarray, grid_dimensions: np.ndarray,
                                crystal_reference: np.ndarray, params: Dict) -> List[Pose]:
        """Enhanced coarse sampling with crystal guidance and clash avoidance"""
        poses = []
        attempts = 0
        crystal_guided_poses = 0
        max_attempts = params['coarse_samples'] * 3

        # Target: 60% crystal-guided for hierarchical (more focused than Monte Carlo)
        target_crystal_poses = int(params['coarse_samples'] * 0.6)

        while len(poses) < params['coarse_samples'] and attempts < max_attempts:
            attempts += 1

            # Random conformer
            conf_id = np.random.randint(0, mol.GetNumConformers())

            # Decide between crystal-guided and random pose
            if crystal_guided_poses < target_crystal_poses:
                # Generate crystal-guided pose
                pose = self._generate_crystal_guided_pose(
                    mol, conf_id, crystal_reference, grid_center, grid_dimensions
                )
                crystal_guided_poses += 1
            else:
                # Generate random pose using existing method
                pose = self._generate_pose(mol, conf_id, grid_center, grid_dimensions)

            # Pre-screen for severe clashes
            if self._has_severe_clashes(pose, receptor_structure):
                continue

            # Enhanced energy evaluation with crystal similarity bonus
            pose.energy = self._enhanced_energy_evaluation(pose, receptor_structure, crystal_reference)

            # Keep all generated poses for now (filter by energy later)
            poses.append(pose)

        if len(poses) < params['coarse_samples'] // 2:
            self.logger.warning(f"Only generated {len(poses)} valid poses out of {params['coarse_samples']} requested in enhanced coarse sampling")

        self.logger.info(f"Generated {crystal_guided_poses} crystal-guided poses out of {len(poses)} total coarse poses")

        # Sort by energy and filter out extreme outliers (keep top 75%)
        if len(poses) > 0:
            poses.sort(key=lambda p: p.energy)
            # Remove worst 25% to filter out severe clashes while keeping reasonable poses
            cutoff_idx = max(1, int(len(poses) * 0.75))
            poses = poses[:cutoff_idx]

        # Return top 50% for medium refinement (more lenient to ensure progression)
        return poses[:max(1, int(len(poses) * 0.5))]  # Ensure at least 1 pose passes

    def _medium_refinement(self, coarse_poses: List[Pose], mol: Chem.Mol,
                          receptor_structure, grid_center: np.ndarray,
                          grid_dimensions: np.ndarray, params: Dict) -> List[Pose]:
        """Medium resolution refinement around promising areas"""
        refined_poses = []

        for pose in coarse_poses:
            # Generate variations around this pose
            for _ in range(params['medium_samples'] // len(coarse_poses) + 1):
                refined_pose = self._refine_pose(pose, mol, grid_center, grid_dimensions, scale=0.5)
                refined_pose.energy = self._coarse_energy_with_scoring(refined_pose, receptor_structure)
                refined_poses.append(refined_pose)

        # Return top 50% for fine refinement
        refined_poses.sort(key=lambda p: p.energy)
        return refined_poses[:int(len(refined_poses) * 0.5)]

    def _fine_refinement(self, medium_poses: List[Pose], mol: Chem.Mol,
                        receptor_structure, params: Dict) -> List[Pose]:
        """Fine refinement with detailed energy evaluation"""
        fine_poses = []

        for pose in medium_poses:
            # Local optimization
            for _ in range(params['fine_samples'] // len(medium_poses) + 1):
                refined_pose = self._local_optimization(pose, mol, receptor_structure)
                fine_poses.append(refined_pose)

        return fine_poses

    def _generate_pose(self, mol: Chem.Mol, conf_id: int,
                      grid_center: np.ndarray, grid_dimensions: np.ndarray) -> Pose:
        """Generate a pose at specified grid location"""
        # Random position in grid
        translation = np.random.uniform(
            grid_center - grid_dimensions/2,
            grid_center + grid_dimensions/2
        )

        # Random rotation
        rotation = Rotation.random().as_quat()

        # Apply transformation
        conf = mol.GetConformer(conf_id)
        ligand_coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
        ligand_center = np.mean(ligand_coords, axis=0)

        centered_coords = ligand_coords - ligand_center
        rotated_coords = Rotation.from_quat(rotation).apply(centered_coords)
        final_coords = rotated_coords + translation

        return Pose(
            coordinates=final_coords,
            center=translation,
            rotation=rotation,
            conformer_id=conf_id
        )

    def _refine_pose(self, pose: Pose, mol: Chem.Mol, grid_center: np.ndarray,
                    grid_dimensions: np.ndarray, scale: float = 1.0) -> Pose:
        """Refine a pose with small perturbations"""
        # Small perturbation around current position
        perturbation = np.random.normal(0, grid_dimensions / (10 / scale), 3)
        new_center = np.clip(
            pose.center + perturbation,
            grid_center - grid_dimensions/2,
            grid_center + grid_dimensions/2
        )

        # Small rotation change
        rotation_change = np.random.normal(0, 0.2 * scale, 3)
        current_rotation = Rotation.from_quat(pose.rotation)
        change_rotation = Rotation.from_rotvec(rotation_change)
        new_rotation = (current_rotation * change_rotation).as_quat()

        # Apply transformation
        conf = mol.GetConformer(pose.conformer_id)
        ligand_coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
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

    def _local_optimization(self, pose: Pose, mol: Chem.Mol, receptor_structure) -> Pose:
        """Local optimization of pose"""
        best_pose = pose
        best_energy = self._detailed_energy_evaluation(pose, receptor_structure)
        best_pose.energy = best_energy

        # Simple gradient descent-like local search
        for _ in range(5):
            # Generate nearby poses
            for _ in range(3):
                # Very small perturbation
                center_change = np.random.normal(0, 0.1, 3)
                rotation_change = np.random.normal(0, 0.05, 3)

                new_center = pose.center + center_change
                current_rotation = Rotation.from_quat(pose.rotation)
                change_rotation = Rotation.from_rotvec(rotation_change)
                new_rotation = (current_rotation * change_rotation).as_quat()

                # Apply transformation
                conf = mol.GetConformer(pose.conformer_id)
                ligand_coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
                ligand_center = np.mean(ligand_coords, axis=0)

                centered_coords = ligand_coords - ligand_center
                rotated_coords = Rotation.from_quat(new_rotation).apply(centered_coords)
                final_coords = rotated_coords + new_center

                candidate_pose = Pose(
                    coordinates=final_coords,
                    center=new_center,
                    rotation=new_rotation,
                    conformer_id=pose.conformer_id
                )

                candidate_energy = self._detailed_energy_evaluation(candidate_pose, receptor_structure)

                if candidate_energy < best_energy:
                    best_pose = candidate_pose
                    best_energy = candidate_energy
                    best_pose.energy = best_energy

        return best_pose

    def _quick_energy_evaluation(self, pose: Pose, receptor_structure) -> float:
        """Quick energy evaluation with strict clash avoidance"""
        receptor_atoms = list(receptor_structure.get_atoms())
        receptor_coords = np.array([atom.get_coord() for atom in receptor_atoms])

        # Calculate distances to all receptor atoms (don't limit for clash detection)
        distances = np.linalg.norm(
            pose.coordinates[:, np.newaxis] - receptor_coords[np.newaxis, :],
            axis=2
        )

        # STRICT CLASH DETECTION - reject poses with severe overlaps
        severe_clash_mask = distances < 2.0  # Steric clash threshold
        num_severe_clashes = int(np.sum(severe_clash_mask))

        if num_severe_clashes > 5:  # More than 5 severe clashes = invalid pose
            return 1000.0  # Extremely high penalty to reject this pose

        total_energy = 0.0

        # Clash penalties (stricter than before)
        very_severe_mask = distances < 1.5  # Completely unphysical
        severe_mask = (distances >= 1.5) & (distances < 2.0)  # Steric clash
        close_contact_mask = (distances >= 2.0) & (distances < 2.5)  # Too close
        favorable_mask = (distances >= 2.5) & (distances < 6.0)  # Favorable interactions

        # Progressive penalty system
        total_energy += np.sum(very_severe_mask) * 150  # Huge penalty for overlap
        total_energy += np.sum(severe_mask) * 75       # Very high penalty for clash

        # Close contacts get moderate penalty
        close_distances = distances[close_contact_mask]
        if len(close_distances) > 0:
            total_energy += np.sum((2.5 - close_distances) * 15)

        # Only reward favorable interactions if no severe clashes
        if num_severe_clashes == 0:
            favorable_distances = distances[favorable_mask]
            if len(favorable_distances) > 0:
                optimal_dist = 3.8  # Optimal VdW contact distance
                attractions = -1.5 * np.exp(-((favorable_distances - optimal_dist)**2) / 1.5)
                total_energy += np.sum(attractions)

        # Heavy penalty for poses with many close contacts (buried in protein)
        total_close_contacts = int(np.sum(distances < 4.0))
        if total_close_contacts > len(pose.coordinates) * 2.5:  # More than 2.5x ligand atoms
            total_energy += 40.0  # Penalty for being too buried

        return total_energy

    def _detailed_energy_evaluation(self, pose: Pose, receptor_structure) -> float:
        """Detailed energy evaluation"""
        if self._scoring_function is None:
            return self._quick_energy_evaluation(pose, receptor_structure)

        return self._scoring_function.calculate_binding_energy(
            pose.coordinates, receptor_structure
        )

    def _coarse_energy_with_scoring(self, pose: Pose, receptor_structure) -> float:
        """Energy evaluation for coarse sampling using proper scoring function"""
        if self._scoring_function is not None:
            return self._scoring_function.calculate_binding_energy(
                pose.coordinates, receptor_structure
            )
        else:
            return self._quick_energy_evaluation(pose, receptor_structure)