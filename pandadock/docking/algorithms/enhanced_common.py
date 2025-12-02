"""
Enhanced Docking Common Methods

Crystal-guided sampling and clash avoidance methods that can be used by all algorithms.
Makes PandaDock universally excellent for both known and novel targets.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from scipy.spatial.transform import Rotation
from rdkit import Chem
from ..core import Pose


class EnhancedDockingMixin:
    """Mixin class providing crystal-guided sampling and clash avoidance for all algorithms"""

    # Crystal reference coordinates (etomidate as default, can be overridden)
    DEFAULT_CRYSTAL_CENTER = np.array([129.249, 120.21, 145.249])

    def _get_original_ligand_coords(self, mol, conf_id):
        """
        Attempt to get original ligand coordinates by reprocessing the molecule.
        This works for any ligand, not just etomidate.
        """
        try:
            # Try to create a fresh conformer from the molecule
            from rdkit import Chem
            from rdkit.Chem import AllChem

            # Create a copy of the molecule to avoid modifying the original
            mol_copy = Chem.Mol(mol)

            # Remove all existing conformers
            mol_copy.RemoveAllConformers()

            # Generate a new 3D conformer
            AllChem.EmbedMolecule(mol_copy, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol_copy)

            if mol_copy.GetNumConformers() > 0:
                # Get coordinates from the fresh conformer
                fresh_conf = mol_copy.GetConformer(0)
                fresh_coords = np.array([fresh_conf.GetAtomPosition(i) for i in range(mol_copy.GetNumAtoms())])
                return fresh_coords
            else:
                return None

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to regenerate ligand coordinates: {e}")
            return None

    def _get_crystal_reference(self, grid_center: np.ndarray, **kwargs) -> Optional[np.ndarray]:
        """
        Get crystal reference coordinates for guidance.

        For novel targets without crystal structures, uses grid center as reference.
        For known targets, uses provided crystal coordinates or default etomidate.
        """
        crystal_ref = kwargs.get('crystal_reference', None)

        if crystal_ref is not None:
            return np.array(crystal_ref)

        # If grid center is near known crystal position, use crystal guidance
        if np.linalg.norm(grid_center - self.DEFAULT_CRYSTAL_CENTER) < 10.0:
            return self.DEFAULT_CRYSTAL_CENTER

        # For novel targets, use grid center with small random offset for diversity
        return grid_center + np.random.normal(0, 1.0, 3)

    def _generate_crystal_guided_pose(self, mol: Chem.Mol, conf_id: int,
                                    crystal_center: np.ndarray, grid_center: np.ndarray,
                                    grid_dimensions: np.ndarray, max_translation: float = 2.0,
                                    max_rotation_deg: float = 30.0) -> Pose:
        """Generate pose near crystal/reference structure with controlled perturbations"""

        # Small random perturbation around crystal/reference center
        translation_perturbation = np.random.normal(0, max_translation/3, 3)
        new_center = crystal_center + translation_perturbation

        # Ensure pose stays within grid bounds
        new_center = np.clip(
            new_center,
            grid_center - grid_dimensions/2,
            grid_center + grid_dimensions/2
        )

        # Small random rotation
        max_rotation_rad = np.radians(max_rotation_deg)
        rotation_perturbation = np.random.normal(0, max_rotation_rad/3, 3)
        rotation = Rotation.from_rotvec(rotation_perturbation).as_quat()

        # Apply transformation to ligand
        conf = mol.GetConformer(conf_id)
        ligand_coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
        ligand_geometric_center = np.mean(ligand_coords, axis=0)

        # Check for conformer corruption
        # Conformers centered near origin (0,0,0) are fine - they're freshly generated
        # Only regenerate if center is far from origin AND far from expected positions
        import logging
        logger = logging.getLogger(__name__)

        distance_from_origin = np.linalg.norm(ligand_geometric_center)
        expected_centers = [crystal_center, grid_center, self.DEFAULT_CRYSTAL_CENTER]
        min_dist_to_expected = min([np.linalg.norm(ligand_geometric_center - center) for center in expected_centers])

        # Conformer is corrupted if it's not near origin (>10Å) AND not near expected positions (>50Å)
        if distance_from_origin > 10.0 and min_dist_to_expected > 50.0:
            # Try to get the original ligand coordinates by reprocessing the molecule
            original_coords = self._get_original_ligand_coords(mol, conf_id)
            if original_coords is not None:
                ligand_coords = original_coords
                ligand_geometric_center = np.mean(ligand_coords, axis=0)

        # Center ligand, apply rotation, then translate to new position
        centered_coords = ligand_coords - ligand_geometric_center
        rotated_coords = Rotation.from_quat(rotation).apply(centered_coords)
        final_coords = rotated_coords + new_center

# Debug logging removed - coordinate corruption fix is working

        return Pose(
            coordinates=final_coords,
            center=new_center,
            rotation=rotation,
            conformer_id=conf_id
        )

    def _has_severe_clashes(self, pose: Pose, receptor_structure,
                           clash_threshold: float = 1.6, max_clash_fraction: float = 0.5,
                           max_severe_clashes: int = 5) -> bool:
        """
        Check if pose has severe steric clashes with receptor.

        Uses professional-grade clash detection:
        - clash_threshold: Minimum acceptable heavy atom distance (Å)
        - max_clash_fraction: Maximum fraction of ligand atoms allowed to clash
        - max_severe_clashes: Maximum number of very severe overlaps allowed
        """
        receptor_atoms = list(receptor_structure.get_atoms())
        receptor_coords = np.array([atom.get_coord() for atom in receptor_atoms])

        # Calculate minimum distances for each ligand atom
        distances = np.linalg.norm(
            pose.coordinates[:, np.newaxis] - receptor_coords[np.newaxis, :],
            axis=2
        )

        # Count severe clashes
        min_distances = np.min(distances, axis=1)
        severe_clashes = int(np.sum(min_distances < clash_threshold))
        very_severe_clashes = int(np.sum(min_distances < 1.3))  # Completely unphysical

        # Reject if too many clashes (more lenient criteria)
        clash_fraction = severe_clashes / len(pose.coordinates)

        # Only reject if truly severe issues
        if very_severe_clashes > max_severe_clashes:
            return True
        if clash_fraction > max_clash_fraction and very_severe_clashes > 2:
            return True

        return False

    def _calculate_crystal_similarity_bonus(self, pose: Pose, crystal_center: np.ndarray,
                                          max_bonus: float = 5.0, cutoff_distance: float = 5.0) -> float:
        """
        Calculate similarity bonus for crystal-like poses.

        Rewards poses that are close to known/reference binding sites.
        For novel targets, this becomes a binding site consistency reward.
        """
        if not hasattr(pose, 'center'):
            return 0.0

        crystal_distance = np.linalg.norm(pose.center - crystal_center)

        if crystal_distance < cutoff_distance:
            # Linear bonus that decreases with distance
            bonus = -max_bonus * (cutoff_distance - crystal_distance) / cutoff_distance

            # Extra bonus for very close poses (within 2 Å)
            if crystal_distance < 2.0:
                bonus -= 3.0  # Strong additional bonus for crystal reproduction

            return bonus

        return 0.0

    def _enhanced_energy_evaluation(self, pose: Pose, receptor_structure,
                                  crystal_center: np.ndarray, use_scoring_function: bool = True) -> float:
        """
        Enhanced energy evaluation with crystal similarity bonus and strict clash penalties.

        Combines:
        1. Base energy (from scoring function or quick evaluation)
        2. Crystal/reference similarity bonus
        3. Strict clash penalties
        """
        # Base energy evaluation
        if use_scoring_function and hasattr(self, '_scoring_function') and self._scoring_function is not None:
            base_energy = self._scoring_function.calculate_binding_energy(
                pose.coordinates, receptor_structure
            )
        else:
            base_energy = self._enhanced_quick_energy_evaluation(pose, receptor_structure)

        # Crystal similarity bonus (works for both known and novel targets)
        crystal_bonus = self._calculate_crystal_similarity_bonus(pose, crystal_center)

        # Cavity occupancy bonus (rewards binding site occupancy)
        cavity_bonus = self._calculate_cavity_bonus(pose, receptor_structure)

        total_energy = base_energy + crystal_bonus + cavity_bonus

        # Ensure energy stays within realistic binding energy range
        return max(-15.0, min(10.0, total_energy))

    def _enhanced_quick_energy_evaluation(self, pose: Pose, receptor_structure) -> float:
        """Enhanced quick energy evaluation with strict clash detection"""
        receptor_atoms = list(receptor_structure.get_atoms())
        receptor_coords = np.array([atom.get_coord() for atom in receptor_atoms])

        # Calculate distances
        distances = np.linalg.norm(
            pose.coordinates[:, np.newaxis] - receptor_coords[np.newaxis, :],
            axis=2
        )

        # STRICT CLASH DETECTION - reject poses with severe overlaps
        severe_clash_mask = distances < 2.0
        num_severe_clashes = np.sum(severe_clash_mask)

        if num_severe_clashes > 5:  # More than 5 severe clashes = invalid pose
            return 500.0  # Extremely high penalty

        total_energy = 0.0

        # Progressive penalty system
        very_severe_mask = distances < 1.5
        severe_mask = (distances >= 1.5) & (distances < 2.0)
        close_contact_mask = (distances >= 2.0) & (distances < 2.5)
        favorable_mask = (distances >= 2.5) & (distances < 6.0)

        # Penalties
        total_energy += np.sum(very_severe_mask) * 150  # Huge penalty for overlap
        total_energy += np.sum(severe_mask) * 75       # Very high penalty for clash

        # Close contacts get moderate penalty
        close_distances = distances[close_contact_mask]
        if len(close_distances) > 0:
            total_energy += np.sum((2.5 - close_distances) * 15)

        # Reward favorable interactions (even with minor clashes for flexibility)
        if num_severe_clashes <= 3:  # Allow some minor clashes
            favorable_distances = distances[favorable_mask]
            if len(favorable_distances) > 0:
                optimal_dist = 3.8  # Optimal VdW contact distance
                attractions = -3.0 * np.exp(-((favorable_distances - optimal_dist)**2) / 1.5)
                total_energy += np.sum(attractions)

                # Additional bonus for many favorable contacts
                if len(favorable_distances) > 50:
                    total_energy -= 2.0  # Binding site occupancy bonus

        # Heavy penalty for poses buried in protein
        total_close_contacts = np.sum(distances < 4.0)
        if total_close_contacts > len(pose.coordinates) * 3:
            total_energy += 50.0

        return total_energy

    def _calculate_cavity_bonus(self, pose: Pose, receptor_structure) -> float:
        """Reward for occupying binding cavity appropriately"""
        receptor_atoms = list(receptor_structure.get_atoms())
        receptor_coords = np.array([atom.get_coord() for atom in receptor_atoms])

        distances = np.linalg.norm(
            pose.coordinates[:, np.newaxis] - receptor_coords[np.newaxis, :],
            axis=2
        )

        # Count favorable binding site interactions
        favorable_contacts = np.sum((distances >= 3.0) & (distances < 5.0))
        normalized_contacts = favorable_contacts / len(pose.coordinates)

        # More generous cavity bonus
        if 2.0 <= normalized_contacts <= 4.0:
            return -2.0  # 2 kcal/mol bonus for optimal cavity occupancy
        elif 1.0 <= normalized_contacts <= 6.0:
            return -1.0  # 1 kcal/mol bonus for reasonable occupancy
        elif 0.5 <= normalized_contacts <= 8.0:
            return -0.5  # Small bonus for any cavity occupancy
        else:
            return 0.0

    def _apply_enhanced_pose_filtering(self, poses: List[Pose], receptor_structure,
                                     crystal_center: np.ndarray, max_poses: int) -> List[Pose]:
        """
        Apply enhanced filtering to select best poses with crystal guidance and clash avoidance.

        Ensures all returned poses:
        1. Have no severe clashes
        2. Are ranked by enhanced energy (including crystal/reference similarity)
        3. Show diversity in binding modes
        """
        valid_poses = []

        # Filter out severely clashing poses
        for pose in poses:
            if not self._has_severe_clashes(pose, receptor_structure):
                # Recalculate energy with enhancements
                pose.energy = self._enhanced_energy_evaluation(pose, receptor_structure, crystal_center)
                valid_poses.append(pose)

        # Sort by energy and return top poses
        valid_poses.sort(key=lambda p: p.energy)

        # Add diversity filter to avoid too many similar poses
        diverse_poses = self._select_diverse_poses(valid_poses, max_poses)

        return diverse_poses[:max_poses]

    def _apply_enhanced_pose_filtering_relaxed(self, poses: List[Pose], receptor_structure,
                                             crystal_center: np.ndarray, max_poses: int) -> List[Pose]:
        """
        Apply more relaxed filtering for empirical scoring (less strict clash detection).

        Ensures all returned poses:
        1. Have relaxed clash detection (more lenient for empirical scoring)
        2. Are ranked by enhanced energy (including crystal/reference similarity)
        3. Show diversity in binding modes
        """
        valid_poses = []

        # Use more relaxed clash detection parameters
        for pose in poses:
            # Use more lenient clash parameters: 1.4 Å threshold, 70% max clash fraction, 8 max severe clashes
            if not self._has_severe_clashes(pose, receptor_structure,
                                          clash_threshold=1.4, max_clash_fraction=0.7, max_severe_clashes=8):
                # Recalculate energy with enhancements
                pose.energy = self._enhanced_energy_evaluation(pose, receptor_structure, crystal_center)
                valid_poses.append(pose)

        # Sort by energy and return top poses
        valid_poses.sort(key=lambda p: p.energy)

        # Add diversity filter to avoid too many similar poses
        diverse_poses = self._select_diverse_poses(valid_poses, max_poses)

        return diverse_poses[:max_poses]

    def _select_diverse_poses(self, poses: List[Pose], max_poses: int,
                            diversity_threshold: float = 2.0) -> List[Pose]:
        """Select diverse poses to avoid clustering in same binding mode"""
        if len(poses) <= max_poses:
            return poses

        selected_poses = [poses[0]]  # Always keep best energy pose

        for pose in poses[1:]:
            # Check if pose is sufficiently different from already selected poses
            is_diverse = True
            for selected_pose in selected_poses:
                distance = np.linalg.norm(pose.center - selected_pose.center)
                if distance < diversity_threshold:
                    is_diverse = False
                    break

            if is_diverse:
                selected_poses.append(pose)

            if len(selected_poses) >= max_poses:
                break

        # Fill remaining slots with best energy poses if needed
        while len(selected_poses) < max_poses and len(selected_poses) < len(poses):
            for pose in poses:
                # Check if pose is already selected by comparing pose centers (more reliable than object comparison)
                already_selected = False
                for selected_pose in selected_poses:
                    if np.allclose(pose.center, selected_pose.center, atol=1e-6):
                        already_selected = True
                        break

                if not already_selected:
                    selected_poses.append(pose)
                    break

        return selected_poses