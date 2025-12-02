"""
Simulated Annealing Refinement for Molecular Docking

Implements simulated annealing optimization for pose refinement with:
1. Temperature-based acceptance criteria
2. Adaptive step sizes
3. Multiple perturbation types (translation, rotation, torsion)
4. Convergence detection
5. Energy-based optimization

Based on the Metropolis-Hastings algorithm with exponential cooling schedule.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
import logging
import random
import math
from scipy.spatial.transform import Rotation as R
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
import copy


class SimulatedAnnealingRefiner:
    """Simulated annealing refinement for molecular poses"""

    def __init__(self, **params):
        self.logger = logging.getLogger("pandadock.refinement.simulated_annealing")

        # Annealing parameters
        self.initial_temperature = params.get('initial_temp', 1000.0)  # K
        self.final_temperature = params.get('final_temp', 0.1)  # K
        self.cooling_rate = params.get('cooling_rate', 0.95)  # Geometric cooling
        self.max_iterations = params.get('max_iterations', 5000)
        self.equilibration_steps = params.get('equilibration_steps', 100)

        # Perturbation parameters
        self.max_translation = params.get('max_translation', 0.5)  # Å
        self.max_rotation = params.get('max_rotation', 15.0)  # degrees
        self.max_torsion = params.get('max_torsion', 30.0)  # degrees

        # Step size adaptation
        self.adaptive_steps = params.get('adaptive_steps', True)
        self.target_acceptance = params.get('target_acceptance', 0.44)  # Optimal for continuous problems
        self.step_adjust_rate = params.get('step_adjust_rate', 1.05)

        # Convergence criteria
        self.energy_tolerance = params.get('energy_tolerance', 0.001)  # kcal/mol
        self.convergence_window = params.get('convergence_window', 50)

        # Perturbation weights
        self.perturbation_weights = params.get('perturbation_weights', {
            'translation': 0.4,
            'rotation': 0.4,
            'torsion': 0.2
        })

        # Statistics tracking
        self.reset_stats()

        self.logger.info(f"Simulated annealing initialized: T0={self.initial_temperature}K, "
                        f"Tf={self.final_temperature}K, max_iter={self.max_iterations}")

    def refine_pose(self, ligand_coords: np.ndarray, ligand_mol: Chem.Mol,
                   scoring_function: Callable, **kwargs) -> Tuple[np.ndarray, float, Dict]:
        """
        Refine ligand pose using simulated annealing

        Args:
            ligand_coords: Initial ligand coordinates (N x 3)
            ligand_mol: RDKit molecule object
            scoring_function: Energy evaluation function
            **kwargs: Additional parameters for scoring function

        Returns:
            Tuple of (refined_coords, best_energy, optimization_stats)
        """
        self.reset_stats()

        # Initialize
        current_coords = ligand_coords.copy()
        current_energy = scoring_function(current_coords, ligand_mol=ligand_mol, **kwargs)

        best_coords = current_coords.copy()
        best_energy = current_energy

        temperature = self.initial_temperature
        energy_history = []
        acceptance_history = []

        # Get rotatable bonds for torsion perturbations
        rotatable_bonds = self._get_rotatable_bonds(ligand_mol)

        self.logger.info(f"Starting refinement with initial energy: {current_energy:.3f} kcal/mol")

        iteration = 0
        while iteration < self.max_iterations and temperature > self.final_temperature:

            # Equilibration at current temperature
            accepted_moves = 0
            total_moves = 0

            for _ in range(self.equilibration_steps):
                total_moves += 1

                # Generate perturbation
                perturbed_coords, perturbation_type = self._generate_perturbation(
                    current_coords, ligand_mol, rotatable_bonds
                )

                # Evaluate energy
                try:
                    perturbed_energy = scoring_function(
                        perturbed_coords, ligand_mol=ligand_mol, **kwargs
                    )
                except:
                    # Skip invalid perturbations
                    continue

                # Acceptance criterion
                delta_energy = perturbed_energy - current_energy

                if self._accept_move(delta_energy, temperature):
                    current_coords = perturbed_coords.copy()
                    current_energy = perturbed_energy
                    accepted_moves += 1

                    # Update best pose
                    if current_energy < best_energy:
                        best_coords = current_coords.copy()
                        best_energy = current_energy
                        self.stats['best_iteration'] = iteration

                # Track statistics
                energy_history.append(current_energy)
                self.stats['perturbation_counts'][perturbation_type] += 1

                iteration += 1
                if iteration >= self.max_iterations:
                    break

            # Calculate acceptance rate
            acceptance_rate = accepted_moves / max(total_moves, 1)
            acceptance_history.append(acceptance_rate)

            # Adaptive step size adjustment
            if self.adaptive_steps and len(acceptance_history) >= 10:
                recent_acceptance = np.mean(acceptance_history[-10:])
                self._adjust_step_sizes(recent_acceptance)

            # Check convergence
            if len(energy_history) >= self.convergence_window:
                recent_energies = energy_history[-self.convergence_window:]
                energy_std = np.std(recent_energies)

                if energy_std < self.energy_tolerance:
                    self.logger.info(f"Converged at iteration {iteration} (energy std: {energy_std:.6f})")
                    break

            # Cool down
            temperature *= self.cooling_rate

            # Logging
            if iteration % 500 == 0 or temperature <= self.final_temperature:
                self.logger.debug(f"Iter {iteration}: T={temperature:.2f}K, "
                                f"E={current_energy:.3f}, Best={best_energy:.3f}, "
                                f"Accept={acceptance_rate:.3f}")

        # Final statistics
        self.stats.update({
            'final_iteration': iteration,
            'final_temperature': temperature,
            'initial_energy': scoring_function(ligand_coords, ligand_mol=ligand_mol, **kwargs),
            'final_energy': best_energy,
            'energy_improvement': self.stats['initial_energy'] - best_energy,
            'acceptance_rate': np.mean(acceptance_history) if acceptance_history else 0.0,
            'convergence_achieved': len(energy_history) >= self.convergence_window and
                                  np.std(energy_history[-self.convergence_window:]) < self.energy_tolerance
        })

        self.logger.info(f"Refinement completed: {self.stats['energy_improvement']:.3f} kcal/mol improvement "
                        f"in {iteration} iterations")

        return best_coords, best_energy, self.stats

    def _generate_perturbation(self, coords: np.ndarray, ligand_mol: Chem.Mol,
                             rotatable_bonds: List[Tuple[int, int]]) -> Tuple[np.ndarray, str]:
        """Generate random perturbation of ligand coordinates"""

        # Choose perturbation type
        rand_val = random.random()
        cumulative = 0

        for pert_type, weight in self.perturbation_weights.items():
            cumulative += weight
            if rand_val <= cumulative:
                perturbation_type = pert_type
                break
        else:
            perturbation_type = 'translation'

        new_coords = coords.copy()

        if perturbation_type == 'translation':
            # Random translation
            translation = np.random.uniform(
                -self.max_translation, self.max_translation, size=3
            )
            new_coords += translation

        elif perturbation_type == 'rotation':
            # Random rotation around center of mass
            center = np.mean(coords, axis=0)

            # Generate random rotation
            angle = np.random.uniform(-self.max_rotation, self.max_rotation)
            axis = np.random.uniform(-1, 1, size=3)
            axis = axis / np.linalg.norm(axis)

            # Apply rotation
            rotation = R.from_rotvec(np.radians(angle) * axis)
            centered_coords = coords - center
            rotated_coords = rotation.apply(centered_coords)
            new_coords = rotated_coords + center

        elif perturbation_type == 'torsion' and rotatable_bonds:
            # Random torsion rotation
            bond_idx = random.randint(0, len(rotatable_bonds) - 1)
            atom1_idx, atom2_idx = rotatable_bonds[bond_idx]

            # Get atoms to rotate
            atoms_to_rotate = self._get_atoms_to_rotate(ligand_mol, atom1_idx, atom2_idx)

            if atoms_to_rotate:
                # Random torsion angle
                angle = np.random.uniform(-self.max_torsion, self.max_torsion)

                # Rotation axis (bond vector)
                bond_vector = coords[atom2_idx] - coords[atom1_idx]
                bond_vector = bond_vector / np.linalg.norm(bond_vector)

                # Rotate atoms
                rotation = R.from_rotvec(np.radians(angle) * bond_vector)
                pivot = coords[atom1_idx]

                for atom_idx in atoms_to_rotate:
                    if atom_idx < len(new_coords):
                        centered = new_coords[atom_idx] - pivot
                        rotated = rotation.apply(centered)
                        new_coords[atom_idx] = rotated + pivot
            else:
                # Fallback to translation if torsion fails
                translation = np.random.uniform(
                    -self.max_translation, self.max_translation, size=3
                )
                new_coords += translation
                perturbation_type = 'translation'

        return new_coords, perturbation_type

    def _accept_move(self, delta_energy: float, temperature: float) -> bool:
        """Metropolis acceptance criterion"""
        if delta_energy <= 0:
            return True

        if temperature <= 0:
            return False

        probability = math.exp(-delta_energy / (0.001987 * temperature))  # kT in kcal/mol
        return random.random() < probability

    def _adjust_step_sizes(self, acceptance_rate: float):
        """Adaptively adjust step sizes based on acceptance rate"""
        ratio = acceptance_rate / self.target_acceptance

        if ratio > 1.1:  # Too many acceptances - increase step sizes
            self.max_translation *= self.step_adjust_rate
            self.max_rotation *= self.step_adjust_rate
            self.max_torsion *= self.step_adjust_rate
        elif ratio < 0.9:  # Too few acceptances - decrease step sizes
            self.max_translation /= self.step_adjust_rate
            self.max_rotation /= self.step_adjust_rate
            self.max_torsion /= self.step_adjust_rate

        # Bounds checking
        self.max_translation = np.clip(self.max_translation, 0.01, 2.0)
        self.max_rotation = np.clip(self.max_rotation, 1.0, 90.0)
        self.max_torsion = np.clip(self.max_torsion, 5.0, 180.0)

    def _get_rotatable_bonds(self, mol: Chem.Mol) -> List[Tuple[int, int]]:
        """Identify rotatable bonds in the molecule"""
        rotatable_bonds = []

        for bond in mol.GetBonds():
            if (bond.GetBondType() == Chem.BondType.SINGLE and
                not bond.IsInRing() and
                bond.GetBeginAtom().GetDegree() > 1 and
                bond.GetEndAtom().GetDegree() > 1):

                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                rotatable_bonds.append((begin_idx, end_idx))

        return rotatable_bonds

    def _get_atoms_to_rotate(self, mol: Chem.Mol, atom1_idx: int, atom2_idx: int) -> List[int]:
        """Get atoms that should be rotated around a bond"""
        # Simple implementation - rotate smaller fragment
        visited = set()
        fragment1 = self._get_connected_atoms(mol, atom1_idx, atom2_idx, visited)

        visited = set()
        fragment2 = self._get_connected_atoms(mol, atom2_idx, atom1_idx, visited)

        # Rotate the smaller fragment
        if len(fragment1) <= len(fragment2):
            return list(fragment1)
        else:
            return list(fragment2)

    def _get_connected_atoms(self, mol: Chem.Mol, start_atom: int,
                           exclude_atom: int, visited: set) -> set:
        """Get all atoms connected to start_atom, excluding exclude_atom"""
        if start_atom in visited:
            return set()

        visited.add(start_atom)
        connected = {start_atom}

        atom = mol.GetAtomWithIdx(start_atom)
        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx != exclude_atom and neighbor_idx not in visited:
                connected.update(
                    self._get_connected_atoms(mol, neighbor_idx, exclude_atom, visited)
                )

        return connected

    def reset_stats(self):
        """Reset optimization statistics"""
        self.stats = {
            'initial_energy': 0.0,
            'final_energy': 0.0,
            'best_energy': float('inf'),
            'best_iteration': 0,
            'final_iteration': 0,
            'energy_improvement': 0.0,
            'acceptance_rate': 0.0,
            'convergence_achieved': False,
            'perturbation_counts': {
                'translation': 0,
                'rotation': 0,
                'torsion': 0
            }
        }

    def get_optimization_stats(self) -> Dict:
        """Get detailed optimization statistics"""
        return self.stats.copy()


class AdaptiveSimulatedAnnealing(SimulatedAnnealingRefiner):
    """Enhanced simulated annealing with adaptive parameters"""

    def __init__(self, **params):
        super().__init__(**params)

        # Enhanced adaptive parameters
        self.temperature_schedule = params.get('temp_schedule', 'exponential')  # or 'linear', 'logarithmic'
        self.reheat_threshold = params.get('reheat_threshold', 100)  # iterations without improvement
        self.reheat_factor = params.get('reheat_factor', 2.0)
        self.min_reheat_temp = params.get('min_reheat_temp', 50.0)

        # Multiple trajectory support
        self.num_trajectories = params.get('num_trajectories', 1)

        self.logger.info(f"Adaptive SA initialized with {self.num_trajectories} trajectory(ies)")

    def refine_pose_adaptive(self, ligand_coords: np.ndarray, ligand_mol: Chem.Mol,
                           scoring_function: Callable, **kwargs) -> Tuple[np.ndarray, float, Dict]:
        """
        Multi-trajectory adaptive simulated annealing refinement
        """
        if self.num_trajectories == 1:
            return self.refine_pose(ligand_coords, ligand_mol, scoring_function, **kwargs)

        # Multiple trajectories
        best_coords = None
        best_energy = float('inf')
        all_stats = []

        for traj in range(self.num_trajectories):
            # Add small random perturbation to initial coordinates
            perturbed_coords = ligand_coords + np.random.normal(0, 0.1, ligand_coords.shape)

            coords, energy, stats = self.refine_pose(
                perturbed_coords, ligand_mol, scoring_function, **kwargs
            )

            if energy < best_energy:
                best_coords = coords.copy()
                best_energy = energy
                best_stats = stats

            all_stats.append(stats)
            self.logger.debug(f"Trajectory {traj+1}/{self.num_trajectories}: "
                            f"Energy = {energy:.3f} kcal/mol")

        # Combine statistics
        best_stats['all_trajectories'] = all_stats
        best_stats['num_trajectories'] = self.num_trajectories

        self.logger.info(f"Multi-trajectory refinement completed: "
                        f"Best energy = {best_energy:.3f} kcal/mol")

        return best_coords, best_energy, best_stats