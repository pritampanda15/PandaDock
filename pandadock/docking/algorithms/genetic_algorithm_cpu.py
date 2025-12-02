"""
Genetic Algorithm Docking (CPU Implementation)

Implements genetic algorithm optimization for molecular docking:
1. Population initialization with diverse poses
2. Selection based on energy fitness
3. Crossover and mutation operations
4. Elitism to preserve best solutions
"""

import numpy as np
from typing import List, Dict, Any
import random
import math
from scipy.spatial.transform import Rotation
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio.PDB import PDBParser
from ..core import BaseDockingAlgorithm, DockingResult, Pose


class GeneticAlgorithmDocker(BaseDockingAlgorithm):
    """Genetic algorithm docking (CPU implementation)"""

    def __init__(self):
        super().__init__("genetic_algorithm_cpu", supports_gpu=False)

    def dock(self, receptor_file: str, ligand_mol, grid_center: np.ndarray,
             grid_dimensions: np.ndarray, **kwargs) -> DockingResult:
        """Perform genetic algorithm docking"""
        self._validate_inputs(receptor_file, ligand_mol, grid_center, grid_dimensions)

        # Extract parameters
        params = {
            'population_size': kwargs.get('population_size', 20),
            'generations': kwargs.get('generations', 10),
            'mutation_rate': kwargs.get('mutation_rate', 0.3),
            'crossover_rate': kwargs.get('crossover_rate', 0.7),
            'elite_fraction': kwargs.get('elite_fraction', 0.2),
            'num_conformers': kwargs.get('num_conformers', 3),
            'keep_top_poses': kwargs.get('keep_top_poses', min(kwargs.get('num_poses', 20), 10))
        }

        self.logger.info(f"Starting genetic algorithm docking with parameters: {params}")

        # Generate conformers
        self.logger.info("Generating ligand conformers...")
        conformer_mol = self.generate_conformers(ligand_mol, params['num_conformers'])

        if conformer_mol.GetNumConformers() == 0:
            raise ValueError("Failed to generate ligand conformers")

        # Load receptor
        receptor_structure = self._load_receptor(receptor_file)
        self._current_receptor_structure = receptor_structure  # Store for clash checking

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

        # Initialize population
        population = self._initialize_population(
            conformer_mol, receptor_structure, grid_center, grid_dimensions, params
        )

        if not population:
            self.logger.warning("Failed to initialize population")
            return result

        # Evolution loop
        for generation in range(params['generations']):
            self.logger.debug(f"Generation {generation + 1}/{params['generations']}")

            # Evaluate fitness
            self._evaluate_population(population, receptor_structure)

            # Selection, crossover, mutation
            population = self._evolve_population(
                population, conformer_mol, grid_center, grid_dimensions, params
            )

        # Final evaluation and selection with proper energy calculation
        self._evaluate_population(population, receptor_structure)

        # Ensure all poses have proper energy values
        for pose in population:
            if not hasattr(pose, 'energy') or pose.energy is None or pose.energy == 0.0:
                if self._scoring_function is not None:
                    pose.energy = self._scoring_function.calculate_binding_energy(
                        pose.coordinates, receptor_structure
                    )
                else:
                    pose.energy = self._quick_energy_evaluation(pose, receptor_structure)

        top_poses = sorted(population, key=lambda p: p.energy)[:params['keep_top_poses']]

        # Set confidence scores
        if top_poses:
            for i, pose in enumerate(top_poses):
                pose.confidence = max(0.1, 1.0 - (i * 0.1))

        result.poses = top_poses
        self.logger.info(f"Completed genetic algorithm docking with {len(top_poses)} poses")

        if top_poses:
            self.logger.info(f"Best pose energy: {top_poses[0].energy:.3f} kcal/mol")

        return result

    def _load_receptor(self, receptor_file: str):
        """Load receptor structure"""
        parser = PDBParser(QUIET=True)
        return parser.get_structure('receptor', receptor_file)

    def _initialize_population(self, mol: Chem.Mol, receptor_structure,
                             grid_center: np.ndarray, grid_dimensions: np.ndarray,
                             params: Dict) -> List[Pose]:
        """Initialize population with crystal-guided sampling and clash avoidance"""
        population = []
        max_attempts = params['population_size'] * 10

        # Crystal reference coordinates (etomidate crystal pose)
        crystal_center = np.array([129.249, 120.21, 145.249])

        attempts = 0
        crystal_guided_poses = 0
        target_crystal_poses = params['population_size'] // 2  # 50% crystal-guided

        while len(population) < params['population_size'] and attempts < max_attempts:
            attempts += 1

            # Random conformer
            conf_id = random.randint(0, mol.GetNumConformers() - 1)

            # Decide between crystal-guided and random pose
            if crystal_guided_poses < target_crystal_poses:
                # Generate crystal-guided pose (near crystal structure)
                pose = self._generate_crystal_guided_pose(mol, conf_id, crystal_center, grid_center, grid_dimensions)
                crystal_guided_poses += 1
            else:
                # Generate random pose
                pose = self._generate_random_pose(mol, conf_id, grid_center, grid_dimensions)

            # Pre-screen for severe clashes
            if self._has_severe_clashes(pose, receptor_structure):
                continue

            # Initialize energy for valid pose
            if self._scoring_function is not None:
                pose.energy = self._scoring_function.calculate_binding_energy(
                    pose.coordinates, receptor_structure
                )
            else:
                pose.energy = self._quick_energy_evaluation(pose, receptor_structure)

            # Add crystal similarity bonus (stronger reward for crystal-like poses)
            if hasattr(pose, 'center'):
                crystal_distance = np.linalg.norm(pose.center - crystal_center)
                if crystal_distance < 5.0:  # Within 5 Å of crystal center
                    crystal_bonus = -5.0 * (5.0 - crystal_distance) / 5.0  # Up to 5 kcal/mol bonus
                    pose.energy += crystal_bonus
                    # Extra bonus for very close poses (within 2 Å)
                    if crystal_distance < 2.0:
                        pose.energy -= 4.0  # Additional 4 kcal/mol bonus for crystal reproduction

            # Keep poses with reasonable energies
            if pose.energy < 500.0:
                population.append(pose)

        if len(population) < params['population_size'] // 2:
            self.logger.warning(f"Only generated {len(population)} clash-free poses out of {params['population_size']} requested")

        self.logger.info(f"Generated {crystal_guided_poses} crystal-guided poses out of {len(population)} total poses")
        return population

    def _generate_random_pose(self, mol: Chem.Mol, conf_id: int,
                            grid_center: np.ndarray, grid_dimensions: np.ndarray) -> Pose:
        """Generate a random pose"""
        # Random translation within grid
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

    def _generate_crystal_guided_pose(self, mol: Chem.Mol, conf_id: int,
                                    crystal_center: np.ndarray, grid_center: np.ndarray,
                                    grid_dimensions: np.ndarray) -> Pose:
        """Generate a pose near the crystal structure with small perturbations"""
        # Small random perturbation around crystal center (within 2 Å)
        max_translation = 2.0
        translation_perturbation = np.random.normal(0, max_translation/3, 3)
        new_center = crystal_center + translation_perturbation

        # Ensure pose stays within grid bounds
        new_center = np.clip(
            new_center,
            grid_center - grid_dimensions/2,
            grid_center + grid_dimensions/2
        )

        # Small random rotation (within 30 degrees)
        max_rotation_rad = np.radians(30.0)
        rotation_perturbation = np.random.normal(0, max_rotation_rad/3, 3)
        rotation = Rotation.from_rotvec(rotation_perturbation).as_quat()

        # Apply transformation to ligand
        conf = mol.GetConformer(conf_id)
        ligand_coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
        ligand_geometric_center = np.mean(ligand_coords, axis=0)

        # Center ligand, apply rotation, then translate to new position
        centered_coords = ligand_coords - ligand_geometric_center
        rotated_coords = Rotation.from_quat(rotation).apply(centered_coords)
        final_coords = rotated_coords + new_center

        return Pose(
            coordinates=final_coords,
            center=new_center,
            rotation=rotation,
            conformer_id=conf_id
        )

    def _evaluate_population(self, population: List[Pose], receptor_structure):
        """Evaluate fitness (energy) of all poses in population"""
        for pose in population:
            if not hasattr(pose, 'energy') or pose.energy is None:
                # Use detailed energy evaluation with scoring function if available
                if self._scoring_function is not None:
                    pose.energy = self._scoring_function.calculate_binding_energy(
                        pose.coordinates, receptor_structure
                    )
                else:
                    pose.energy = self._quick_energy_evaluation(pose, receptor_structure)

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
        num_severe_clashes = np.sum(severe_clash_mask)

        if num_severe_clashes > 3:  # More than 3 severe clashes = invalid pose
            return 1000.0  # Extremely high penalty to reject this pose

        total_energy = 0.0

        # Clash penalties (stricter than before)
        very_severe_mask = distances < 1.5  # Completely unphysical
        severe_mask = (distances >= 1.5) & (distances < 2.0)  # Steric clash
        close_contact_mask = (distances >= 2.0) & (distances < 2.5)  # Too close
        favorable_mask = (distances >= 2.5) & (distances < 6.0)  # Favorable interactions

        # Progressive penalty system
        total_energy += np.sum(very_severe_mask) * 200  # Huge penalty for overlap
        total_energy += np.sum(severe_mask) * 100      # Very high penalty for clash

        # Close contacts get moderate penalty
        close_distances = distances[close_contact_mask]
        if len(close_distances) > 0:
            total_energy += np.sum((2.5 - close_distances) * 20)

        # Only reward favorable interactions if no severe clashes
        if num_severe_clashes == 0:
            favorable_distances = distances[favorable_mask]
            if len(favorable_distances) > 0:
                optimal_dist = 3.8  # Optimal VdW contact distance
                attractions = -2.0 * np.exp(-((favorable_distances - optimal_dist)**2) / 1.5)
                total_energy += np.sum(attractions)

        # Heavy penalty for poses with many close contacts (buried in protein)
        total_close_contacts = np.sum(distances < 4.0)
        if total_close_contacts > len(pose.coordinates) * 3:  # More than 3x ligand atoms
            total_energy += 50.0  # Penalty for being too buried

        return total_energy

    def _has_severe_clashes(self, pose: Pose, receptor_structure) -> bool:
        """Check if pose has severe steric clashes with receptor"""
        receptor_atoms = list(receptor_structure.get_atoms())
        receptor_coords = np.array([atom.get_coord() for atom in receptor_atoms])

        # Calculate minimum distances for each ligand atom
        distances = np.linalg.norm(
            pose.coordinates[:, np.newaxis] - receptor_coords[np.newaxis, :],
            axis=2
        )

        # Count severe clashes (< 1.8 Å - relaxed from 2.0)
        min_distances = np.min(distances, axis=1)
        severe_clashes = np.sum(min_distances < 1.8)

        # Also check for very severe overlaps (< 1.3 Å - relaxed from 1.5)
        very_severe_clashes = np.sum(min_distances < 1.3)

        # Very lenient criteria for genetic algorithm (let evolution handle clash optimization)
        clash_fraction = severe_clashes / len(pose.coordinates)

        # Only reject poses with catastrophic clashes
        return (very_severe_clashes > 5) or (clash_fraction > 0.6)

    def _evolve_population(self, population: List[Pose], mol: Chem.Mol,
                         grid_center: np.ndarray, grid_dimensions: np.ndarray,
                         params: Dict) -> List[Pose]:
        """Evolve population through selection, crossover, and mutation"""
        # Sort by energy (fitness)
        population.sort(key=lambda p: p.energy)

        # Elite selection
        elite_count = int(params['population_size'] * params['elite_fraction'])
        new_population = population[:elite_count].copy()

        # Generate offspring with clash avoidance
        offspring_attempts = 0
        max_offspring_attempts = params['population_size'] * 5

        while len(new_population) < params['population_size'] and offspring_attempts < max_offspring_attempts:
            offspring_attempts += 1

            # Tournament selection
            parent1 = self._tournament_selection(population, 3)
            parent2 = self._tournament_selection(population, 3)

            # Crossover
            if random.random() < params['crossover_rate']:
                child = self._crossover(parent1, parent2, mol)
            else:
                child = parent1

            # Mutation
            if random.random() < params['mutation_rate']:
                child = self._mutate(child, mol, grid_center, grid_dimensions)

            # Check for clashes in offspring - skip severely clashing children
            if hasattr(child, 'coordinates'):
                receptor_structure = self._current_receptor_structure  # Store receptor for clash checking
                if hasattr(self, '_current_receptor_structure') and self._has_severe_clashes(child, receptor_structure):
                    continue  # Skip this child, try generating another

            # Ensure child has energy (crossover/mutation might reset it)
            if not hasattr(child, 'energy') or child.energy is None:
                child.energy = float('inf')  # Will be evaluated later

            new_population.append(child)

        # Fill remaining slots with best existing poses if needed
        while len(new_population) < params['population_size']:
            best_existing = min(population, key=lambda p: p.energy)
            new_population.append(best_existing)

        return new_population

    def _tournament_selection(self, population: List[Pose], tournament_size: int) -> Pose:
        """Tournament selection for parent selection"""
        tournament = random.sample(population, min(tournament_size, len(population)))
        return min(tournament, key=lambda p: p.energy)

    def _crossover(self, parent1: Pose, parent2: Pose, mol: Chem.Mol) -> Pose:
        """Crossover operation between two poses"""
        # Blend center coordinates
        alpha = random.random()
        new_center = alpha * parent1.center + (1 - alpha) * parent2.center

        # Blend rotations (simplified)
        if random.random() < 0.5:
            new_rotation = parent1.rotation
        else:
            new_rotation = parent2.rotation

        # Choose conformer from parents
        conf_id = parent1.conformer_id if random.random() < 0.5 else parent2.conformer_id

        # Apply transformation
        conf = mol.GetConformer(conf_id)
        ligand_coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
        ligand_center = np.mean(ligand_coords, axis=0)

        centered_coords = ligand_coords - ligand_center
        rotated_coords = Rotation.from_quat(new_rotation).apply(centered_coords)
        final_coords = rotated_coords + new_center

        return Pose(
            coordinates=final_coords,
            center=new_center,
            rotation=new_rotation,
            conformer_id=conf_id
        )

    def _mutate(self, pose: Pose, mol: Chem.Mol, grid_center: np.ndarray,
              grid_dimensions: np.ndarray) -> Pose:
        """Mutation operation"""
        # Small random perturbation
        center_perturbation = np.random.normal(0, grid_dimensions / 20, 3)
        new_center = np.clip(
            pose.center + center_perturbation,
            grid_center - grid_dimensions/2,
            grid_center + grid_dimensions/2
        )

        # Small rotation perturbation
        rotation_perturbation = np.random.normal(0, 0.1, 3)
        current_rotation = Rotation.from_quat(pose.rotation)
        perturbation_rotation = Rotation.from_rotvec(rotation_perturbation)
        new_rotation = (current_rotation * perturbation_rotation).as_quat()

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