"""
Parallel search algorithms for PandaDock.
This module provides parallel implementations of search algorithms for molecular docking
that leverage multi-core CPUs for improved performance.
"""

import numpy as np
import copy
import random
import time
import multiprocessing as mp
from pathlib import Path
from scipy.spatial.transform import Rotation, Slerp

from .search import GeneticAlgorithm, RandomSearch
from .utils import is_within_grid, detect_steric_clash

class ParallelGeneticAlgorithm(GeneticAlgorithm):
    def __init__(self, scoring_function, max_iterations=100, population_size=50, 
                 mutation_rate=0.2, crossover_rate=0.8, tournament_size=3, 
                 n_processes=None, batch_size=None, process_pool=None, 
                 output_dir=None, perform_local_opt=False, grid_spacing=0.375, grid_radius=10.0, grid_center=None):
        super().__init__(scoring_function, max_iterations, population_size, mutation_rate)
        self.scoring_function = scoring_function  # Ensure this is set
        self.output_dir = output_dir
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.perform_local_opt = perform_local_opt
        self.grid_spacing = grid_spacing  # Add grid_spacing as an attribute
        self.grid_radius = grid_radius  # Add grid_radius as an attribute
        self.grid_center = np.array(grid_center) if grid_center is not None else np.array([0.0, 0.0, 0.0])  # Default grid center

        if n_processes is None:
            self.n_processes = mp.cpu_count()
        else:
            self.n_processes = n_processes

        if batch_size is None:
            self.batch_size = max(1, self.population_size // (self.n_processes * 2))
        else:
            self.batch_size = batch_size

        self.process_pool = process_pool
        self.own_pool = False

        self.eval_time = 0.0
        self.total_time = 0.0
        self.best_score = float('inf')
        self.best_pose = None

        self.grid_center = np.array([0.0, 0.0, 0.0])  # Default grid center
    


    def initialize_population(self, protein, ligand):
        """
        Initialize random population for genetic algorithm within spherical grid.

        Parameters:
        -----------
        protein : Protein
            Protein object
        ligand : Ligand object

        Returns:
        --------
        list
            List of (pose, score) tuples
        """
        population = []

        # Determine search space
        if protein.active_site:
            center = protein.active_site['center']
            radius = protein.active_site['radius']
        else:
            center = np.mean(protein.xyz, axis=0)
            radius = 15.0  # Arbitrary default

        self.initialize_grid_points(center)

        print(f"Using {self.n_processes} CPU cores for evaluation")
        print(f"Using {self.batch_size} poses per process for evaluation")
        print(f"Using {self.population_size} poses in total")
        print(f"Using {self.mutation_rate} mutation rate")
        print(f"Using {self.crossover_rate} crossover rate")
        print(f"Using {self.tournament_size} tournament size")
        print(f"Performing local optimization: {self.perform_local_opt}")
        print(f"Grid spacing: {self.grid_spacing}")
        print(f"Grid radius: {self.grid_radius}")

        for _ in range(self.population_size):
            pose = copy.deepcopy(ligand)

            # 🟡 Select a random point from precomputed spherical grid
            random_grid_point = random.choice(self.grid_points)

            # 🟡 Move the ligand centroid to that random point
            centroid = np.mean(pose.xyz, axis=0)
            translation = random_grid_point - centroid
            pose.translate(translation)

            # 🟡 Apply random rotation around the ligand center
            # Normal random rotation
            rotation = Rotation.random()

            # Add bias: rotate toward center of pocket
            centroid = np.mean(pose.xyz, axis=0)
            vector_to_center = center - centroid
            vector_to_center /= np.linalg.norm(vector_to_center)

            # Small rotation (~10 degrees) toward pocket center
            bias_rotation = Rotation.from_rotvec(0.2 * vector_to_center)  # 0.2 rad ≈ 11 degrees
            biased_rotation = rotation * bias_rotation

            rotation_matrix = biased_rotation.as_matrix()

            # Apply
            pose.translate(-centroid)
            pose.rotate(rotation_matrix)
            pose.translate(centroid)


            population.append((pose, None))

        return population

    
    def initialize_grid_points(self, center):
        from .utils import generate_spherical_grid
        if self.grid_points is None:
            self.grid_points = generate_spherical_grid(
                center=center,
                radius=self.grid_radius,
                spacing=self.grid_spacing
            )
            self.logger.info(
                f"Initialized spherical grid with {len(self.grid_points)} points "
                f"(spacing: {self.grid_spacing}, radius: {self.grid_radius})"
            )
            # Save Light Sphere PDB
            subsample_rate = 20  # <-- Only keep 1 out of every 5 points

            if self.output_dir is not None:
                sphere_path = Path(self.output_dir) / "sphere.pdb"
                sphere_path.parent.mkdir(parents=True, exist_ok=True)
                with open(sphere_path, 'w') as f:
                    for idx, point in enumerate(self.grid_points):
                        if idx % subsample_rate == 0:  # <-- Only write every Nth point
                            f.write(
                                f"HETATM{idx+1:5d} {'S':<2s}   SPH A   1    "
                                f"{point[0]:8.3f}{point[1]:8.3f}{point[2]:8.3f}  1.00  0.00          S\n"
                            )
                self.logger.info(f"Sphere grid written to {sphere_path} (subsampled every {subsample_rate} points)")
    
    def _adjust_search_radius(self, initial_radius, generation, total_generations):
        """
        Shrink the search radius over generations (parallel version).
        """
        decay_rate = 0.5  # You can tune it (0.3 or 0.7)
        factor = 1.0 - (generation / total_generations) * decay_rate
        return max(initial_radius * factor, initial_radius * 0.5)

    def _generate_orientations(self, ligand, protein):
        orientations = []
        
        # Get active site center and radius
        if protein.active_site:
            center = protein.active_site['center']
            radius = protein.active_site['radius']
        else:
            center = np.mean(protein.xyz, axis=0)
            radius = 15.0

        bad_orientations = 0
        max_bad_orientations = self.num_orientations * 10

        while len(orientations) < self.num_orientations and bad_orientations < max_bad_orientations:
            pose = copy.deepcopy(ligand)
            pose.random_rotate()

            displacement = np.random.normal(0, 1, size=3)
            pose.translate(center + displacement)

            # Check if pose is inside the pocket
            centroid = np.mean(pose.xyz, axis=0)
            distance = np.linalg.norm(centroid - center)

            if distance <= radius:
                # 🧪 Add clash filter
                is_valid = self._check_pose_validity(pose, protein)
                if is_valid:
                    orientations.append(pose)
                else:
                    bad_orientations += 1
            else:
                bad_orientations += 1

        return orientations

    def _check_pose_validity(self, pose, protein):
        """
        Check whether a pose has acceptable steric clashes or energy.
        """
        if hasattr(self.scoring_function, '_calculate_clashes'):
            clash_score = self.scoring_function._calculate_clashes(protein, pose)
            return clash_score < 10.0  # or make this configurable
        else:
            score = self.scoring_function.score(protein, pose)
            return score < 100.0


    def search(self, protein, ligand):
        start_time = time.time()
        
        if protein.active_site:
            center = protein.active_site['center']
            radius = protein.active_site['radius']
        else:
            center = np.mean(protein.xyz, axis=0)
            radius = 10.0
        self.initialize_grid_points(center)

        
        # ✨ Insert here
        if not hasattr(protein, 'active_site') or protein.active_site is None:
            protein.active_site = {
                'center': center,
                'radius': radius
            }
        if 'atoms' not in protein.active_site or protein.active_site['atoms'] is None:
            protein.active_site['atoms'] = [
                atom for atom in protein.atoms
                if np.linalg.norm(atom['coords'] - center) <= radius
            ]
            print(f"[INFO] Added {len(protein.active_site['atoms'])} atoms into active_site region")

        print(f"Searching around center {center} with radius {radius}")
        
        # ✅ Save sphere.pdb
        self.initialize_grid_points(center)
    
    # Then continue normal pose generation...

        
        # Initialize population
        population = self.initialize_population(protein, ligand)
        
        # Evaluate initial population
        evaluated_population = self._evaluate_population(protein, population)
        
        # Sort population by score
        evaluated_population.sort(key=lambda x: x[1])
        
        # Store best individual
        best_individual = evaluated_population[0]
        self.best_pose = best_individual[0]
        self.best_score = best_individual[1]
        
        print(f"Generation 0: Best score = {self.best_score:.4f}")
        
        # Track all individuals if population is diverse
        all_individuals = [evaluated_population[0]]
        
        # Main evolutionary loop
        for generation in range(self.max_iterations):
            current_radius = self._adjust_search_radius(radius, generation, self.max_iterations)

            gen_start = time.time()
            
            # Select parents
            parents = self._selection(evaluated_population)
            
            # Create offspring through crossover and mutation
            offspring = []
            
            # Apply genetic operators
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    parent1 = parents[i][0]
                    parent2 = parents[i+1][0]
                    
                    # Crossover with probability
                    if random.random() < self.crossover_rate:
                        child1, child2 = self._crossover_pair(parent1, parent2)
                    else:
                        child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
                    
                    # Mutation
                    self._mutate(child1, copy.deepcopy(parent1), center, current_radius)
                    self._mutate(child2, copy.deepcopy(parent2), center, current_radius)

                    
                    offspring.append((child1, None))
                    offspring.append((child2, None))
            
            # Evaluate offspring
            eval_start = time.time()
            evaluated_offspring = self._evaluate_population(protein, offspring)
            self.eval_time += time.time() - eval_start
            
            # Combine parent and offspring populations (μ + λ)
            combined = evaluated_population + evaluated_offspring
            
            # Keep only the best individuals (elitism)
            combined.sort(key=lambda x: x[1])
            evaluated_population = combined[:self.population_size]
            
            # Update best solution
            if evaluated_population[0][1] < self.best_score:
                self.best_pose = evaluated_population[0][0]
                self.best_score = evaluated_population[0][1]
                all_individuals.append(evaluated_population[0])
            
            # Display progress
            gen_time = time.time() - gen_start
            print(f"Generation {generation + 1}/{self.max_iterations}: "
                  f"Best score = {self.best_score:.4f}, "
                  f"Current best = {evaluated_population[0][1]:.4f}, "
                  f"Time = {gen_time:.2f}s")
            
            # Apply local search to the best individual occasionally
            if hasattr(self, '_local_optimization') and generation % 5 == 0:
                #print("Applying local optimization to best individual...")
                best_pose, best_score = self._local_optimization(evaluated_population[0][0], protein)
                
                if best_score < self.best_score:
                    self.best_pose = best_pose
                    self.best_score = best_score
                    #print(f"Improved score after optimization: {self.best_score:.4f}")
                    
                    # Replace best individual in population
                    evaluated_population[0] = (best_pose, best_score)
                    evaluated_population.sort(key=lambda x: x[1])
                    all_individuals.append((best_pose, best_score))
        
        # Return unique solutions, best first
        self.total_time = time.time() - start_time
        print(f"\nSearch completed in {self.total_time:.2f} seconds")
        print(f"Evaluation time: {self.eval_time:.2f} seconds ({self.eval_time/self.total_time*100:.1f}%)")
        
        # Sort all_individuals by score and ensure uniqueness
        all_individuals.sort(key=lambda x: x[1])
        
        # Return results
        return all_individuals

    def _evaluate_population(self, protein, population):
        """
        Evaluate population sequentially to avoid multiprocessing issues.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        population : list
            List of (pose, score) tuples
        
        Returns:
        --------
        list
            Evaluated population as (pose, score) tuples
        """
        results = []
        
        # Process all poses
        for i, (pose, _) in enumerate(population):
            # Show progress for large populations
            if i % 10 == 0 and i > 0 and len(population) > 50:
                print(f"  Evaluating pose {i}/{len(population)}...")
                
            score = self.scoring_function.score(protein, pose)
            results.append((copy.deepcopy(pose), score))

            if detect_steric_clash(protein.atoms, pose.atoms):
                score = float('inf')  # or apply a large clash penalty
            else:
                score = self.scoring_function.score(protein, pose)
                
        return results
    
    def _selection(self, population):
        """
        Tournament selection of parents.
        
        Parameters:
        -----------
        population : list
            List of (pose, score) tuples
        
        Returns:
        --------
        list
            Selected parents as (pose, score) tuples
        """
        selected = []
        
        for _ in range(self.population_size):
            # Select random individuals for tournament
            tournament = random.sample(population, min(self.tournament_size, len(population)))
            
            # Select the best from tournament
            tournament.sort(key=lambda x: x[1])
            selected.append(tournament[0])
        
        return selected
    
    def _crossover_pair(self, parent1, parent2):
        """
        Perform crossover between two parents using a more sophisticated approach.
        
        Parameters:
        -----------
        parent1 : Ligand
            First parent
        parent2 : Ligand
            Second parent
        
        Returns:
        --------
        tuple
            (child1, child2) as Ligand objects
        """
        # Create deep copies to avoid modifying parents
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        
        # Calculate centroids
        centroid1 = np.mean(parent1.xyz, axis=0)
        centroid2 = np.mean(parent2.xyz, axis=0)
        
        # Weighted centroid crossover
        alpha = random.uniform(0.3, 0.7)  # Random weight for variability
        new_centroid1 = alpha * centroid1 + (1 - alpha) * centroid2
        new_centroid2 = (1 - alpha) * centroid1 + alpha * centroid2
        
        # Apply translation to children
        child1.translate(new_centroid1 - centroid1)
        child2.translate(new_centroid2 - centroid2)
        
        # Fragment-based crossover
        fragment_indices = random.sample(range(len(parent1.xyz)), len(parent1.xyz) // 2)
        for idx in fragment_indices:
            child1.xyz[idx], child2.xyz[idx] = child2.xyz[idx], child1.xyz[idx]
        
        # Rotation interpolation
        rotation1 = Rotation.random()
        rotation2 = Rotation.random()
        key_times = [0, 1]
        rotations = Rotation.concatenate([rotation1, rotation2])
        slerp = Slerp(key_times, rotations)
        interpolated_rotation = slerp([alpha])[0]  # Interpolate at alpha
        
        # Apply interpolated rotation to children
        centroid1 = np.mean(child1.xyz, axis=0)
        centroid2 = np.mean(child2.xyz, axis=0)
        
        child1.translate(-centroid1)
        child1.rotate(interpolated_rotation.as_matrix())
        child1.translate(centroid1)
        
        child2.translate(-centroid2)
        child2.rotate(interpolated_rotation.as_matrix())
        child2.translate(centroid2)
        
        # Validate children
        if not self._validate_conformation(child1):
            #print("Child1 failed validation. Attempting repair...")
            child1 = self._repair_conformation(child1)
        
        if not self._validate_conformation(child2):
            #print("Child2 failed validation. Attempting repair...")
            child2 = self._repair_conformation(child2)
        
        return child1, child2

    def _validate_conformation(self, ligand):
        """
        Validate a ligand conformation to ensure no overlapping atoms or invalid bond lengths.
        
        Parameters:
        -----------
        ligand : Ligand
            Ligand to validate
        
        Returns:
        --------
        bool
            True if the conformation is valid, False otherwise
        """
        # Check for overlapping atoms
        for i in range(len(ligand.xyz)):
            for j in range(i + 1, len(ligand.xyz)):
                distance = np.linalg.norm(ligand.xyz[i] - ligand.xyz[j])
                if distance < 1.2:  # Adjusted threshold for atom overlap (in Å)
                    #print(f"Validation failed: Overlapping atoms at indices {i} and {j} (distance: {distance:.2f} Å)")
                    return False
        
        # Check for valid bond lengths
        for bond in ligand.bonds:
            atom1 = ligand.xyz[bond['begin_atom_idx']]
            atom2 = ligand.xyz[bond['end_atom_idx']]
            bond_length = np.linalg.norm(atom1 - atom2)
            if bond_length < 0.9 or bond_length > 2.0:  # Adjusted bond length range (in Å)
                #print(f"Validation failed: Invalid bond length ({bond_length:.2f} Å) between atoms {bond['begin_atom_idx']} and {bond['end_atom_idx']}")
                return False
        
        return True

    def _repair_conformation(self, ligand, max_attempts=5):
        """
        Attempt to repair an invalid ligand conformation.

        Parameters:
        -----------
        ligand : Ligand
            Ligand to repair
        max_attempts : int
            Maximum number of repair attempts

        Returns:
        --------
        Ligand
            Repaired ligand or a new random pose if repair fails
        """
        #print("Attempting to repair ligand conformation...")
        
        for attempt in range(max_attempts):
            #print(f"Repair attempt {attempt + 1}/{max_attempts}...")
            # Apply small random perturbations to atom positions
            perturbation = np.random.normal(0, 0.2, ligand.xyz.shape)  # 0.2 Å standard deviation
            ligand.xyz += perturbation
            
            # Revalidate after perturbation
            if self._validate_conformation(ligand):
                #print("Repair successful after random perturbation.")
                return ligand
            
            # Attempt to resolve steric clashes by energy minimization
            try:
                #print("Applying energy minimization to repair ligand...")
                ligand = self._minimize_energy(ligand, max_iterations=200)
                if self._validate_conformation(ligand):
                    #print("Repair successful after energy minimization.")
                    return ligand
            except Exception as e:
                print(f"Energy minimization failed: {e}")
        
        # If repair fails, generate a new random pose
        print("Repair failed after maximum attempts. Generating a new random pose...")
        return self._generate_random_pose(ligand, np.mean(ligand.xyz, axis=0), 15.0)  # Example radius
    
    def _minimize_energy(self, ligand, max_iterations=100):
        """
        Perform energy minimization to resolve steric clashes and optimize ligand geometry.

        Parameters:
        -----------
        ligand : Ligand
            Ligand to minimize
        max_iterations : int
            Maximum number of optimization iterations

        Returns:
        --------
        Ligand
            Minimized ligand
        """
        from scipy.optimize import minimize

        def energy_function(coords):
            # Example energy function: penalize overlapping atoms and bond length deviations
            coords = coords.reshape(ligand.xyz.shape)
            energy = 0.0
            
            # Penalize overlapping atoms
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    distance = np.linalg.norm(coords[i] - coords[j])
                    if distance < 1.2:  # Overlap threshold
                        energy += (1.2 - distance) ** 2
            
            # Penalize invalid bond lengths
            for bond in ligand.bonds:
                atom1 = coords[bond['begin_atom_idx']]
                atom2 = coords[bond['end_atom_idx']]
                bond_length = np.linalg.norm(atom1 - atom2)
                if bond_length < 0.9:
                    energy += (0.9 - bond_length) ** 2
                elif bond_length > 2.0:
                    energy += (bond_length - 2.0) ** 2
            
            return energy

        # Flatten coordinates for optimization
        initial_coords = ligand.xyz.flatten()
        result = minimize(energy_function, initial_coords, method='L-BFGS-B', options={'maxiter': max_iterations})
        
        # Update ligand coordinates with minimized values
        ligand.xyz = result.x.reshape(ligand.xyz.shape)
        return ligand
    

    def _generate_random_pose(self, ligand, center, radius):
        while True:
            r = radius * random.random() ** (1.0 / 3.0)
            theta = random.uniform(0, 2 * np.pi)
            phi = random.uniform(0, np.pi)

            x = center[0] + r * np.sin(phi) * np.cos(theta)
            y = center[1] + r * np.sin(phi) * np.sin(theta)
            z = center[2] + r * np.cos(phi)

            pose = copy.deepcopy(ligand)
            centroid = np.mean(pose.xyz, axis=0)
            translation = np.array([x, y, z]) - centroid
            pose.translate(translation)

            if is_within_grid(pose, center, radius):
                return pose

    

    # Apply translation and rotation as usual
    ...
    
    ##############
    # Mutation
    ##############

    def _mutate(self, individual, original_individual, center, radius):
        """
        Mutate an individual with probability mutation_rate and respect current radius.
        """

        if random.random() >= self.mutation_rate:
            return  # No mutation

        # Perform either translation, rotation, or both
        mutation_type = random.choice(['translation', 'rotation', 'both'])

        if mutation_type in ['translation', 'both']:
            translation = np.random.normal(0, 2.0, 3)  # 2.0 Å standard deviation
            individual.translate(translation)

        if mutation_type in ['rotation', 'both']:
            angle = np.random.normal(0, 0.5)  # ~30 degrees std dev
            axis = np.random.randn(3)
            axis = axis / np.linalg.norm(axis)

            rotation = Rotation.from_rotvec(angle * axis)
            centroid = np.mean(individual.xyz, axis=0)
            individual.translate(-centroid)
            individual.rotate(rotation.as_matrix())
            individual.translate(centroid)

        # ✅ NEW: Check inside the *current shrinking radius*
        if not is_within_grid(individual, center, radius):
            # If out of bounds, revert to original
            individual.xyz = original_individual.xyz.copy()

        return individual


    def _local_optimization(self, pose, protein):
        """
        Perform local optimization of pose using gradient descent with clash detection.
        
        Parameters:
        -----------
        pose : Ligand
            Ligand pose to optimize
        protein : Protein
            Protein target
        
        Returns:
        --------
        tuple
            (optimized_pose, optimized_score)
        """
        return super()._local_optimization(pose, protein)   
    
class ParallelRandomSearch(RandomSearch):
    """
    Parallel implementation of random search for molecular docking.
    """
    
    def __init__(self, scoring_function, max_iterations=100, n_processes=None, 
                 batch_size=None, process_pool=None, output_dir=None):
        super().__init__(scoring_function, max_iterations)
        self.output_dir = output_dir

        if n_processes is None:
            self.n_processes = mp.cpu_count()
        else:
            self.n_processes = n_processes

        if batch_size is None:
            self.batch_size = max(10, self.max_iterations // (self.n_processes * 5))
        else:
            self.batch_size = batch_size

        self.process_pool = process_pool
        self.own_pool = False

        self.eval_time = 0.0
        self.total_time = 0.0
    
    def _adjust_search_radius(self, initial_radius, iteration, total_iterations):
        """Shrink search radius over iterations."""
        decay_rate = 0.5
        factor = 1.0 - (iteration / total_iterations) * decay_rate
        return max(initial_radius * factor, initial_radius * 0.5)

    def search(self, protein, ligand):
        start_time = time.time()

        if protein.active_site:
            center = protein.active_site['center']
            radius = protein.active_site['radius']
        else:
            center = np.mean(protein.xyz, axis=0)
            radius = 10.0
        self.initialize_grid_points(center)
        
        # ✨ Insert here
        if not hasattr(protein, 'active_site') or protein.active_site is None:
            protein.active_site = {
                'center': center,
                'radius': radius
            }
        if 'atoms' not in protein.active_site or protein.active_site['atoms'] is None:
            protein.active_site['atoms'] = [
                atom for atom in protein.atoms
                if np.linalg.norm(atom['coords'] - center) <= radius
            ]
            print(f"[INFO] Added {len(protein.active_site['atoms'])} atoms into active_site region")

        print(f"Searching around center {center} with radius {radius}")
        print(f"Using {self.n_processes} CPU cores for evaluation")

        # Save sphere grid
        self.initialize_grid_points(center)

        results = []
        # New variables to track clash failures
        fail_counter = 0
        max_failures = 30  # After 30 consecutive fails, expand radius
        radius_increment = 1.0  # How much to expand each time
        
        for i in range(self.max_iterations):
            if i % 25 == 0 and i > 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                remaining = avg_time * (self.max_iterations - i)
                print(f"Progress: {i}/{self.max_iterations} poses evaluated ({i/self.max_iterations*100:.1f}%) - "
                      f"Est. remaining: {remaining:.1f}s")

            # Adjust radius dynamically
            current_radius = self._adjust_search_radius(radius, i, self.max_iterations)

            pose = self._generate_random_pose(ligand, center, current_radius)

            # Clash prefilter
            # Quick clash check before expensive scoring
            from .utils import detect_steric_clash
            if detect_steric_clash(protein.atoms, pose.atoms):
                fail_counter += 1
                if fail_counter >= max_failures:
                    radius += radius_increment
                    fail_counter = 0
                    print(f"⚡ Auto-expanding search radius to {radius:.2f} Å due to repeated clashes!")
                continue  # Skip pose, try again
            else:
                fail_counter = 0  # Reset fail counter on success

            # Score the good pose
            score = self.scoring_function.score(protein, pose)
            results.append((pose, score))
            # Check if the score is valid
            if not self._check_pose_validity(pose, protein):
                fail_counter += 1
                if fail_counter >= max_failures:
                    radius += radius_increment
                    fail_counter = 0
                    print(f"⚡ Auto-expanding search radius to {radius:.2f} Å due to repeated clashes!")
                continue  # Skip pose, try again

            # Evaluate the pose
            # Score
            score = self.scoring_function.score(protein, pose)

            results.append((pose, score))
        
        results.sort(key=lambda x: x[1])

        self.total_time = time.time() - start_time

        if not results:
            print("⚠️ No valid poses generated! All poses clashed or failed. Returning empty result.")
            return []

        # Otherwise, continue
        results.sort(key=lambda x: x[1])
        print(f"Search completed in {self.total_time:.2f} seconds")
        print(f"Best score: {results[0][1]:.4f}")

        return results

    def _generate_orientations(self, ligand, protein):
        orientations = []
        
        # Get active site center and radius
        if protein.active_site:
            center = protein.active_site['center']
            radius = protein.active_site['radius']
        else:
            center = np.mean(protein.xyz, axis=0)
            radius = 15.0

        bad_orientations = 0
        max_bad_orientations = self.num_orientations * 10

        while len(orientations) < self.num_orientations and bad_orientations < max_bad_orientations:
            pose = copy.deepcopy(ligand)
            pose.random_rotate()

            displacement = np.random.normal(0, 1, size=3)
            pose.translate(center + displacement)

            # Check if pose is inside the pocket
            centroid = np.mean(pose.xyz, axis=0)
            distance = np.linalg.norm(centroid - center)

            if distance <= radius:
                # 🧪 Add clash filter
                is_valid = self._check_pose_validity(pose, protein)
                if is_valid:
                    orientations.append(pose)
                else:
                    bad_orientations += 1
            else:
                bad_orientations += 1

        return orientations

    def _check_pose_validity(self, pose, protein):
        """
        Check whether a pose has acceptable steric clashes or energy.
        """
        if hasattr(self.scoring_function, '_calculate_clashes'):
            clash_score = self.scoring_function._calculate_clashes(protein, pose)
            return clash_score < 10.0
        else:
            score = self.scoring_function.score(protein, pose)
            return score < 100.0



    def _generate_random_pose(self, ligand, center, radius):
        # More uniform sampling within sphere volume
        r = radius * random.betavariate(2, 5) ** (1/3)
        theta = random.uniform(0, 2 * np.pi)
        phi = random.uniform(0, np.pi)

        x = center[0] + r * np.sin(phi) * np.cos(theta)
        y = center[1] + r * np.sin(phi) * np.sin(theta)
        z = center[2] + r * np.cos(phi)

        pose = copy.deepcopy(ligand)
        centroid = np.mean(pose.xyz, axis=0)
        translation = np.array([x, y, z]) - centroid
        pose.translate(translation)

        # Apply random rotation
        rotation = Rotation.random()
        rotation_matrix = rotation.as_matrix()
        centroid = np.mean(pose.xyz, axis=0)
        pose.translate(-centroid)
        pose.rotate(rotation_matrix)
        pose.translate(centroid)

        return pose
