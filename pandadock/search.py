# -*- coding: utf-8 -*-
# search.py
"""
Search algorithm implementations for PandaDock molecular docking.

This module provides base and concrete implementations of search algorithms
for exploring the conformational space of ligands during molecular docking.
It includes:
1. Base DockingSearch class defining the common interface
2. RandomSearch for simple stochastic sampling
3. GeneticAlgorithm for evolutionary optimization

For multi-core parallel implementations, see parallel_search.py.
"""

import numpy as np
from scipy.spatial.transform import Rotation
import pathlib as Path
import random
import copy
import time
import os

# Import necessary utility functions
from .utils import setup_logging, generate_spherical_grid, is_inside_sphere, random_point_in_sphere
from .utils import save_intermediate_result, update_status, detect_steric_clash

class DockingSearch:
    """
    Base class for all docking search algorithms.
    
    This class defines the common interface and shared functionality
    for all molecular docking search algorithms in PandaDock.
    Concrete search algorithms should inherit from this class and
    implement the search() method.
    """
    
    def __init__(self, scoring_function, max_iterations=100, output_dir=None, grid_spacing=0.375, grid_radius=10.0, grid_center=None):
        """
        Initialize search algorithm.
        
        Parameters:
        -----------
        scoring_function : ScoringFunction
            Scoring function to evaluate poses
        max_iterations : int
            Maximum number of iterations
        grid_spacing : float
            Spacing between grid points
        grid_radius : float
            Radius of the search sphere
        """
      
    
        self.scoring_function = scoring_function
        self.max_iterations = max_iterations
        self.output_dir = Path(output_dir) if output_dir else None
        
        # Grid parameters
        self.grid_spacing = grid_spacing
        self.grid_radius = grid_radius
        self.grid_center = grid_center
        self.grid_points = None

        # Set up logging if output_dir is provided
        self.logger = setup_logging(self.output_dir) if self.output_dir else None 
    

    # Define a method to set up the grid
    def initialize_grid_points(self, center, protein=None):
        """
        Generate a grid of points within a sphere for search space sampling.
        
        Parameters:
        -----------
        center : array-like
            Center coordinates of the search sphere
        protein : Protein, optional
            Protein object for context-aware grid generation
            
        Notes:
        ------
        This method populates self.grid_points with coordinates within
        a sphere centered at 'center' with radius self.grid_radius
        and spacing self.grid_spacing.
        """
        # Return if grid already initialized
        if self.grid_points is not None:
            return
            
        # Generate spherical grid
        self.grid_points = generate_spherical_grid(
            center=center,
            radius=self.grid_radius,
            spacing=self.grid_spacing
        )
        
        if self.logger:
            self.logger.info(
                f"Initialized spherical grid with {len(self.grid_points)} points "
                f"(spacing: {self.grid_spacing}, radius: {self.grid_radius})"
            )
            
        # Save grid visualization if output directory exists
        if self.output_dir is not None:
            subsample_rate = 20  # Only save a subset for visualization
            sphere_path = self.output_dir / "sphere.pdb"
            sphere_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(sphere_path, 'w') as f:
                for idx, point in enumerate(self.grid_points):
                    if idx % subsample_rate == 0:
                        f.write(
                            f"HETATM{idx+1:5d} {'S':<2s}   SPH A   1    "
                            f"{point[0]:8.3f}{point[1]:8.3f}{point[2]:8.3f}  1.00  0.00          S\n"
                        )
            
            if self.logger:
                self.logger.info(f"Sphere grid written to {sphere_path} (subsampled every {subsample_rate} points)")   
    
    def search(self, protein, ligand):
        """
        Perform docking search to identify optimal ligand poses.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        ligand : Ligand
            Ligand object
        
        Returns:
        --------
        list
            List of (pose, score) tuples, sorted by score (best first)
            
        Notes:
        ------
        This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def _adjust_search_radius(self, initial_radius, generation, total_generations):
        """
        Adaptively shrink the search radius as the search progresses.
        
        Parameters:
        -----------
        initial_radius : float
            Starting radius for the search
        iteration : int
            Current iteration number
        total_iterations : int
            Total number of iterations
            
        Returns:
        --------
        float
            Adjusted radius for the current iteration
            
        Notes:
        ------
        This implements a linear decay strategy to focus the search
        on promising regions of the conformational space.
        """
        decay_rate = 0.5  # How much radius should shrink overall (50% smaller at end)
        factor = 1.0 - (generation / total_generations) * decay_rate
        return max(initial_radius * factor, initial_radius * 0.5)  # Do not shrink below 50%

    def _check_pose_validity(self, ligand, protein, clash_threshold=1.5):
        """
        Check if ligand pose clashes with protein atoms.
        
        Parameters:
            ligand: Ligand object with .atoms
            protein: Protein object with .atoms or active_site['atoms']
            clash_threshold: Ångström cutoff for hard clash
            
        Returns:
            bool: True if pose is valid (no severe clash), False otherwise
        """
        ligand_coords = np.array([atom['coords'] for atom in ligand.atoms])
        
        # Use active site atoms if defined
        if hasattr(protein, 'active_site') and protein.active_site and 'atoms' in protein.active_site:
            protein_coords = np.array([atom['coords'] for atom in protein.active_site['atoms']])
        else:
            protein_coords = np.array([atom['coords'] for atom in protein.atoms])
        
        for lig_coord in ligand_coords:
            distances = np.linalg.norm(protein_coords - lig_coord, axis=1)
            if np.any(distances < clash_threshold):
                return False  # Clash detected
        
        return True
    def _check_pose_clash(self, ligand, protein, clash_threshold=1.5):
        """
        Check if ligand pose clashes with protein atoms.
        
        Parameters:
            ligand: Ligand object with .atoms
            protein: Protein object with .atoms or active_site['atoms']
            clash_threshold: Ångström cutoff for hard clash
            
        Returns:
            bool: True if pose is valid (no severe clash), False otherwise
        """
        ligand_coords = np.array([atom['coords'] for atom in ligand.atoms])
        
        # Use active site atoms if defined
        if hasattr(protein, 'active_site') and protein.active_site and 'atoms' in protein.active_site:
            protein_coords = np.array([atom['coords'] for atom in protein.active_site['atoms']])
        else:
            protein_coords = np.array([atom['coords'] for atom in protein.atoms])
        
        for lig_coord in ligand_coords:
            distances = np.linalg.norm(protein_coords - lig_coord, axis=1)
            if np.any(distances < clash_threshold):
                return True  # Clash detected
        
        return False
    
    def _calculate_clash_score(self, protein, ligand):
        """
        Calculate a score specifically for steric clashes between protein and ligand.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        ligand : Ligand
            Ligand pose
        
        Returns:
        --------
        float
            Clash score (lower is better)
        """
        # Get protein atoms
        if protein.active_site and 'atoms' in protein.active_site:
            protein_atoms = protein.active_site['atoms']
        else:
            protein_atoms = protein.atoms
        
        clash_score = 0.0
        
        for p_atom in protein_atoms:
            p_coords = p_atom['coords']
            p_symbol = p_atom.get('element', p_atom.get('name', 'C'))[0]
            p_radius = self.scoring_function.vdw_radii.get(p_symbol, 1.7)
            
            for l_atom in ligand.atoms:
                l_coords = l_atom['coords']
                l_symbol = l_atom.get('symbol', 'C')
                l_radius = self.scoring_function.vdw_radii.get(l_symbol, 1.7)
                
                # Calculate distance
                distance = np.linalg.norm(p_coords - l_coords)
                
                # Define minimum allowed distance (typically 70-80% of sum of vdW radii)
                min_allowed = (p_radius + l_radius) * 0.7
                
                # Penalize severe clashes
                if distance < min_allowed:
                    # Calculate penalty proportional to the overlap
                    overlap = (min_allowed - distance) / min_allowed
                    clash_score += overlap**2  # Square to emphasize severe clashes
        
        return clash_score
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

            sampled_point = random_point_in_sphere(center, radius)
            pose.translate(sampled_point)

            if is_inside_sphere(pose, center, radius):
                if self._check_pose_validity(pose, protein):
                    orientations.append(pose)
                else:
                    bad_orientations += 1
            else:
                bad_orientations += 1

        return orientations
    
    def _local_optimization(self, pose, protein):
        """
        Perform local optimization of a pose using gradient descent.
        
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
            
        Notes:
        ------
        This method implements a simple gradient descent optimization
        by trying small movements in different directions and accepting
        improvements.
        """
        # Set optimization parameters
        step_size = 0.1  # Angstroms for translation
        angle_step = 0.05  # Radians for rotation
        max_steps = 50  # Maximum optimization steps
        
        # Make copies to avoid modifying originals
        current_pose = copy.deepcopy(pose)
        current_score = self.scoring_function.score(protein, current_pose)
        best_pose = copy.deepcopy(current_pose)
        best_score = current_score
        
        if self.logger:
            self.logger.info(f"Starting local optimization from score: {current_score:.2f}")
        
        # Main optimization loop
        for step in range(max_steps):
            improved = False
            
            # Try translations in 6 directions (+/- x, y, z)
            directions = [
                np.array([step_size, 0, 0]),
                np.array([-step_size, 0, 0]),
                np.array([0, step_size, 0]),
                np.array([0, -step_size, 0]),
                np.array([0, 0, step_size]),
                np.array([0, 0, -step_size])
            ]
            
            # Test translations
            for direction in directions:
                test_pose = copy.deepcopy(current_pose)
                test_pose.translate(direction)
                test_score = self.scoring_function.score(protein, test_pose)
                
                if test_score < best_score:
                    best_pose = copy.deepcopy(test_pose)
                    best_score = test_score
                    improved = True
                    break
            
            # If no improvement with translation, try rotations
            if not improved:
                # Try rotations around 3 primary axes
                axes = [
                    np.array([1, 0, 0]),  # X axis
                    np.array([0, 1, 0]),  # Y axis
                    np.array([0, 0, 1])   # Z axis
                ]
                
                for axis in axes:
                    for angle in [angle_step, -angle_step]:
                        test_pose = copy.deepcopy(current_pose)
                        centroid = np.mean(test_pose.xyz, axis=0)
                        
                        # Rotate around centroid
                        test_pose.translate(-centroid)
                        rotation = Rotation.from_rotvec(axis * angle)
                        test_pose.rotate(rotation.as_matrix())
                        test_pose.translate(centroid)
                        
                        test_score = self.scoring_function.score(protein, test_pose)
                        
                        if test_score < best_score:
                            best_pose = copy.deepcopy(test_pose)
                            best_score = test_score
                            improved = True
                            break
                    
                    if improved:
                        break
            
            # Update current pose if improved
            if improved:
                current_pose = copy.deepcopy(best_pose)
                current_score = best_score
                
                # Reduce step sizes for finer search
                step_size *= 0.9
                angle_step *= 0.9
            else:
                # No improvement found, terminate early
                break
        
        if self.logger:
            self.logger.info(f"Local optimization completed. Score improved from {self.scoring_function.score(protein, pose):.2f} to {best_score:.2f}")
        
        return best_pose, best_score


    def _enhanced_local_optimization(self, protein, pose, step_size=0.2, angle_step=0.05, max_steps=50):
        """
        Enhanced local optimization with more sophisticated sampling.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        pose : Ligand
            Ligand pose to optimize
        step_size : float
            Translation step size in Angstroms
        angle_step : float
            Rotation step size in radians
        max_steps : int
            Maximum optimization steps
        
        Returns:
        --------
        tuple
            (optimized_pose, optimized_score)
            
        Notes:
        ------
        This method implements a more thorough local optimization
        compared to _local_optimization with adaptive step sizes
        and more sophisticated sampling.
        """
        # Make copies to avoid modifying originals
        current_pose = copy.deepcopy(pose)
        current_score = self.scoring_function.score(protein, current_pose)
        best_pose = copy.deepcopy(current_pose)
        best_score = current_score
        
        if self.logger:
            self.logger.info(f"Starting enhanced local optimization from score: {current_score:.2f}")
        
        # Main optimization loop
        for step in range(max_steps):
            improved = False
            
            # Try translations in 6 directions
            directions = [
                np.array([step_size, 0, 0]),
                np.array([-step_size, 0, 0]),
                np.array([0, step_size, 0]),
                np.array([0, -step_size, 0]),
                np.array([0, 0, step_size]),
                np.array([0, 0, -step_size])
            ]
            
            # Test translations
            for direction in directions:
                test_pose = copy.deepcopy(current_pose)
                test_pose.translate(direction)
                test_score = self.scoring_function.score(protein, test_pose)
                
                if test_score < best_score:
                    best_pose = copy.deepcopy(test_pose)
                    best_score = test_score
                    improved = True
            
            # Try rotations around 3 primary axes
            axes = [
                np.array([1, 0, 0]),   # X axis
                np.array([0, 1, 0]),   # Y axis
                np.array([0, 0, 1]),   # Z axis
            ]
            
            # Test rotations
            for axis in axes:
                for angle in [angle_step, -angle_step]:
                    test_pose = copy.deepcopy(current_pose)
                    pose_center = np.mean(test_pose.xyz, axis=0)
                    
                    # Translate to origin, rotate, translate back
                    test_pose.translate(-pose_center)
                    
                    # Create rotation matrix
                    rotation = Rotation.from_rotvec(axis * angle)
                    rotation_matrix = rotation.as_matrix()
                    
                    test_pose.rotate(rotation_matrix)
                    test_pose.translate(pose_center)
                    
                    test_score = self.scoring_function.score(protein, test_pose)
                    
                    if test_score < best_score:
                        best_pose = copy.deepcopy(test_pose)
                        best_score = test_score
                        improved = True
            
            # Update current pose if improved
            if improved:
                current_pose = copy.deepcopy(best_pose)
                current_score = best_score
                
                # Progress reporting
                if (step + 1) % 10 == 0:
                    if self.logger:
                        self.logger.info(f"  Optimization step {step + 1}, score: {best_score:.2f}")
            else:
                # Reduce step sizes and continue
                step_size *= 0.8
                angle_step *= 0.8
                
                # Stop if step size gets too small
                if step_size < 0.01:
                    break
        
        if self.logger:
            self.logger.info(f"Enhanced optimization complete: Score improved from {self.scoring_function.score(protein, pose):.2f} to {best_score:.2f}")
        
        return best_pose, best_score


    def improve_rigid_docking(self, protein, ligand, args):
        """
        Improved rigid docking implementation with more focused search.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        ligand : Ligand
            Ligand object
        args : argparse.Namespace
            Command-line arguments with docking parameters
        
        Returns:
        --------
        list
            List of (pose, score) tuples, sorted by score (best first)
            
        Notes:
        ------
        This method implements a more sophisticated docking protocol
        that combines random sampling with genetic algorithm optimization
        and local refinement.
        """
        print("\nPerforming enhanced rigid docking...")
        
        # Ensure active site is properly defined
        if not protein.active_site:
            if hasattr(args, 'site') and args.site:
                protein.define_active_site(args.site, args.radius if hasattr(args, 'radius') else 10.0)
                print(f"Using provided active site center with radius {args.radius if hasattr(args, 'radius') else 10.0}Å")
            elif hasattr(args, 'detect_pockets') and args.detect_pockets:
                pockets = protein.detect_pockets()
                if pockets:
                    print(f"Using detected binding pocket as active site")
                    protein.define_active_site(pockets[0]['center'], pockets[0]['radius'])
                else:
                    print("No pockets detected, using protein center")
                    center = np.mean(protein.xyz, axis=0)
                    protein.define_active_site(center, 15.0)
            else:
                print("WARNING: No active site specified. Defining one based on protein center.")
                center = np.mean(protein.xyz, axis=0)
                protein.define_active_site(center, 15.0)
       
        # Get active site parameters
        active_site_center = protein.active_site['center']
        active_site_radius = protein.active_site['radius']
        
        # Initialize grid points
        self.initialize_grid_points(center=active_site_center)
        
        # Check if ligand is within the active site
        if not is_inside_sphere(ligand, active_site_center, active_site_radius):
            print("WARNING: Ligand centroid is outside the active site. Adjusting position...")
            # Move ligand to the center of the active site
            centroid = np.mean(ligand.xyz, axis=0)
            translation = active_site_center - centroid
            ligand.translate(translation)
            print(f"Moved ligand centroid to active site center: {active_site_center}")
        else:
            print("Ligand centroid is within the active site. No adjustment needed.")
        
        # 1. Perform targeted random sampling within the active site
        print("Performing targeted random sampling...")
        
        # Adjust search parameters for better exploration
        n_initial_random = min(self.max_iterations // 4, 1000)
        random_results = []
        
        # Create initial samples with a focus on the center
        for i in range(n_initial_random):
            pose = copy.deepcopy(ligand)
            
            # Sample points within the active site with bias toward center
            random_point = active_site_center + np.random.normal(0, active_site_radius / 2, size=3)
            
            # Ensure point is within the sphere
            dist_to_center = np.linalg.norm(random_point - active_site_center)
            if dist_to_center > active_site_radius:
                # Scale back to the radius
                random_point = active_site_center + (random_point - active_site_center) * (active_site_radius / dist_to_center)
            
            # Move ligand centroid to this point
            centroid = np.mean(pose.xyz, axis=0)
            translation = random_point - centroid
            pose.translate(translation)
            
            # Apply random rotation
            rotation = Rotation.random()
            rotation_matrix = rotation.as_matrix()
            centroid = np.mean(pose.xyz, axis=0)
            pose.translate(-centroid)
            pose.rotate(rotation_matrix)
            pose.translate(centroid)
            
            # Score the pose
            score = self.scoring_function.score(protein, pose)
            random_results.append((pose, score))
            
            if (i + 1) % 100 == 0 and i > 0:
                print(f"  Completed {i + 1}/{n_initial_random} random poses")
        
        # Sort by score
        random_results.sort(key=lambda x: x[1])
        all_results = random_results.copy()
        
        # 2. Take top poses from random results as initial seeds for genetic algorithm
        if hasattr(args, 'algorithm') and args.algorithm == 'genetic':
            print("Performing genetic algorithm optimization...")
            
            # Take top 20% of random results as initial population
            top_random = random_results[:min(len(random_results) // 5, 50)]
            
            # Create a genetic algorithm with enhanced parameters
            if hasattr(self, '_selection') and hasattr(self, '_crossover') and hasattr(self, '_mutation'):
                population = top_random.copy()
                
                # Fill in population if needed
                population_size = min(len(top_random) * 2, 100)
                while len(population) < population_size:
                    idx = random.randint(0, len(top_random) - 1)
                    population.append(top_random[idx])
                
                # Sort initial population
                population.sort(key=lambda x: x[1])
                best_poses = [population[0]]
                
                # Main GA loop for a subset of iterations
                for generation in range(self.max_iterations // 5):
                    # Selection, crossover, and mutation
                    parents = self._selection(population)
                    offspring = self._crossover(parents)
                    self._mutation(offspring, active_site_radius, active_site_center)
                    
                    # Evaluate offspring
                    for i, (pose, _) in enumerate(offspring):
                        if hasattr(args, 'local_opt') and args.local_opt and i < len(offspring) // 4:
                            # Apply local optimization to top 25% if enabled
                            optimized_pose, optimized_score = self._local_optimization(pose, protein)
                            offspring[i] = (optimized_pose, optimized_score)
                        else:
                            # Otherwise just score
                            score = self.scoring_function.score(protein, pose)
                            offspring[i] = (pose, score)
                    
                    # Combine and select new population
                    combined = population + offspring
                    combined.sort(key=lambda x: x[1])
                    population = combined[:population_size]
                    
                    # Update best solution
                    if population[0][1] < best_poses[-1][1]:
                        best_poses.append(population[0])
                    
                    print(f"  Generation {generation + 1}, best score: {population[0][1]}")
                
                all_results.extend(population)
            else:
                print("Warning: Genetic algorithm methods not available. Skipping GA optimization.")
        
        # 3. Apply more aggressive local optimization to best poses
        if hasattr(args, 'local_opt') and args.local_opt:
            print("Applying enhanced local optimization (enabled by --local-opt)...")
            optimized_results = []

            # Sort all results by score and take top unique poses
            all_results.sort(key=lambda x: x[1])
            unique_poses = []
            seen_scores = set()

            for pose, score in all_results:
                rounded_score = round(score, 2)
                if rounded_score not in seen_scores:
                    unique_poses.append((pose, score))
                    seen_scores.add(rounded_score)
                    if len(unique_poses) >= 20:
                        break

            # Apply more aggressive local optimization to top poses
            poses_to_optimize_count = min(10, len(unique_poses))
            for i, (pose, score) in enumerate(unique_poses[:poses_to_optimize_count]):
                print(f"  Optimizing pose {i+1}/{poses_to_optimize_count} (initial score: {score:.2f})...")

                # Apply multiple stages of optimization with decreasing step sizes
                optimized_pose, optimized_score = self._enhanced_local_optimization(
                    protein, pose, step_size=0.5, angle_step=0.1, max_steps=20
                )
                optimized_pose, optimized_score = self._enhanced_local_optimization(
                    protein, optimized_pose, step_size=0.2, angle_step=0.05, max_steps=20
                )
                optimized_pose, optimized_score = self._enhanced_local_optimization(
                    protein, optimized_pose, step_size=0.1, angle_step=0.02, max_steps=20
                )
                
                optimized_results.append((optimized_pose, optimized_score))

            # Add remaining poses that were not optimized
            optimized_results.extend(unique_poses[poses_to_optimize_count:])

            # Sort final results
            optimized_results.sort(key=lambda x: x[1])
            print(f"Enhanced rigid docking completed. Best score: {optimized_results[0][1]:.2f}")
            return optimized_results
        else:
            # If local opt not enabled, just sort and return the results
            print("Skipping enhanced local optimization (--local-opt not specified).")
            all_results.sort(key=lambda x: x[1])
            print(f"Enhanced rigid docking completed without local optimization. Best score: {all_results[0][1]:.2f}")
            return all_results

    # Reference-guided docking with optional local optimization
        
    def reference_guided_docking(self, protein, ligand, reference_ligand, skip_optimization=False):
            """
            Perform docking with guidance from a reference ligand pose.
            
            Parameters:
            -----------
            protein : Protein
                Protein object
            ligand : Ligand
                Ligand object to dock
            reference_ligand : Ligand
                Reference ligand pose (e.g., from crystal structure)
            skip_optimization : bool
                Whether to skip local optimization
                
            Returns:
            --------
            list
                List of (pose, score) tuples, sorted by score
            """
            import copy
            import random
            from scipy.spatial.transform import Rotation
            
            print("\nPerforming reference-guided docking...")
            
            # Extract reference information
            ref_centroid = np.mean(reference_ligand.xyz, axis=0)
            ref_radius = 2.0  # Allow some flexibility around reference
            
            # Initialize result container
            results = []
            
            # Generate poses near the reference
            for i in range(100):  # Reduced iterations for faster results
                # Create a new pose
                pose = copy.deepcopy(ligand)
                
                # Position near reference centroid with some randomness
                r = ref_radius * (random.random() ** 0.5)  # Bias toward center
                theta = random.uniform(0, 2 * np.pi)
                phi = random.uniform(0, np.pi)
                
                x = ref_centroid[0] + r * np.sin(phi) * np.cos(theta)
                y = ref_centroid[1] + r * np.sin(phi) * np.sin(theta)
                z = ref_centroid[2] + r * np.sin(phi) * np.cos(phi)
                
                # Calculate translation
                pose_centroid = np.mean(pose.xyz, axis=0)
                translation = np.array([x, y, z]) - pose_centroid
                pose.translate(translation)
                
                # Apply random rotation, but with smaller angles
                # This keeps the overall orientation similar to starting pose
                angle = random.uniform(-np.pi/4, np.pi/4)  # Limit to ±45 degrees
                axis = np.random.rand(3)
                axis = axis / np.linalg.norm(axis)
                
                rotation = Rotation.from_rotvec(angle * axis)
                pose_centroid = np.mean(pose.xyz, axis=0)
                pose.translate(-pose_centroid)
                pose.rotate(rotation.as_matrix())
                pose.translate(pose_centroid)
                
                # Score the pose
                score = self.scoring_function.score(protein, pose)
                
                # Store result
                results.append((pose, score))
                
                if (i + 1) % 25 == 0:
                    print(f"Generated {i + 1}/100 reference-guided poses")
            
                # Skip local optimization if requested (i.e., skip_optimization is True)
            if skip_optimization:
                print("Skipping local optimization as requested (use --local-opt to enable)")
                results.sort(key=lambda x: x[1])
                return results

            # Apply local optimization to best poses (only runs if skip_optimization is False)
            print("Performing local optimization on top poses (enabled by --local-opt)...")
            results.sort(key=lambda x: x[1])
            optimized_results = []
            poses_to_optimize_count = min(10, len(results))

            for i, (pose, score) in enumerate(results[:poses_to_optimize_count]):
                print(f"Optimizing pose {i+1}/{poses_to_optimize_count} (initial score: {score:.2f})")
                if hasattr(self, '_enhanced_local_optimization'):
                    opt_pose, opt_score = self._enhanced_local_optimization(protein, pose, step_size=0.2, max_steps=30)
                elif hasattr(self, '_local_optimization'): # Fallback to standard if enhanced not available
                    opt_pose, opt_score = self._local_optimization(pose, protein)
                else: # If no optimization method, keep original
                    print("  Warning: No local optimization method found. Keeping original.")
                    opt_pose, opt_score = pose, score
                optimized_results.append((opt_pose, opt_score))

            # Combine results and sort
            all_results = optimized_results + results[poses_to_optimize_count:]
            all_results.sort(key=lambda x: x[1])

            return all_results
    
    def exact_reference_docking(self, protein, ligand, reference_ligand, skip_optimization=False):
        """
        Perform docking with exact alignment to a reference ligand pose.
        With aggressive refinement to improve scoring while maintaining alignment.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        ligand : Ligand
            Ligand object to dock
        reference_ligand : Ligand
            Reference ligand pose (from crystal structure)
        skip_optimization : bool
            Whether to skip the local optimization step
            
        Returns:
        --------
        list
            List of (pose, score) tuples, with the reference-aligned pose first
        """
        print("\nPerforming exact reference-based docking...")
        
        # Create a perfect superimposed pose
        import copy
        aligned_pose = copy.deepcopy(ligand)
        
        # Step 1: Calculate centroids
        ref_centroid = np.mean(reference_ligand.xyz, axis=0)
        ligand_centroid = np.mean(aligned_pose.xyz, axis=0)
        
        # Step 2: Translate ligand to origin
        aligned_pose.translate(-ligand_centroid)
        
        # Step 3: Check if atom counts match for the Kabsch algorithm
        if aligned_pose.xyz.shape[0] == reference_ligand.xyz.shape[0]:
            # Center reference coordinates
            ref_coords_centered = reference_ligand.xyz - ref_centroid
            
            # Calculate covariance matrix
            covariance = np.dot(aligned_pose.xyz.T, ref_coords_centered)
            
            # SVD decomposition
            U, S, Vt = np.linalg.svd(covariance)
            
            # Calculate rotation matrix
            rotation_matrix = np.dot(Vt.T, U.T)
            
            # Check for reflection case (determinant should be 1)
            if np.linalg.det(rotation_matrix) < 0:
                Vt[-1, :] *= -1
                rotation_matrix = np.dot(Vt.T, U.T)
            
            # Apply rotation
            aligned_pose.rotate(rotation_matrix)
        else:
            print("Warning: Atom count mismatch between ligand and reference")
            print(f"Ligand atoms: {aligned_pose.xyz.shape[0]}, Reference atoms: {reference_ligand.xyz.shape[0]}")
            print("Using centroid alignment only without rotation")
        
        # Step 4: Translate to reference position
        aligned_pose.translate(ref_centroid)
        
        # Store the exact aligned pose before any refinement
        exact_aligned_pose = copy.deepcopy(aligned_pose)
        
        # Score the exact aligned pose with normal scoring function to see baseline
        baseline_score = self.scoring_function.score(protein, aligned_pose)
        print(f"Exact reference-aligned pose baseline score: {baseline_score:.2f}")
        
        # Skip refinement/optimization if requested
        if skip_optimization:
            print("Skipping refinement as requested (--no-local-optimization)")
            return [(aligned_pose, baseline_score)]
        
        # AGGRESSIVE REFINEMENT: Move ligand out slightly and then move it back in slowly
        print("Applying aggressive refinement to find better scoring pose while preserving alignment...")
        
        # Identify the binding pocket center and ligand center
        if protein.active_site and 'center' in protein.active_site:
            pocket_center = protein.active_site['center']
        else:
            pocket_center = ref_centroid
        
        # Try different relaxation approaches and select the best result
        refined_poses = []
        
        # 1. Try aggressive atom-by-atom adjustment
        print("Approach 1: Atom-by-atom adjustment")
        relaxed_pose1 = self._aggressive_atom_adjustment_aggressive_atom_adjustment(protein, exact_aligned_pose, max_steps=50)
        relaxed_score1 = self.scoring_function.score(protein, relaxed_pose1)
        refined_poses.append((relaxed_pose1, relaxed_score1))
        
        # 2. Try slight systematic shifts in different directions
        print("Approach 2: Systematic directional shifts")
        for shift_distance in [0.2, 0.4, 0.6]:
            for direction_vector in [
                np.array([1, 0, 0]),
                np.array([-1, 0, 0]),
                np.array([0, 1, 0]),
                np.array([0, -1, 0]),
                np.array([0, 0, 1]),
                np.array([0, 0, -1]),
                np.array([0.577, 0.577, 0.577]),  # Diagonal
                np.array([-0.577, -0.577, -0.577])
            ]:
                # Normalize vector
                direction = direction_vector / np.linalg.norm(direction_vector)
                
                # Create a shifted copy
                shifted_pose = copy.deepcopy(exact_aligned_pose)
                shifted_pose.translate(direction * shift_distance)
                
                # Score the shifted pose
                shifted_score = self.scoring_function.score(protein, shifted_pose)
                
                # If score improves significantly, keep it
                if shifted_score < baseline_score - 1.0:
                    refined_poses.append((shifted_pose, shifted_score))
        
        # 3. Try very small random rotations
        print("Approach 3: Small rotational adjustments")
        from scipy.spatial.transform import Rotation
        for angle in [0.05, 0.1, 0.15]:  # Small angles in radians
            for _ in range(6):  # Try multiple random axes
                # Random rotation axis
                axis = np.random.randn(3)
                axis = axis / np.linalg.norm(axis)
                
                # Create rotated copy
                rotated_pose = copy.deepcopy(exact_aligned_pose)
                centroid = np.mean(rotated_pose.xyz, axis=0)
                
                # Apply rotation
                rotated_pose.translate(-centroid)
                rotation = Rotation.from_rotvec(axis * angle)
                rotated_pose.rotate(rotation.as_matrix())
                rotated_pose.translate(centroid)
                
                # Score the rotated pose
                rotated_score = self.scoring_function.score(protein, rotated_pose)
                
                # If score improves significantly, keep it
                if rotated_score < baseline_score - 1.0:
                    refined_poses.append((rotated_pose, rotated_score))
        
        # 4. Try moving away from clashes and back (retreat and approach)
        print("Approach 4: Retreat and approach strategy")
        # Get vector from pocket center to ligand center
        ligand_center = np.mean(exact_aligned_pose.xyz, axis=0)
        retreat_vector = ligand_center - pocket_center
        retreat_vector = retreat_vector / np.linalg.norm(retreat_vector)
        
        # Move ligand slightly away from pocket
        retreat_pose = copy.deepcopy(exact_aligned_pose)
        retreat_pose.translate(retreat_vector * 1.0)  # 1Å outward
        
        # Now approach back in small steps
        for approach_distance in [0.8, 0.6, 0.4, 0.2, 0.0]:
            approach_pose = copy.deepcopy(retreat_pose)
            approach_pose.translate(-retreat_vector * approach_distance)
            
            # Score the approach pose
            approach_score = self.scoring_function.score(protein, approach_pose)
            
            # If score improves, keep it
            if approach_score < baseline_score:
                refined_poses.append((approach_pose, approach_score))
        
        # Sort refined poses by score
        refined_poses.sort(key=lambda x: x[1])
        
        # Check if we found any better poses
        if refined_poses and refined_poses[0][1] < baseline_score:
            best_refined_pose, best_refined_score = refined_poses[0]
            print(f"Found better pose with score: {best_refined_score:.2f} (improved by {baseline_score - best_refined_score:.2f})")
            
            # Calculate RMSD from original aligned pose
            from .utils import calculate_rmsd
            rmsd = calculate_rmsd(best_refined_pose.xyz, exact_aligned_pose.xyz)
            print(f"RMSD from exact alignment: {rmsd:.3f} Å")
            
            # Create final result list
            results = [(best_refined_pose, best_refined_score)]
            
            # Add the exact aligned pose as backup
            results.append((exact_aligned_pose, baseline_score))
            
            # Add a few other good refined poses if available
            for pose, score in refined_poses[1:min(len(refined_poses), 4)]:
                results.append((pose, score))
        else:
            print("No refinement improved the score. Using exact aligned pose.")
            results = [(exact_aligned_pose, baseline_score)]
        
        return results

    def _aggressive_atom_adjustment(self, protein, pose, max_steps=50, max_atom_shift=0.5):
        """
        Aggressively adjust individual atoms to improve score while maintaining overall structure.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        pose : Ligand
            Ligand pose to refine
        max_steps : int
            Maximum number of refinement steps
        max_atom_shift : float
            Maximum allowed movement for any atom (Å)
        
        Returns:
        --------
        Ligand
            Refined pose
        """
        import copy
        import random
        print("Applying aggressive refinement (enabled by --local-opt)...")
        # Make a working copy
        working_pose = copy.deepcopy(pose)
        original_xyz = working_pose.xyz.copy()
        current_score = self.scoring_function.score(protein, working_pose)
        best_pose = copy.deepcopy(working_pose)
        best_score = current_score
        
        print(f"Starting aggressive atom adjustment (max {max_steps} steps)")
        print(f"Initial score: {current_score:.2f}")
        
        # Identify high-energy atoms (likely causing bad interactions)
        bad_atoms = self._identify_high_energy_atoms(protein, working_pose)
        
        if not bad_atoms:
            print("No problematic atoms detected")
            return working_pose
        
        print(f"Identified {len(bad_atoms)} atoms with potentially unfavorable interactions")
        
        # Iteratively adjust atoms to improve score
        for step in range(max_steps):
            improved = False
            
            # Choose a random problematic atom to adjust
            if bad_atoms:
                atom_idx = random.choice(bad_atoms)
            else:
                atom_idx = random.randint(0, len(working_pose.atoms) - 1)
            
            # Get original position for this atom
            original_pos = original_xyz[atom_idx]
            current_pos = working_pose.atoms[atom_idx]['coords']
            
            # Try multiple random adjustments for this atom
            for _ in range(10):
                # Create test pose
                test_pose = copy.deepcopy(working_pose)
                
                # Random direction
                direction = np.random.randn(3)
                direction = direction / np.linalg.norm(direction)
                
                # Random distance (up to max_atom_shift)
                move_distance = np.random.uniform(0.05, max_atom_shift)
                
                # Move the atom
                new_pos = current_pos + direction * move_distance
                
                # Ensure we don't move too far from original position
                total_shift = np.linalg.norm(new_pos - original_pos)
                if total_shift > max_atom_shift:
                    # Scale back the movement to stay within limits
                    vector_to_original = original_pos - current_pos
                    scaling_factor = (max_atom_shift - np.linalg.norm(current_pos - original_pos)) / move_distance
                    if scaling_factor <= 0:
                        continue  # Skip this attempt if we can't move
                    
                    # Apply scaled movement
                    new_pos = current_pos + direction * move_distance * scaling_factor
                
                # Update atom position
                test_pose.atoms[atom_idx]['coords'] = new_pos
                test_pose.xyz[atom_idx] = new_pos
                
                # Score the test pose
                test_score = self.scoring_function.score(protein, test_pose)
                
                # If improved, update the working pose
                if test_score < best_score:
                    best_pose = copy.deepcopy(test_pose)
                    best_score = test_score
                    improved = True
                    break  # Found improvement for this atom
            
            # Update working pose if improved
            if improved:
                working_pose = copy.deepcopy(best_pose)
                current_score = best_score
                
                # Recalculate problem atoms every 5 steps
                if step % 5 == 0:
                    bad_atoms = self._identify_high_energy_atoms(protein, working_pose)
                
                if (step + 1) % 10 == 0:
                    print(f"  Step {step+1}: Score improved to {current_score:.2f}")
            elif step % 20 == 0 and step > 0:
                # If no progress, try different atoms
                bad_atoms = self._identify_high_energy_atoms(protein, working_pose, 
                                                        energy_threshold=step/max_steps)  # Gradually include more atoms
        
        # Compare final to initial
        improvement = current_score - self.scoring_function.score(protein, pose)
        print(f"Atom adjustment complete: Score {current_score:.2f} (improved by {improvement:.2f})")
        
        return best_pose

    def _identify_high_energy_atoms(self, protein, pose, energy_threshold=0.8):
        """
        Identify atoms in the ligand that have high interaction energies with the protein.
        This is a more sophisticated approach than just looking for clashes.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        pose : Ligand
            Ligand pose to check
        energy_threshold : float
            Threshold to determine high-energy atoms, higher value includes more atoms
        
        Returns:
        --------
        list
            Indices of high-energy atoms in the ligand
        """
        # Get parameters for energy calculations
        vdw_radii = {
            'H': 1.2, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8,
            'P': 1.8, 'F': 1.47, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98
        }
        
        atom_energies = {}
        
        # Get protein atoms in active site
        if hasattr(protein, 'active_site') and protein.active_site and 'atoms' in protein.active_site:
            protein_atoms = protein.active_site['atoms']
        else:
            protein_atoms = protein.atoms
        
        # Calculate per-atom interaction energies
        for lig_idx, lig_atom in enumerate(pose.atoms):
            lig_coords = lig_atom['coords']
            lig_symbol = lig_atom.get('symbol', 'C')
            lig_radius = vdw_radii.get(lig_symbol, 1.7)
            
            # Track total energy for this atom
            atom_energy = 0.0
            
            for prot_atom in protein_atoms:
                prot_coords = prot_atom['coords']
                prot_symbol = prot_atom.get('element', prot_atom.get('name', 'C'))[0]
                prot_radius = vdw_radii.get(prot_symbol, 1.7)
                
                # Calculate distance
                distance = np.linalg.norm(lig_coords - prot_coords)
                
                # Skip atoms that are too far away
                if distance > 5.0:
                    continue
                
                # Calculate VDW energy using simple Lennard-Jones
                sigma = (lig_radius + prot_radius) * 0.5
                if distance < 0.1:  # Avoid division by zero
                    atom_energy += 100.0  # Large repulsive energy
                else:
                    ratio = sigma / distance
                    vdw_energy = (ratio**12 - 2 * ratio**6)
                    atom_energy += vdw_energy
            
            # Store energy for this atom
            atom_energies[lig_idx] = atom_energy
        
        # Sort atoms by energy (highest first)
        sorted_atoms = sorted(atom_energies.items(), key=lambda x: x[1], reverse=True)
        
        # Take atoms above threshold (proportional to list size)
        threshold_idx = max(1, int(len(sorted_atoms) * energy_threshold))
        high_energy_atoms = [idx for idx, _ in sorted_atoms[:threshold_idx]]
    
        return high_energy_atoms

    def _gentle_clash_relief(self, protein, pose, reference=None, max_steps=20, max_movement=0.2):
        """
        Perform a gentle relaxation to relieve steric clashes while preserving alignment.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        pose : Ligand
            Ligand pose to optimize
        reference : Ligand
            Reference pose to maintain alignment with
        max_steps : int
            Maximum number of minimization steps
        max_movement : float
            Maximum allowed atom movement from original position (Å)
        
        Returns:
        --------
        Ligand
            Gently relaxed pose
        """
        import copy
        from scipy.spatial.transform import Rotation
        
        # Make a work copy
        working_pose = copy.deepcopy(pose)
        current_score = self.scoring_function.score(protein, working_pose)
        best_pose = copy.deepcopy(working_pose)
        best_score = current_score
        
        # Create a modified scoring function that penalizes movement from reference
        # This is done implicitly through constraints in the minimization
        
        print(f"Starting gentle clash relief (max {max_steps} steps, max movement {max_movement} Å)")
        print(f"Initial score: {current_score:.2f}")
        
        # Identify problematic atoms (those likely causing clashes)
        clash_atoms = self._identify_clashing_atoms(protein, working_pose)
        if not clash_atoms:
            print("No significant clashes detected - skipping relaxation")
            return working_pose
        
        print(f"Detected {len(clash_atoms)} potentially clashing atoms")
        
        # First pass: try moving only clashing atoms
        for step in range(max_steps):
            improved = False
            
            for atom_idx in clash_atoms:
                # Try small random movements for the clashing atom
                for _ in range(3):  # Try multiple random directions
                    test_pose = copy.deepcopy(working_pose)
                    atom_coords = test_pose.atoms[atom_idx]['coords']
                    
                    # Random direction
                    direction = np.random.randn(3)
                    direction = direction / np.linalg.norm(direction)
                    
                    # Random distance (up to max_movement)
                    distance = np.random.uniform(0.01, max_movement)
                    movement = direction * distance
                    
                    # Move the atom
                    test_pose.atoms[atom_idx]['coords'] = atom_coords + movement
                    
                    # Update xyz array
                    test_pose.xyz[atom_idx] = test_pose.atoms[atom_idx]['coords']
                    
                    # Check if we've moved too far from reference
                    if reference is not None:
                        atom_displacement = np.linalg.norm(reference.atoms[atom_idx]['coords'] - test_pose.atoms[atom_idx]['coords'])
                        if atom_displacement > max_movement:
                            continue  # Skip if moved too far
                    
                    # Score the adjusted pose
                    test_score = self.scoring_function.score(protein, test_pose)
                    
                    if test_score < best_score:
                        best_pose = copy.deepcopy(test_pose)
                        best_score = test_score
                        improved = True
                        break  # Found improvement for this atom
            
            # Update current pose if improved
            if improved:
                working_pose = copy.deepcopy(best_pose)
                current_score = best_score
                print(f"  Step {step+1}: Score improved to {current_score:.2f}")
            else:
                break  # No further improvement possible
        
        # Second pass: try small movements of the entire ligand
        for step in range(min(5, max_steps // 4)):  # Fewer steps for global movements
            improved = False
            
            # Try small translations
            for direction in [
                np.array([0.05, 0, 0]),
                np.array([-0.05, 0, 0]),
                np.array([0, 0.05, 0]),
                np.array([0, -0.05, 0]),
                np.array([0, 0, 0.05]),
                np.array([0, 0, -0.05])
            ]:
                test_pose = copy.deepcopy(working_pose)
                test_pose.translate(direction)
                
                # Check if we've moved too far from reference
                if reference is not None:
                    max_displacement = np.max(np.linalg.norm(reference.xyz - test_pose.xyz, axis=1))
                    if max_displacement > max_movement:
                        continue  # Skip if moved too far
                
                test_score = self.scoring_function.score(protein, test_pose)
                
                if test_score < best_score:
                    best_pose = copy.deepcopy(test_pose)
                    best_score = test_score
                    improved = True
            
            # Try very small rotations
            if not improved:
                for axis in [
                    np.array([1, 0, 0]),
                    np.array([0, 1, 0]),
                    np.array([0, 0, 1])
                ]:
                    for angle in [0.01, -0.01]:  # Very small rotations
                        test_pose = copy.deepcopy(working_pose)
                        
                        centroid = np.mean(test_pose.xyz, axis=0)
                        test_pose.translate(-centroid)
                        
                        rotation = Rotation.from_rotvec(axis * angle)
                        test_pose.rotate(rotation.as_matrix())
                        
                        test_pose.translate(centroid)
                        
                        # Check if we've moved too far from reference
                        if reference is not None:
                            max_displacement = np.max(np.linalg.norm(reference.xyz - test_pose.xyz, axis=1))
                            if max_displacement > max_movement:
                                continue  # Skip if moved too far
                        
                        test_score = self.scoring_function.score(protein, test_pose)
                        
                        if test_score < best_score:
                            best_pose = copy.deepcopy(test_pose)
                            best_score = test_score
                            improved = True
            
            # Update current pose if improved
            if improved:
                working_pose = copy.deepcopy(best_pose)
                current_score = best_score
                print(f"  Global refinement step {step+1}: Score improved to {current_score:.2f}")
            else:
                break  # No further improvement possible
        
        print(f"Relaxation complete: Final score {best_score:.2f} (improved by {current_score - best_score:.2f})")
        return best_pose

    def _identify_clashing_atoms(self, protein, pose, clash_cutoff=0.7):
        """
        Identify atoms in the ligand that are clashing with protein atoms.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        pose : Ligand
            Ligand pose to check
        clash_cutoff : float
            Factor to determine clash distance (0.7 means atoms closer than
            0.7 * (sum of vdW radii) are considered clashing)
        
        Returns:
        --------
        list
            Indices of clashing atoms in the ligand
        """
        # Get VDW radii
        vdw_radii = {
            'H': 1.2, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8,
            'P': 1.8, 'F': 1.47, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98
        }
        
        clashing_atoms = set()
        
        # Get protein atoms in active site
        if hasattr(protein, 'active_site') and protein.active_site and 'atoms' in protein.active_site:
            protein_atoms = protein.active_site['atoms']
        else:
            protein_atoms = protein.atoms
        
        # Check each ligand atom against all protein atoms
        for lig_idx, lig_atom in enumerate(pose.atoms):
            lig_coords = lig_atom['coords']
            lig_symbol = lig_atom.get('symbol', 'C')
            lig_radius = vdw_radii.get(lig_symbol, 1.7)
            
            for prot_atom in protein_atoms:
                prot_coords = prot_atom['coords']
                prot_symbol = prot_atom.get('element', prot_atom.get('name', 'C'))[0]
                prot_radius = vdw_radii.get(prot_symbol, 1.7)
                
                # Calculate distance
                distance = np.linalg.norm(lig_coords - prot_coords)
                
                # Check for clash
                min_allowed = (lig_radius + prot_radius) * clash_cutoff
                if distance < min_allowed:
                    clashing_atoms.add(lig_idx)
                    break  # This ligand atom is clashing, move to next
        
        return list(clashing_atoms)

    def _write_ligand(self, ligand, filename):
        """Write ligand to SDF file."""
        with open(filename, 'w') as f:
            f.write("Reference\n")
            f.write("  PandaDock\n\n")
            
            # Number of atoms and bonds
            f.write(f"{len(ligand.atoms):3d}{len(ligand.bonds):3d}  0  0  0  0  0  0  0  0999 V2000\n")
            
            # Atoms
            for atom in ligand.atoms:
                coords = atom['coords']
                symbol = atom.get('symbol', 'C')
                
                # PDB ATOM format
                f.write(f"{coords[0]:10.4f}{coords[1]:10.4f}{coords[2]:10.4f} {symbol:<3}  0  0  0  0  0  0  0  0  0  0  0  0\n")
            
            # Bonds
            for bond in ligand.bonds:
                a1 = bond['begin_atom_idx'] + 1  # 1-based indexing in SDF
                a2 = bond['end_atom_idx'] + 1
                type_num = bond.get('bond_type', 1)
                if isinstance(type_num, str):
                    type_num = 1  # Default to single bond
                f.write(f"{a1:3d}{a2:3d}{type_num:3d}  0  0  0  0\n")
            
            # Terminator
            f.write("M  END\n$$$$\n")

    def _read_ligand(self, filename):
        """Read ligand from SDF file."""
        from .ligand import Ligand
        return Ligand(filename)
        

    def exact_reference_docking_with_tethering(self, protein, ligand, reference_ligand, 
                                            tether_weight=10.0, skip_optimization=False):
        """
        Perform docking with exact alignment to a reference and tethered optimization.
        """
        print("\nPerforming tethered reference-based docking...")
        
        # Create a perfect superimposed pose
        import copy
        aligned_pose = copy.deepcopy(ligand)
        
        # Step 1: Calculate centroids
        ref_centroid = np.mean(reference_ligand.xyz, axis=0)
        ligand_centroid = np.mean(aligned_pose.xyz, axis=0)
        
        # Step 2: Translate ligand to origin
        aligned_pose.translate(-ligand_centroid)
        
        # Step 3: Check if atom counts match for the Kabsch algorithm
        if aligned_pose.xyz.shape[0] == reference_ligand.xyz.shape[0]:
            # Center reference coordinates
            ref_coords_centered = reference_ligand.xyz - ref_centroid
            
            # Calculate covariance matrix
            covariance = np.dot(aligned_pose.xyz.T, ref_coords_centered)
            
            # SVD decomposition
            U, S, Vt = np.linalg.svd(covariance)
            
            # Calculate rotation matrix
            rotation_matrix = np.dot(Vt.T, U.T)
            
            # Check for reflection case (determinant should be 1)
            if np.linalg.det(rotation_matrix) < 0:
                Vt[-1, :] *= -1
                rotation_matrix = np.dot(Vt.T, U.T)
            
            # Apply rotation
            aligned_pose.rotate(rotation_matrix)
        else:
            print("Warning: Atom count mismatch between ligand and reference")
            print(f"Ligand atoms: {aligned_pose.xyz.shape[0]}, Reference atoms: {reference_ligand.xyz.shape[0]}")
            print("Using centroid alignment only without rotation")
        
        # Step 4: Translate to reference position
        aligned_pose.translate(ref_centroid)
        
        # Store the exact aligned pose before any refinement
        exact_aligned_pose = copy.deepcopy(aligned_pose)
        
        # Import the tethered scoring function
        from .unified_scoring import TetheredScoringFunction
        
        # Create a tethered scoring function
        # Create a tethered scoring function using the unified approach
        tethered_scoring = TetheredScoringFunction(
            self.scoring_function,
            reference_ligand,
            weight=tether_weight
        )
        
        # Score the exact aligned pose with normal scoring function
        baseline_score = self.scoring_function.score(protein, aligned_pose)
        print(f"Exact reference-aligned pose baseline score: {baseline_score:.2f}")
        
        # Skip refinement/optimization if requested
        if skip_optimization:
            print("Skipping refinement (use --local-opt to enable)")
            return [(aligned_pose, baseline_score)]
        
        # Perform optimization with tethered scoring
        print("Applying tethered optimization to improve score while preserving alignment...")
        
        # Store original scoring function temporarily
        original_scoring = self.scoring_function
        
        # Replace with tethered scoring function
        self.scoring_function = tethered_scoring
        
        # Perform optimization
        optimized_pose, optimized_score = self._enhanced_local_optimization(
            protein, exact_aligned_pose, step_size=0.2, angle_step=0.05, max_steps=50
        )
        
        # Restore original scoring function
        self.scoring_function = original_scoring
        if hasattr(self, '_enhanced_local_optimization'):
             optimized_pose, _ = self._enhanced_local_optimization( # Score here is tethered score
                 protein, exact_aligned_pose, step_size=0.2, angle_step=0.05, max_steps=50
             )
        elif hasattr(self, '_local_optimization'):
            print("  Warning: Using standard local optimization for tethering.")
            optimized_pose, _ = self._local_optimization(protein, exact_aligned_pose)
        else:
            print("  Warning: No local optimization method found for tethering. Using aligned pose.")
            optimized_pose = exact_aligned_pose # Fallback
        # Score the optimized pose with original scoring function
        final_score = original_scoring.score(protein, optimized_pose)
        
        # Calculate RMSD between optimized and reference
        from .utils import calculate_rmsd
        rmsd = calculate_rmsd(optimized_pose.xyz, reference_ligand.xyz)
        
        print(f"Tethered optimization complete:")
        print(f" - Final score: {final_score:.2f}")
        print(f" - RMSD from reference: {rmsd:.3f} Å")
        
        # Return results
        return [(optimized_pose, final_score), (exact_aligned_pose, baseline_score)]
    
    
class RandomSearch(DockingSearch):
    """
    Simple random search algorithm.
    
    This class implements a basic stochastic sampling approach for 
    molecular docking, generating random poses within the active site
    and evaluating them.
    """
    
    def search(self, protein, ligand):
        """
        Perform random search to find optimal ligand poses.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        ligand : Ligand
            Ligand object
        
        Returns:
        --------
        list
            List of (pose, score) tuples, sorted by score
        """

        # Determine search space
        if protein.active_site:
            center = protein.active_site['center']
            radius = protein.active_site['radius']
        else:
            center = np.mean(protein.xyz, axis=0)
            radius = 15.0  # Default
        
        # Initialize grid points
        self.initialize_grid_points(center)
        
        # Ensure active site atoms are defined for faster scoring
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

        print(f"Starting random search with {self.max_iterations} iterations")
        print(f"Searching around center {center} with radius {radius} Å")

        # Store results
        results = []
        
        # Track progress data
        start_time = time.time()
        best_score = float('inf')
        best_pose = None
        
        # Count failures to potentially adjust search space
        failures = 0
        max_failures_before_adjust = 50
        
        # Main search loop
        for iteration in range(self.max_iterations):
            # Adjust radius to focus search as we progress
            current_radius = self._adjust_search_radius(radius, iteration, self.max_iterations)
            
            # Generate a random pose
            pose = copy.deepcopy(ligand)
            
            # Choose a random grid point
            random_point = random.choice(self.grid_points)
            
            # Move ligand centroid to this point
            centroid = np.mean(pose.xyz, axis=0)
            translation = random_point - centroid
            pose.translate(translation)
            
            # Apply random rotation
            rotation = Rotation.random()
            rotation_matrix = rotation.as_matrix()
            pose.translate(-centroid)
            pose.rotate(rotation_matrix)
            pose.translate(centroid)
            
            # Pre-filtering for obvious issues
            if not self._check_pose_validity(pose, protein):
                failures += 1
                if failures >= max_failures_before_adjust:
                    # If too many failures, temporarily increase radius
                    print(f"[INFO] High failure rate. Temporarily increasing search radius.")
                    current_radius *= 1.2
                    failures = 0
                continue
            
            # Reset failure counter on success
            failures = 0
            
            # Score the pose
            score = self.scoring_function.score(protein, pose)
            
            # Add to results
            results.append((pose, score))
            
            # Update best pose
            if score < best_score:
                best_score = score
                best_pose = copy.deepcopy(pose)
                
                # Save intermediate result if output_dir is specified
                if self.output_dir:
                    save_intermediate_result(
                        best_pose, best_score, iteration + 1, 
                        self.output_dir, self.max_iterations
                    )
                    
                    # Update status file
                    update_status(
                        self.output_dir,
                        current_iteration=iteration + 1,
                        best_score=best_score,
                        total_iterations=self.max_iterations,
                        progress=(iteration + 1) / self.max_iterations
                    )
            
            # Progress reporting
            if (iteration + 1) % 100 == 0 or iteration == self.max_iterations - 1:
                elapsed = time.time() - start_time
                remaining = (elapsed / (iteration + 1)) * (self.max_iterations - iteration - 1)
                print(f"Iteration {iteration + 1}/{self.max_iterations}, "
                      f"Best score: {best_score:.2f}, "
                      f"Time: {elapsed:.1f}s, ETA: {remaining:.1f}s")
        
        # Sort results by score
        results.sort(key=lambda x: x[1])
        
        print(f"Random search completed. "
              f"Best score: {results[0][1]:.2f}, "
              f"Total time: {time.time() - start_time:.1f}s")
        
        return results
        



class GeneticAlgorithm(DockingSearch):
    """
    Genetic algorithm for conformational search in molecular docking.
    
    This class implements an evolutionary optimization approach, evolving
    a population of ligand poses through selection, crossover, and mutation
    operations to identify optimal binding modes.
    """
    
    def __init__(self, scoring_function, max_iterations=100, 
                 population_size=50, mutation_rate=0.2, perform_local_opt=False,
                 output_dir=None, grid_spacing=0.375, grid_radius=10.0):
        """
        Initialize genetic algorithm with specific parameters.
        
        Parameters:
        -----------
        scoring_function : ScoringFunction
            Scoring function to evaluate poses
        max_iterations : int
            Maximum number of generations
        population_size : int
            Size of the population
        mutation_rate : float
            Probability of mutation (0.0 to 1.0)
        perform_local_opt : bool
            Whether to perform local optimization on selected individuals
        output_dir : str or Path
            Directory for saving output files
        grid_spacing : float
            Spacing between grid points in Angstroms
        grid_radius : float
            Radius of search sphere in Angstroms
        """
        super().__init__(scoring_function, max_iterations, output_dir, 
                         grid_spacing, grid_radius)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.perform_local_opt = perform_local_opt
        self.grid_radius = grid_radius
        self.grid_spacing = grid_spacing

    def search(self, protein, ligand):
        """
        Perform genetic algorithm search to find optimal ligand poses.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        ligand : Ligand
            Ligand object
        
        Returns:
        --------
        list
            List of (pose, score) tuples, sorted by score
        """

        # Determine search space
        if protein.active_site:
            center = protein.active_site['center']
            radius = protein.active_site['radius']
        else:
            center = np.mean(protein.xyz, axis=0)
            radius = 10.0  # Default
        # Initialize grid points
        self.initialize_grid_points(center)
        
        # Ensure active site atoms are defined for faster scoring
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
        print(f"Starting genetic algorithm search with population {self.population_size}")
        print(f"Local optimization within GA generations is {'ENABLED' if self.perform_local_opt else 'DISABLED'}.")

        # Initialize population
        population = []
        for _ in range(self.population_size):
            pose = copy.deepcopy(ligand)

            #  Pick a random point from grid
            random_grid_point = random.choice(self.grid_points)

            #  Move ligand centroid to grid point
            centroid = np.mean(pose.xyz, axis=0)
            translation = random_grid_point - centroid
            pose.translate(translation)

            #  Apply random rotation
            rotation = Rotation.random()
            rotation_matrix = rotation.as_matrix()
            centroid = np.mean(pose.xyz, axis=0)
            pose.translate(-centroid)
            pose.rotate(rotation_matrix)
            pose.translate(centroid)

            #  Evaluate Score
            score = self.scoring_function.score(protein, pose)
            population.append((pose, score))

        # Sort initial population
        population.sort(key=lambda x: x[1])
        best_poses = [population[0]]
        best_score = population[0][1]

        # Track progress timing
        start_time = time.time()

        # Main evolutionary loop
        for generation in range(self.max_iterations):
            # Adjust search radius to focus on promising regions
            current_radius = self._adjust_search_radius(radius, generation, self.max_iterations)

            # Selection: tournament selection
            parents = self._selection(population)

            # Crossover: create offspring
            offspring = self._crossover(parents)

            # Mutation: apply random modifications
            self._mutation(offspring, current_radius, center)

            # Enforce boundary constraints
            for i, (pose, _) in enumerate(offspring):
                # Check if the pose is within the sphere
                if not is_inside_sphere(pose, center, current_radius):
                    # If outside, regenerate this individual
                    new_pose = copy.deepcopy(ligand)
                    
                    # Sample within the current radius
                    r = current_radius * np.random.random() ** (1/3)
                    theta = np.random.uniform(0, 2 * np.pi)
                    phi = np.random.uniform(0, np.pi)
                    
                    x = center[0] + r * np.sin(phi) * np.cos(theta)
                    y = center[1] + r * np.sin(phi) * np.sin(theta)
                    z = center[2] + r * np.sin(phi) * np.cos(phi)
                    
                    new_centroid = np.mean(new_pose.xyz, axis=0)
                    new_pose.translate(np.array([x, y, z]) - new_centroid)
                    
                    # Random rotation
                    rotation = Rotation.random()
                    new_centroid = np.mean(new_pose.xyz, axis=0)
                    new_pose.translate(-new_centroid)
                    new_pose.rotate(rotation.as_matrix())
                    new_pose.translate(new_centroid)
                    
                    offspring[i] = (new_pose, 0)  # Will be scored later
            
            # Evaluate offspring
            for i, (pose, _) in enumerate(offspring):
                # Apply local optimization to top offspring if enabled
                if self.perform_local_opt and i < len(offspring) // 4:  # Optimize top 25%
                    optimized_pose, optimized_score = self._local_optimization(pose, protein)
                    offspring[i] = (optimized_pose, optimized_score)
                else:
                    # Just score
                    score = self.scoring_function.score(protein, pose)
                    offspring[i] = (pose, score)
            
            # Combine parents and offspring, then select new population (elitism)
            combined = population + offspring
            combined.sort(key=lambda x: x[1])
            population = combined[:self.population_size]
            
            # Update best solution
            if population[0][1] < best_score:
                best_score = population[0][1]
                best_poses.append(population[0])
                
                # Save intermediate result if output_dir is specified
                if self.output_dir:
                    save_intermediate_result(
                        population[0][0], population[0][1], generation + 1, 
                        self.output_dir, self.max_iterations
                    )
                    
                    # Update status file
                    update_status(
                        self.output_dir,
                        current_generation=generation + 1,
                        best_score=best_score,
                        total_generations=self.max_iterations,
                        progress=(generation + 1) / self.max_iterations
                    )
            
            # Progress reporting
            elapsed = time.time() - start_time
            remaining = (elapsed / (generation + 1)) * (self.max_iterations - generation - 1)
            print(f"Generation {generation + 1}/{self.max_iterations}, "
                  f"Best score: {best_score:.2f}, "
                  f"Time: {elapsed:.1f}s, ETA: {remaining:.1f}s")
        
        # Final local optimization of top poses if enabled
        if self.perform_local_opt:
            print("Performing final local optimization on top poses...")
            optimized_results = []
            
            # Take top 10 unique poses
            population.sort(key=lambda x: x[1])
            unique_poses = []
            seen_scores = set()
            
            for pose, score in population:
                rounded_score = round(score, 2)
                if rounded_score not in seen_scores:
                    unique_poses.append((pose, score))
                    seen_scores.add(rounded_score)
                    if len(unique_poses) >= 10:
                        break
            
            # Optimize each pose
            for i, (pose, score) in enumerate(unique_poses):
                print(f"  Optimizing pose {i+1}/10 (initial score: {score:.2f})...")
                optimized_pose, optimized_score = self._enhanced_local_optimization(
                    protein, pose, step_size=0.2, angle_step=0.05, max_steps=50
                )
                optimized_results.append((optimized_pose, optimized_score))
            
            # Combine with other good poses
            optimized_results.extend(population[len(optimized_results):20])
            optimized_results.sort(key=lambda x: x[1])
            
            print(f"Genetic algorithm completed with optimization. "
                  f"Best score: {optimized_results[0][1]:.2f}")
            
            return optimized_results
        else:
            # Return best unique poses if no final optimization
            population.sort(key=lambda x: x[1])
            
            print(f"Genetic algorithm completed. "
                  f"Best score: {population[0][1]:.2f}")
            
            return population

    def _selection(self, population):
        """
        Tournament selection to choose parents for reproduction.
        
        Parameters:
        -----------
        population : list
            List of (pose, score) tuples
        
        Returns:
        --------
        list
            Selected parents as (pose, score) tuples
        """
        parents = []
        for _ in range(self.population_size):
            # Select random individuals for tournament
            tournament_size = max(2, int(self.population_size * 0.1))
            tournament = random.sample(population, tournament_size)
            
            # Select best from tournament
            tournament.sort(key=lambda x: x[1])
            parents.append(tournament[0])
        
        return parents
    
    def _crossover(self, parents):
        """
        Crossover operation to create offspring from parents.
        
        Parameters:
        -----------
        parents : list
            List of (pose, score) tuples
        
        Returns:
        --------
        list
            Offspring as (pose, score) tuples with scores set to 0
        """
        offspring = []
        random.shuffle(parents)
        
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                parent1, _ = parents[i]
                parent2, _ = parents[i + 1]
                
                # Create new poses by combining aspects of parents
                child1 = copy.deepcopy(parent1)
                child2 = copy.deepcopy(parent2)
                
                # Average position
                centroid1 = np.mean(parent1.xyz, axis=0)
                centroid2 = np.mean(parent2.xyz, axis=0)
                avg_centroid = (centroid1 + centroid2) / 2.0
                
                # Move child1 to average position
                child1_centroid = np.mean(child1.xyz, axis=0)
                child1.translate(avg_centroid - child1_centroid)
                
                # Move child2 to average position
                child2_centroid = np.mean(child2.xyz, axis=0)
                child2.translate(avg_centroid - child2_centroid)
                
                offspring.append((child1, 0))
                offspring.append((child2, 0))
        
        return offspring
    
    def _mutation(self, offspring, radius, center):
        """
        Mutation operation to introduce random variations.
        
        Parameters:
        -----------
        offspring : list
            List of (pose, score) tuples to mutate
        radius : float
            Current search radius
        center : array-like
            Center coordinates of search sphere
        """
        for i, (pose, _) in enumerate(offspring):
            original_pose = copy.deepcopy(pose)
            self._mutate(pose)  # Try mutation
            if not is_inside_sphere(pose, center, radius):
                # Outside the sphere? Revert
                offspring[i] = (original_pose, 0)


    def _mutate(self, individual):
        """
        Mutate an individual with probability mutation_rate.
        Also handles flexible residues if available.
        
        Parameters:
        -----------
        individual : Ligand
            The ligand pose to mutate
        """
        # Apply standard mutation (replaces the call to super()._mutate)
        if random.random() < self.mutation_rate:
            # Random translation mutation
            translation = np.random.normal(0, 0.5, 3)
            individual.translate(translation)
            
            # Random rotation mutation
            angle = np.random.normal(0, 0.2)  # ~10 degrees std dev
            axis = np.random.rand(3)
            axis = axis / np.linalg.norm(axis)
            
            rotation = Rotation.from_rotvec(angle * axis)
            rotation_matrix = rotation.as_matrix()
            
            centroid = np.mean(individual.xyz, axis=0)
            individual.translate(-centroid)
            individual.rotate(rotation_matrix)
            individual.translate(centroid)
        
        # Mutate flexible residues if available
        # Check if the protein has flexible residues
        if hasattr(self, 'protein') and hasattr(self.protein, 'flexible_residues') and self.protein.flexible_residues:
            # Probability of mutating a flexible residue
            if random.random() < self.mutation_rate:
                # Randomly select a flexible residue
                residue = random.choice(self.protein.flexible_residues)
                
                # Randomly select a rotatable bond in the residue
                if residue.rotatable_bonds:
                    bond_idx = random.randint(0, len(residue.rotatable_bonds) - 1)
                    
                    # Random rotation angle between -60 and +60 degrees
                    angle = random.uniform(-np.pi/3, np.pi/3)
                    
                    # Apply rotation
                    residue.rotate_bond(bond_idx, angle)


                        