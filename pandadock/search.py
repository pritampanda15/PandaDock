# search.py
import numpy as np
from scipy.spatial.transform import Rotation
import random
import copy
from .utils import setup_logging

class DockingSearch:
    """Base class for docking search algorithms."""
    
    def __init__(self, scoring_function, max_iterations=1000, output_dir=None):
        """
        Initialize search algorithm.
        
        Parameters:
        -----------
        scoring_function : ScoringFunction
            Scoring function to evaluate poses
        max_iterations : int
            Maximum number of iterations
        """
        from .utils import setup_logging
    
        self.scoring_function = scoring_function
        self.max_iterations = max_iterations
        self.output_dir = output_dir
    
    # Set up logger
        self.logger = setup_logging(output_dir)

    
    def search(self, protein, ligand):
        """
        Perform docking search.
        
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
        raise NotImplementedError("Subclasses must implement this method")

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
            Command-line arguments
        
        Returns:
        --------
        list
            List of (pose, score) tuples, sorted by score (best first)
        """
        print("\nPerforming enhanced rigid docking...")

        # --- Active site radius enforcement and adjustment ---
        if not protein.active_site:
            if hasattr(args, 'site') and args.site:
                radius = args.radius if hasattr(args, 'radius') else 15.0  # Increased default radius
                # Enforce minimum radius
                if radius < 12.0:
                    print(f"Provided radius {radius}Å is small; increasing to 15.0Å to avoid close contacts")
                    radius = 15.0
                protein.define_active_site(args.site, radius)
                print(f"Using provided active site center with radius {radius}Å")
            elif hasattr(args, 'detect_pockets') and args.detect_pockets:
                pockets = protein.detect_pockets()
                if pockets:
                    print(f"Using detected binding pocket as active site")
                    # Enforce minimum radius on detected pocket
                    pocket_radius = pockets[0]['radius']
                    if pocket_radius < 12.0:
                        print(f"Detected pocket radius {pocket_radius}Å is small; increasing to 15.0Å")
                        pocket_radius = 15.0
                    protein.define_active_site(pockets[0]['center'], pocket_radius)
                else:
                    print("No pockets detected, using protein center")
                    center = np.mean(protein.xyz, axis=0)
                    protein.define_active_site(center, 15.0)
            else:
                print("WARNING: No active site specified. Defining one based on protein center.")
                center = np.mean(protein.xyz, axis=0)
                protein.define_active_site(center, 15.0)

        # Create a more focused initial pose generation
        active_site_center = protein.active_site['center']
        active_site_radius = protein.active_site['radius']
        
        # Use the current scoring function
        scoring_function = self.scoring_function
        
        # 1. Perform targeted random sampling within the active site
        print("Performing targeted random sampling...")
        
        # Adjust search parameters for better exploration
        n_initial_random = min(self.max_iterations // 4, 1000)
        random_results = []
        
        # Create initial samples with a focus on the center
        for i in range(n_initial_random):
            pose = copy.deepcopy(ligand)
            
            # Use more samples near the center of the active site
            # with reduced radius to ensure better placement
            distance_factor = np.random.random() ** 0.5  # Bias toward center
            r = active_site_radius * distance_factor
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            
            x = active_site_center[0] + r * np.sin(phi) * np.cos(theta)
            y = active_site_center[1] + r * np.sin(phi) * np.sin(theta)
            z = active_site_center[2] + r * np.cos(phi)
            
            # Move the ligand to this position
            centroid = np.mean(pose.xyz, axis=0)
            translation = np.array([x, y, z]) - centroid
            pose.translate(translation)
            
            # Apply a random rotation
            rotation = Rotation.random()
            
            centroid = np.mean(pose.xyz, axis=0)
            pose.translate(-centroid)
            pose.rotate(rotation.as_matrix())
            pose.translate(centroid)
            
            # Score the pose
            score = scoring_function.score(protein, pose)
            random_results.append((pose, score))
            
            if (i + 1) % 100 == 0 and i > 0:
                print(f"  Completed {i + 1}/{n_initial_random} random poses")
        
        # Sort by score
        random_results.sort(key=lambda x: x[1])
        all_results = random_results.copy()
        
        # 2. Take top 20% of random results as initial seeds for genetic algorithm
        if hasattr(args, 'algorithm') and args.algorithm == 'genetic':
            print("Performing genetic algorithm optimization...")
            
            # Take top 20% of random results as initial population
            top_random = random_results[:min(len(random_results) // 5, 50)]
            
            # Create a genetic algorithm with enhanced parameters
            from .search import GeneticAlgorithm
            ga = GeneticAlgorithm(
                scoring_function=scoring_function,
                max_iterations=self.max_iterations // 5,
                population_size=min(len(top_random) * 2, 100),
                mutation_rate=0.3,  # Higher mutation rate for better exploration
                # Pass the optimization flag to the GA
                perform_local_opt=args.local_opt if hasattr(args, 'local_opt') else False
            )
            
            # Pass protein to GA for flexible residue mutation if needed
            ga.protein = protein
            
            ga_results = []
            
            # Create initial population from top random poses
            population = top_random.copy()
            
            # Fill in population if needed
            while len(population) < ga.population_size:
                idx = random.randint(0, len(top_random) - 1)
                population.append(top_random[idx])
            
            # Sort initial population
            population.sort(key=lambda x: x[1])
            best_poses = [population[0]]
            
            # Main GA loop
            for generation in range(ga.max_iterations):
                # Selection, crossover, and mutation
                parents = ga._selection(population)
                offspring = ga._crossover(parents)
                ga._mutation(offspring, active_site_radius, active_site_center)
                
                # Evaluate offspring
                for i, (pose, _) in enumerate(offspring):
                    if ga.perform_local_opt and i < len(offspring) // 4:
                        optimized_pose, optimized_score = self._local_optimization(pose, protein)
                        offspring[i] = (optimized_pose, optimized_score)
                    else: # Otherwise, just score
                        score = scoring_function.score(protein, pose)
                        offspring[i] = (pose, score)
                
                # Combine and select new population
                combined = population + offspring
                combined.sort(key=lambda x: x[1])
                population = combined[:ga.population_size]
                
                # Update best solution
                if population[0][1] < best_poses[-1][1]:
                    best_poses.append(population[0])
                
                print(f"  Generation {generation + 1}, best score: {population[0][1]}")
            
            ga_results = population
            all_results.extend(ga_results)
        
        # 3. Apply more aggressive local optimization to best poses ONLY IF args.local_opt is TRUE
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

            # Apply more aggressive local optimization to top 10 poses
            poses_to_optimize_count = min(10, len(unique_poses))
            for i, (pose, score) in enumerate(unique_poses[:poses_to_optimize_count]):
                print(f"  Optimizing pose {i+1}/{poses_to_optimize_count} (initial score: {score:.2f})...")

                optimized_pose = copy.deepcopy(pose)
                optimized_score = score
                optimized_pose, optimized_score = self._enhanced_local_optimization(
                    protein, optimized_pose, step_size=0.5, angle_step=0.1, max_steps=20
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

            # --- APPLY GENTLE CLASH RELIEF POST-OPTIMIZATION ---
            print("Applying gentle clash relief to optimized poses...")
            relaxed_poses = []
            for pose, score in optimized_results:
                relaxed_pose = self._gentle_clash_relief(protein, pose, max_steps=20, max_movement=0.2)
                relaxed_score = scoring_function.score(protein, relaxed_pose)
                relaxed_poses.append((relaxed_pose, relaxed_score))
            relaxed_poses.sort(key=lambda x: x[1])
            print(f"Post-docking clash relief applied. Best relaxed score: {relaxed_poses[0][1]:.2f}")

            return relaxed_poses
        else:
            center = np.mean(protein.xyz, axis=0)
            radius = 15.0
        
        current_pose = copy.deepcopy(pose)
        current_score = self.scoring_function.score(protein, current_pose)
        best_pose = copy.deepcopy(current_pose)
        best_score = current_score
        
        # Counters for monitoring progress
        step_count = 0
        no_improvement_count = 0
        
        while step_count < max_steps and no_improvement_count < 10:
            step_count += 1
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
                
                # Ensure pose stays within active site
                pose_center = np.mean(test_pose.xyz, axis=0)
                if np.linalg.norm(pose_center - center) > radius:
                    continue
                    
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
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Reduce step sizes as optimization progresses
            if step_count % 10 == 0 and step_count > 0:
                step_size *= 0.8
                angle_step *= 0.8
        
        return best_pose, best_score
        
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
            ref_radius = 5.0  # Allow some flexibility around reference
            
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

            # --- APPLY GENTLE CLASH RELIEF WHEN LOCAL OPT IS NOT ENABLED ---
            print("Applying gentle clash relief to poses (local optimization disabled)...")
            relaxed_poses = []
            for pose, score in all_results:
                relaxed_pose = self._gentle_clash_relief(protein, pose, max_steps=20, max_movement=0.2)
                relaxed_score = scoring_function.score(protein, relaxed_pose)
                relaxed_poses.append((relaxed_pose, relaxed_score))
            relaxed_poses.sort(key=lambda x: x[1])
            print(f"Post-docking clash relief applied. Best relaxed score: {relaxed_poses[0][1]:.2f}")

            return relaxed_poses

    # ... rest of your methods unchanged ...

class GeneticAlgorithm(DockingSearch):
    """Genetic algorithm for docking search."""
    
    def __init__(self, scoring_function, max_iterations=1000, 
                 population_size=50, mutation_rate=0.2, perform_local_opt=False, output_dir=None):
        """Initialize genetic algorithm."""
        super().__init__(scoring_function, max_iterations)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.perform_local_opt = perform_local_opt
        
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
        step_size = 0.1  # Angstroms for translation
        angle_step = 0.05  # Radians for rotation
        max_steps = 50
        converged = False
        clash_threshold = 1.0  # Threshold for acceptable clash score
        
        # Make a copy of the pose to avoid modifying the original
        current_pose = copy.deepcopy(pose)
        current_score = self.scoring_function.score(protein, pose) # Get score before opt
        current_score = self.scoring_function.score(protein, current_pose)
        best_pose = copy.deepcopy(current_pose)
        best_score = current_score
        
        print(f"Starting local optimization from score: {current_score:.2f}")
        
        for step in range(max_steps):
            improved = False
            
            # Try small translations in 6 directions (+/- x, y, z)
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
                
                # Calculate clash score specifically
                clash_score = self._calculate_clash_score(protein, test_pose)
                
                # Only accept if both overall score is better AND clash score is below threshold
                if test_score < best_score and clash_score < clash_threshold:
                    best_pose = copy.deepcopy(test_pose)
                    best_score = test_score
                    improved = True
            
            # Try small rotations around 3 axes
            axes = [
                np.array([1, 0, 0]),
                np.array([0, 1, 0]),
                np.array([0, 0, 1])
            ]
            
            # Test rotations
            for axis in axes:
                for angle in [angle_step, -angle_step]:
                    test_pose = copy.deepcopy(current_pose)
                    
                    # Get centroid
                    centroid = np.mean(test_pose.xyz, axis=0)
                    
                    # Translate to origin, rotate, translate back
                    test_pose.translate(-centroid)
                    
                    # Create rotation matrix
                    rotation = Rotation.from_rotvec(axis * angle)
                    rotation_matrix = rotation.as_matrix()
                    
                    test_pose.rotate(rotation_matrix)
                    test_pose.translate(centroid)
                    
                    test_score = self.scoring_function.score(protein, test_pose)
                    
                    # Calculate clash score specifically
                    clash_score = self._calculate_clash_score(protein, test_pose)
                    
                    # Only accept if both overall score is better AND clash score is below threshold
                    if test_score < best_score and clash_score < clash_threshold:
                        best_pose = copy.deepcopy(test_pose)
                        best_score = test_score
                        improved = True
            
            # Update current pose if improved
            if improved:
                current_pose = copy.deepcopy(best_pose)
                current_score = best_score
                
                # Reduce step size for finer search
                step_size *= 0.9
                angle_step *= 0.9
            else:
                # No improvement found, terminate
                converged = True
                break
            
            if (step + 1) % 10 == 0:
                print(f"  Optimization step {step + 1}, score: {best_score:.2f}")
        
        if converged:
            print(f"Local optimization converged after {step + 1} steps")
        else:
            print(f"Local optimization reached maximum steps ({max_steps})")
        
        print(f"Score improved from {self.scoring_function.score(protein, pose):.2f} to {best_score:.2f}")
        
        return best_pose, best_score

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

    def search(self, protein, ligand):
        """Perform genetic algorithm search."""
        self.protein = protein
        # Determine search space
        if protein.active_site:
            center = protein.active_site['center']
            radius = protein.active_site['radius']
        else:
            # Use protein center of mass
            center = np.mean(protein.xyz, axis=0)
            radius = 15.0  # Arbitrary search radius
        
        print(f"Starting genetic algorithm search with population {self.population_size}")
        if self.perform_local_opt:
            print("Local optimization within GA generations is ENABLED.")
        else:
            print("Local optimization within GA generations is DISABLED.")
        
        # Initialize population
        population = []
        for _ in range(self.population_size):
            # Make a deep copy of the ligand
            pose = copy.deepcopy(ligand)
            
            # Generate random position within sphere
            r = radius * random.random() ** (1.0/3.0)
            theta = random.uniform(0, 2 * np.pi)
            phi = random.uniform(0, np.pi)
            
            x = center[0] + r * np.sin(phi) * np.cos(theta)
            y = center[1] + r * np.sin(phi) * np.sin(theta)
            z = center[2] + r * np.cos(phi)
            
            # Calculate translation vector
            centroid = np.mean(pose.xyz, axis=0)
            translation = np.array([x, y, z]) - centroid
            
            # Apply translation
            pose.translate(translation)
            
            # Generate random rotation
            rotation = Rotation.random()
            rotation_matrix = rotation.as_matrix()
            
            # Apply rotation
            centroid = np.mean(pose.xyz, axis=0)
            pose.translate(-centroid)
            pose.rotate(rotation_matrix)
            pose.translate(centroid)
            
            # Evaluate pose
            score = self.scoring_function.score(protein, pose)
            
            # Store pose
            population.append((pose, score))
        
        # Sort initial population
        population.sort(key=lambda x: x[1])
        best_poses = [population[0]]
        
        # Main GA loop
        for generation in range(self.max_iterations):
            # Select parents (tournament selection)
            parents = self._selection(population)
            
            # Generate offspring
            offspring = self._crossover(parents)
            
            # Mutate offspring
            self._mutation(offspring, radius, center)
                
            # Evaluate offspring
            # Evaluate offspring
            for i, (pose, _) in enumerate(offspring):
                # Apply local optimization ONLY IF self.perform_local_opt is TRUE
                # Use the stored instance variable
                if self.perform_local_opt and i < len(offspring) // 4: # Optimize top 25% if enabled
                    optimized_pose, optimized_score = self._local_optimization(pose, protein)
                    offspring[i] = (optimized_pose, optimized_score)
                else: # Otherwise, just score
                    score = self.scoring_function.score(protein, pose)
                    offspring[i] = (pose, score)
            
            # Combine and select new population
            combined = population + offspring
            combined.sort(key=lambda x: x[1])
            population = combined[:self.population_size]
            
            # Update best solution
            # After this line in GeneticAlgorithm.search:
            if population[0][1] < best_poses[-1][1]:
                best_poses.append(population[0])

            # Insert the progress tracking code:
            if self.output_dir:
                from .utils import save_intermediate_result, update_status
                
                # Save current best pose
                save_intermediate_result(
                    population[0][0],  # Current best pose
                    population[0][1],  # Current best score
                    generation + 1,    # Current iteration
                    self.output_dir, 
                    self.max_iterations
                )
                
                # Update status
                update_status(
                    self.output_dir,
                    current_iteration=generation + 1,
                    best_score=population[0][1],
                    generation=generation + 1
                )
            
            print(f"Generation {generation + 1}, best score: {population[0][1]}")
            
        return best_poses
    
    def _selection(self, population):
        """Tournament selection."""
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
        """Crossover operation."""
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
        Mutation operation for the population.
        
        Parameters:
        -----------
        offspring : list
            List of (pose, score) tuples to mutate
        radius : float
            Radius of search space
        center : array-like
            Center of search space
        """
        for i, (pose, _) in enumerate(offspring):
            self._mutate(pose)  # Call the new _mutate method for each pose

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
            translation = np.random.normal(0, 0.2, 3)  # Reduced from 0.5 to 0.2
            individual.translate(translation)
            
            angle = np.random.normal(0, 0.1)  # Reduced from 0.2 to 0.1 radians (~5.7 degrees)
            axis = np.random.rand(3)
            axis = axis / np.linalg.norm(axis)
            
            rotation = Rotation.from_rotvec(angle * axis)
            rotation_matrix = rotation.as_matrix()
            
            centroid = np.mean(individual.xyz, axis=0)
            individual.translate(-centroid)
            individual.rotate(rotation_matrix)
            individual.translate(centroid)
        
        # Mutate flexible residues if available
        # Note: We need to have protein as a class attribute or pass it as a parameter
        # For now, we'll check if it exists as a class attribute
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


                        