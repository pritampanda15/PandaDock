# search.py
import numpy as np
from scipy.spatial.transform import Rotation
import random
import copy

class DockingSearch:
    """Base class for docking search algorithms."""
    
    def __init__(self, scoring_function, max_iterations=1000):
        """
        Initialize search algorithm.
        
        Parameters:
        -----------
        scoring_function : ScoringFunction
            Scoring function to evaluate poses
        max_iterations : int
            Maximum number of iterations
        """
        self.scoring_function = scoring_function
        self.max_iterations = max_iterations
    
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


class RandomSearch(DockingSearch):
    """Simple random search algorithm."""
    
    def search(self, protein, ligand):
        """Perform random search."""
        best_poses = []
        
        # Determine search space
        if protein.active_site:
            center = protein.active_site['center']
            radius = protein.active_site['radius']
        else:
            # Use protein center of mass
            center = np.mean(protein.xyz, axis=0)
            radius = 15.0  # Arbitrary search radius
        
        print(f"Searching around center {center} with radius {radius}")
        
        for i in range(self.max_iterations):
            # Make a deep copy of the ligand to avoid modifying the original
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
            
            # Apply rotation around the new center
            centroid = np.mean(pose.xyz, axis=0)
            pose.translate(-centroid)
            pose.rotate(rotation_matrix)
            pose.translate(centroid)
            
            # Evaluate pose
            score = self.scoring_function.score(protein, pose)
            
            # Store pose
            best_poses.append((pose, score))
            
            if (i + 1) % 100 == 0:
                print(f"Completed {i + 1} iterations")
        
        # Sort poses by score (lower is better)
        best_poses.sort(key=lambda x: x[1])
        
        return best_poses


class GeneticAlgorithm(DockingSearch):
    """Genetic algorithm for docking search."""
    
    def __init__(self, scoring_function, max_iterations=100, 
                 population_size=50, mutation_rate=0.2):
        """Initialize genetic algorithm."""
        super().__init__(scoring_function, max_iterations)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
    
    def _local_optimization(self, pose, protein):
        """
        Perform local optimization of pose using gradient descent.
        
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
        
        # Make a copy of the pose to avoid modifying the original
        current_pose = copy.deepcopy(pose)
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
                
                if test_score < best_score:
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
                    
                    if test_score < best_score:
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

    def search(self, protein, ligand):
        """Perform genetic algorithm search."""
        # Determine search space
        if protein.active_site:
            center = protein.active_site['center']
            radius = protein.active_site['radius']
        else:
            # Use protein center of mass
            center = np.mean(protein.xyz, axis=0)
            radius = 15.0  # Arbitrary search radius
        
        print(f"Starting genetic algorithm search with population {self.population_size}")
        
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
            for i, (pose, _) in enumerate(offspring):
                score = self.scoring_function.score(protein, pose)
                
                # Apply local optimization to the best individuals
                if i < len(offspring) // 4:  # Optimize top 25% of offspring
                    optimized_pose, optimized_score = self._local_optimization(pose, protein)
                    offspring[i] = (optimized_pose, optimized_score)
                else:
                    offspring[i] = (pose, score)
            
            # Combine and select new population
            combined = population + offspring
            combined.sort(key=lambda x: x[1])
            population = combined[:self.population_size]
            
            # Update best solution
            if population[0][1] < best_poses[-1][1]:
                best_poses.append(population[0])
            
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
        """Mutation operation."""
        for i, (pose, _) in enumerate(offspring):
            if random.random() < self.mutation_rate:
                # Random translation mutation
                translation = np.random.normal(0, radius * 0.1, 3)
                pose.translate(translation)
                
                # Random rotation mutation
                angle = np.random.normal(0, 0.2)  # ~10 degrees std dev
                axis = np.random.rand(3)
                axis = axis / np.linalg.norm(axis)
                
                rotation = Rotation.from_rotvec(angle * axis)
                rotation_matrix = rotation.as_matrix()
                
                centroid = np.mean(pose.xyz, axis=0)
                pose.translate(-centroid)
                pose.rotate(rotation_matrix)
                pose.translate(centroid)

    # Add to search.py, in the GeneticAlgorithm class

    def _mutate(self, individual):
        """
        Mutate an individual with probability mutation_rate.
        Now handles flexible residues too.
        """
        # Original mutation of ligand pose
        super()._mutate(individual)
    
        # Now also mutate flexible residues if available
        if hasattr(self.protein, 'flexible_residues') and self.protein.flexible_residues:
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



                    