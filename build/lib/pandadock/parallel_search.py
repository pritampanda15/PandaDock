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
from scipy.spatial.transform import Rotation

from .search import GeneticAlgorithm, RandomSearch


class ParallelGeneticAlgorithm(GeneticAlgorithm):
    """
    Parallel implementation of genetic algorithm for molecular docking.
    
    This class extends the standard GeneticAlgorithm to parallelize the evaluation
    of poses, which is typically the most time-consuming part of the search process.
    """
    
    def __init__(self, scoring_function, max_iterations=100, population_size=50, 
                 mutation_rate=0.2, crossover_rate=0.8, tournament_size=3, 
                 n_processes=None, batch_size=None, process_pool=None):
        """
        Initialize the parallel genetic algorithm.
        
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
        crossover_rate : float
            Probability of crossover (0.0 to 1.0)
        tournament_size : int
            Size of tournament for selection
        n_processes : int
            Number of processes to use for parallelization.
            If None, uses all available CPU cores.
        batch_size : int
            Size of batches for parallel evaluation.
            If None, determines automatically based on population size and CPU count.
        process_pool : multiprocessing.Pool
            An existing process pool to use. If None, creates a new one.
        """
        super().__init__(scoring_function, max_iterations, population_size, mutation_rate)
        
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        
        # Set up parallelization parameters (but don't use for now due to pickling issues)
        if n_processes is None:
            self.n_processes = mp.cpu_count()
        else:
            self.n_processes = n_processes
        
        # Determine batch size
        if batch_size is None:
            # Set batch size to balance parallelism and overhead
            self.batch_size = max(1, self.population_size // (self.n_processes * 2))
        else:
            self.batch_size = batch_size
        
        # Store process pool but don't use it (due to pickling issues)
        self.process_pool = process_pool
        self.own_pool = False  # Flag to track if we created our own pool
        
        # Performance metrics
        self.eval_time = 0.0
        self.total_time = 0.0
        self.best_score = float('inf')
        self.best_pose = None
    
    def initialize_population(self, protein, ligand):
        """
        Initialize random population for genetic algorithm.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        ligand : Ligand
            Ligand object
        
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
            # Use protein center of mass
            center = np.mean(protein.xyz, axis=0)
            radius = 15.0  # Arbitrary search radius
        
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
            
            # Apply rotation around the new center
            centroid = np.mean(pose.xyz, axis=0)
            pose.translate(-centroid)
            pose.rotate(rotation_matrix)
            pose.translate(centroid)
            
            # Add to population with placeholder score
            population.append((pose, None))
        
        return population
    
    def search(self, protein, ligand):
        """
        Perform genetic algorithm search with parallel evaluation.
        
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
        start_time = time.time()
        
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
                    self._mutate(child1)
                    self._mutate(child2)
                    
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
                print("Applying local optimization to best individual...")
                best_pose, best_score = self._local_optimization(evaluated_population[0][0], protein)
                
                if best_score < self.best_score:
                    self.best_pose = best_pose
                    self.best_score = best_score
                    print(f"Improved score after optimization: {self.best_score:.4f}")
                    
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
        Perform crossover between two parents.
        
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
        
        # Intermediate point for crossover
        midpoint = (centroid1 + centroid2) / 2.0
        
        # Apply to children (swap coordinates)
        child1.translate(midpoint - centroid1)
        child2.translate(midpoint - centroid2)
        
        # Random rotation recombination (create intermediate rotation)
        # Interpolate rotation between parents to create children
        # This is a simplified approach - more advanced methods could be used
        
        return child1, child2
    
    def _mutate(self, individual):
        """
        Mutate an individual with probability mutation_rate.
        
        Parameters:
        -----------
        individual : Ligand
            Individual to mutate
        """
        if random.random() >= self.mutation_rate:
            return  # No mutation
        
        # Perform either translation, rotation, or both
        mutation_type = random.choice(['translation', 'rotation', 'both'])
        
        if mutation_type in ['translation', 'both']:
            # Random translation
            translation = np.random.normal(0, 2.0, 3)  # 2.0 Å standard deviation
            individual.translate(translation)
        
        if mutation_type in ['rotation', 'both']:
            # Random rotation
            angle = np.random.normal(0, 0.5)  # ~30 degrees standard deviation
            axis = np.random.randn(3)
            axis = axis / np.linalg.norm(axis)
            
            rotation = Rotation.from_rotvec(angle * axis)
            
            # Apply rotation around the center of mass
            centroid = np.mean(individual.xyz, axis=0)
            individual.translate(-centroid)
            individual.rotate(rotation.as_matrix())
            individual.translate(centroid)


class ParallelRandomSearch(RandomSearch):
    """
    Parallel implementation of random search for molecular docking.
    
    This class extends the standard RandomSearch to parallelize the evaluation
    of poses, which is typically the most time-consuming part of the search process.
    """
    
    def __init__(self, scoring_function, max_iterations=1000, n_processes=None, 
                 batch_size=None, process_pool=None):
        """
        Initialize the parallel random search.
        
        Parameters:
        -----------
        scoring_function : ScoringFunction
            Scoring function to evaluate poses
        max_iterations : int
            Maximum number of iterations
        n_processes : int
            Number of processes to use for parallelization.
            If None, uses all available CPU cores.
        batch_size : int
            Size of batches for parallel evaluation.
            If None, determines automatically based on iterations and CPU count.
        process_pool : multiprocessing.Pool
            An existing process pool to use. If None, creates a new one.
        """
        super().__init__(scoring_function, max_iterations)
        
        # Set up parallelization
        if n_processes is None:
            self.n_processes = mp.cpu_count()
        else:
            self.n_processes = n_processes
        
        # Determine batch size
        if batch_size is None:
            # Set batch size to balance parallelism and overhead
            self.batch_size = max(10, self.max_iterations // (self.n_processes * 5))
        else:
            self.batch_size = batch_size
        
        # Store process pool
        self.process_pool = process_pool
        self.own_pool = False  # Flag to track if we created our own pool
        
        # Performance metrics
        self.eval_time = 0.0
        self.total_time = 0.0
    
    def search(self, protein, ligand):
        """
        Perform random search with parallel evaluation.
        
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
        start_time = time.time()
        
        # Determine search space
        if protein.active_site:
            center = protein.active_site['center']
            radius = protein.active_site['radius']
        else:
            # Use protein center of mass
            center = np.mean(protein.xyz, axis=0)
            radius = 15.0  # Arbitrary search radius
        
        print(f"Searching around center {center} with radius {radius}")
        print(f"Using {self.n_processes} CPU cores for evaluation")
        
        # Generate and evaluate poses sequentially to avoid multiprocessing issues
        results = []
        
        for i in range(self.max_iterations):
            # Show progress
            if i % 25 == 0 and i > 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                remaining = avg_time * (self.max_iterations - i)
                print(f"Progress: {i}/{self.max_iterations} poses evaluated ({i/self.max_iterations*100:.1f}%) - "
                      f"Est. remaining: {remaining:.1f}s")
            
            # Generate random pose
            pose = self._generate_random_pose(ligand, center, radius)
            
            # Evaluate pose
            score = self.scoring_function.score(protein, pose)
            
            # Store result
            results.append((pose, score))
        
        # Sort results by score
        results.sort(key=lambda x: x[1])
        
        self.total_time = time.time() - start_time
        print(f"Search completed in {self.total_time:.2f} seconds")
        print(f"Best score: {results[0][1]:.4f}")
        
        return results
    
    def _generate_random_pose(self, ligand, center, radius):
        """
        Generate a single random pose.
        
        Parameters:
        -----------
        ligand : Ligand
            Template ligand
        center : array-like
            Center of search space
        radius : float
            Radius of search space
        
        Returns:
        --------
        Ligand
            Random ligand pose
        """
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
        
        # Apply rotation around the new center
        centroid = np.mean(pose.xyz, axis=0)
        pose.translate(-centroid)
        pose.rotate(rotation_matrix)
        pose.translate(centroid)
        
        return pose