"""
PANDADOCK implementation for PandaDock.
A CHARMm based docking algorithm using molecular dynamics for pose generation and refinement.
"""

import numpy as np
import copy
import random
from scipy.spatial.transform import Rotation
from .search import DockingSearch
import time as time
import pathlib as Path
import os

class PANDADOCKAlgorithm(DockingSearch):
    """
    Implementation of PANDADOCK - A CHARMm based docking algorithm.
    
    PANDADOCK uses a rigid receptor and performs molecular dynamics simulations
    with the CHARMm force field to generate and refine ligand poses.
    
    Protocol steps:
    1. Generate ligand conformers using high-temperature molecular dynamics
    2. Generate random orientations within the receptor active site
    3. Run simulated annealing molecular dynamics
    4. Perform final minimization
    5. Score and rank poses
    """
    
    def __init__(self, scoring_function, max_iterations=100, 
                 high_temp=1000, target_temp=300, 
                 num_conformers=10, num_orientations=10,
                 md_steps=1000, minimize_steps=200, 
                 use_grid=True, output_dir=None, grid_spacing=0.375, grid_radius=10.0, grid_center=None):
        self.grid_spacing = grid_spacing  # Add grid_spacing as an attribute
        self.grid_radius = grid_radius  # Add grid_radius as an attribute
        self.grid_center = np.array(grid_center) if grid_center is not None else np.array([0.0, 0.0, 0.0])  # Default grid center

        """
        Initialize PANDADOCK algorithm.
        
        Parameters:
        -----------
        scoring_function : ScoringFunction
            Scoring function to evaluate poses
        max_iterations : int
            Maximum number of iterations
        high_temp : float
            High temperature for MD simulations (K)
        target_temp : float
            Target temperature for cooling (K)
        num_conformers : int
            Number of ligand conformers to generate
        num_orientations : int
            Number of orientations to try for each conformer
        md_steps : int
            Number of MD steps for simulated annealing
        minimize_steps : int
            Number of minimization steps for final refinement
        use_grid : bool
            Whether to use grid-based energy calculations
        """
        super().__init__(scoring_function, max_iterations, output_dir)
        
        self.high_temp = high_temp
        self.target_temp = target_temp
        self.num_conformers = num_conformers
        self.num_orientations = num_orientations
        self.md_steps = md_steps
        self.minimize_steps = minimize_steps
        self.use_grid = use_grid
        self.output_dir = output_dir

    def _save_sphere_pdb(self, center, radius, filename="sphere.pdb"):
        """
        Save a PDB file with a sphere of dummy atoms centered at `center` with radius `radius`.
        
        Parameters:
        -----------
        center : np.ndarray
            3D center of the sphere
        radius : float
            Radius of the sphere
        filename : str
            Output PDB filename
        """
        with open(filename, 'w') as f:
            for i in range(100):  # Sample 100 points on the sphere
                theta = np.random.uniform(0, 2*np.pi)
                phi = np.random.uniform(0, np.pi)
                
                x = center[0] + radius * np.sin(phi) * np.cos(theta)
                y = center[1] + radius * np.sin(phi) * np.sin(theta)
                z = center[2] + radius * np.cos(phi)
                
                f.write(
                    f"ATOM  {i+1:5d}  X   SPH A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          X\n"
                )
            f.write("END\n")

    def search(self, protein, ligand):
        """
        Perform PANDADOCK-based search.
        
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
        print(f"Starting PANDADOCK docking protocol...")
        all_poses = []

        # Step 1: Define center and radius
        if protein.active_site:
            center = protein.active_site['center']
            radius = protein.active_site['radius']
        else:
            center = np.mean(protein.xyz, axis=0)
            radius = 10.0  # Default fallback

        # ✅ Save sphere.pdb
        sphere_filename = os.path.join(self.output_dir, "sphere.pdb") if self.output_dir else "sphere.pdb"
        self._save_sphere_pdb(center, radius, filename=sphere_filename)
        print(f"[INFO] Sphere file saved at {sphere_filename}")
        
        # Step 1: Generate ligand conformers using high-temperature MD
        # Note: This is a simplified approach without actual MD
        conformers = self._generate_conformers(ligand)
        print(f"Generated {len(conformers)} ligand conformers")
        
        # Step 2: Generate random orientations for each conformer
        for i, conf in enumerate(conformers):
            print(f"Processing conformer {i+1}/{len(conformers)}")
            orientations = self._generate_orientations(conf, protein)
            print(f"  Generated {len(orientations)} orientations")
            
            # Step 3: Perform simulated annealing MD for each orientation
            for j, orient in enumerate(orientations):
                refined_pose = self._simulated_annealing(orient, protein)
                
                # Step 4: Final minimization
                minimized_pose = self._final_minimization(refined_pose, protein)
                
                # Score the final pose
                score = self.scoring_function.score(protein, minimized_pose)
                all_poses.append((minimized_pose, score))
                
                # Print progress
                if (j + 1) % 5 == 0 or (j + 1) == len(orientations):
                    print(f"  Processed {j+1}/{len(orientations)} orientations")
        
        # Sort poses by score
        all_poses.sort(key=lambda x: x[1])
        
        # Return the top poses
        return all_poses
    
    def _generate_conformers(self, ligand):
        """
        Generate ligand conformers using high-temperature molecular dynamics.
        
        Parameters:
        -----------
        ligand : Ligand
            Ligand object
        
        Returns:
        --------
        list
            List of ligand conformers
        """
        # Add the original conformer
        conformers = [copy.deepcopy(ligand)]
        
        # If no rotatable bonds, return just the original
        if not hasattr(ligand, 'rotatable_bonds') or not ligand.rotatable_bonds:
            return conformers
        
        # Systematic rotation of rotatable bonds to generate diverse conformers
        for _ in range(self.num_conformers - 1):
            # Create a new conformer
            conf = copy.deepcopy(ligand)
            
            # Rotate each rotatable bond by a random angle
            for bond_idx in range(len(conf.rotatable_bonds)):
                # Random rotation angle (0 to 360 degrees)
                angle = random.uniform(0, 2 * np.pi)
                
                # Apply rotation if the ligand has this method
                if hasattr(conf, 'rotate_bond'):
                    conf.rotate_bond(bond_idx, angle)
            
            conformers.append(conf)
        
        return conformers
    
    def _generate_orientations(self, ligand, protein):
        """
        Generate random orientations of the ligand within the protein active site.
        
        Parameters:
        -----------
        ligand : Ligand
            Ligand conformer
        protein : Protein
            Protein object
        
        Returns:
        --------
        list
            List of oriented ligand poses
        """
        orientations = []
        
        # Get active site center and radius
        if protein.active_site:
            center = protein.active_site['center']
            radius = protein.active_site['radius']
        else:
            # Use protein center of mass
            center = np.mean(protein.xyz, axis=0)
            radius = 15.0  # Default radius
        
        bad_orientations = 0
        max_bad_orientations = self.num_orientations * 10  # Allow 10x the number of desired orientations
        
        while len(orientations) < self.num_orientations and bad_orientations < max_bad_orientations:
            # Create a copy of the ligand
            pose = copy.deepcopy(ligand)
            
            # Translate to active site center first
            centroid = np.mean(pose.xyz, axis=0)
            translation = center - centroid
            pose.translate(translation)
            
            # Apply random rotation
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
            
            # Check if pose is valid (soft energy check)
            is_valid = self._check_pose_validity(pose, protein)
            
            if is_valid:
                orientations.append(pose)
            else:
                bad_orientations += 1
        
        return orientations
    
    def _check_pose_validity(self, pose, protein):
        """
        Check if a pose has acceptable initial energy.
        Uses a "softened" energy function to allow some clashes.
        
        Parameters:
        -----------
        pose : Ligand
            Ligand pose
        protein : Protein
            Protein object
        
        Returns:
        --------
        bool
            True if pose is valid, False otherwise
        """
        # We'll use a clash check as a proxy for soft energy
        if hasattr(self.scoring_function, '_calculate_clashes'):
            clash_score = self.scoring_function._calculate_clashes(protein, pose)
            
            # Allow some clashes, but reject poses with severe clashes
            return clash_score < 10.0
        else:
            # If no clash function, use the full scoring function with a relaxed threshold
            score = self.scoring_function.score(protein, pose)
            
            # The threshold depends on the scoring function, but generally
            # we want to reject only very bad poses
            return score < 100.0  # Arbitrary threshold
    
    def _simulated_annealing(self, pose, protein):
        """
        Perform simulated annealing molecular dynamics.
        
        Parameters:
        -----------
        pose : Ligand
            Initial ligand pose
        protein : Protein
            Protein object
        
        Returns:
        --------
        Ligand
            Refined ligand pose
        """
        # Create a Monte Carlo protocol as an approximation of MD
        current_pose = copy.deepcopy(pose)
        current_score = self.scoring_function.score(protein, current_pose)
        best_pose = copy.deepcopy(current_pose)
        best_score = current_score
        
        # Set up temperature schedule (linear cooling)
        temp_schedule = np.linspace(self.high_temp, self.target_temp, self.md_steps)
        
        # Monte Carlo steps as an approximation of MD
        for step in range(self.md_steps):
            # Get current temperature
            temp = temp_schedule[step]
            # Find the main loop where progress is measured and add:
            if self.output_dir:
                from .utils import save_intermediate_result, update_status
                
                # For MD steps, you might want to save periodically rather than every step
                if step % 100 == 0:  # Save every 10 steps
                    save_intermediate_result(
                        current_pose,
                        current_score,
                        step,
                        self.output_dir,
                        self.md_steps
                    )
                    # Update status for progress tracking
                    update_status(
                        self.output_dir,
                        current_md_step=step,
                        temperature=temp,
                        current_score=current_score,
                        progress=step/self.md_steps
                    )

            # Create a candidate pose with small random movement
            candidate = copy.deepcopy(current_pose)
            
            # Random translation (scaled by temperature)
            scaling = temp / self.high_temp
            max_translation = 0.5 * scaling  # Max 0.5 Å at high temp
            translation = np.random.uniform(-max_translation, max_translation, 3)
            candidate.translate(translation)
            
            # Random rotation (scaled by temperature)
            max_rotation = 0.1 * scaling  # Max 0.1 radians at high temp
            axis = np.random.randn(3)
            axis = axis / np.linalg.norm(axis)
            angle = np.random.uniform(-max_rotation, max_rotation)
            
            rotation = Rotation.from_rotvec(axis * angle)
            
            # Apply rotation around center of mass
            centroid = np.mean(candidate.xyz, axis=0)
            candidate.translate(-centroid)
            candidate.rotate(rotation.as_matrix())
            candidate.translate(centroid)
            
            # Evaluate candidate
            candidate_score = self.scoring_function.score(protein, candidate)
            
            # Metropolis criterion
            delta_score = candidate_score - current_score
            if delta_score < 0 or np.random.random() < np.exp(-delta_score / (0.001987 * temp)):
                current_pose = candidate
                current_score = candidate_score
                
                # Update best if improved
                if current_score < best_score:
                    best_pose = copy.deepcopy(current_pose)
                    best_score = current_score
        
        return best_pose
    
    def _final_minimization(self, pose, protein):
        """
        Perform final energy minimization of the ligand in the rigid receptor.
        
        Parameters:
        -----------
        pose : Ligand
            Ligand pose to minimize
        protein : Protein
            Protein object
        
        Returns:
        --------
        Ligand
            Minimized ligand pose
        """
        # Use a simplified gradient descent approach for minimization
        current_pose = copy.deepcopy(pose)
        current_score = self.scoring_function.score(protein, current_pose)
        
        step_size = 0.1  # Initial step size (Å)
        angle_step = 0.05  # Initial rotation step (radians)
        
        for step in range(self.minimize_steps):
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
            
            for direction in directions:
                test_pose = copy.deepcopy(current_pose)
                test_pose.translate(direction)
                test_score = self.scoring_function.score(protein, test_pose)
                
                if test_score < current_score:
                    current_pose = test_pose
                    current_score = test_score
                    improved = True
                    break
            
            # If no improvement with translation, try rotations
            if not improved:
                axes = [
                    np.array([1, 0, 0]),
                    np.array([0, 1, 0]),
                    np.array([0, 0, 1])
                ]
                
                for axis in axes:
                    for angle in [angle_step, -angle_step]:
                        test_pose = copy.deepcopy(current_pose)
                        
                        # Get centroid
                        centroid = np.mean(test_pose.xyz, axis=0)
                        
                        # Rotate around centroid
                        rotation = Rotation.from_rotvec(axis * angle)
                        test_pose.translate(-centroid)
                        test_pose.rotate(rotation.as_matrix())
                        test_pose.translate(centroid)
                        
                        test_score = self.scoring_function.score(protein, test_pose)
                        
                        if test_score < current_score:
                            current_pose = test_pose
                            current_score = test_score
                            improved = True
                            break
                    
                    if improved:
                        break
            
            # If no improvement, reduce step size
            if not improved:
                step_size *= 0.9
                angle_step *= 0.9
                
                # Stop if step size gets too small
                if step_size < 0.001:
                    break
        
        return current_pose
    
    def _local_optimization(self, pose, protein):
        """
        Public method for local optimization of a pose, to be compatible with other search algorithms.
        
        Parameters:
        -----------
        pose : Ligand
            Ligand pose to optimize
        protein : Protein
            Protein object
        
        Returns:
        --------
        tuple
            (optimized_pose, optimized_score)
        """
        optimized_pose = self._final_minimization(pose, protein)
        optimized_score = self.scoring_function.score(protein, optimized_pose)
        return optimized_pose, optimized_score
