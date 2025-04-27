"""
Advanced search algorithms for PandaDock.
This module provides sophisticated search methods beyond the standard genetic
and random search algorithms, including gradient-based optimizers,
replica exchange methods, and machine learning guided approaches.
"""

import numpy as np
import os
import sys
import copy
import time
import random
from scipy.optimize import minimize
from .search import DockingSearch
from .utils import calculate_rmsd


class GradientBasedSearch(DockingSearch):
    """
    Gradient-based optimization for molecular docking using L-BFGS-B algorithm.
    """
    
    def __init__(self, scoring_function, max_iterations=100, 
                 gradient_step=0.1, convergence_threshold=0.01, output_dir=None):
        """
        Initialize the gradient-based search algorithm.

        Parameters:
        -----------
        scoring_function : ScoringFunction
            Scoring function to evaluate poses.
        max_iterations : int, optional
            Maximum number of iterations (default is 100).
        gradient_step : float, optional
            Step size for gradient calculation (default is 0.1).
        convergence_threshold : float, optional
            Threshold to determine convergence (default is 0.01).
        output_dir : str, optional
            Directory to save output files.
        """

        super().__init__(scoring_function, max_iterations, output_dir)
        self.gradient_step = gradient_step
        self.convergence_threshold = convergence_threshold
        self.output_dir = output_dir             
        
    def _calculate_gradient(self, protein, pose, delta=0.1, center=None, current_radius=None):
        """
        Calculate numerical gradient for a pose using finite differences.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        pose : Ligand
            Ligand pose
        delta : float
            Step size for finite difference calculation
            
        Returns:
        --------
        numpy.ndarray
            Gradient vector [tx, ty, tz, rx, ry, rz]
        """
        # Make a copy of the pose to avoid modifying the original
        base_pose = copy.deepcopy(pose)

        
        # Calculate base score
        base_score = self.scoring_function.score(protein, base_pose)
        
        # Initialize gradient vector
        gradient = np.zeros(6)  # [tx, ty, tz, rx, ry, rz]
        
        # Calculate translational gradients (x, y, z)
        for i in range(3):
            # Forward step
            test_pose = copy.deepcopy(base_pose)
            translation = np.zeros(3)
            translation[i] = delta
            test_pose.translate(translation)
            forward_score = self.scoring_function.score(protein, test_pose)
            
            # Backward step
            test_pose = copy.deepcopy(base_pose)
            translation = np.zeros(3)
            translation[i] = -delta
            test_pose.translate(translation)
            backward_score = self.scoring_function.score(protein, test_pose)
            
            # Central difference
            gradient[i] = (forward_score - backward_score) / (2 * delta)
        
        # Calculate rotational gradients (rx, ry, rz)
        from scipy.spatial.transform import Rotation
        for i in range(3):
            # Forward rotation
            test_pose = copy.deepcopy(base_pose)
            axis = np.zeros(3)
            axis[i] = 1.0
            rotation = Rotation.from_rotvec(axis * delta)
            
            # Apply rotation around center of mass
            centroid = np.mean(test_pose.xyz, axis=0)
            test_pose.translate(-centroid)
            test_pose.rotate(rotation.as_matrix())
            test_pose.translate(centroid)
            
            forward_score = self.scoring_function.score(protein, test_pose)
            
            # Backward rotation
            test_pose = copy.deepcopy(base_pose)
            rotation = Rotation.from_rotvec(axis * -delta)
            
            # Apply rotation around center of mass
            centroid = np.mean(test_pose.xyz, axis=0)
            test_pose.translate(-centroid)
            test_pose.rotate(rotation.as_matrix())
            test_pose.translate(centroid)
            
            backward_score = self.scoring_function.score(protein, test_pose)
            
            # Central difference
            gradient[i+3] = (forward_score - backward_score) / (2 * delta)
        
        return gradient
    
    def search(self, protein, ligand, current_radius=None):
        """
        Perform gradient-based search using L-BFGS-B algorithm.
        
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
        print(f"Starting gradient-based search with max {self.max_iterations} iterations")
        start_time = time.time()
        
        # Set up intermediate results directory if output_dir is specified
        if self.output_dir:
            gradient_dir = os.path.join(self.output_dir, "intermediate", "gradient_based")
            os.makedirs(gradient_dir, exist_ok=True)


        # Create multiple starting poses to avoid local minima
        n_starting_poses = 10
        starting_poses = []
        
        # Determine search space
        if protein.active_site:
            center = protein.active_site['center']
            radius = protein.active_site['radius']
        else:
            # Use protein center of mass
            center = np.mean(protein.xyz, axis=0)
            radius = 10.0  # Arbitrary search radius
            self.initialize_grid_points(center)

            if current_radius is None:
                current_radius = radius

            # Initialize consecutive failures counter
            consecutive_failures = 0
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
        
        # Generate random starting poses
        for i in range(n_starting_poses):
            pose = copy.deepcopy(ligand)
            from .utils import generate_valid_random_pose

            pose = generate_valid_random_pose(protein, pose, center, current_radius)
            if pose is None:
                consecutive_failures += 1
                if consecutive_failures > 10:
                    current_radius += 1.0
                    consecutive_failures = 0
                    print(f"⚡ Expanding search radius to {current_radius:.2f} Å")
                continue
            starting_poses.append(pose)
            
            # Random position within sphere
            r = radius * random.betavariate(2, 5) ** (1/3)
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            
            x = center[0] + r * np.sin(phi) * np.cos(theta)
            y = center[1] + r * np.sin(phi) * np.sin(theta)
            z = center[2] + r * np.cos(phi)
            
            # Calculate translation vector
            centroid = np.mean(pose.xyz, axis=0)
            translation = np.array([x, y, z]) - centroid
            
            # Apply translation
            pose.translate(translation)
            
            # Random rotation
            from scipy.spatial.transform import Rotation
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

            
            # Add to starting poses
            starting_poses.append(pose)
        
        # Optimize each starting pose
        all_results = []
        
        for i, start_pose in enumerate(starting_poses):
            print(f"Optimizing starting pose {i+1}/{n_starting_poses}")
            

            # Create optimization tracking directory
            if self.output_dir:
                opt_dir = os.path.join(gradient_dir, f"pose_{i+1}_optimization")
                os.makedirs(opt_dir, exist_ok=True)


            # Get initial pose parameters
            centroid = np.mean(start_pose.xyz, axis=0)
            initial_params = np.zeros(6)  # [tx, ty, tz, rx, ry, rz]
            initial_params[:3] = centroid
            
            # Make a copy for optimization
            pose = copy.deepcopy(start_pose)
            
            """
            This code defines a callback function used to track the progress of an optimization process for refining a molecular pose. The callback function is invoked during each iteration of the optimization algorithm, allowing the program to evaluate and adjust the pose parameters iteratively.

            ### Key Steps in the Code:

            1. **Iteration Counter**: 
            A list `iteration_count` is initialized with a single element `[0]`. This allows the counter to be modified within the nested `callback` function, as lists are mutable in Python.

            2. **Pose Adjustment**:
            - A deep copy of the starting pose (`start_pose`) is created to avoid modifying the original pose directly.
            - The translation vector is calculated as the difference between the current parameters (`params[:3]`) and the centroid of the starting pose. This translation is applied to the copied pose using the `translate` method.
            - If the rotation vector (`params[3:6]`) has a non-zero magnitude, a rotation is applied. The rotation vector is normalized to determine the axis of rotation, and a rotation matrix is computed using `scipy.spatial.transform.Rotation`. The pose is then rotated around its center of mass by translating it to the origin, applying the rotation, and translating it back.

            3. **Scoring**:
            The adjusted pose is evaluated using a scoring function (`self.scoring_function.score`), which likely measures how well the pose fits a target structure or satisfies certain constraints.

            4. **Saving Intermediate Results**:
            If an output directory (`self.output_dir`) is specified, the pose is saved as a PDB file every 5 iterations or during the first iteration. The filename includes the iteration number and the current score for traceability.

            5. **Progress Reporting**:
            The iteration counter is incremented after each call to the callback. Additionally, progress is printed to the console every 5 iterations, showing the current iteration number and score.

            ### Purpose:
            This callback function is designed to facilitate optimization by:
            - Tracking and saving intermediate results for analysis.
            - Providing feedback on the optimization progress.
            - Ensuring that the pose adjustments (translation and rotation) are applied systematically and correctly.

            This approach is particularly useful in computational biology or structural modeling tasks, where iterative refinement of molecular poses is common.
            6. **Optimization**:
            The `scipy.optimize.minimize` function is used to perform the optimization, applying the callback function as the `callback` argument.
            7. **Final Pose**:
            The final optimized pose is stored in the `final_pose` variable.
            8. **Results**:
            The final pose, score, and iteration count are added to the `all_results` list, which is returned at the end of the function.
            9. **Output**:
            The final pose, score, and iteration count are returned as a tuple.
            10. **Sorting**:
            The `all_results` list is sorted in descending order based on the scores.
            11. **Summary**:
            The function prints a summary of the optimization process, including the final pose, score, and iteration count.
            """
            # Define callback function to track progress
            from scipy.optimize import OptimizeResult
            from scipy.optimize import OptimizeWarning

            iteration_count = [0]  # Using a list to allow modification in nested function
            
            def callback(params):
                # Create a test pose with current parameters
                test_pose = copy.deepcopy(start_pose)
                
                # Extract translation and rotation
                translation = params[:3] - centroid
                rotation_vec = params[3:6]
                
                # Apply translation
                test_pose.translate(translation)
                
                # Apply rotation if rotation vector is not zero
                rot_magnitude = np.linalg.norm(rotation_vec)
                if rot_magnitude > 1e-6:
                    # Normalize rotation vector
                    axis = rotation_vec / rot_magnitude
                    
                    # Create rotation matrix
                    rotation = Rotation.from_rotvec(axis * rot_magnitude)
                    rotation_matrix = rotation.as_matrix()
                    
                    # Apply rotation around center of mass
                    new_centroid = np.mean(test_pose.xyz, axis=0)
                    test_pose.translate(-new_centroid)
                    test_pose.rotate(rotation_matrix)
                    test_pose.translate(new_centroid)
                
                # Evaluate score
                current_score = self.scoring_function.score(protein, test_pose)
                
                # Save intermediate pose if output_dir is specified and every few iterations
                if self.output_dir and (iteration_count[0] % 5 == 0 or iteration_count[0] == 0):
                    self._save_pose_to_pdb(
                        test_pose,
                        os.path.join(opt_dir, f"iter_{iteration_count[0]}_score_{current_score:.2f}.pdb"),
                        score=current_score,
                        remark=f"Gradient optimization iteration {iteration_count[0]}"
                    )
                
                # Increment iteration counter
                iteration_count[0] += 1
                
                # Print progress occasionally
                if iteration_count[0] % 5 == 0:
                    print(f"  Iteration {iteration_count[0]}, score: {current_score:.4f}")

            # Define objective function for L-BFGS-B
            def objective(params):
                # Create a new pose for evaluation
                test_pose = copy.deepcopy(pose)
                
                # Extract translation and rotation
                translation = params[:3] - centroid
                rotation_vec = params[3:6]
                
                # Apply translation
                test_pose.translate(translation)
                
                # Apply rotation if rotation vector is not zero
                rot_magnitude = np.linalg.norm(rotation_vec)
                if rot_magnitude > 1e-6:
                    from scipy.spatial.transform import Rotation
                    
                    # Normalize rotation vector
                    axis = rotation_vec / rot_magnitude
                    
                    # Create rotation matrix
                    rotation = Rotation.from_rotvec(axis * rot_magnitude)
                    rotation_matrix = rotation.as_matrix()
                    
                    # Apply rotation around center of mass
                    new_centroid = np.mean(test_pose.xyz, axis=0)
                    test_pose.translate(-new_centroid)
                    test_pose.rotate(rotation_matrix)
                    test_pose.translate(new_centroid)
                
                # Evaluate score
                score = self.scoring_function.score(protein, test_pose)
                return score
            
            # Define gradient function
            def gradient(params):
                # Create a new pose with the given parameters
                test_pose = copy.deepcopy(pose)
                
                # Extract translation and rotation
                translation = params[:3] - centroid
                rotation_vec = params[3:6]
                
                # Apply translation
                test_pose.translate(translation)
                
                # Apply rotation if rotation vector is not zero
                rot_magnitude = np.linalg.norm(rotation_vec)
                if rot_magnitude > 1e-6:
                    from scipy.spatial.transform import Rotation
                    
                    # Normalize rotation vector
                    axis = rotation_vec / rot_magnitude
                    
                    # Create rotation matrix
                    rotation = Rotation.from_rotvec(axis * rot_magnitude)
                    rotation_matrix = rotation.as_matrix()
                    
                    # Apply rotation around center of mass
                    new_centroid = np.mean(test_pose.xyz, axis=0)
                    test_pose.translate(-new_centroid)
                    test_pose.rotate(rotation_matrix)
                    test_pose.translate(new_centroid)
                
                # Calculate numerical gradient
                return self._calculate_gradient(protein, test_pose, delta=self.gradient_step)
            
            # Run L-BFGS-B optimization
            result = minimize(
                objective,
                initial_params,
                method='L-BFGS-B',
                jac=gradient,
                options={
                    'maxiter': self.max_iterations,
                    'ftol': self.convergence_threshold,
                    'disp': True
                }
            )
            
            # Create optimized pose
            optimized_pose = copy.deepcopy(pose)
            
            # Apply optimal transformation
            final_params = result.x
            translation = final_params[:3] - centroid
            rotation_vec = final_params[3:6]
            
            # Apply translation
            optimized_pose.translate(translation)
            
            # Apply rotation if rotation vector is not zero
            rot_magnitude = np.linalg.norm(rotation_vec)
            if rot_magnitude > 1e-6:
                from scipy.spatial.transform import Rotation
                
                # Normalize rotation vector
                axis = rotation_vec / rot_magnitude
                
                # Create rotation matrix
                rotation = Rotation.from_rotvec(axis * rot_magnitude)
                rotation_matrix = rotation.as_matrix()
                
                # Apply rotation around center of mass
                new_centroid = np.mean(optimized_pose.xyz, axis=0)
                optimized_pose.translate(-new_centroid)
                optimized_pose.rotate(rotation_matrix)
                optimized_pose.translate(new_centroid)
            
            # Evaluate final score
            final_score = self.scoring_function.score(protein, optimized_pose)
            
            # Add to results
            all_results.append((optimized_pose, final_score))
            print(f"  Optimization completed: score = {final_score:.4f}")
        
        # Sort results by score
        all_results.sort(key=lambda x: x[1])
        
        elapsed_time = time.time() - start_time
        print(f"Gradient-based search completed in {elapsed_time:.2f} seconds")
        print(f"Best score: {all_results[0][1]:.4f}")
        
        return all_results


class ReplicaExchangeDocking(DockingSearch):
    """
    Replica Exchange Monte Carlo for enhanced sampling of docking poses.
    Maintains multiple replicas at different temperatures and exchanges them.
    """
    
    def __init__(self, scoring_function, n_replicas=4, 
                 temperatures=None, exchange_steps=10, steps_per_exchange=100, output_dir=None):
        super().__init__(scoring_function, steps_per_exchange * exchange_steps, output_dir)
        self.n_replicas = n_replicas
        self.output_dir = output_dir
        # Set up temperatures if not provided
        if temperatures is None:
            # Geometric progression of temperatures
            base_temp = 300.0  # K
            max_temp = 1200.0  # K
            self.temperatures = np.geomspace(base_temp, max_temp, n_replicas)
        else:
            self.temperatures = np.array(temperatures)
            if len(self.temperatures) != n_replicas:
                raise ValueError(f"Number of temperatures ({len(self.temperatures)}) "
                                f"must match number of replicas ({n_replicas})")
        
        self.exchange_steps = exchange_steps
        self.steps_per_exchange = steps_per_exchange
        
        # Physical constants
        self.k_boltzmann = 1.9872e-3  # kcal/(mol·K)
    
    def search(self, protein, ligand, current_radius=None):
        """
        Perform replica exchange Monte Carlo docking.
        
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
        print(f"Starting Replica Exchange Monte Carlo with {self.n_replicas} replicas")
        print(f"Temperatures: {', '.join([f'{t:.1f}K' for t in self.temperatures])}")
        print(f"Exchange steps: {self.exchange_steps}, Steps per exchange: {self.steps_per_exchange}")
        
        # Determine search space
        if protein.active_site:
            center = protein.active_site['center']
            radius = protein.active_site['radius']
        else:
            # Use protein center of mass
            center = np.mean(protein.xyz, axis=0)
            radius = 15.0  # Arbitrary search radius
            self.initialize_grid_points(center)

            if current_radius is None:
                current_radius = radius

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

        # Save sphere grid
        from .utils import save_sphere_grid
        import pathlib as Path
        self.initialize_grid_points(center)
        sphere_path = Path(self.output_dir) / "sphere.pdb"
        save_sphere_grid(sphere_path, self.grid_points)
        
        # Initialize replicas with random poses
        replicas = []
        consecutive_failures = 0

        for i in range(self.n_replicas):
            pose = copy.deepcopy(ligand)
            pose = self.initialize_pose(protein, pose, center, current_radius)
            if pose is None:
                consecutive_failures += 1
                if consecutive_failures > 10:
                    current_radius += 1.0
                    consecutive_failures = 0
                    print(f"\u26a1 Expanding search radius to {current_radius:.2f} Å!")
                continue
            score = self.scoring_function.score(protein, pose)
            replicas.append({
                'pose': pose,
                'score': score,
                'temperature': self.temperatures[i],
                'history': [(pose, score)]
            })
            print(f"Initialized replica {i+1} with score {score:.4f} at {self.temperatures[i]:.1f}K")

        best_pose = None
        best_score = float('inf')

        for exchange_step in range(self.exchange_steps):
            print(f"Exchange step {exchange_step+1}/{self.exchange_steps}")

            for replica in replicas:
                for _ in range(self.steps_per_exchange):
                    candidate_pose = copy.deepcopy(replica['pose'])
                    max_translation = 2.0 * (replica['temperature'] / self.temperatures[-1])
                    translation = np.random.uniform(-max_translation, max_translation, 3)
                    candidate_pose.translate(translation)

                    max_rotation = 0.3 * (replica['temperature'] / self.temperatures[-1])
                    axis = np.random.randn(3)
                    axis /= np.linalg.norm(axis)
                    angle = np.random.uniform(-max_rotation, max_rotation)
                    rotation = Rotation.from_rotvec(axis * angle)
                    centroid = np.mean(candidate_pose.xyz, axis=0)
                    candidate_pose.translate(-centroid)
                    candidate_pose.rotate(rotation.as_matrix())
                    candidate_pose.translate(centroid)

                    candidate_score = self.scoring_function.score(protein, candidate_pose)

                    delta_score = candidate_score - replica['score']
                    if delta_score <= 0 or np.random.random() < np.exp(-delta_score / (self.k_boltzmann * replica['temperature'])):
                        replica['pose'] = candidate_pose
                        replica['score'] = candidate_score
                        replica['history'].append((candidate_pose, candidate_score))

                        if candidate_score < best_score:
                            best_score = candidate_score
                            best_pose = candidate_pose

            for i in range(self.n_replicas - 1):
                r1, r2 = i, i+1
                delta = (1.0 / (self.k_boltzmann * replicas[r1]['temperature']) - 1.0 / (self.k_boltzmann * replicas[r2]['temperature'])) * (replicas[r2]['score'] - replicas[r1]['score'])
                if delta <= 0 or np.random.random() < np.exp(-delta):
                    replicas[r1]['pose'], replicas[r2]['pose'] = replicas[r2]['pose'], replicas[r1]['pose']
                    replicas[r1]['score'], replicas[r2]['score'] = replicas[r2]['score'], replicas[r1]['score']

        all_poses = []
        for replica in replicas:
            all_poses.extend(replica['history'])

        all_poses.sort(key=lambda x: x[1])
        elapsed_time = time.time() - start_time
        print(f"Replica Exchange search completed in {elapsed_time:.2f} seconds")
        print(f"Best score: {all_poses[0][1]:.4f}")

        return all_poses



class MLGuidedSearch(DockingSearch):
    """
    Machine learning guided search that uses a surrogate model to guide docking.
    """
    
    def __init__(self, scoring_function, max_iterations=100, 
                 surrogate_model_type='rf', exploitation_factor=0.8, output_dir=None):
        """
        Initialize ML-guided search algorithm.
        
        Parameters:
        -----------
        scoring_function : ScoringFunction
            Scoring function to evaluate poses
        max_iterations : int
            Maximum number of iterations
        surrogate_model_type : str
            Type of surrogate model ('rf', 'gp', or 'nn')
        exploitation_factor : float
            Factor controlling exploitation vs exploration (0.0-1.0)
        output_dir : str, optional
            Directory to save output files
        """
        super().__init__(scoring_function, max_iterations, output_dir)
        self.surrogate_model_type = surrogate_model_type
        self.exploitation_factor = exploitation_factor
        self.ml_model = None
        self.feature_scaler = None
        self.output_dir = output_dir
        
    def _extract_features(self, protein, pose):
        """
        Extract features from a protein-ligand pose for ML prediction.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        pose : Ligand
            Ligand pose
        
        Returns:
        --------
        numpy.ndarray
            Feature vector
        """
        # Calculate centroid and principal axes
        centroid = np.mean(pose.xyz, axis=0)
        
        # Calculate distance to protein center or active site
        if protein.active_site:
            site_center = protein.active_site['center']
            
        else:
            site_center = np.mean(protein.xyz, axis=0)
        
        
        distance_to_center = np.linalg.norm(centroid - site_center)
        
        # Calculate principal components of pose
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            pca.fit(pose.xyz - centroid)
            principal_axes = pca.components_
            variances = pca.explained_variance_
        except:
            # Fallback if sklearn not available
            cov_matrix = np.cov((pose.xyz - centroid).T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            # Sort by eigenvalue in descending order
            idx = eigenvalues.argsort()[::-1]
            variances = eigenvalues[idx]
            principal_axes = eigenvectors[:, idx].T
        
        # Calculate alignment with protein principal axes
        try:
            protein_pca = PCA(n_components=3)
            if protein.active_site and 'atoms' in protein.active_site:
                protein_coords = np.array([atom['coords'] for atom in protein.active_site['atoms']])
            else:
                protein_coords = protein.xyz
            protein_centroid = np.mean(protein_coords, axis=0)
            protein_pca.fit(protein_coords - protein_centroid)
            protein_axes = protein_pca.components_
            
            # Calculate alignment scores
            alignments = np.abs([np.dot(axis1, axis2) for axis1 in principal_axes for axis2 in protein_axes])
        except:
            alignments = np.zeros(9)  # Fallback
        
        # Combine features
        features = np.concatenate([
            [distance_to_center],  # Distance to center
            centroid - site_center,  # Relative position (3)
            variances,  # Shape descriptors (3)
            alignments,  # Alignment with protein (9)
        ])
        
        return features
    
    def _train_surrogate_model(self, features, scores):
        """
        Train a surrogate model based on existing samples and scores.
        
        Parameters:
        -----------
        features : list
            List of feature vectors
        scores : list
            Corresponding scores
        
        Returns:
        --------
        object
            Trained surrogate model
        """
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(scores)
        
        try:
            # Scale features
            from sklearn.preprocessing import StandardScaler
            self.feature_scaler = StandardScaler()
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # Train surrogate model based on type
            if self.surrogate_model_type == 'rf':
                # Random Forest
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_scaled, y)
                
            elif self.surrogate_model_type == 'gp':
                # Gaussian Process
                from sklearn.gaussian_process import GaussianProcessRegressor
                from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
                kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
                model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
                model.fit(X_scaled, y)
                
            elif self.surrogate_model_type == 'nn':
                # Neural Network
                from sklearn.neural_network import MLPRegressor
                model = MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42)
                model.fit(X_scaled, y)
                
            else:
                raise ValueError(f"Unknown surrogate model type: {self.surrogate_model_type}")
                
            return model
            
        except ImportError:
            print("Warning: scikit-learn not available. Using simple linear regression.")
            
            # Simple linear regression as fallback
            from scipy.stats import linregress
            
            # Use single feature (distance to center) if multiple not available
            if X.shape[1] > 1:
                X_simplified = X[:, 0]  # Use first feature
            else:
                X_simplified = X
                
            slope, intercept, r_value, p_value, std_err = linregress(X_simplified, y)
            
            # Create a simple model object
            class SimpleModel:
                def __init__(self, slope, intercept):
                    self.slope = slope
                    self.intercept = intercept
                    
                def predict(self, X):
                    if X.ndim > 1:
                        X_simplified = X[:, 0]
                    else:
                        X_simplified = X
                    return self.slope * X_simplified + self.intercept
                    
            return SimpleModel(slope, intercept)
    
    def search(self, protein, ligand, current_radius=None):
        """
        Perform ML-guided search with clash detection and adaptive radius expansion.
        """
        print(f"Starting ML-guided search with {self.max_iterations} iterations")
        print(f"Using {self.surrogate_model_type} surrogate model with exploitation factor {self.exploitation_factor}")

        start_time = time.time()

        if protein.active_site:
            center = protein.active_site['center']
            radius = protein.active_site['radius']
        else:
            center = np.mean(protein.xyz, axis=0)
            radius = 15.0
            if current_radius is None:
                current_radius = radius

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

        if current_radius is None:
            current_radius = radius

        from .utils import generate_valid_random_pose

        # Track failures
        consecutive_failures = 0

        # === Step 1: Initial sampling ===
        n_initial_samples = min(20, self.max_iterations // 5)
        print(f"Generating {n_initial_samples} initial samples for model training")

        initial_poses = []
        initial_features = []
        initial_scores = []

        while len(initial_poses) < n_initial_samples:
            pose = copy.deepcopy(ligand)
            pose = generate_valid_random_pose(protein, pose, center, current_radius)
            if pose is None:
                consecutive_failures += 1
                if consecutive_failures > 10:
                    current_radius += 1.0
                    consecutive_failures = 0
                    print(f"⚡ Expanding search radius to {current_radius:.2f} Å")
                continue

            score = self.scoring_function.score(protein, pose)
            features = self._extract_features(protein, pose)

            initial_poses.append((pose, score))
            initial_features.append(features)
            initial_scores.append(score)

            print(f"  Initial sample {len(initial_poses)}/{n_initial_samples}: score = {score:.4f}")

        # Train initial surrogate model
        self.ml_model = self._train_surrogate_model(initial_features, initial_scores)
        print(f"Trained initial surrogate model on {len(initial_poses)} samples")

        all_poses = initial_poses.copy()
        all_features = initial_features.copy()
        all_scores = initial_scores.copy()

        # === Step 2: Main optimization loop ===
        remaining_iterations = self.max_iterations - n_initial_samples

        for iteration in range(remaining_iterations):
            print(f"Iteration {iteration+1}/{remaining_iterations}")

            if np.random.random() < self.exploitation_factor:
                # === Exploitation phase ===
                print("  Exploitation phase (using surrogate model)")

                candidates = []
                while len(candidates) < 10:
                    pose = copy.deepcopy(ligand)
                    pose = generate_valid_random_pose(protein, pose, center, current_radius)
                    if pose is None:
                        consecutive_failures += 1
                        if consecutive_failures > 10:
                            current_radius += 1.0
                            consecutive_failures = 0
                            print(f"⚡ Expanding search radius to {current_radius:.2f} Å")
                        continue

                    features = self._extract_features(protein, pose)
                    if self.feature_scaler:
                        features_scaled = self.feature_scaler.transform(features.reshape(1, -1))
                        predicted_score = self.ml_model.predict(features_scaled)[0]
                    else:
                        predicted_score = self.ml_model.predict(features.reshape(1, -1))[0]

                    candidates.append((pose, features, predicted_score))

                # Select best predicted pose
                candidates.sort(key=lambda x: x[2])
                selected_pose, selected_features, _ = candidates[0]
                true_score = self.scoring_function.score(protein, selected_pose)
                print(f"  Selected candidate: predicted = {candidates[0][2]:.4f}, actual = {true_score:.4f}")

            else:
                # === Exploration phase ===
                print("  Exploration phase (random sampling)")

                while True:
                    pose = copy.deepcopy(ligand)
                    pose = generate_valid_random_pose(protein, pose, center, current_radius)
                    if pose is not None:
                        break
                    consecutive_failures += 1
                    if consecutive_failures > 10:
                        current_radius += 1.0
                        consecutive_failures = 0
                        print(f"⚡ Expanding search radius to {current_radius:.2f} Å")

                selected_pose = pose
                true_score = self.scoring_function.score(protein, selected_pose)
                selected_features = self._extract_features(protein, selected_pose)
                print(f"  Random exploration pose: score = {true_score:.4f}")

            # Add selected pose
            all_poses.append((selected_pose, true_score))
            all_features.append(selected_features)
            all_scores.append(true_score)

            # Retrain surrogate every 5 iterations
            if (iteration + 1) % 5 == 0 or iteration == remaining_iterations - 1:
                print("  Retraining surrogate model")
                self.ml_model = self._train_surrogate_model(all_features, all_scores)

        # === Step 3: Finalize ===
        all_poses.sort(key=lambda x: x[1])

        elapsed_time = time.time() - start_time
        print(f"ML-guided search completed in {elapsed_time:.2f} seconds")
        print(f"Best score: {all_poses[0][1]:.4f}")

        return all_poses



class FragmentBasedDocking(DockingSearch):
    """
    Incremental construction of ligand binding poses through fragments.
    """
    
    def __init__(self, scoring_function, fragment_min_size=5, 
                 growth_steps=3, poses_per_fragment=10, output_dir=None):
        super().__init__(scoring_function, poses_per_fragment * growth_steps, output_dir)
        self.fragment_min_size = fragment_min_size
        self.growth_steps = growth_steps
        self.poses_per_fragment = poses_per_fragment
        self.output_dir = output_dir
    def _decompose_ligand(self, ligand):
        """
        Decompose ligand into fragments for incremental docking.
        
        Parameters:
        -----------
        ligand : Ligand
            Ligand object
        
        Returns:
        --------
        list
            List of fragments as (atom_indices, bond_indices) tuples
        """
        # This is a simplified implementation
        # A real implementation would use chemoinformatics tools like RDKit
        
        # Try to use RDKit if available
        try:
            import rdkit
            from rdkit import Chem
            from rdkit.Chem import BRICS
            
            # Convert ligand to RDKit molecule (this is a simplified approach)
            mol_block = self._ligand_to_molblock(ligand)
            mol = Chem.MolFromMolBlock(mol_block)
            
            if mol is not None:
                # Use BRICS decomposition
                fragments = list(BRICS.BRICSDecompose(mol))
                
                # Convert fragments back to atom and bond indices
                result = []
                for frag in fragments:
                    frag_mol = Chem.MolFromSmiles(frag)
                    if frag_mol and frag_mol.GetNumAtoms() >= self.fragment_min_size:
                        # This is simplified - in practice, you'd need to match atoms and bonds
                        atom_indices = list(range(frag_mol.GetNumAtoms()))
                        bond_indices = list(range(frag_mol.GetNumBonds()))
                        result.append((atom_indices, bond_indices))
                
                if result:
                    return result
                
                # Fall back to simple decomposition if BRICS fails
                print("BRICS decomposition failed. Using simplified approach.")
        
        except ImportError:
            print("RDKit not available. Using simplified fragmentation approach.")
        
        # Simple decomposition: split into roughly equal parts
        n_atoms = len(ligand.atoms)
        n_fragments = max(2, n_atoms // self.fragment_min_size)
        
        atoms_per_fragment = n_atoms // n_fragments
        
        fragments = []
        for i in range(n_fragments):
            start_idx = i * atoms_per_fragment
            end_idx = start_idx + atoms_per_fragment if i < n_fragments - 1 else n_atoms
            
            # Get atom indices for this fragment
            atom_indices = list(range(start_idx, end_idx))
            
            # Find bonds within this fragment
            bond_indices = []
            for j, bond in enumerate(ligand.bonds):
                begin_atom = bond['begin_atom_idx']
                end_atom = bond['end_atom_idx']
                if begin_atom in atom_indices and end_atom in atom_indices:
                    bond_indices.append(j)
            
            fragments.append((atom_indices, bond_indices))
        
        return fragments
    
    def _ligand_to_molblock(self, ligand):
        """
        Convert ligand to MOL block format for RDKit.
        
        Parameters:
        -----------
        ligand : Ligand
            Ligand object
        
        Returns:
        --------
        str
            MOL block string
        """
        # This is a very simplified MOL block generator
        mol_block = []
        
        # Header
        mol_block.append("Ligand")
        mol_block.append("  PandaDock")
        mol_block.append("")
        
        # Counts line
        n_atoms = len(ligand.atoms)
        n_bonds = len(ligand.bonds)
        mol_block.append(f"{n_atoms:3d}{n_bonds:3d}  0  0  0  0  0  0  0  0999 V2000")
        
        # Atoms
        for atom in ligand.atoms:
            coords = atom['coords']
            symbol = atom.get('symbol', 'C')
            mol_block.append(f"{coords[0]:10.4f}{coords[1]:10.4f}{coords[2]:10.4f} {symbol:<3}  0  0  0  0  0  0  0  0  0  0  0  0")
        
        # Bonds
        for bond in ligand.bonds:
            a1 = bond['begin_atom_idx'] + 1  # 1-based indexing in MOL
            a2 = bond['end_atom_idx'] + 1
            type_num = 1  # Assume single bond
            mol_block.append(f"{a1:3d}{a2:3d}{type_num:3d}  0  0  0  0")
        
        # Terminator
        mol_block.append("M  END")
        
        return "\n".join(mol_block)
    
    def _extract_fragment(self, ligand, atom_indices, bond_indices):
        """
        Extract a fragment from the ligand.
        
        Parameters:
        -----------
        ligand : Ligand
            Ligand object
        atom_indices : list
            List of atom indices in the fragment
        bond_indices : list
            List of bond indices in the fragment
        
        Returns:
        --------
        dict
            Fragment dictionary with atoms, bonds, and original indices
        """
        fragment = {
            'atoms': [ligand.atoms[i] for i in atom_indices],
            'bonds': [ligand.bonds[i] for i in bond_indices],
            'atom_indices': atom_indices,
            'bond_indices': bond_indices,
            'xyz': ligand.xyz[atom_indices]
        }
        
        return fragment
    
    def _dock_fragment(self, protein, fragment, center, radius):
        """
        Dock a fragment to the protein.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        fragment : dict
            Fragment dictionary
        center : numpy.ndarray
            Center of search space
        radius : float
            Radius of search space
        
        Returns:
        --------
        list
            List of (pose, score) tuples for the fragment
        """
        # Create a simple scoring function for fragments
        # This is a simplified version of the main scoring function
        def fragment_score(frag_pose):
            score = 0
            
            # Calculate potential with protein atoms
            for frag_atom in frag_pose['atoms']:
                frag_coords = frag_atom['coords']
                frag_element = frag_atom.get('symbol', 'C')
                
                # Get appropriate protein atoms
                if protein.active_site and 'atoms' in protein.active_site:
                    protein_atoms = protein.active_site['atoms']
                else:
                    protein_atoms = protein.atoms
                
                for prot_atom in protein_atoms:
                    prot_coords = prot_atom['coords']
                    prot_element = prot_atom.get('element', prot_atom.get('name', 'C'))[0]
                    
                    # Simple distance-based score
                    dist = np.linalg.norm(frag_coords - prot_coords)
                    
                    # Skip if too far
                    if dist > 10.0:
                        continue
                    
                    # Simplified VDW term
                    vdw_cutoff = 3.5
                    if dist < vdw_cutoff:
                        vdw_score = ((vdw_cutoff / dist) ** 12 - 2 * (vdw_cutoff / dist) ** 6)
                        score += vdw_score
            
            return score
        
        # Generate and score random poses
        poses = []
        
        for _ in range(self.poses_per_fragment):
            # Copy fragment
            frag_pose = copy.deepcopy(fragment)
            
            # Random position within sphere
            r = radius * random.betavariate(2, 5) ** (1/3)
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            
            x = center[0] + r * np.sin(phi) * np.cos(theta)
            y = center[1] + r * np.sin(phi) * np.sin(theta)
            z = center[2] + r * np.cos(phi)
            
            # Calculate centroid
            centroid = np.mean(frag_pose['xyz'], axis=0)
            
            # Calculate translation vector
            translation = np.array([x, y, z]) - centroid
            
            # Apply translation
            for atom in frag_pose['atoms']:
                atom['coords'] += translation
            
            frag_pose['xyz'] += translation
            
            # Random rotation
            from scipy.spatial.transform import Rotation
            rotation = Rotation.random()
            rotation_matrix = rotation.as_matrix()
            
            # Apply rotation around new center
            new_centroid = np.mean(frag_pose['xyz'], axis=0)
            
            for atom in frag_pose['atoms']:
                atom['coords'] -= new_centroid
                atom['coords'] = np.dot(atom['coords'], rotation_matrix.T)
                atom['coords'] += new_centroid
            
            frag_pose['xyz'] = np.array([atom['coords'] for atom in frag_pose['atoms']])
            
            # Score pose
            score = fragment_score(frag_pose)
            
            poses.append((frag_pose, score))
        
        # Sort by score (lowest first)
        poses.sort(key=lambda x: x[1])
        
        return poses
    
    def _grow_fragment(self, protein, base_pose, fragment, connection_points):
        """
        Grow a fragment onto an existing base pose.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        base_pose : dict
            Base pose fragment
        fragment : dict
            Fragment to attach
        connection_points : list
            List of (base_atom_idx, frag_atom_idx) tuples for connection
        
        Returns:
        --------
        list
            List of (pose, score) tuples for the combined fragment
        """
        # This is a simplified implementation
        # In practice, you'd need sophisticated fragment joining code
        
        poses = []
        
        for _ in range(self.poses_per_fragment):
            # Copy fragments
            base_copy = copy.deepcopy(base_pose)
            frag_copy = copy.deepcopy(fragment)
            
            # Pick random connection points (simplified)
            if connection_points:
                base_atom_idx, frag_atom_idx = random.choice(connection_points)
            else:
                # No connection info - just pick random atoms from each fragment
                base_atom_idx = random.choice(range(len(base_copy['atoms'])))
                frag_atom_idx = random.choice(range(len(frag_copy['atoms'])))
            
            # Get coordinates
            base_atom_coords = base_copy['atoms'][base_atom_idx]['coords']
            frag_atom_coords = frag_copy['atoms'][frag_atom_idx]['coords']
            
            # Calculate translation to align connection points
            translation = base_atom_coords - frag_atom_coords
            
            # Apply translation to fragment
            for atom in frag_copy['atoms']:
                atom['coords'] += translation
            
            frag_copy['xyz'] += translation
            
            # Apply random rotation around connection point
            from scipy.spatial.transform import Rotation
            rotation = Rotation.random()
            rotation_matrix = rotation.as_matrix()
            
            # Rotate around connection point
            for atom in frag_copy['atoms']:
                if atom != frag_copy['atoms'][frag_atom_idx]:  # Don't rotate connection atom
                    atom['coords'] -= frag_atom_coords
                    atom['coords'] = np.dot(atom['coords'], rotation_matrix.T)
                    atom['coords'] += frag_atom_coords
            
            frag_copy['xyz'] = np.array([atom['coords'] for atom in frag_copy['atoms']])
            
            # Combine fragments
            combined = {
                'atoms': base_copy['atoms'] + frag_copy['atoms'],
                'bonds': base_copy['bonds'] + frag_copy['bonds'],
                'atom_indices': base_copy['atom_indices'] + frag_copy['atom_indices'],
                'bond_indices': base_copy['bond_indices'] + frag_copy['bond_indices'],
                'xyz': np.vstack([base_copy['xyz'], frag_copy['xyz']])
            }
            
            # Add a new bond between connection points
            combined['bonds'].append({
                'begin_atom_idx': base_atom_idx,
                'end_atom_idx': len(base_copy['atoms']) + frag_atom_idx,
                'bond_type': 1  # Single bond
            })
            
            # Score the combined pose
            # In a real implementation, you'd use the proper scoring function
            # but for this simplified version, we'll use a custom function
            def combined_score(pose):
                score = 0
                
                # Calculate potential with protein atoms
                for atom in pose['atoms']:
                    coords = atom['coords']
                    element = atom.get('symbol', 'C')
                    
                    # Get appropriate protein atoms
                    if protein.active_site and 'atoms' in protein.active_site:
                        protein_atoms = protein.active_site['atoms']
                    else:
                        protein_atoms = protein.atoms
                    
                    for prot_atom in protein_atoms:
                        prot_coords = prot_atom['coords']
                        prot_element = prot_atom.get('element', prot_atom.get('name', 'C'))[0]
                        
                        # Simple distance-based score
                        dist = np.linalg.norm(coords - prot_coords)
                        
                        # Skip if too far
                        if dist > 10.0:
                            continue
                        
                        # Simplified VDW term
                        vdw_cutoff = 3.5
                        if dist < vdw_cutoff:
                            vdw_score = ((vdw_cutoff / dist) ** 12 - 2 * (vdw_cutoff / dist) ** 6)
                            score += vdw_score
                
                return score
            
            score = combined_score(combined)
            
            poses.append((combined, score))
        
        # Sort by score (lowest first)
        poses.sort(key=lambda x: x[1])
        
        return poses
    
    def _fragments_to_ligand(self, ligand_template, final_fragment):
        """
        Convert a final fragment back to a proper Ligand object.
        
        Parameters:
        -----------
        ligand_template : Ligand
            Template ligand object
        final_fragment : dict
            Final fragment pose
        
        Returns:
        --------
        Ligand
            Reconstructed ligand object
        """
        # Create a new ligand
        result = copy.deepcopy(ligand_template)
        
        # Update atom coordinates
        for i, atom_idx in enumerate(final_fragment['atom_indices']):
            if atom_idx < len(result.atoms):
                result.atoms[atom_idx]['coords'] = final_fragment['atoms'][i]['coords'].copy()
        
        # Update xyz array
        result.xyz = np.array([atom['coords'] for atom in result.atoms])
        
        return result
    
    def search(self, protein, ligand, current_radius=None):
        """
        Perform fragment-based incremental docking.
        
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
        print(f"Starting fragment-based docking with {self.growth_steps} growth steps")
        start_time = time.time()
        
        # Determine search space
        if protein.active_site:
            center = protein.active_site['center']
            radius = protein.active_site['radius']
        else:
            # Use protein center of mass
            center = np.mean(protein.xyz, axis=0)
            radius = 15.0  # Arbitrary search radius
            self.initialize_grid_points(center)

            if current_radius is None:
                current_radius = radius

            # Set the current search radius
            print(f"Using current search radius: {current_radius}")
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

        
        # Decompose ligand into fragments
        print("Decomposing ligand into fragments")
        fragments = self._decompose_ligand(ligand)
        
        print(f"Ligand decomposed into {len(fragments)} fragments")
        
        if not fragments:
            print("Warning: Could not decompose ligand. Using standard docking.")
            # Fall back to standard random search
            from .search import RandomSearch
            random_search = RandomSearch(self.scoring_function, self.max_iterations)
            return random_search.search(protein, ligand)
        
        # Extract fragment objects
        fragment_objects = [self._extract_fragment(ligand, atom_indices, bond_indices) 
                           for atom_indices, bond_indices in fragments]
        
        # Sort fragments by size (largest first)
        fragment_objects.sort(key=lambda x: len(x['atoms']), reverse=True)
        
        # Start with the largest fragment
        anchor_fragment = fragment_objects[0]
        remaining_fragments = fragment_objects[1:]
        
        print(f"Starting with anchor fragment ({len(anchor_fragment['atoms'])} atoms)")
        
        # Dock the anchor fragment
        print("Docking anchor fragment")
        anchor_poses = self._dock_fragment(protein, anchor_fragment, center, radius)
        
        print(f"Generated {len(anchor_poses)} anchor poses, best score: {anchor_poses[0][1]:.4f}")
        
        # Keep only the top poses
        top_anchor_poses = anchor_poses[:self.poses_per_fragment]
        
        # Build the molecule incrementally
        current_poses = top_anchor_poses
        
        # Keep track of all complete poses
        all_poses = []
        
        # Add remaining fragments one by one
        for i, fragment in enumerate(remaining_fragments):
            print(f"Adding fragment {i+1}/{len(remaining_fragments)} ({len(fragment['atoms'])} atoms)")
            
            next_poses = []
            
            for base_pose, base_score in current_poses:
                # Determine connection points (simplified)
                # In a real implementation, you'd use chemical knowledge
                connection_points = []  # (base_atom_idx, frag_atom_idx) pairs
                
                # Grow fragment onto base pose
                grown_poses = self._grow_fragment(protein, base_pose, fragment, connection_points)
                
                # Keep top poses
                next_poses.extend(grown_poses[:3])  # Keep top 3 from each base pose
            
            # Sort combined poses by score
            next_poses.sort(key=lambda x: x[1])
            
            # Keep only the top poses for next iteration
            current_poses = next_poses[:self.poses_per_fragment]
            
            print(f"After adding fragment {i+1}, best score: {current_poses[0][1]:.4f}")
        
        # Convert final fragments to ligand objects
        print("Converting final fragments to ligand objects")
        
        for final_fragment, score in current_poses:
            # Convert to ligand
            final_ligand = self._fragments_to_ligand(ligand, final_fragment)
            
            # Evaluate with full scoring function
            final_score = self.scoring_function.score(protein, final_ligand)
            
            all_poses.append((final_ligand, final_score))
        
        # Sort all poses by score
        all_poses.sort(key=lambda x: x[1])
        
        elapsed_time = time.time() - start_time
        print(f"Fragment-based docking completed in {elapsed_time:.2f} seconds")
        print(f"Best score: {all_poses[0][1]:.4f}")
        
        return all_poses


class HybridSearch(DockingSearch):
    """
    Hybrid search combining genetic algorithm with gradient-based optimization.
    Uses a genetic algorithm for global search followed by L-BFGS for local optimization.
    """
    
    def __init__(self, scoring_function, ga_iterations=50, lbfgs_iterations=50, 
                 population_size=100, mutation_rate=0.2, crossover_rate=0.8,
                 top_n_for_local=10, output_dir=None):
        """
        Initialize hybrid search algorithm.
        
        Parameters:
        -----------
        scoring_function : ScoringFunction
            Scoring function to evaluate poses
        ga_iterations : int
            Maximum number of genetic algorithm generations
        lbfgs_iterations : int
            Maximum number of L-BFGS iterations for local optimization
        population_size : int
            Size of the genetic algorithm population
        mutation_rate : float
            Probability of mutation (0.0 to 1.0)
        crossover_rate : float
            Probability of crossover (0.0 to 1.0)
        top_n_for_local : int
            Number of top GA solutions to optimize with L-BFGS
        output_dir : str, optional
            Directory to save output files
        """
        super().__init__(scoring_function, ga_iterations + lbfgs_iterations, output_dir)
        self.ga_iterations = ga_iterations
        self.lbfgs_iterations = lbfgs_iterations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.top_n_for_local = top_n_for_local
        self.output_dir = output_dir
        # Import necessary modules
        from .search import GeneticAlgorithm
        self.ga = GeneticAlgorithm(
            scoring_function=scoring_function,
            max_iterations=ga_iterations,
            population_size=population_size,
            mutation_rate=mutation_rate
        )
    
    def search(self, protein, ligand, current_radius=None):
        """
        Perform hybrid search with GA global search and L-BFGS local optimization.
        
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
            # Use protein center of mass
            center = np.mean(protein.xyz, axis=0)
            radius = 15.0  # Arbitrary search radius
            self.initialize_grid_points(center)

            if current_radius is None:
                current_radius = radius

            # Set the current search radius
            print(f"Using current search radius: {current_radius}")
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
        print(f"Starting hybrid search with {self.ga_iterations} GA iterations and "
              f"{self.lbfgs_iterations} L-BFGS iterations")
        
        # Phase 1: Genetic algorithm for global search
        print("\nPhase 1: Genetic Algorithm global search")
        ga_results = self.ga.search(protein, ligand)
        
        # Sort results by score
        ga_results.sort(key=lambda x: x[1])
        
        # Phase 2: L-BFGS local optimization on top poses
        print(f"\nPhase 2: L-BFGS local optimization on top {self.top_n_for_local} poses")
        optimized_results = []
        
        for i, (pose, score) in enumerate(ga_results[:self.top_n_for_local]):
            print(f"Optimizing pose {i+1}/{self.top_n_for_local} (score: {score:.4f})")
            
            # Perform local optimization
            opt_pose, opt_score = self._lbfgs_optimize(protein, pose)
            
            # Add to results
            optimized_results.append((opt_pose, opt_score))
            print(f"  Optimization result: {score:.4f} → {opt_score:.4f}")
        
        # Combine and sort all results
        all_results = optimized_results + ga_results[self.top_n_for_local:]
        all_results.sort(key=lambda x: x[1])
        
        print(f"\nHybrid search completed. Best score: {all_results[0][1]:.4f}")
        return all_results
    
    def _lbfgs_optimize(self, protein, pose):
        """
        Optimize pose using L-BFGS algorithm.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        pose : Ligand
            Ligand pose to optimize
        
        Returns:
        --------
        tuple
            (optimized_pose, optimized_score)
        """
        # Make a copy of the pose to avoid modifying original
        ligand_pose = copy.deepcopy(pose)
        
        # Get initial pose parameters (position and orientation)
        centroid = np.mean(ligand_pose.xyz, axis=0)
        initial_params = np.zeros(6)  # [tx, ty, tz, rx, ry, rz]
        initial_params[:3] = centroid
        
        # Define objective function for L-BFGS
        def objective(params):
            # Create a new pose for evaluation
            test_pose = copy.deepcopy(ligand_pose)
            
            # Extract translation and rotation
            translation = params[:3] - centroid
            rotation_vec = params[3:6]
            
            # Apply translation
            test_pose.translate(translation)
            
            # Apply rotation if rotation vector is not zero
            rot_magnitude = np.linalg.norm(rotation_vec)
            if rot_magnitude > 1e-6:
                from scipy.spatial.transform import Rotation
                
                # Normalize rotation vector
                axis = rotation_vec / rot_magnitude
                
                # Create rotation matrix
                rotation = Rotation.from_rotvec(axis * rot_magnitude)
                rotation_matrix = rotation.as_matrix()
                
                # Apply rotation around center of mass
                new_centroid = np.mean(test_pose.xyz, axis=0)
                test_pose.translate(-new_centroid)
                test_pose.rotate(rotation_matrix)
                test_pose.translate(new_centroid)
            
            # Evaluate score
            score = self.scoring_function.score(protein, test_pose)
            return score
        
        # Run L-BFGS optimization
        result = minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            options={
                'maxiter': self.lbfgs_iterations,
                'disp': False
            }
        )
        
        # Create final optimized pose
        optimized_pose = copy.deepcopy(ligand_pose)
        
        # Apply optimal transformation
        final_params = result.x
        translation = final_params[:3] - centroid
        rotation_vec = final_params[3:6]
        
        # Apply translation
        optimized_pose.translate(translation)
        
        # Apply rotation if rotation vector is not zero
        rot_magnitude = np.linalg.norm(rotation_vec)
        if rot_magnitude > 1e-6:
            from scipy.spatial.transform import Rotation
            
            # Normalize rotation vector
            axis = rotation_vec / rot_magnitude
            
            # Create rotation matrix
            rotation = Rotation.from_rotvec(axis * rot_magnitude)
            rotation_matrix = rotation.as_matrix()
            
            # Apply rotation around center of mass
            new_centroid = np.mean(optimized_pose.xyz, axis=0)
            optimized_pose.translate(-new_centroid)
            optimized_pose.rotate(rotation_matrix)
            optimized_pose.translate(new_centroid)
        
        # Get final score
        optimized_score = self.scoring_function.score(protein, optimized_pose)
        
        return optimized_pose, optimized_score


def create_advanced_search_algorithm(algorithm_type, scoring_function, **kwargs):
    """Factory function to create the appropriate advanced search algorithm."""
    if algorithm_type == 'gradient':
        return GradientBasedSearch(scoring_function, **kwargs)
    elif algorithm_type == 'replica-exchange':
        return ReplicaExchangeDocking(scoring_function, **kwargs)
    elif algorithm_type == 'ml-guided':
        return MLGuidedSearch(scoring_function, **kwargs)
    elif algorithm_type == 'fragment-based':
        return FragmentBasedDocking(scoring_function, **kwargs)
    elif algorithm_type == 'hybrid':
        return HybridSearch(scoring_function, **kwargs)
    else:
        raise ValueError(f"Unknown advanced algorithm type: {algorithm_type}")