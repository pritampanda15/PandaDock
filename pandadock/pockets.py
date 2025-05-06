import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN

class OptimizedCastpDetector:
    """
    An optimized CASTp-inspired algorithm for protein pocket detection.
    """
    def __init__(self, probe_radius=1.4, grid_spacing=0.8):
        self.probe_radius = probe_radius
        self.grid_spacing = grid_spacing
        self.pockets = []
        
    def detect_pockets(self, protein):
        """Detect pockets using an optimized grid-based approach."""
        print("Starting optimized pocket detection...")
        
        # Step 1: Identify surface atoms to focus our search
        surface_atoms, surface_indices = self._get_surface_atoms(protein)
        print(f"Identified {len(surface_atoms)} surface atoms")
        
        # Step 2: Create grid focused around surface atoms only
        grid_points = self._generate_focused_grid(protein, surface_atoms)
        print(f"Created optimized grid with {len(grid_points)} points")
        
        # Step 3: Identify pocket points using vectorized operations
        pocket_points, buriedness = self._identify_pocket_points_vectorized(grid_points, protein)
        print(f"Identified {len(pocket_points)} potential pocket points")
        
        # Step 4: Cluster pocket points using density-based clustering
        pocket_clusters = self._cluster_pocket_points(pocket_points, buriedness)
        print(f"Clustered into {len(pocket_clusters)} distinct pockets")
        
        # Step 5: Calculate pocket properties
        pockets = self._calculate_pocket_properties(pocket_clusters, protein)
        print(f"Calculated properties for {len(pockets)} pockets")
        
        self.pockets = pockets
        return pockets
    
    def _get_surface_atoms(self, protein):
        """Identify surface atoms using a simple accessibility criterion."""
        coords = protein.xyz
        radii = np.array([atom.get('radius', 1.7) for atom in protein.atoms])
        
        # Build KD-tree for fast distance queries
        tree = KDTree(coords)
        
        surface_indices = []
        surface_coords = []
        
        # For each atom, check if it's accessible
        for i, coord in enumerate(coords):
            # Find neighbors within a certain radius
            neighbors = tree.query_ball_point(coord, radii[i] + 2*self.probe_radius)
            
            # If atom has fewer neighbors than a threshold, it's likely on the surface
            if len(neighbors) < 25:  # Arbitrary threshold, can be tuned
                surface_indices.append(i)
                surface_coords.append(coord)
        
        return np.array(surface_coords), surface_indices
    
    def _generate_focused_grid(self, protein, surface_atoms):
        """Generate grid points focused around surface atoms only."""
        # Build a KD-tree for surface atoms
        surface_tree = KDTree(surface_atoms)
        
        # Create a coarse grid covering the protein
        coords = protein.xyz
        min_coords = np.min(coords, axis=0) - 4.0
        max_coords = np.max(coords, axis=0) + 4.0
        
        # Use larger spacing for the initial grid
        x = np.arange(min_coords[0], max_coords[0], self.grid_spacing*2)
        y = np.arange(min_coords[1], max_coords[1], self.grid_spacing*2)
        z = np.arange(min_coords[2], max_coords[2], self.grid_spacing*2)
        
        # Create meshgrid for vectorized operations
        X, Y, Z = np.meshgrid(x, y, z)
        grid_points_coarse = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
        
        # Filter grid points that are close to surface atoms
        distances, _ = surface_tree.query(grid_points_coarse)
        potential_pocket_area = grid_points_coarse[distances < 6.0]  # 6Å from any surface atom
        
        # Create a finer grid around potential pocket areas
        fine_grid_points = []
        for point in potential_pocket_area:
            # Generate finer grid points around this coarse point
            x_fine = np.arange(point[0]-self.grid_spacing, point[0]+self.grid_spacing, self.grid_spacing)
            y_fine = np.arange(point[1]-self.grid_spacing, point[1]+self.grid_spacing, self.grid_spacing)
            z_fine = np.arange(point[2]-self.grid_spacing, point[2]+self.grid_spacing, self.grid_spacing)
            
            X_fine, Y_fine, Z_fine = np.meshgrid(x_fine, y_fine, z_fine)
            fine_points = np.vstack((X_fine.ravel(), Y_fine.ravel(), Z_fine.ravel())).T
            fine_grid_points.append(fine_points)
        
        # Combine all fine grid points and remove duplicates
        if fine_grid_points:
            combined_grid = np.vstack(fine_grid_points)
            # Remove duplicates by rounding to grid spacing and using unique rows
            rounded_grid = np.round(combined_grid / self.grid_spacing) * self.grid_spacing
            unique_grid = np.unique(rounded_grid, axis=0)
            return unique_grid
        else:
            return np.array([])
    
    def _identify_pocket_points_vectorized(self, grid_points, protein):
        """Identify pocket points using vectorized operations."""
        if len(grid_points) == 0:
            return np.array([]), np.array([])
            
        coords = protein.xyz
        radii = np.array([atom.get('radius', 1.7) for atom in protein.atoms])
        
        # Build KD-tree for protein atoms
        atom_tree = KDTree(coords)
        
        # Process grid points in batches to avoid memory issues
        batch_size = 10000
        pocket_points = []
        buriedness_scores = []
        
        for i in range(0, len(grid_points), batch_size):
            batch = grid_points[i:i+batch_size]
            
            # Find nearest atoms and distances for each grid point
            distances, indices = atom_tree.query(batch, k=15)  # Get closest 15 atoms
            
            # Check if point is outside all atoms
            nearest_atom_dist = distances[:, 0]
            nearest_atom_idx = indices[:, 0]
            nearest_atom_radius = radii[nearest_atom_idx]
            outside_atoms = nearest_atom_dist > nearest_atom_radius
            
            # Check if point is near protein surface
            near_surface = nearest_atom_dist < nearest_atom_radius + 2*self.probe_radius
            
            # Check if point is not too far from protein
            not_far = nearest_atom_dist < 4.0
            
            # Calculate buriedness as number of atoms within 8Å
            atoms_within_8A = (distances < 8.0).sum(axis=1)
            
            # Select points that meet all criteria
            mask = outside_atoms & near_surface & not_far
            selected_points = batch[mask]
            selected_buriedness = atoms_within_8A[mask]
            
            pocket_points.append(selected_points)
            buriedness_scores.append(selected_buriedness)
        
        # Combine results from all batches
        if pocket_points:
            pocket_points = np.vstack(pocket_points)
            buriedness_scores = np.concatenate(buriedness_scores)
            return pocket_points, buriedness_scores
        else:
            return np.array([]), np.array([])
    
    def _cluster_pocket_points(self, pocket_points, buriedness):
        """Cluster pocket points into distinct pockets with informed clustering parameters."""
        if len(pocket_points) == 0:
            return []
            
        # Use more informed DBSCAN parameters
        # Higher eps for point in well-buried regions
        if len(pocket_points) > 10000:
            # For large datasets, use a two-step approach
            # First cluster with looser parameters
            clustering = DBSCAN(eps=2.5, min_samples=5).fit(pocket_points)
            labels = clustering.labels_
            
            # Find cluster centers
            clusters = {}
            for i, label in enumerate(labels):
                if label == -1:  # Skip noise points
                    continue
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append((pocket_points[i], buriedness[i]))
            
            # Process each cluster separately with parameters based on buriedness
            refined_clusters = []
            for cluster_points in clusters.values():
                points = np.array([p[0] for p in cluster_points])
                burial = np.array([p[1] for p in cluster_points])
                
                # Skip small clusters
                if len(points) < 10:
                    continue
                
                # Use tighter clustering for highly buried regions
                avg_burial = np.mean(burial)
                eps = 1.2 if avg_burial > 10 else 2.0
                min_samples = 3 if avg_burial > 10 else 5
                
                # Recluster with refined parameters
                sub_clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
                sub_labels = sub_clustering.labels_
                
                for sub_label in set(sub_labels):
                    if sub_label == -1:
                        continue
                    refined_clusters.append(points[sub_labels == sub_label])
        else:
            # For smaller datasets, use a single clustering step
            clustering = DBSCAN(eps=2.0, min_samples=5).fit(pocket_points)
            labels = clustering.labels_
            
            # Group points by cluster label
            refined_clusters = []
            for label in set(labels):
                if label == -1:  # Skip noise points
                    continue
                refined_clusters.append(pocket_points[labels == label])
        
        # Sort clusters by size (largest first)
        refined_clusters.sort(key=len, reverse=True)
        
        return refined_clusters
    
    def _calculate_pocket_properties(self, pocket_clusters, protein):
        """Calculate properties for each pocket with improved property estimation."""
        pockets = []
        coords = protein.xyz
        
        # Build KD-tree for protein atoms
        atom_tree = KDTree(coords)
        
        for i, cluster in enumerate(pocket_clusters):
            # Skip small clusters
            if len(cluster) < 10:
                continue
                
            # Calculate center as center of mass
            center = np.mean(cluster, axis=0)
            
            # Calculate convex hull for better volume estimation
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(cluster)
                volume = hull.volume
            except:
                # Fallback to sphere approximation
                radius = np.max(np.linalg.norm(cluster - center, axis=1))
                volume = (4/3) * np.pi * radius**3
            
            # Find atoms that form the pocket (improved algorithm)
            # Use KD-tree for efficient neighbor finding
            indices = atom_tree.query_ball_point(cluster, r=3.5)
            pocket_atoms = list(set([item for sublist in indices for item in sublist]))
            
            # Identify residues forming the pocket
            pocket_residues = set()
            for atom_idx in pocket_atoms:
                if atom_idx < len(protein.atoms):
                    atom = protein.atoms[atom_idx]
                    if 'residue_id' in atom:
                        pocket_residues.add(atom['residue_id'])
            
            # Estimate pocket mouth openings
            # Project points to a sphere centered at center and look for clusters in spherical coordinates
            vectors = cluster - center
            distances = np.linalg.norm(vectors, axis=1)
            unit_vectors = vectors / distances[:, np.newaxis]
            
            # Calculate spherical coordinates
            theta = np.arccos(unit_vectors[:, 2])  # polar angle
            phi = np.arctan2(unit_vectors[:, 1], unit_vectors[:, 0])  # azimuthal angle
            
            # Cluster in spherical coordinates to identify mouth openings
            spherical_coords = np.column_stack((theta, phi))
            try:
                mouth_clustering = DBSCAN(eps=0.3, min_samples=3).fit(spherical_coords)
                labels = mouth_clustering.labels_
                n_mouths = len(set(labels)) - (1 if -1 in labels else 0)
            except:
                n_mouths = 1  # Default to 1
            
            # Store pocket information
            pocket_info = {
                'id': i + 1,
                'center': center,
                'volume': volume,
                'n_points': len(cluster),
                'atoms': pocket_atoms,
                'residues': list(pocket_residues),
                'n_mouths': n_mouths,
                'solvent_exposure': n_mouths > 0
            }
            
            pockets.append(pocket_info)
        
        return pockets





#Advanced pocket detection algorithm inspired by CASTp
# import numpy as np
# from scipy.spatial import Delaunay, ConvexHull
# from sklearn.cluster import DBSCAN
# import networkx as nx

# class CastpLikeDetector:
#     """
#     A simplified CASTp-inspired algorithm for protein pocket detection.
#     """
#     def __init__(self, probe_radius=1.4, grid_spacing=0.8):
#         """
#         Initialize the pocket detector.
        
#         Parameters:
#         -----------
#         probe_radius : float
#             Radius of the probe sphere (default: 1.4 Å for water molecule)
#         grid_spacing : float
#             Spacing of the 3D grid for pocket detection (default: 0.8 Å)
#         """
#         self.probe_radius = probe_radius
#         self.grid_spacing = grid_spacing
#         self.pockets = []
        
#     def detect_pockets(self, protein):
#         """
#         Detect pockets using a grid-based approach inspired by CASTp.
        
#         Parameters:
#         -----------
#         protein : Protein
#             Protein object with atom coordinates
            
#         Returns:
#         --------
#         list
#             List of detected pockets with properties
#         """
#         print("Starting pocket detection...")
#         # Generate grid around protein
#         grid_points, grid_spacing = self._generate_grid(protein)
#         print(f"Created grid with {len(grid_points)} points at {grid_spacing}Å spacing")
        
#         # Identify pocket grid points
#         pocket_points = self._identify_pocket_points(grid_points, protein)
#         print(f"Identified {len(pocket_points)} potential pocket points")
        
#         # Cluster pocket points into distinct pockets
#         pocket_clusters = self._cluster_pocket_points(pocket_points)
#         print(f"Clustered into {len(pocket_clusters)} distinct pockets")
        
#         # Calculate properties for each pocket
#         pockets = self._calculate_pocket_properties(pocket_clusters, protein)
#         print(f"Calculated properties for {len(pockets)} pockets")
        
#         self.pockets = pockets
#         return pockets
    
#     def _generate_grid(self, protein):
#         """Generate a 3D grid around the protein."""
#         # Get protein bounds
#         coords = protein.xyz
#         min_coords = np.min(coords, axis=0) - 4.0  # 4Å buffer
#         max_coords = np.max(coords, axis=0) + 4.0
        
#         # Create grid points
#         x = np.arange(min_coords[0], max_coords[0], self.grid_spacing)
#         y = np.arange(min_coords[1], max_coords[1], self.grid_spacing)
#         z = np.arange(min_coords[2], max_coords[2], self.grid_spacing)
        
#         grid_points = []
#         for i in x:
#             for j in y:
#                 for k in z:
#                     grid_points.append([i, j, k])
        
#         return np.array(grid_points), self.grid_spacing
    
#     def _identify_pocket_points(self, grid_points, protein):
#         """Identify grid points that are potential pocket points."""
#         # Get protein atoms
#         coords = protein.xyz
#         radii = np.array([atom.get('radius', 1.7) for atom in protein.atoms])
        
#         # For each grid point, check if it's:
#         # 1. Outside protein atoms (not inside any atom)
#         # 2. Near protein surface (within probe radius of at least one atom)
#         # 3. Not far from protein (within 4Å of any atom)
#         pocket_points = []
        
#         for point in grid_points:
#             # Calculate distances to all atoms
#             distances = np.linalg.norm(coords - point, axis=1)
            
#             # Check if point is outside all atoms
#             outside_atoms = np.all(distances > radii)
            
#             # Check if point is near protein surface
#             near_surface = np.any(distances < radii + 2*self.probe_radius)
            
#             # Check if point is not too far from protein
#             not_far = np.any(distances < 4.0)
            
#             if outside_atoms and near_surface and not_far:
#                 # Calculate a "buriedness" score
#                 # Count atoms within 8Å
#                 buriedness = np.sum(distances < 8.0)
#                 pocket_points.append((point, buriedness))
        
#         # Sort by buriedness score (most buried first)
#         pocket_points.sort(key=lambda x: x[1], reverse=True)
        
#         # Return the points (discard buriedness scores for now)
#         return np.array([p[0] for p in pocket_points])
    
#     def _cluster_pocket_points(self, pocket_points):
#         """Cluster pocket points into distinct pockets."""
#         if len(pocket_points) == 0:
#             return []
            
#         # Use DBSCAN clustering
#         clustering = DBSCAN(eps=2.0, min_samples=5).fit(pocket_points)
#         labels = clustering.labels_
        
#         # Group points by cluster label
#         clusters = {}
#         for i, label in enumerate(labels):
#             if label == -1:  # Skip noise points
#                 continue
#             if label not in clusters:
#                 clusters[label] = []
#             clusters[label].append(pocket_points[i])
        
#         # Convert to list of numpy arrays
#         pocket_clusters = [np.array(cluster) for cluster in clusters.values()]
        
#         # Sort clusters by size (largest first)
#         pocket_clusters.sort(key=len, reverse=True)
        
#         return pocket_clusters
    
#     def _calculate_pocket_properties(self, pocket_clusters, protein):
#         """Calculate properties for each pocket cluster."""
#         pockets = []
        
#         for i, cluster in enumerate(pocket_clusters):
#             # Skip small clusters
#             if len(cluster) < 10:
#                 continue
                
#             # Calculate center and radius
#             center = np.mean(cluster, axis=0)
            
#             # Calculate distances to protein atoms
#             coords = protein.xyz
#             distances = []
#             for point in cluster:
#                 # Find closest atom
#                 min_dist = np.min(np.linalg.norm(coords - point, axis=1))
#                 distances.append(min_dist)
            
#             # Approximate radius based on cluster extent
#             radius = np.max(np.linalg.norm(cluster - center, axis=1))
            
#             # Calculate volume (approximate as volume of sphere)
#             volume = (4/3) * np.pi * radius**3
            
#             # Identify atoms that form the pocket
#             pocket_atoms = []
#             for j, atom_coord in enumerate(coords):
#                 # If atom is close to any pocket point, it's part of the pocket
#                 if np.any(np.linalg.norm(cluster - atom_coord, axis=1) < 3.5):
#                     pocket_atoms.append(j)
            
#             # Identify residues forming the pocket
#             pocket_residues = set()
#             for atom_idx in pocket_atoms:
#                 if atom_idx < len(protein.atoms):
#                     atom = protein.atoms[atom_idx]
#                     if 'residue_id' in atom:
#                         pocket_residues.add(atom['residue_id'])
            
#             # Store pocket information
#             pocket_info = {
#                 'id': i + 1,
#                 'center': center,
#                 'radius': radius,
#                 'volume': volume,
#                 'n_points': len(cluster),
#                 'atoms': pocket_atoms,
#                 'residues': list(pocket_residues)
#             }
            
#             pockets.append(pocket_info)
        
#         return pockets

