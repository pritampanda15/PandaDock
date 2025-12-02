import numpy as np
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
import logging
from typing import List, Tuple, Dict, Any
from .core import GridBox, GridBoxGenerator


class CavityDetectionGridBoxGenerator(GridBoxGenerator):
    """Generate grid boxes based on cavity detection algorithms"""

    def __init__(self):
        super().__init__()
        self.probe_radius = 1.4  # Water molecule radius
        self.grid_spacing = 1.5  # Increased for better performance
        self.min_cavity_volume = 50.0  # Minimum cavity volume in Ų

    def generate(self, receptor_file: str, max_cavities: int = 5,
                 min_volume: float = None, spacing: float = 0.375, **kwargs) -> List[GridBox]:
        """Generate grid boxes for detected cavities"""

        if min_volume is None:
            min_volume = self.min_cavity_volume

        structure = self._parse_receptor(receptor_file)
        atoms = list(structure.get_atoms())

        # Get protein coordinates
        protein_coords = np.array([atom.get_coord() for atom in atoms])
        protein_radii = np.array([self._get_atomic_radius(atom.element) for atom in atoms])

        # Detect cavities
        cavities = self._detect_cavities(protein_coords, protein_radii)

        # Filter and rank cavities
        cavities = self._filter_cavities(cavities, min_volume)
        cavities = self._rank_cavities(cavities)

        # Convert top cavities to grid boxes
        grid_boxes = []
        for i, cavity in enumerate(cavities[:max_cavities]):
            center = tuple(cavity['center'])
            dimensions = tuple(cavity['dimensions'])

            grid_box = GridBox(
                center=center,
                dimensions=dimensions,
                spacing=spacing,
                receptor_file=receptor_file,
                method_used="cavity_detection",
                confidence_score=cavity['score'],
                binding_site_info={
                    'cavity_volume': cavity['volume'],
                    'cavity_depth': cavity['depth'],
                    'druggability_score': cavity['druggability'],
                    'cavity_rank': i + 1,
                    'num_points': len(cavity['points'])
                }
            )
            grid_boxes.append(grid_box)

        return grid_boxes

    def _detect_cavities(self, protein_coords: np.ndarray, protein_radii: np.ndarray) -> List[Dict[str, Any]]:
        """Detect cavities using grid-based approach"""

        # Create bounding box with smaller padding for efficiency
        min_coords = np.min(protein_coords, axis=0) - 5.0
        max_coords = np.max(protein_coords, axis=0) + 5.0

        # Generate grid points more efficiently
        x_range = np.arange(min_coords[0], max_coords[0], self.grid_spacing)
        y_range = np.arange(min_coords[1], max_coords[1], self.grid_spacing)
        z_range = np.arange(min_coords[2], max_coords[2], self.grid_spacing)

        # Limit grid size for performance
        max_points_per_dim = 50
        if len(x_range) > max_points_per_dim:
            x_range = x_range[::len(x_range)//max_points_per_dim]
        if len(y_range) > max_points_per_dim:
            y_range = y_range[::len(y_range)//max_points_per_dim]
        if len(z_range) > max_points_per_dim:
            z_range = z_range[::len(z_range)//max_points_per_dim]

        # Use meshgrid for efficient generation
        xx, yy, zz = np.meshgrid(x_range, y_range, z_range, indexing='ij')
        grid_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

        # Filter points that are inside cavities (not occupied by protein)
        self.logger.info(f"Analyzing {len(grid_points)} grid points for cavities...")
        cavity_points = self._filter_cavity_points(grid_points, protein_coords, protein_radii)
        self.logger.info(f"Found {len(cavity_points)} potential cavity points")

        if len(cavity_points) < 10:
            return []

        # Cluster cavity points
        clusters = self._cluster_cavity_points(cavity_points)

        # Analyze each cluster
        cavities = []
        for cluster_id in np.unique(clusters):
            if cluster_id == -1:  # Noise points
                continue

            cluster_points = cavity_points[clusters == cluster_id]
            cavity_info = self._analyze_cavity_cluster(cluster_points, protein_coords)

            if cavity_info['volume'] >= self.min_cavity_volume:
                cavities.append(cavity_info)

        return cavities

    def _filter_cavity_points(self, grid_points: np.ndarray, protein_coords: np.ndarray,
                            protein_radii: np.ndarray) -> np.ndarray:
        """Filter grid points to find cavity points"""

        cavity_points = []

        # Process in smaller batches for better performance
        batch_size = 5000
        total_batches = (len(grid_points) + batch_size - 1) // batch_size

        for i in range(0, len(grid_points), batch_size):
            if i // batch_size % 10 == 0:  # Progress every 10 batches
                self.logger.info(f"Processing batch {i//batch_size + 1}/{total_batches}")

            batch = grid_points[i:i+batch_size]

            # Calculate distances to all protein atoms (vectorized)
            distances = cdist(batch, protein_coords)

            # Quick filtering: points must be close to protein but not inside
            min_distances_raw = np.min(distances, axis=1)

            # Filter points that are too far from protein (> 8 Å)
            close_mask = min_distances_raw <= 8.0
            if not np.any(close_mask):
                continue

            batch_close = batch[close_mask]
            distances_close = distances[close_mask]

            # Check if points are outside all protein atoms (with probe radius)
            min_distances = np.min(distances_close - protein_radii[np.newaxis, :], axis=1)
            valid_mask = min_distances >= self.probe_radius

            if np.any(valid_mask):
                cavity_points.append(batch_close[valid_mask])

        if cavity_points:
            return np.vstack(cavity_points)
        else:
            return np.array([]).reshape(0, 3)

    def _cluster_cavity_points(self, cavity_points: np.ndarray) -> np.ndarray:
        """Cluster cavity points using DBSCAN"""

        if len(cavity_points) < 10:
            return np.array([-1] * len(cavity_points))

        # Use tighter clustering for binding-site-sized cavities
        dbscan = DBSCAN(eps=3.0, min_samples=20)  # Larger clusters for binding sites
        clusters = dbscan.fit_predict(cavity_points)

        # Post-process to split large clusters into binding-site-sized pieces
        refined_clusters = []
        next_cluster_id = 0

        for cluster_id in np.unique(clusters):
            if cluster_id == -1:  # Noise
                continue

            cluster_mask = clusters == cluster_id
            cluster_points = cavity_points[cluster_mask]

            if len(cluster_points) > 2000:  # Split large clusters
                # Sub-cluster large clusters
                sub_dbscan = DBSCAN(eps=5.0, min_samples=50)
                sub_clusters = sub_dbscan.fit_predict(cluster_points)

                for sub_id in np.unique(sub_clusters):
                    if sub_id == -1:
                        continue
                    refined_clusters.extend([next_cluster_id] * np.sum(sub_clusters == sub_id))
                    next_cluster_id += 1
            else:
                refined_clusters.extend([next_cluster_id] * len(cluster_points))
                next_cluster_id += 1

        # Pad with -1 for noise points
        refined_array = np.full(len(cavity_points), -1)
        refined_idx = 0
        for cluster_id in np.unique(clusters):
            if cluster_id == -1:
                continue
            cluster_size = np.sum(clusters == cluster_id)
            refined_array[clusters == cluster_id] = refined_clusters[refined_idx:refined_idx + cluster_size]
            refined_idx += cluster_size

        return refined_array

    def _analyze_cavity_cluster(self, cluster_points: np.ndarray, protein_coords: np.ndarray) -> Dict[str, Any]:
        """Analyze a cavity cluster to extract properties"""

        center = np.mean(cluster_points, axis=0)

        # Calculate approximate volume
        volume = len(cluster_points) * (self.grid_spacing ** 3)

        # Calculate depth (distance from surface)
        distances_to_protein = cdist([center], protein_coords)[0]
        depth = np.min(distances_to_protein)

        # Calculate bounding box with padding for binding site
        min_coords = np.min(cluster_points, axis=0)
        max_coords = np.max(cluster_points, axis=0)
        dimensions = max_coords - min_coords

        # Add padding for ligand binding (5-8 Å on each side)
        padding = 6.0
        dimensions = dimensions + 2 * padding

        # Ensure reasonable binding site dimensions (15-30 Å)
        dimensions = np.maximum(dimensions, [15.0, 15.0, 15.0])
        dimensions = np.minimum(dimensions, [30.0, 30.0, 30.0])

        # Calculate druggability score (simple heuristic)
        druggability = self._calculate_druggability_score(volume, depth, dimensions)

        return {
            'center': center,
            'dimensions': dimensions,
            'volume': volume,
            'depth': depth,
            'druggability': druggability,
            'points': cluster_points,
            'score': druggability  # Use druggability as initial score
        }

    def _calculate_druggability_score(self, volume: float, depth: float, dimensions: np.ndarray) -> float:
        """Calculate a simple druggability score"""

        # Volume score (optimal range 200-800 Ų)
        if volume < 100:
            volume_score = volume / 100.0
        elif volume <= 800:
            volume_score = 1.0
        else:
            volume_score = 800.0 / volume

        # Depth score (prefer buried sites)
        depth_score = min(depth / 5.0, 1.0)

        # Shape score (prefer more spherical cavities)
        shape_ratio = np.max(dimensions) / np.min(dimensions)
        shape_score = 1.0 / (1.0 + (shape_ratio - 1.0) * 0.5)

        # Combined score
        score = (volume_score * 0.5 + depth_score * 0.3 + shape_score * 0.2)
        return min(score, 1.0)

    def _filter_cavities(self, cavities: List[Dict[str, Any]], min_volume: float) -> List[Dict[str, Any]]:
        """Filter cavities based on volume and other criteria"""
        return [cavity for cavity in cavities if cavity['volume'] >= min_volume]

    def _rank_cavities(self, cavities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank cavities by druggability score"""
        return sorted(cavities, key=lambda x: x['druggability'], reverse=True)

    def _get_atomic_radius(self, element: str) -> float:
        """Get van der Waals radius for an element"""
        radii = {
            'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47,
            'P': 1.80, 'S': 1.80, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98,
            'Fe': 2.00, 'Zn': 1.39, 'Ca': 2.31, 'Mg': 1.73, 'Na': 2.27,
            'K': 2.75, 'Mn': 1.73, 'Cu': 1.40, 'Co': 1.67, 'Ni': 1.63
        }
        return radii.get(element.upper(), 1.80)  # Default radius