"""
Pose Analysis and Statistics

Provides advanced analysis of docking results:
- RMSD calculations
- Clustering analysis
- Binding site characterization
- Statistical comparisons
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
from ..core import Pose, DockingResult


class PoseAnalyzer:
    """Advanced analysis of docking poses and results"""

    def __init__(self):
        self.logger = logging.getLogger("pandadock.visualization.analyzer")

    def analyze_poses(self, poses: List[Pose], reference_pose: Pose = None) -> Dict[str, Any]:
        """
        Comprehensive analysis of pose ensemble

        Args:
            poses: List of poses to analyze
            reference_pose: Reference pose for RMSD calculation

        Returns:
            Dictionary with analysis results
        """
        if not poses:
            return {}

        analysis = {}

        # Basic statistics
        energies = [pose.energy for pose in poses]
        analysis['energy_stats'] = {
            'min': np.min(energies),
            'max': np.max(energies),
            'mean': np.mean(energies),
            'std': np.std(energies),
            'range': np.max(energies) - np.min(energies)
        }

        # Pose clustering
        analysis['clustering'] = self._cluster_poses(poses)

        # RMSD analysis if reference provided
        if reference_pose:
            analysis['rmsd_stats'] = self._calculate_rmsd_statistics(poses, reference_pose)

        # Binding site characterization
        analysis['binding_site'] = self._characterize_binding_site(poses)

        return analysis

    def _cluster_poses(self, poses: List[Pose]) -> Dict[str, Any]:
        """Cluster poses based on coordinate similarity"""
        if len(poses) < 2:
            return {'num_clusters': 1, 'cluster_labels': [0]}

        # Calculate pairwise RMSD matrix
        n_poses = len(poses)
        rmsd_matrix = np.zeros((n_poses, n_poses))

        for i in range(n_poses):
            for j in range(i+1, n_poses):
                rmsd = self._calculate_rmsd(poses[i].coordinates, poses[j].coordinates)
                rmsd_matrix[i, j] = rmsd
                rmsd_matrix[j, i] = rmsd

        # Cluster using DBSCAN
        clustering = DBSCAN(eps=2.0, min_samples=2, metric='precomputed')
        cluster_labels = clustering.fit_predict(rmsd_matrix)

        return {
            'num_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
            'cluster_labels': cluster_labels.tolist(),
            'rmsd_matrix': rmsd_matrix.tolist()
        }

    def _calculate_rmsd_statistics(self, poses: List[Pose], reference: Pose) -> Dict[str, float]:
        """Calculate RMSD statistics relative to reference"""
        rmsds = []
        for pose in poses:
            rmsd = self._calculate_rmsd(pose.coordinates, reference.coordinates)
            rmsds.append(rmsd)

        return {
            'min_rmsd': np.min(rmsds),
            'max_rmsd': np.max(rmsds),
            'mean_rmsd': np.mean(rmsds),
            'std_rmsd': np.std(rmsds)
        }

    def _calculate_rmsd(self, coords1: np.ndarray, coords2: np.ndarray) -> float:
        """Calculate RMSD between two coordinate sets"""
        if coords1.shape != coords2.shape:
            return float('inf')

        diff = coords1 - coords2
        rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
        return rmsd

    def _characterize_binding_site(self, poses: List[Pose]) -> Dict[str, Any]:
        """Characterize the binding site based on pose distribution"""
        all_coords = np.vstack([pose.coordinates for pose in poses])
        centers = np.array([pose.center for pose in poses])

        return {
            'site_volume': self._estimate_binding_site_volume(all_coords),
            'center_spread': np.std(centers, axis=0).tolist(),
            'mean_center': np.mean(centers, axis=0).tolist()
        }

    def _estimate_binding_site_volume(self, coordinates: np.ndarray) -> float:
        """Estimate binding site volume from coordinate distribution"""
        # Simple convex hull volume estimation
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(coordinates)
            return hull.volume
        except:
            return 0.0