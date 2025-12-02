"""
Ensemble Refinement for Molecular Docking

Implements ensemble-based pose refinement strategies including:
1. Multiple conformer optimization
2. Consensus scoring across multiple algorithms
3. Clustering-based refinement
4. Boltzmann-weighted ensemble generation
5. Cross-validation refinement
6. Multi-objective optimization

Designed to improve pose accuracy through ensemble methods and reduce overfitting.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Any
import logging
from dataclasses import dataclass
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN, KMeans
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
import copy
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


@dataclass
class EnsembleResult:
    """Container for ensemble refinement results"""
    refined_poses: List[np.ndarray]
    pose_energies: List[float]
    pose_scores: List[Dict[str, float]]
    consensus_pose: np.ndarray
    consensus_energy: float
    cluster_assignments: List[int]
    ensemble_statistics: Dict[str, Any]
    refinement_history: List[Dict[str, Any]]


class EnsembleRefiner:
    """Multi-algorithm ensemble pose refinement"""

    def __init__(self, **params):
        self.logger = logging.getLogger("pandadock.refinement.ensemble")

        # Ensemble parameters
        self.num_refinement_rounds = params.get('num_rounds', 3)
        self.refinement_methods = params.get('methods', ['local_search', 'monte_carlo'])
        self.consensus_method = params.get('consensus_method', 'boltzmann_weighted')

        # Clustering parameters
        self.clustering_method = params.get('clustering_method', 'hierarchical')  # or 'dbscan', 'kmeans'
        self.rmsd_threshold = params.get('rmsd_threshold', 2.0)  # Å for clustering
        self.min_cluster_size = params.get('min_cluster_size', 3)

        # Scoring parameters
        self.scoring_weights = params.get('scoring_weights', {
            'primary': 0.6,
            'secondary': 0.3,
            'tertiary': 0.1
        })
        self.temperature = params.get('temperature', 298.15)  # K for Boltzmann weighting

        # Optimization parameters
        self.max_poses_per_round = params.get('max_poses_per_round', 50)
        self.convergence_threshold = params.get('convergence_threshold', 0.01)  # kcal/mol
        self.max_parallel_jobs = params.get('max_parallel_jobs', 4)

        # Diversity parameters
        self.diversity_weight = params.get('diversity_weight', 0.1)
        self.novelty_threshold = params.get('novelty_threshold', 1.0)  # Å

        self.logger.info(f"Ensemble refinement initialized: {self.num_refinement_rounds} rounds, "
                        f"methods={self.refinement_methods}, consensus={self.consensus_method}")

    def refine_ensemble(self, initial_poses: List[np.ndarray],
                       ligand_mol: Chem.Mol,
                       scoring_functions: List[Callable],
                       refinement_algorithms: List[Callable],
                       **kwargs) -> EnsembleResult:
        """
        Perform ensemble refinement of poses

        Args:
            initial_poses: List of initial pose coordinates
            ligand_mol: RDKit molecule object
            scoring_functions: List of scoring functions to use
            refinement_algorithms: List of refinement algorithms
            **kwargs: Additional parameters for scoring/refinement

        Returns:
            EnsembleResult containing refined poses and statistics
        """
        self.logger.info(f"Starting ensemble refinement with {len(initial_poses)} initial poses")

        refinement_history = []
        current_poses = [pose.copy() for pose in initial_poses]

        # Initialize result tracking
        all_poses = []
        all_energies = []
        all_scores = []

        for round_idx in range(self.num_refinement_rounds):
            self.logger.info(f"Refinement round {round_idx + 1}/{self.num_refinement_rounds}")

            round_start_time = time.time()
            round_results = self._perform_refinement_round(
                current_poses, ligand_mol, scoring_functions,
                refinement_algorithms, round_idx, **kwargs
            )

            # Collect poses from this round
            round_poses = round_results['poses']
            round_energies = round_results['energies']
            round_scores = round_results['scores']

            all_poses.extend(round_poses)
            all_energies.extend(round_energies)
            all_scores.extend(round_scores)

            # Clustering and selection for next round
            if round_idx < self.num_refinement_rounds - 1:
                selected_poses = self._select_diverse_poses(
                    round_poses, round_energies, ligand_mol
                )
                current_poses = selected_poses

            # Track refinement progress
            round_stats = {
                'round': round_idx + 1,
                'num_poses': len(round_poses),
                'best_energy': min(round_energies),
                'mean_energy': np.mean(round_energies),
                'energy_std': np.std(round_energies),
                'runtime': time.time() - round_start_time
            }
            refinement_history.append(round_stats)

            self.logger.info(f"Round {round_idx + 1} completed: "
                           f"{len(round_poses)} poses, "
                           f"best energy: {min(round_energies):.3f} kcal/mol")

            # Check convergence
            if self._check_convergence(refinement_history):
                self.logger.info(f"Converged after {round_idx + 1} rounds")
                break

        # Final ensemble analysis
        return self._analyze_ensemble(
            all_poses, all_energies, all_scores,
            ligand_mol, refinement_history, **kwargs
        )

    def _perform_refinement_round(self, poses: List[np.ndarray],
                                 ligand_mol: Chem.Mol,
                                 scoring_functions: List[Callable],
                                 refinement_algorithms: List[Callable],
                                 round_idx: int, **kwargs) -> Dict[str, Any]:
        """Perform one round of ensemble refinement"""

        refined_poses = []
        pose_energies = []
        pose_scores = []

        # Parallel refinement of poses
        if self.max_parallel_jobs > 1:
            results = self._parallel_refinement(
                poses, ligand_mol, scoring_functions,
                refinement_algorithms, **kwargs
            )
        else:
            results = self._sequential_refinement(
                poses, ligand_mol, scoring_functions,
                refinement_algorithms, **kwargs
            )

        for result in results:
            refined_poses.extend(result['poses'])
            pose_energies.extend(result['energies'])
            pose_scores.extend(result['scores'])

        return {
            'poses': refined_poses,
            'energies': pose_energies,
            'scores': pose_scores
        }

    def _parallel_refinement(self, poses: List[np.ndarray],
                           ligand_mol: Chem.Mol,
                           scoring_functions: List[Callable],
                           refinement_algorithms: List[Callable],
                           **kwargs) -> List[Dict[str, Any]]:
        """Perform parallel refinement of poses"""

        results = []

        with ThreadPoolExecutor(max_workers=self.max_parallel_jobs) as executor:
            futures = []

            for pose in poses:
                for refine_alg in refinement_algorithms:
                    future = executor.submit(
                        self._refine_single_pose,
                        pose, ligand_mol, scoring_functions,
                        refine_alg, **kwargs
                    )
                    futures.append(future)

            for future in as_completed(futures):
                try:
                    result = future.result(timeout=300)  # 5 minute timeout
                    results.append(result)
                except Exception as e:
                    self.logger.warning(f"Refinement task failed: {e}")

        return results

    def _sequential_refinement(self, poses: List[np.ndarray],
                             ligand_mol: Chem.Mol,
                             scoring_functions: List[Callable],
                             refinement_algorithms: List[Callable],
                             **kwargs) -> List[Dict[str, Any]]:
        """Perform sequential refinement of poses"""

        results = []

        for pose in poses:
            for refine_alg in refinement_algorithms:
                try:
                    result = self._refine_single_pose(
                        pose, ligand_mol, scoring_functions,
                        refine_alg, **kwargs
                    )
                    results.append(result)
                except Exception as e:
                    self.logger.warning(f"Single pose refinement failed: {e}")

        return results

    def _refine_single_pose(self, pose: np.ndarray,
                           ligand_mol: Chem.Mol,
                           scoring_functions: List[Callable],
                           refinement_algorithm: Callable,
                           **kwargs) -> Dict[str, Any]:
        """Refine a single pose using specified algorithm"""

        # Apply refinement algorithm
        try:
            if hasattr(refinement_algorithm, 'refine_pose'):
                refined_coords, energy, stats = refinement_algorithm.refine_pose(
                    pose, ligand_mol, scoring_functions[0], **kwargs
                )
                refined_poses = [refined_coords]
            else:
                # Assume it's a docking algorithm that returns multiple poses
                result = refinement_algorithm(pose, ligand_mol, **kwargs)
                refined_poses = [p.coordinates for p in result.poses[:5]]  # Top 5 poses

        except Exception as e:
            self.logger.warning(f"Refinement algorithm failed: {e}")
            refined_poses = [pose]  # Return original pose if refinement fails

        # Score all refined poses with all scoring functions
        pose_results = {
            'poses': [],
            'energies': [],
            'scores': []
        }

        for refined_pose in refined_poses:
            scores = {}
            total_energy = 0.0

            for i, score_func in enumerate(scoring_functions):
                try:
                    energy = score_func(refined_pose, ligand_mol=ligand_mol, **kwargs)
                    score_name = f'score_{i}'
                    scores[score_name] = energy

                    # Weight scores for consensus
                    if i == 0:  # Primary scoring function
                        total_energy += energy * self.scoring_weights['primary']
                    elif i == 1:  # Secondary scoring function
                        total_energy += energy * self.scoring_weights.get('secondary', 0.3)
                    else:  # Tertiary and beyond
                        total_energy += energy * self.scoring_weights.get('tertiary', 0.1)

                except Exception as e:
                    self.logger.warning(f"Scoring function {i} failed: {e}")
                    scores[f'score_{i}'] = float('inf')

            pose_results['poses'].append(refined_pose)
            pose_results['energies'].append(total_energy)
            pose_results['scores'].append(scores)

        return pose_results

    def _select_diverse_poses(self, poses: List[np.ndarray],
                             energies: List[float],
                             ligand_mol: Chem.Mol) -> List[np.ndarray]:
        """Select diverse poses for next refinement round"""

        if len(poses) <= self.max_poses_per_round:
            return poses

        # Cluster poses by structural similarity
        clusters = self._cluster_poses(poses, ligand_mol)

        # Select representative poses from each cluster
        selected_poses = []
        cluster_energies = {}

        # Group poses by cluster
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in cluster_energies:
                cluster_energies[cluster_id] = []
            cluster_energies[cluster_id].append((i, energies[i]))

        # Select best pose from each cluster
        for cluster_id, pose_energy_pairs in cluster_energies.items():
            # Sort by energy and select best
            pose_energy_pairs.sort(key=lambda x: x[1])
            best_pose_idx = pose_energy_pairs[0][0]
            selected_poses.append(poses[best_pose_idx])

        # If we need more poses, add next best poses
        while len(selected_poses) < self.max_poses_per_round:
            remaining_indices = []
            for cluster_poses in cluster_energies.values():
                if len(cluster_poses) > 1:
                    remaining_indices.append(cluster_poses[1])  # Second best from each cluster

            if not remaining_indices:
                break

            remaining_indices.sort(key=lambda x: x[1])
            best_remaining_idx = remaining_indices[0][0]
            selected_poses.append(poses[best_remaining_idx])

            # Remove this pose from consideration
            for cluster_poses in cluster_energies.values():
                cluster_poses[:] = [p for p in cluster_poses if p[0] != best_remaining_idx]

        self.logger.debug(f"Selected {len(selected_poses)} diverse poses from {len(poses)} total")
        return selected_poses

    def _cluster_poses(self, poses: List[np.ndarray],
                       ligand_mol: Chem.Mol) -> List[int]:
        """Cluster poses based on structural similarity"""

        if len(poses) <= 1:
            return [0] * len(poses)

        # Calculate RMSD matrix
        rmsd_matrix = np.zeros((len(poses), len(poses)))

        for i in range(len(poses)):
            for j in range(i+1, len(poses)):
                rmsd = self._calculate_rmsd(poses[i], poses[j])
                rmsd_matrix[i, j] = rmsd
                rmsd_matrix[j, i] = rmsd

        # Perform clustering
        if self.clustering_method == 'hierarchical':
            # Hierarchical clustering
            condensed_distances = squareform(rmsd_matrix)
            linkage_matrix = linkage(condensed_distances, method='average')
            clusters = fcluster(linkage_matrix, self.rmsd_threshold, criterion='distance')

        elif self.clustering_method == 'dbscan':
            # DBSCAN clustering
            clustering = DBSCAN(
                eps=self.rmsd_threshold,
                min_samples=self.min_cluster_size,
                metric='precomputed'
            )
            clusters = clustering.fit_predict(rmsd_matrix)
            # Convert -1 (noise) to separate clusters
            noise_points = np.where(clusters == -1)[0]
            max_cluster = max(clusters) if len(clusters) > 0 else 0
            for i, point in enumerate(noise_points):
                clusters[point] = max_cluster + i + 1

        elif self.clustering_method == 'kmeans':
            # K-means clustering (estimate number of clusters)
            n_clusters = min(len(poses) // 3, 10)  # Heuristic for number of clusters
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(rmsd_matrix)

        else:
            # Default: each pose in its own cluster
            clusters = list(range(len(poses)))

        return clusters.tolist() if hasattr(clusters, 'tolist') else list(clusters)

    def _calculate_rmsd(self, coords1: np.ndarray, coords2: np.ndarray) -> float:
        """Calculate RMSD between two coordinate sets"""
        try:
            # Center coordinates
            center1 = np.mean(coords1, axis=0)
            center2 = np.mean(coords2, axis=0)

            centered1 = coords1 - center1
            centered2 = coords2 - center2

            # Calculate RMSD
            diff = centered1 - centered2
            rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

            return rmsd
        except:
            return float('inf')

    def _check_convergence(self, history: List[Dict[str, Any]]) -> bool:
        """Check if refinement has converged"""
        if len(history) < 2:
            return False

        recent_energies = [h['best_energy'] for h in history[-3:]]

        if len(recent_energies) >= 2:
            energy_change = abs(recent_energies[-1] - recent_energies[-2])
            return energy_change < self.convergence_threshold

        return False

    def _analyze_ensemble(self, poses: List[np.ndarray],
                         energies: List[float],
                         scores: List[Dict[str, float]],
                         ligand_mol: Chem.Mol,
                         history: List[Dict[str, Any]],
                         **kwargs) -> EnsembleResult:
        """Analyze final ensemble and generate consensus"""

        # Cluster final poses
        clusters = self._cluster_poses(poses, ligand_mol)

        # Generate consensus pose
        consensus_pose, consensus_energy = self._generate_consensus_pose(
            poses, energies, clusters
        )

        # Calculate ensemble statistics
        ensemble_stats = {
            'total_poses': len(poses),
            'unique_clusters': len(set(clusters)),
            'energy_range': max(energies) - min(energies),
            'energy_mean': np.mean(energies),
            'energy_std': np.std(energies),
            'best_energy': min(energies),
            'worst_energy': max(energies),
            'consensus_method': self.consensus_method,
            'clustering_method': self.clustering_method,
            'refinement_rounds': len(history)
        }

        return EnsembleResult(
            refined_poses=poses,
            pose_energies=energies,
            pose_scores=scores,
            consensus_pose=consensus_pose,
            consensus_energy=consensus_energy,
            cluster_assignments=clusters,
            ensemble_statistics=ensemble_stats,
            refinement_history=history
        )

    def _generate_consensus_pose(self, poses: List[np.ndarray],
                               energies: List[float],
                               clusters: List[int]) -> Tuple[np.ndarray, float]:
        """Generate consensus pose from ensemble"""

        if self.consensus_method == 'best_energy':
            # Simply return the best energy pose
            best_idx = np.argmin(energies)
            return poses[best_idx].copy(), energies[best_idx]

        elif self.consensus_method == 'cluster_centroid':
            # Return centroid of largest cluster
            cluster_counts = {}
            for i, cluster_id in enumerate(clusters):
                if cluster_id not in cluster_counts:
                    cluster_counts[cluster_id] = []
                cluster_counts[cluster_id].append(i)

            # Find largest cluster
            largest_cluster = max(cluster_counts.keys(), key=lambda x: len(cluster_counts[x]))
            cluster_poses = [poses[i] for i in cluster_counts[largest_cluster]]
            cluster_energies = [energies[i] for i in cluster_counts[largest_cluster]]

            # Calculate centroid
            centroid = np.mean(cluster_poses, axis=0)
            mean_energy = np.mean(cluster_energies)

            return centroid, mean_energy

        elif self.consensus_method == 'boltzmann_weighted':
            # Boltzmann-weighted average
            kT = 0.001987 * self.temperature  # kcal/mol
            min_energy = min(energies)

            # Calculate Boltzmann weights
            weights = []
            for energy in energies:
                weight = np.exp(-(energy - min_energy) / kT)
                weights.append(weight)

            weights = np.array(weights)
            weights = weights / np.sum(weights)  # Normalize

            # Weighted average of coordinates
            consensus_coords = np.zeros_like(poses[0])
            for i, pose in enumerate(poses):
                consensus_coords += weights[i] * pose

            # Weighted average of energies
            consensus_energy = np.sum(weights * np.array(energies))

            return consensus_coords, consensus_energy

        else:
            # Default: return best energy pose
            best_idx = np.argmin(energies)
            return poses[best_idx].copy(), energies[best_idx]


class MultiObjectiveEnsembleRefiner(EnsembleRefiner):
    """Ensemble refinement with multi-objective optimization"""

    def __init__(self, **params):
        super().__init__(**params)

        # Multi-objective parameters
        self.objectives = params.get('objectives', ['energy', 'diversity', 'druglikeness'])
        self.objective_weights = params.get('objective_weights', [0.6, 0.3, 0.1])
        self.pareto_fraction = params.get('pareto_fraction', 0.2)

        self.logger.info(f"Multi-objective ensemble refiner initialized with objectives: {self.objectives}")

    def _select_diverse_poses(self, poses: List[np.ndarray],
                             energies: List[float],
                             ligand_mol: Chem.Mol) -> List[np.ndarray]:
        """Multi-objective pose selection"""

        # Calculate multiple objectives
        objectives = self._calculate_multiple_objectives(poses, energies, ligand_mol)

        # Pareto-optimal selection
        pareto_indices = self._find_pareto_optimal(objectives)

        # Select poses
        if len(pareto_indices) >= self.max_poses_per_round:
            # Use only Pareto-optimal poses
            selected_indices = pareto_indices[:self.max_poses_per_round]
        else:
            # Add more poses based on weighted objectives
            remaining_poses = len(poses) - len(pareto_indices)
            additional_poses = min(remaining_poses,
                                 self.max_poses_per_round - len(pareto_indices))

            # Calculate weighted scores for non-Pareto poses
            non_pareto_indices = [i for i in range(len(poses)) if i not in pareto_indices]
            weighted_scores = []

            for i in non_pareto_indices:
                score = sum(w * objectives[i][j]
                           for j, w in enumerate(self.objective_weights))
                weighted_scores.append((score, i))

            weighted_scores.sort()
            additional_indices = [idx for _, idx in weighted_scores[:additional_poses]]

            selected_indices = pareto_indices + additional_indices

        return [poses[i] for i in selected_indices]

    def _calculate_multiple_objectives(self, poses: List[np.ndarray],
                                     energies: List[float],
                                     ligand_mol: Chem.Mol) -> List[List[float]]:
        """Calculate multiple objective values for each pose"""

        objectives = []

        for i, pose in enumerate(poses):
            obj_values = []

            for obj_name in self.objectives:
                if obj_name == 'energy':
                    obj_values.append(energies[i])  # Minimize
                elif obj_name == 'diversity':
                    # Diversity from other poses (maximize, so negate)
                    diversity = self._calculate_pose_diversity(pose, poses, i)
                    obj_values.append(-diversity)
                elif obj_name == 'druglikeness':
                    # Simple druglikeness score (maximize, so negate)
                    druglike = self._calculate_druglikeness(ligand_mol)
                    obj_values.append(-druglike)
                else:
                    obj_values.append(0.0)

            objectives.append(obj_values)

        return objectives

    def _find_pareto_optimal(self, objectives: List[List[float]]) -> List[int]:
        """Find Pareto-optimal solutions"""

        pareto_indices = []

        for i, obj_i in enumerate(objectives):
            is_dominated = False

            for j, obj_j in enumerate(objectives):
                if i == j:
                    continue

                # Check if obj_j dominates obj_i (all objectives better or equal, at least one strictly better)
                dominates = all(obj_j[k] <= obj_i[k] for k in range(len(obj_i)))
                strictly_better = any(obj_j[k] < obj_i[k] for k in range(len(obj_i)))

                if dominates and strictly_better:
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_indices.append(i)

        return pareto_indices

    def _calculate_pose_diversity(self, pose: np.ndarray,
                                 all_poses: List[np.ndarray],
                                 pose_index: int) -> float:
        """Calculate diversity of a pose relative to others"""

        diversities = []

        for i, other_pose in enumerate(all_poses):
            if i != pose_index:
                rmsd = self._calculate_rmsd(pose, other_pose)
                diversities.append(rmsd)

        return np.mean(diversities) if diversities else 0.0

    def _calculate_druglikeness(self, mol: Chem.Mol) -> float:
        """Simple druglikeness score"""

        try:
            mw = Chem.rdMolDescriptors.CalcExactMolWt(mol)
            logp = Chem.rdMolDescriptors.CalcCrippenDescriptors(mol)[0]
            hbd = Chem.rdMolDescriptors.CalcNumHBD(mol)
            hba = Chem.rdMolDescriptors.CalcNumHBA(mol)
            rotb = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)

            # Simple Lipinski-like scoring
            score = 1.0
            if mw > 500: score *= 0.5
            if logp > 5 or logp < -2: score *= 0.5
            if hbd > 5: score *= 0.5
            if hba > 10: score *= 0.5
            if rotb > 10: score *= 0.7

            return score

        except:
            return 0.5  # Default moderate score