"""
Ensemble Boltzmann Averaging for Binding Free Energy Calculation

Implements the core Boltzmann ensemble methods as specified in your algorithm document:
- Boltzmann weighting of poses
- Ensemble binding free energy calculation
- Confidence estimation
- Calibration against experimental data
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Optional
import logging
from scipy import stats
from sklearn.linear_model import Ridge
from ..core import Pose, DockingResult


class BoltzmannEnsemble:
    """Boltzmann ensemble calculations for binding free energy estimation"""

    def __init__(self, temperature: float = 298.15, **params):
        self.logger = logging.getLogger("pandadock.scoring.ensemble")

        # Physical constants
        self.temperature = temperature  # Kelvin
        self.R = 0.0019872041  # kcal/(mol·K) - gas constant
        self.kT = self.R * self.temperature  # kcal/mol (~0.593 at 298K)

        # Ensemble parameters
        self.max_poses_for_ensemble = params.get('max_poses', 50)
        self.energy_cutoff = params.get('energy_cutoff', 20.0)  # kcal/mol above minimum

        # Calibration parameters (fitted to experimental data)
        self.calibration_slope = params.get('calibration_slope', 1.0)
        self.calibration_intercept = params.get('calibration_intercept', 0.0)
        self.is_calibrated = False

        self.logger.info(f"Initialized Boltzmann ensemble at T={temperature}K, kT={self.kT:.3f} kcal/mol")

    def calculate_ensemble_binding_energy(self, poses: List[Pose],
                                        add_entropy_correction: bool = True) -> Tuple[float, List[float]]:
        """
        Calculate ensemble binding free energy using Boltzmann averaging

        Formula: ΔG_ensemble = -kT * ln(Σ exp(-Ei/kT)) + C

        Args:
            poses: List of Pose objects with calculated energies
            add_entropy_correction: Whether to add rotatable bond entropy penalty

        Returns:
            Tuple of (ensemble_binding_energy, boltzmann_weights)
        """
        if not poses:
            raise ValueError("No poses provided for ensemble calculation")

        # Filter and select poses for ensemble
        selected_poses = self._select_poses_for_ensemble(poses)

        if len(selected_poses) == 0:
            self.logger.warning("No valid poses for ensemble calculation")
            return float('inf'), []

        self.logger.info(f"Calculating ensemble energy from {len(selected_poses)} poses")

        # Extract energies
        energies = np.array([pose.energy for pose in selected_poses])

        # Shift energies to avoid numerical overflow (subtract minimum)
        min_energy = np.min(energies)
        shifted_energies = energies - min_energy

        # Calculate Boltzmann weights
        boltzmann_factors = np.exp(-shifted_energies / self.kT)
        partition_function = np.sum(boltzmann_factors)

        # Normalized weights
        boltzmann_weights = boltzmann_factors / partition_function

        # Ensemble free energy (raw, before calibration)
        ensemble_free_energy_raw = -self.kT * math.log(partition_function) + min_energy

        # Add entropy corrections
        if add_entropy_correction:
            entropy_penalty = self._calculate_entropy_penalty(selected_poses[0])
            ensemble_free_energy_raw += entropy_penalty

        # Apply calibration only if one has actually been fitted.
        #
        # There used to be a "default calibration" here: a hand-written piecewise
        # remap, clamped to [-15, 10], applied whenever no calibration existed. It
        # was not fitted to anything, and it moved the reported ensemble energy
        # outside the range of the pose scores it summarises -- a run whose poses
        # spanned -16.3 to -14.5 kcal/mol reported an ensemble dG of -8.3. Without
        # a fitted calibration the raw Boltzmann free energy is returned, which is
        # on the same scale as the pose scores and is interpretable as such.
        if self.is_calibrated:
            ensemble_free_energy = (self.calibration_slope * ensemble_free_energy_raw +
                                  self.calibration_intercept)
        else:
            ensemble_free_energy = ensemble_free_energy_raw

        self.logger.debug(f"Raw ensemble energy: {ensemble_free_energy_raw:.3f} kcal/mol")
        self.logger.debug(f"Calibrated ensemble energy: {ensemble_free_energy:.3f} kcal/mol")
        self.logger.debug(f"Effective poses in ensemble: {np.sum(boltzmann_weights > 0.01)}")

        return ensemble_free_energy, boltzmann_weights.tolist()

    def calculate_ensemble_confidence(self, poses: List[Pose],
                                    boltzmann_weights: List[float]) -> float:
        """
        Calculate confidence score for ensemble prediction

        Based on:
        1. Energy spread of poses
        2. Number of significant poses
        3. Convergence of Boltzmann weighting
        """
        if not poses or not boltzmann_weights:
            return 0.0

        weights = np.array(boltzmann_weights)
        energies = np.array([pose.energy for pose in poses[:len(weights)]])

        # Factor 1: Energy spread (lower spread = higher confidence)
        energy_spread = np.std(energies)
        spread_factor = math.exp(-energy_spread / 5.0)  # Normalized by 5 kcal/mol

        # Factor 2: Number of significant poses (more poses = higher confidence)
        significant_poses = np.sum(weights > 0.01)  # Poses with >1% weight
        pose_factor = min(significant_poses / 10.0, 1.0)  # Normalized by 10 poses

        # Factor 3: Weight distribution (uniform = higher confidence)
        # Use Shannon entropy of weights
        weights_nonzero = weights[weights > 1e-10]
        shannon_entropy = -np.sum(weights_nonzero * np.log(weights_nonzero))
        max_entropy = math.log(len(weights_nonzero))
        entropy_factor = shannon_entropy / max_entropy if max_entropy > 0 else 0.0

        # Combined confidence (0-1 scale)
        confidence = (spread_factor * 0.4 + pose_factor * 0.3 + entropy_factor * 0.3)

        self.logger.debug(f"Confidence factors - spread: {spread_factor:.3f}, "
                         f"poses: {pose_factor:.3f}, entropy: {entropy_factor:.3f}")

        return min(confidence, 1.0)

    def calibrate_to_experimental_data(self, docking_results: List[DockingResult],
                                     experimental_affinities: List[float],
                                     experimental_type: str = "kd") -> Dict[str, float]:
        """
        Calibrate ensemble energies to experimental binding affinities

        Args:
            docking_results: List of DockingResult objects
            experimental_affinities: List of experimental values
            experimental_type: "kd" (Kd in M), "ki" (Ki in M), or "deltaG" (ΔG in kcal/mol)

        Returns:
            Calibration statistics
        """
        if len(docking_results) != len(experimental_affinities):
            raise ValueError("Number of docking results must match experimental data")

        if len(docking_results) < 5:
            raise ValueError("Need at least 5 data points for calibration")

        # Convert experimental data to ΔG if needed
        exp_delta_g = self._convert_to_delta_g(experimental_affinities, experimental_type)

        # Calculate ensemble energies for all results
        predicted_energies = []
        for result in docking_results:
            if result.poses:
                ensemble_energy, _ = self.calculate_ensemble_binding_energy(
                    result.poses, add_entropy_correction=True
                )
                predicted_energies.append(ensemble_energy)
            else:
                predicted_energies.append(float('inf'))

        predicted_energies = np.array(predicted_energies)
        exp_delta_g = np.array(exp_delta_g)

        # Remove infinite values
        valid_mask = np.isfinite(predicted_energies)
        predicted_energies = predicted_energies[valid_mask]
        exp_delta_g = exp_delta_g[valid_mask]

        if len(predicted_energies) < 3:
            raise ValueError("Not enough valid predictions for calibration")

        # Fit linear calibration: ΔG_exp = slope * ΔG_pred + intercept
        ridge = Ridge(alpha=1.0)  # Small regularization
        ridge.fit(predicted_energies.reshape(-1, 1), exp_delta_g)

        self.calibration_slope = ridge.coef_[0]
        self.calibration_intercept = ridge.intercept_
        self.is_calibrated = True

        # Calculate calibration statistics
        calibrated_predictions = (self.calibration_slope * predicted_energies +
                                self.calibration_intercept)

        rmse = np.sqrt(np.mean((calibrated_predictions - exp_delta_g) ** 2))
        mae = np.mean(np.abs(calibrated_predictions - exp_delta_g))
        r_squared = stats.pearsonr(calibrated_predictions, exp_delta_g)[0] ** 2

        calibration_stats = {
            'slope': self.calibration_slope,
            'intercept': self.calibration_intercept,
            'rmse': rmse,
            'mae': mae,
            'r_squared': r_squared,
            'n_points': len(predicted_energies)
        }

        self.logger.info(f"Calibration completed: slope={self.calibration_slope:.3f}, "
                        f"intercept={self.calibration_intercept:.3f}")
        self.logger.info(f"Calibration statistics: RMSE={rmse:.3f}, MAE={mae:.3f}, "
                        f"R²={r_squared:.3f}")

        return calibration_stats

    def _select_poses_for_ensemble(self, poses: List[Pose]) -> List[Pose]:
        """Select poses for ensemble calculation based on energy filtering"""
        if not poses:
            return []

        # Sort by energy
        sorted_poses = sorted(poses, key=lambda p: p.energy)

        # Remove poses with infinite or very high energies (increased threshold for development)
        valid_poses = [p for p in sorted_poses if np.isfinite(p.energy) and p.energy < 50000.0]

        if not valid_poses:
            return []

        # Filter by energy cutoff relative to best pose
        min_energy = valid_poses[0].energy
        energy_threshold = min_energy + self.energy_cutoff

        filtered_poses = [p for p in valid_poses if p.energy <= energy_threshold]

        # Limit number of poses for computational efficiency
        selected_poses = filtered_poses[:self.max_poses_for_ensemble]

        self.logger.debug(f"Selected {len(selected_poses)} poses from {len(poses)} total poses")
        self.logger.debug(f"Energy range: {min_energy:.3f} to {selected_poses[-1].energy:.3f} kcal/mol")

        return selected_poses

    def _calculate_entropy_penalty(self, representative_pose: Pose,
                                   n_rotatable_bonds: Optional[int] = None) -> float:
        """
        Entropy penalty for ligand binding.

        The rotatable bond count is taken from the pose when available. It was
        previously hardcoded to 5 with a "Placeholder" comment, so every ligand
        received an identical +5.0 kcal/mol correction regardless of its actual
        flexibility -- which defeats the purpose of a flexibility-dependent term.
        """
        if n_rotatable_bonds is None:
            torsions = getattr(representative_pose, 'torsion_angles', None)
            n_rotatable_bonds = len(torsions) if torsions is not None else 0

        entropy_per_bond = 0.5

        # Translational and rotational entropy loss on binding.
        translational_entropy = 1.5
        rotational_entropy = 1.0

        total_entropy_penalty = (n_rotatable_bonds * entropy_per_bond +
                                 translational_entropy + rotational_entropy)

        self.logger.debug(
            f"Entropy penalty: {total_entropy_penalty:.3f} kcal/mol "
            f"({n_rotatable_bonds} rotatable bonds)"
        )

        return total_entropy_penalty

    def _apply_default_calibration(self, raw_energy: float) -> float:
        """Apply default calibration to convert to realistic binding free energies"""
        # Default calibration based on typical docking performance
        # Raw energies typically range widely, but binding energies should be -15 to +5 kcal/mol

        if raw_energy < -50:
            # Very negative energies - likely over-attractive
            calibrated = -12.0 + (raw_energy + 50) * 0.1
        elif raw_energy < -10:
            # Moderately negative - good binding range
            calibrated = -8.0 + (raw_energy + 10) * 0.2
        elif raw_energy < 10:
            # Slightly negative to slightly positive
            calibrated = -2.0 + raw_energy * 0.3
        elif raw_energy < 50:
            # Positive energies - poor binding
            calibrated = 1.0 + (raw_energy - 10) * 0.1
        else:
            # Very high positive energies
            calibrated = 5.0 + (raw_energy - 50) * 0.05

        # Clamp to reasonable binding energy range
        return max(-15.0, min(10.0, calibrated))

    def _convert_to_delta_g(self, values: List[float], value_type: str) -> List[float]:
        """Convert experimental binding affinities to ΔG (kcal/mol)"""
        if value_type.lower() == "deltag":
            return values
        elif value_type.lower() in ["kd", "ki"]:
            # Convert Kd/Ki (M) to ΔG using: ΔG = RT ln(Kd)
            return [self.R * self.temperature * math.log(kd) for kd in values]
        else:
            raise ValueError(f"Unknown experimental value type: {value_type}")

    def update_docking_result_with_ensemble(self, result: DockingResult) -> None:
        """Update DockingResult with ensemble calculations"""
        if not result.poses:
            result.ensemble_binding_energy = float('inf')
            result.ensemble_confidence = 0.0
            result.boltzmann_weights = []
            return

        # Calculate ensemble energy and weights
        ensemble_energy, weights = self.calculate_ensemble_binding_energy(result.poses)
        confidence = self.calculate_ensemble_confidence(result.poses, weights)

        # Update result
        result.ensemble_binding_energy = ensemble_energy
        result.ensemble_confidence = confidence
        result.boltzmann_weights = weights

        self.logger.info(f"Updated {result.ligand_name}: "
                        f"ensemble ΔG = {ensemble_energy:.3f} kcal/mol, "
                        f"confidence = {confidence:.3f}")

    def predict_experimental_affinity(self, ensemble_energy: float,
                                    output_type: str = "kd") -> float:
        """
        Predict experimental binding affinity from ensemble energy

        Args:
            ensemble_energy: Ensemble binding energy (kcal/mol)
            output_type: "kd" (M), "ki" (M), or "deltaG" (kcal/mol)

        Returns:
            Predicted experimental value
        """
        if output_type.lower() == "deltag":
            return ensemble_energy
        elif output_type.lower() in ["kd", "ki"]:
            # Convert ΔG to Kd/Ki using: Kd = exp(ΔG/RT)
            return math.exp(ensemble_energy / (self.R * self.temperature))
        else:
            raise ValueError(f"Unknown output type: {output_type}")