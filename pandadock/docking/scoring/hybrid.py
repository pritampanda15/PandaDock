"""
Hybrid Scoring Function

Combines multiple scoring approaches for enhanced accuracy:
1. Physics-based component (30% weight)
2. Empirical component (40% weight)
3. Precision component (30% weight)

Provides balanced scoring that leverages strengths of each method
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from Bio.PDB import Structure
from rdkit import Chem
from .physics_based import PhysicsBasedScoring
from .empirical import EmpiricalScoring
from .precision_score import PrecisionScoring


class HybridScoring:
    """Hybrid scoring combining physics-based, empirical, and precision methods"""

    def __init__(self, **params):
        self.logger = logging.getLogger("pandadock.scoring.hybrid")

        # Initialize component scoring functions
        self.physics_scorer = PhysicsBasedScoring(**params)
        self.empirical_scorer = EmpiricalScoring(**params)
        self.precision_scorer = PrecisionScoring(**params)

        # Scoring weights (must sum to 1.0)
        self.weights = {
            'physics': params.get('physics_weight', 0.30),
            'empirical': params.get('empirical_weight', 0.40),
            'precision': params.get('precision_weight', 0.30)
        }

        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight != 1.0:
            self.weights = {k: v/total_weight for k, v in self.weights.items()}

        self.logger.debug(f"Hybrid scoring initialized with weights: {self.weights}")

    def calculate_binding_energy(self, ligand_coords: np.ndarray,
                                receptor_structure: Structure,
                                ligand_mol: Optional[Chem.Mol] = None) -> float:
        """
        Calculate hybrid binding score

        Args:
            ligand_coords: 3D coordinates of ligand atoms
            receptor_structure: BioPython receptor structure
            ligand_mol: RDKit molecule (optional, for detailed analysis)

        Returns:
            Hybrid binding score in kcal/mol
        """
        # Calculate individual scoring components
        try:
            physics_score = self.physics_scorer.calculate_binding_energy(
                ligand_coords, receptor_structure, ligand_mol
            )
        except Exception as e:
            self.logger.warning(f"Physics scoring failed: {e}")
            physics_score = 5.0  # Default unfavorable score

        try:
            empirical_score = self.empirical_scorer.calculate_binding_energy(
                ligand_coords, receptor_structure, ligand_mol
            )
        except Exception as e:
            self.logger.warning(f"Empirical scoring failed: {e}")
            empirical_score = 5.0  # Default unfavorable score

        try:
            precision_score = self.precision_scorer.calculate_binding_energy(
                ligand_coords, receptor_structure, ligand_mol
            )
        except Exception as e:
            self.logger.warning(f"Precision scoring failed: {e}")
            precision_score = 5.0  # Default unfavorable score

        # Weighted combination
        hybrid_score = (
            self.weights['physics'] * physics_score +
            self.weights['empirical'] * empirical_score +
            self.weights['precision'] * precision_score
        )

        # Apply consensus bonus for agreement between methods
        consensus_bonus = self._calculate_consensus_bonus(
            physics_score, empirical_score, precision_score
        )

        final_score = hybrid_score + consensus_bonus

        # Clamp to physically reasonable range (wide bounds to avoid hiding issues)
        # Floor at -50 to catch numerical errors only
        # Ceiling at +50 for severe clashes
        if final_score < -50.0:
            self.logger.warning(f"Hybrid score {final_score:.1f} clamped to -50.0 kcal/mol - check scoring!")
            final_score = -50.0
        elif final_score > 50.0:
            self.logger.warning(f"Hybrid score {final_score:.1f} clamped to +50.0 kcal/mol - severe clashes detected")
            final_score = 50.0

        self.logger.debug(f"Hybrid components - Physics: {physics_score:.3f}, "
                         f"Empirical: {empirical_score:.3f}, "
                         f"Precision: {precision_score:.3f}")
        self.logger.debug(f"Consensus bonus: {consensus_bonus:.3f}")
        self.logger.debug(f"Final hybrid score: {final_score:.3f} kcal/mol")

        return final_score

    def _calculate_consensus_bonus(self, physics_score: float,
                                 empirical_score: float,
                                 precision_score: float) -> float:
        """
        Calculate bonus for consensus between scoring methods

        When multiple methods agree on favorable binding, add bonus
        """
        scores = [physics_score, empirical_score, precision_score]

        # Count how many methods predict favorable binding (< 0)
        favorable_count = sum(1 for score in scores if score < 0)

        # Count how many methods predict very favorable binding (< -3)
        very_favorable_count = sum(1 for score in scores if score < -3.0)

        consensus_bonus = 0.0

        # Bonus for consensus on favorable binding
        if favorable_count >= 2:
            consensus_bonus -= 0.5  # Small bonus for agreement

        if favorable_count == 3:
            consensus_bonus -= 0.5  # Additional bonus for full consensus

        # Extra bonus for very favorable consensus
        if very_favorable_count >= 2:
            consensus_bonus -= 0.5

        # Check for agreement within reasonable range
        score_range = max(scores) - min(scores)
        if score_range < 3.0:  # Methods agree within 3 kcal/mol
            consensus_bonus -= 0.3

        return consensus_bonus

    def get_component_scores(self, ligand_coords: np.ndarray,
                           receptor_structure: Structure,
                           ligand_mol: Optional[Chem.Mol] = None) -> Dict[str, float]:
        """
        Get individual component scores for analysis

        Returns:
            Dictionary with individual scoring components
        """
        try:
            physics_score = self.physics_scorer.calculate_binding_energy(
                ligand_coords, receptor_structure, ligand_mol
            )
        except Exception:
            physics_score = 5.0

        try:
            empirical_score = self.empirical_scorer.calculate_binding_energy(
                ligand_coords, receptor_structure, ligand_mol
            )
        except Exception:
            empirical_score = 5.0

        try:
            precision_score = self.precision_scorer.calculate_binding_energy(
                ligand_coords, receptor_structure, ligand_mol
            )
        except Exception:
            precision_score = 5.0

        consensus_bonus = self._calculate_consensus_bonus(
            physics_score, empirical_score, precision_score
        )

        hybrid_score = (
            self.weights['physics'] * physics_score +
            self.weights['empirical'] * empirical_score +
            self.weights['precision'] * precision_score +
            consensus_bonus
        )

        return {
            'physics': physics_score,
            'empirical': empirical_score,
            'precision': precision_score,
            'consensus_bonus': consensus_bonus,
            'hybrid_total': hybrid_score  # No artificial clamping
        }