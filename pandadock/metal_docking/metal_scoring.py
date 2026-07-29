"""
Metal-Aware Scoring Functions

Specialized scoring functions for metal-containing ligands and metalloproteins
following AutoDock and GOLD protocols.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from rdkit import Chem
from ..docking.scoring.physics_based import PhysicsBasedScoring
from .metal_parameters import MetalParameterManager

class MetalAwareScoring:
    """Metal-aware scoring function for docking calculations"""

    def __init__(self, parameter_manager: Optional[MetalParameterManager] = None):
        """
        Initialize metal-aware scoring

        Args:
            parameter_manager: Metal parameter manager instance
        """
        self.logger = logging.getLogger(__name__)

        if parameter_manager is None:
            parameter_manager = MetalParameterManager()

        self.param_manager = parameter_manager
        self.base_scoring = PhysicsBasedScoring()

        # Metal coordination preferences
        self.coordination_preferences = {
            'Zn': {'preferred_atoms': ['N', 'S', 'O'], 'preferred_distance': 2.1, 'coordination_number': 4},
            'ZN': {'preferred_atoms': ['N', 'S', 'O'], 'preferred_distance': 2.1, 'coordination_number': 4},
            'Fe': {'preferred_atoms': ['N', 'S', 'O'], 'preferred_distance': 2.2, 'coordination_number': 6},
            'FE': {'preferred_atoms': ['N', 'S', 'O'], 'preferred_distance': 2.2, 'coordination_number': 6},
            'Cu': {'preferred_atoms': ['N', 'S', 'O'], 'preferred_distance': 2.0, 'coordination_number': 4},
            'Mg': {'preferred_atoms': ['O'], 'preferred_distance': 2.1, 'coordination_number': 6},
            'MG': {'preferred_atoms': ['O'], 'preferred_distance': 2.1, 'coordination_number': 6},
            'Ca': {'preferred_atoms': ['O'], 'preferred_distance': 2.4, 'coordination_number': 8},
            'CA': {'preferred_atoms': ['O'], 'preferred_distance': 2.4, 'coordination_number': 8},
            'Mn': {'preferred_atoms': ['N', 'O'], 'preferred_distance': 2.2, 'coordination_number': 6},
            'MN': {'preferred_atoms': ['N', 'O'], 'preferred_distance': 2.2, 'coordination_number': 6},
        }

    def calculate_metal_coordination_score(self, ligand_coords: np.ndarray,
                                         ligand_atoms: List[str],
                                         protein_metal_sites: List[Dict]) -> float:
        """
        Calculate metal coordination interaction score

        Args:
            ligand_coords: Ligand atom coordinates
            ligand_atoms: Ligand atom types
            protein_metal_sites: List of protein metal site dictionaries

        Returns:
            Metal coordination score (more negative = better)
        """
        total_score = 0.0

        for metal_site in protein_metal_sites:
            metal_coords = metal_site['metal_coords']
            metal_element = metal_site['metal_element']
            net_charge = metal_site.get('net_charge', 2.0)

            # Calculate interactions with ligand atoms
            metal_score = self._calculate_single_metal_interaction(
                metal_coords, metal_element, net_charge,
                ligand_coords, ligand_atoms
            )

            total_score += metal_score

        return total_score

    def _calculate_single_metal_interaction(self, metal_coords: np.ndarray,
                                          metal_element: str, metal_charge: float,
                                          ligand_coords: np.ndarray,
                                          ligand_atoms: List[str]) -> float:
        """Calculate interaction score for a single metal center"""
        score = 0.0

        if metal_element not in self.coordination_preferences:
            return score

        prefs = self.coordination_preferences[metal_element]
        preferred_atoms = prefs['preferred_atoms']
        preferred_distance = prefs['preferred_distance']

        coordinating_count = 0

        for i, (coord, atom_type) in enumerate(zip(ligand_coords, ligand_atoms)):
            distance = np.linalg.norm(coord - metal_coords)

            # Skip if too far for coordination
            if distance > 3.5:
                continue

            # Check if atom type can coordinate
            atom_element = atom_type.replace('A', '').replace('S', '')  # Remove AutoDock suffixes
            if atom_element not in preferred_atoms:
                continue

            # Calculate distance-based score
            if distance < 1.5:  # Too close - steric clash
                score += 50.0  # Penalty
            elif distance <= preferred_distance + 0.3:  # Good coordination distance
                distance_score = -10.0 * np.exp(-(distance - preferred_distance)**2 / (2 * 0.2**2))
                score += distance_score
                coordinating_count += 1

                # Electrostatic contribution
                if metal_charge > 0:
                    if atom_element in ['O', 'S']:  # Anionic/polar donors
                        score -= 5.0 * metal_charge / distance
                    elif atom_element == 'N':  # Neutral donors
                        score -= 2.0 * metal_charge / distance

            else:  # Too far but still interacting
                score += 2.0 * (distance - preferred_distance)  # Penalty for being too far

        # Penalty for incomplete coordination
        if coordinating_count > 0:
            expected_coordination = min(prefs['coordination_number'], len(ligand_atoms))
            coordination_penalty = max(0, expected_coordination - coordinating_count) * 2.0
            score += coordination_penalty

        return score

    def calculate_metal_binding_term(self, ligand_coords: np.ndarray,
                                   ligand_atoms: List[str],
                                   protein_metal_sites: List[Dict],
                                   net_protein_charge: float = 0.0) -> float:
        """
        Calculate metal binding term similar to AutoDock

        Args:
            ligand_coords: Ligand coordinates
            ligand_atoms: Ligand atom types
            protein_metal_sites: Metal sites in protein
            net_protein_charge: Net charge of protein metal sites

        Returns:
            Metal binding score
        """
        metal_term = 0.0

        # Count anionic and polar acceptor atoms in ligand
        anionic_count = 0
        polar_count = 0

        for atom_type in ligand_atoms:
            atom_element = atom_type.replace('A', '').replace('S', '')
            if atom_element in ['O', 'S'] and 'A' in atom_type:  # Acceptor atoms
                if atom_element == 'O':
                    polar_count += 1
                elif atom_element == 'S':
                    anionic_count += 1

        # Apply metal binding preference based on net charge
        if net_protein_charge > 0:  # Positive metal sites prefer anionic/polar ligands
            preference_score = -2.0 * (anionic_count + 0.5 * polar_count)
            metal_term += preference_score
        elif net_protein_charge == 0:  # Neutral sites - suppress preference
            metal_term += 0.0

        return metal_term

    def score_metal_pose(self, ligand_coords: np.ndarray,
                        ligand_mol: Chem.Mol,
                        receptor_structure,
                        receptor_atoms: List[str],
                        protein_metal_sites: List[Dict]) -> Dict[str, float]:
        """
        Score a pose with metal-aware scoring

        Args:
            ligand_coords: Ligand coordinates
            ligand_mol: RDKit molecule object
            receptor_structure: BioPython receptor structure. This parameter was
                previously named `receptor_coords` and callers passed an empty
                array on the assumption it was unused, but it is forwarded
                straight to the base scoring function, which needs a structure.
                Passing an array raised AttributeError on every metal-aware score.
            receptor_atoms: Receptor atom types
            protein_metal_sites: Metal sites in protein

        Returns:
            Dictionary of scoring components
        """
        # Get base physics-based score
        base_score = self.base_scoring.calculate_binding_energy(
            ligand_coords, receptor_structure, ligand_mol
        )

        # Get ligand atom types
        ligand_atoms = [atom.GetSymbol() for atom in ligand_mol.GetAtoms()]

        # Calculate metal-specific components
        metal_coordination_score = self.calculate_metal_coordination_score(
            ligand_coords, ligand_atoms, protein_metal_sites
        )

        # Calculate net protein charge
        net_charge = sum(site.get('net_charge', 0) for site in protein_metal_sites)

        metal_binding_score = self.calculate_metal_binding_term(
            ligand_coords, ligand_atoms, protein_metal_sites, net_charge
        )

        # Combine scores
        total_score = base_score + metal_coordination_score + metal_binding_score

        return {
            'total_score': total_score,
            'base_score': base_score,
            'metal_coordination': metal_coordination_score,
            'metal_binding': metal_binding_score,
            'coordinating_atoms': len([a for a in ligand_atoms if a in ['N', 'O', 'S']])
        }

    def validate_metal_coordination(self, ligand_coords: np.ndarray,
                                  ligand_atoms: List[str],
                                  metal_coords: np.ndarray,
                                  metal_element: str) -> bool:
        """
        Validate that metal coordination is geometrically reasonable

        Args:
            ligand_coords: Ligand coordinates
            ligand_atoms: Ligand atom types
            metal_coords: Metal center coordinates
            metal_element: Metal element symbol

        Returns:
            True if coordination is reasonable
        """
        if metal_element not in self.coordination_preferences:
            return True  # Can't validate unknown metals

        prefs = self.coordination_preferences[metal_element]
        coordinating_atoms = []

        # Find coordinating ligand atoms
        for i, (coord, atom_type) in enumerate(zip(ligand_coords, ligand_atoms)):
            distance = np.linalg.norm(coord - metal_coords)
            atom_element = atom_type.replace('A', '').replace('S', '')

            if (distance <= 3.0 and
                atom_element in prefs['preferred_atoms']):
                coordinating_atoms.append(coord)

        if len(coordinating_atoms) < 2:
            return True  # Too few atoms to validate geometry

        # Check for reasonable bond angles
        for i in range(len(coordinating_atoms)):
            for j in range(i + 1, len(coordinating_atoms)):
                vec1 = coordinating_atoms[i] - metal_coords
                vec2 = coordinating_atoms[j] - metal_coords

                vec1_norm = vec1 / np.linalg.norm(vec1)
                vec2_norm = vec2 / np.linalg.norm(vec2)

                angle = np.arccos(np.clip(np.dot(vec1_norm, vec2_norm), -1.0, 1.0))
                angle_deg = np.degrees(angle)

                # Check for unreasonable angles (too small)
                if angle_deg < 60.0:  # Too acute
                    return False

        return True


class MetalConstrainedScoring(MetalAwareScoring):
    """Scoring with geometric constraints for metal coordination"""

    def __init__(self, parameter_manager: Optional[MetalParameterManager] = None,
                 constraint_weight: float = 10.0):
        """
        Initialize constrained scoring

        Args:
            parameter_manager: Metal parameter manager
            constraint_weight: Weight for constraint violations
        """
        super().__init__(parameter_manager)
        self.constraint_weight = constraint_weight

    def apply_geometric_constraints(self, ligand_coords: np.ndarray,
                                  ligand_atoms: List[str],
                                  protein_metal_sites: List[Dict]) -> float:
        """
        Apply geometric constraints for metal coordination

        Args:
            ligand_coords: Ligand coordinates
            ligand_atoms: Ligand atom types
            protein_metal_sites: Metal sites in protein

        Returns:
            Constraint penalty (positive value)
        """
        total_penalty = 0.0

        for metal_site in protein_metal_sites:
            metal_coords = metal_site['metal_coords']
            metal_element = metal_site['metal_element']

            # Validate coordination geometry
            if not self.validate_metal_coordination(
                ligand_coords, ligand_atoms, metal_coords, metal_element):
                total_penalty += self.constraint_weight

            # Distance constraints
            for i, coord in enumerate(ligand_coords):
                distance = np.linalg.norm(coord - metal_coords)

                if distance < 1.0:  # Severe steric clash with metal
                    total_penalty += self.constraint_weight * 5.0
                elif distance > 4.0 and ligand_atoms[i] in ['N', 'O', 'S']:
                    # Coordinating atom too far from metal
                    total_penalty += self.constraint_weight * 0.5

        return total_penalty

    def score_constrained_pose(self, ligand_coords: np.ndarray,
                             ligand_mol: Chem.Mol,
                             receptor_structure,
                             receptor_atoms: List[str],
                             protein_metal_sites: List[Dict]) -> Dict[str, float]:
        """
        Score pose with geometric constraints

        Args:
            ligand_coords: Ligand coordinates
            ligand_mol: RDKit molecule
            receptor_coords: Receptor coordinates
            receptor_atoms: Receptor atom types
            protein_metal_sites: Metal sites

        Returns:
            Scoring components including constraint penalties
        """
        # Get base metal-aware score
        scores = self.score_metal_pose(
            ligand_coords, ligand_mol, receptor_structure,
            receptor_atoms, protein_metal_sites
        )

        # Add constraint penalties
        ligand_atoms = [atom.GetSymbol() for atom in ligand_mol.GetAtoms()]
        constraint_penalty = self.apply_geometric_constraints(
            ligand_coords, ligand_atoms, protein_metal_sites
        )

        scores['constraint_penalty'] = constraint_penalty
        scores['total_score'] += constraint_penalty

        return scores