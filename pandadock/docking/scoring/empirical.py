"""
Empirical Scoring Function

Implements ChemScore-like empirical scoring based on:
1. Hydrogen bonding contributions
2. Hydrophobic contact rewards
3. Metal coordination bonuses
4. Rotatable bond penalties
5. Clash penalties
6. Solvation corrections

Based on empirical data and statistical analysis of protein-ligand complexes
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from scipy.spatial.distance import cdist
from Bio.PDB import Structure
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors
import math


class EmpiricalScoring:
    """ChemScore-like empirical scoring function"""

    def __init__(self, **params):
        self.logger = logging.getLogger("pandadock.scoring.empirical")

        # Empirical scoring weights (calibrated from experimental data)
        self.weights = {
            'hbond': params.get('hbond_weight', -3.0),  # kcal/mol per H-bond
            'hydrophobic': params.get('hydrophobic_weight', -0.5),  # kcal/mol per contact
            'metal_coordination': params.get('metal_weight', -10.0),  # Strong metal bonus
            'rotatable_bond': params.get('rotb_weight', 1.5),  # Entropy penalty
            'clash_penalty': params.get('clash_weight', 15.0),  # Reduced clash penalty
            'solvation': params.get('solvation_weight', 0.01),  # Small solvation correction
            'size_correction': params.get('size_weight', 0.1)  # Molecular size correction
        }

        # Distance cutoffs (Å)
        self.hbond_cutoff = 3.5
        self.hydrophobic_cutoff = 4.0
        self.metal_cutoff = 3.0
        self.clash_cutoff = 2.0

        # Atom type classifications
        self.hbond_donors = {'N', 'O'}
        self.hbond_acceptors = {'N', 'O', 'S', 'F'}
        self.hydrophobic_atoms = {'C'}
        self.metal_elements = {'MG', 'CA', 'ZN', 'FE', 'MN', 'CO', 'NI', 'CU'}

        # Residue classifications
        self.hydrophobic_residues = {'ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TRP', 'PRO'}
        self.polar_residues = {'SER', 'THR', 'ASN', 'GLN', 'TYR', 'CYS'}
        self.charged_residues = {'ARG', 'LYS', 'ASP', 'GLU', 'HIS'}

        self.logger.debug("Empirical scoring initialized with ChemScore-like parameters")

    def calculate_binding_energy(self, ligand_coords: np.ndarray,
                                receptor_structure: Structure,
                                ligand_mol: Optional[Chem.Mol] = None) -> float:
        """
        Calculate empirical binding score

        Args:
            ligand_coords: 3D coordinates of ligand atoms
            receptor_structure: BioPython receptor structure
            ligand_mol: RDKit molecule (optional, for detailed analysis)

        Returns:
            Empirical binding score in kcal/mol
        """
        # Get receptor information
        receptor_atoms = list(receptor_structure.get_atoms())
        receptor_coords = np.array([atom.get_coord() for atom in receptor_atoms])

        # Calculate individual scoring terms
        score_components = {}

        # 1. Hydrogen bonding score
        score_components['hbond'] = self._calculate_hbond_score(
            ligand_coords, receptor_atoms, ligand_mol
        )

        # 2. Hydrophobic contact score
        score_components['hydrophobic'] = self._calculate_hydrophobic_score(
            ligand_coords, receptor_coords, receptor_atoms, ligand_mol
        )

        # 3. Metal coordination score
        score_components['metal_coordination'] = self._calculate_metal_score(
            ligand_coords, receptor_atoms, ligand_mol
        )

        # 4. Rotatable bond penalty
        score_components['rotatable_bond'] = self._calculate_rotatable_bond_penalty(ligand_mol)

        # 5. Clash penalty
        score_components['clash_penalty'] = self._calculate_clash_penalty(
            ligand_coords, receptor_coords
        )

        # 6. Solvation correction
        score_components['solvation'] = self._calculate_solvation_correction(
            ligand_coords, receptor_coords, ligand_mol
        )

        # 7. Size correction
        score_components['size_correction'] = self._calculate_size_correction(ligand_mol)

        # Calculate total weighted score
        total_score = sum(
            self.weights[term] * score
            for term, score in score_components.items()
        )

        # Apply scaling to get realistic binding energies
        scaled_score = self._scale_empirical_score(total_score)

        # Apply interaction bias
        interaction_corrected_score = self._apply_interaction_bias(
            scaled_score, ligand_coords, receptor_structure
        )

        self.logger.debug(f"Empirical score components: {score_components}")
        self.logger.debug(f"Total empirical score: {total_score:.3f} kcal/mol")
        self.logger.debug(f"Scaled score: {interaction_corrected_score:.3f} kcal/mol")

        return interaction_corrected_score

    def _calculate_hbond_score(self, ligand_coords: np.ndarray, receptor_atoms: List,
                              ligand_mol: Optional[Chem.Mol]) -> float:
        """Calculate hydrogen bonding contribution"""
        if ligand_mol is None:
            return 0.0

        hbond_count = 0

        # Get ligand H-bond atoms
        lig_hb_atoms = self._get_ligand_hb_atoms(ligand_mol)

        for lig_idx, lig_type in lig_hb_atoms.items():
            lig_coord = ligand_coords[lig_idx]

            for rec_idx, rec_atom in enumerate(receptor_atoms):
                # Check if receptor atom can form H-bonds
                if not self._can_form_hbond(rec_atom):
                    continue

                rec_coord = rec_atom.get_coord()
                distance = np.linalg.norm(lig_coord - rec_coord)

                if distance <= self.hbond_cutoff:
                    # Check if donor-acceptor pair
                    rec_type = self._get_receptor_hb_type(rec_atom)

                    if ((lig_type == 'donor' and rec_type == 'acceptor') or
                        (lig_type == 'acceptor' and rec_type == 'donor')):

                        # Distance-dependent H-bond strength
                        strength = self._hbond_strength_function(distance)
                        hbond_count += strength

        return hbond_count

    def _calculate_hydrophobic_score(self, ligand_coords: np.ndarray, receptor_coords: np.ndarray,
                                   receptor_atoms: List, ligand_mol: Optional[Chem.Mol]) -> float:
        """Calculate hydrophobic contact contribution"""
        if ligand_mol is None:
            return 0.0

        contact_count = 0

        # Get hydrophobic atoms in ligand
        lig_hydrophobic = []
        for i, atom in enumerate(ligand_mol.GetAtoms()):
            if atom.GetSymbol() == 'C' and atom.GetTotalNumHs() > 0:
                lig_hydrophobic.append(int(i))

        for lig_idx in lig_hydrophobic:
            lig_coord = ligand_coords[lig_idx]

            for rec_idx, rec_atom in enumerate(receptor_atoms):
                # Check if receptor atom is hydrophobic
                if (rec_atom.element == 'C' and
                    rec_atom.get_parent().get_resname() in self.hydrophobic_residues):

                    rec_coord = receptor_coords[rec_idx]
                    distance = np.linalg.norm(lig_coord - rec_coord)

                    if distance <= self.hydrophobic_cutoff:
                        # Distance-dependent contact strength
                        strength = self._hydrophobic_strength_function(distance)
                        contact_count += strength

        return contact_count

    def _calculate_metal_score(self, ligand_coords: np.ndarray, receptor_atoms: List,
                             ligand_mol: Optional[Chem.Mol]) -> float:
        """Calculate metal coordination contribution"""
        if ligand_mol is None:
            return 0.0

        coordination_count = 0

        # Find metal atoms in receptor
        metal_atoms = []
        for i, atom in enumerate(receptor_atoms):
            if atom.element in self.metal_elements:
                metal_atoms.append((i, atom))

        if not metal_atoms:
            return 0.0

        # Find potential coordinating atoms in ligand
        for lig_idx, atom in enumerate(ligand_mol.GetAtoms()):
            if atom.GetSymbol() in ['O', 'N', 'S']:  # Potential coordinating atoms
                lig_coord = ligand_coords[int(lig_idx)]

                for metal_idx, metal_atom in metal_atoms:
                    metal_coord = metal_atom.get_coord()
                    distance = np.linalg.norm(lig_coord - metal_coord)

                    if distance <= self.metal_cutoff:
                        # Strong coordination bonus
                        coordination_count += 1

        return coordination_count

    def _calculate_rotatable_bond_penalty(self, ligand_mol: Optional[Chem.Mol]) -> float:
        """Calculate entropy penalty for rotatable bonds"""
        if ligand_mol is None:
            return 0.0

        num_rotatable = rdMolDescriptors.CalcNumRotatableBonds(ligand_mol)
        return num_rotatable  # Returns positive penalty

    def _calculate_clash_penalty(self, ligand_coords: np.ndarray, receptor_coords: np.ndarray) -> float:
        """Calculate penalty for severe atomic clashes"""
        clash_penalty = 0.0

        # Calculate all pairwise distances
        distances = cdist(ligand_coords, receptor_coords)

        # Count severe clashes (< 2.0 Å)
        severe_clashes = int(np.sum(distances < self.clash_cutoff))
        clash_penalty = severe_clashes

        return clash_penalty

    def _calculate_solvation_correction(self, ligand_coords: np.ndarray, receptor_coords: np.ndarray,
                                      ligand_mol: Optional[Chem.Mol]) -> float:
        """Calculate simple solvation correction"""
        if ligand_mol is None:
            return 0.0

        # Simple burial correction based on number of contacts
        total_contacts = 0

        for lig_coord in ligand_coords:
            # Count receptor atoms within 5 Å
            distances = np.linalg.norm(receptor_coords - lig_coord, axis=1)
            contacts = int(np.sum(distances < 5.0))
            total_contacts += contacts

        # More buried ligand = less favorable solvation
        return total_contacts

    def _calculate_size_correction(self, ligand_mol: Optional[Chem.Mol]) -> float:
        """Calculate molecular size correction"""
        if ligand_mol is None:
            return 0.0

        # Simple size correction based on heavy atom count
        heavy_atoms = ligand_mol.GetNumHeavyAtoms()

        # Penalty for very large or very small molecules
        optimal_size = 25  # atoms
        size_deviation = abs(heavy_atoms - optimal_size) / optimal_size

        return size_deviation

    # Helper methods
    def _get_ligand_hb_atoms(self, mol: Chem.Mol) -> Dict[int, str]:
        """Identify H-bond donors and acceptors in ligand"""
        hb_atoms = {}

        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            symbol = atom.GetSymbol()

            if symbol in self.hbond_donors:
                if atom.GetTotalNumHs() > 0:
                    hb_atoms[idx] = 'donor'
                else:
                    # Check if it can accept H-bonds
                    if symbol in self.hbond_acceptors:
                        hb_atoms[idx] = 'acceptor'
            elif symbol in self.hbond_acceptors:
                hb_atoms[idx] = 'acceptor'

        return hb_atoms

    def _can_form_hbond(self, atom) -> bool:
        """Check if receptor atom can form hydrogen bonds"""
        return atom.element in self.hbond_donors or atom.element in self.hbond_acceptors

    def _get_receptor_hb_type(self, atom) -> str:
        """Determine if receptor atom is H-bond donor or acceptor"""
        residue_name = atom.get_parent().get_resname()
        atom_name = atom.get_name()

        # Common donor patterns
        if atom_name in ['OG', 'OH', 'NE2'] and atom.element in ['O', 'N']:
            return 'donor'
        # Common acceptor patterns
        elif atom_name in ['OD1', 'OD2', 'OE1', 'OE2', 'ND1'] and atom.element in ['O', 'N']:
            return 'acceptor'
        # Backbone carbonyl oxygen
        elif atom.element == 'O' and atom_name == 'O':
            return 'acceptor'
        else:
            return 'acceptor'  # Default assumption

    def _hbond_strength_function(self, distance: float) -> float:
        """Distance-dependent H-bond strength"""
        optimal_distance = 2.8  # Å
        if distance <= optimal_distance:
            return 1.0
        else:
            # Exponential decay beyond optimal distance
            return math.exp(-(distance - optimal_distance) / 0.5)

    def _hydrophobic_strength_function(self, distance: float) -> float:
        """Distance-dependent hydrophobic contact strength"""
        if distance <= 3.5:
            return 1.0
        elif distance <= self.hydrophobic_cutoff:
            return 1.0 - (distance - 3.5) / (self.hydrophobic_cutoff - 3.5)
        else:
            return 0.0

    def _scale_empirical_score(self, raw_score: float) -> float:
        """Scale empirical score to realistic binding energy range"""
        # Empirical scores can be very different from physics-based
        # Typical empirical scores might range widely

        if raw_score < -50:
            # Very favorable - excellent binding
            scaled = -12.0 + (raw_score + 50) * 0.04
        elif raw_score < -10:
            # Good binding
            scaled = -8.0 + (raw_score + 10) * 0.1
        elif raw_score < 0:
            # Moderate binding
            scaled = -3.0 + raw_score * 0.3
        elif raw_score < 10:
            # Unfavorable but not terrible
            scaled = raw_score * 0.2
        elif raw_score < 100:
            # Poor binding
            scaled = 2.0 + (raw_score - 10) * 0.05
        else:
            # Very poor binding
            scaled = 6.5

        # Return scaled value without artificial clamping
        # Let the interaction bias method handle final range checking
        return scaled

    def _apply_interaction_bias(self, energy: float, ligand_coords, receptor_structure) -> float:
        """Apply bias based on favorable interactions (similar to physics-based)"""
        # Get receptor coordinates
        receptor_atoms = list(receptor_structure.get_atoms())
        receptor_coords = np.array([atom.get_coord() for atom in receptor_atoms])

        # Count favorable interactions
        distances = np.linalg.norm(
            ligand_coords[:, np.newaxis] - receptor_coords[np.newaxis, :], axis=2
        )

        close_contacts = int(np.sum(distances < 4.0))
        very_close = int(np.sum(distances < 2.5))

        # Apply interaction-based bonus (more aggressive)
        if close_contacts > 50:
            interaction_bonus = -3.5 - (close_contacts - 50) * 0.07  # Increased from -2.0
            if very_close > 20:
                interaction_bonus -= 1.5  # Increased from -0.8
        elif close_contacts > 20:
            interaction_bonus = -1.2 - (close_contacts - 20) * 0.04  # Increased from -0.4
        elif close_contacts > 10:  # New tier
            interaction_bonus = -0.4 - (close_contacts - 10) * 0.02
        else:
            interaction_bonus = 0.0

        # Cap interaction bonus to prevent unrealistic scores
        # Max bonus: -6.0 kcal/mol (slightly more aggressive than precision_score)
        interaction_bonus = max(interaction_bonus, -6.0)

        # Apply bonus
        if energy > 3.0:
            corrected_energy = energy + interaction_bonus * 0.2
        elif energy > 0:
            corrected_energy = energy + interaction_bonus * 0.4
        else:
            corrected_energy = energy + interaction_bonus

        # Clamp to physically reasonable range
        # Floor at -50 to catch numerical errors only (not to hide scoring issues)
        # Ceiling at +50 for severe clashes
        if corrected_energy < -50.0:
            self.logger.warning(f"Energy {corrected_energy:.1f} clamped to -50.0 kcal/mol - check scoring!")
            return -50.0
        elif corrected_energy > 50.0:
            self.logger.warning(f"Energy {corrected_energy:.1f} clamped to +50.0 kcal/mol - severe clashes detected")
            return 50.0

        return corrected_energy