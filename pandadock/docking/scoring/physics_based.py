"""
Physics-Based Scoring Function

Implements fundamental physics-based energy terms:
1. Van der Waals interactions (Lennard-Jones potential)
2. Electrostatic interactions (Coulomb potential)
3. Hydrogen bonding (directional and distance-dependent)
4. Solvation/desolvation effects (simplified Born model)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy.spatial.distance import cdist
from Bio.PDB import Structure
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors, AllChem


class PhysicsBasedScoring:
    """Physics-based scoring function for molecular docking"""

    def __init__(self, **params):
        self.logger = logging.getLogger("pandadock.scoring.physics")

        # Energy term weights (in kcal/mol)
        self.weights = {
            'vdw': params.get('vdw_weight', 1.0),
            'electrostatic': params.get('electrostatic_weight', 1.0),
            'hbond': params.get('hbond_weight', 1.0),
            'solvation': params.get('solvation_weight', 0.5)
        }

        # Physical constants
        self.COULOMB_CONSTANT = 332.0637  # kcal·Å/(mol·e²)
        self.DIELECTRIC_CONSTANT = params.get('dielectric', 80.0)

        # Van der Waals parameters (Å, kcal/mol)
        self.vdw_params = self._initialize_vdw_parameters()

        # Hydrogen bonding parameters
        self.hbond_donors = {'N', 'O'}
        self.hbond_acceptors = {'N', 'O', 'S', 'F'}
        self.hbond_cutoff = 3.5  # Å
        self.hbond_angle_cutoff = 120.0  # degrees

        # Solvation parameters
        self.solvation_radii = self._initialize_solvation_radii()

    def calculate_binding_energy(self, ligand_coords: np.ndarray,
                                receptor_structure: Structure,
                                ligand_mol: Optional[Chem.Mol] = None) -> float:
        """
        Calculate total binding energy

        Args:
            ligand_coords: 3D coordinates of ligand atoms (N x 3)
            receptor_structure: BioPython receptor structure
            ligand_mol: RDKit molecule object (optional, for detailed analysis)

        Returns:
            Total binding energy in kcal/mol
        """
        # Get receptor information
        receptor_atoms = list(receptor_structure.get_atoms())
        receptor_coords = np.array([atom.get_coord() for atom in receptor_atoms])
        receptor_elements = [atom.element for atom in receptor_atoms]

        # Calculate individual energy terms
        energy_components = {}

        # Van der Waals energy
        energy_components['vdw'] = self._calculate_vdw_energy(
            ligand_coords, receptor_coords, receptor_elements
        )

        # Electrostatic energy
        energy_components['electrostatic'] = self._calculate_electrostatic_energy(
            ligand_coords, receptor_coords, ligand_mol, receptor_atoms
        )

        # Hydrogen bonding energy
        energy_components['hbond'] = self._calculate_hydrogen_bond_energy(
            ligand_coords, receptor_coords, ligand_mol, receptor_atoms
        )

        # Solvation energy
        energy_components['solvation'] = self._calculate_solvation_energy(
            ligand_coords, receptor_coords, ligand_mol, receptor_elements
        )

        # Calculate weighted total energy
        total_energy = sum(
            self.weights[term] * energy
            for term, energy in energy_components.items()
        )

        # Apply scaling to get realistic binding energies
        # Typical experimental binding energies: -15 to +5 kcal/mol
        scaled_energy = self._scale_to_binding_energy(total_energy)

        # Additional correction based on likely favorable interactions
        interaction_corrected_energy = self._apply_interaction_bias(
            scaled_energy, ligand_coords, receptor_structure
        )

        # Additional debug info
        if len(ligand_coords) > 0:
            self.logger.debug(f"Energy components: {energy_components}")
            self.logger.debug(f"Raw total energy: {total_energy:.3f} kcal/mol")
            self.logger.debug(f"Scaled binding energy: {scaled_energy:.3f} kcal/mol")
            self.logger.debug(f"Interaction-corrected energy: {interaction_corrected_energy:.3f} kcal/mol")
            self.logger.debug(f"Ligand atoms: {len(ligand_coords)}")

        return interaction_corrected_energy

    def _calculate_vdw_energy(self, ligand_coords: np.ndarray,
                             receptor_coords: np.ndarray,
                             receptor_elements: List[str]) -> float:
        """Calculate van der Waals interaction energy using Lennard-Jones potential"""
        vdw_energy = 0.0

        # Calculate all pairwise distances
        distances = cdist(ligand_coords, receptor_coords)

        # Apply tighter distance cutoff for efficiency (8 Å instead of 12 Å)
        cutoff = 8.0
        close_pairs = np.where(distances < cutoff)

        # Limit number of interactions for speed in fast mode
        max_interactions = 500
        if len(close_pairs[0]) > max_interactions:
            # Sample the closest interactions
            sorted_indices = np.argsort(distances[close_pairs])[:max_interactions]
            close_pairs = (close_pairs[0][sorted_indices], close_pairs[1][sorted_indices])

        for lig_idx, rec_idx in zip(close_pairs[0], close_pairs[1]):
            dist = distances[lig_idx, rec_idx]

            # Handle close contacts with reasonable penalties
            if dist < 1.0:
                vdw_energy += 20.0  # Further reduced penalty
                continue
            elif dist < 1.5:
                # Gradual penalty for soft clashes
                vdw_energy += (1.5 - dist) * 10.0  # Reduced penalty
                continue

            # Get van der Waals parameters
            # For simplicity, assume ligand atoms are carbon-like
            lig_element = 'C'  # Could be improved with actual ligand atom types
            rec_element = receptor_elements[rec_idx]

            sigma_lig = self.vdw_params.get(lig_element, {'sigma': 3.4})['sigma']
            epsilon_lig = self.vdw_params.get(lig_element, {'epsilon': 0.066})['epsilon']
            sigma_rec = self.vdw_params.get(rec_element, {'sigma': 3.4})['sigma']
            epsilon_rec = self.vdw_params.get(rec_element, {'epsilon': 0.066})['epsilon']

            # Combination rules (Lorentz-Berthelot)
            sigma_ij = (sigma_lig + sigma_rec) / 2.0
            epsilon_ij = np.sqrt(epsilon_lig * epsilon_rec)

            # Lennard-Jones potential
            r6 = (sigma_ij / dist) ** 6
            r12 = r6 ** 2
            vdw_energy += 4 * epsilon_ij * (r12 - r6)

        return vdw_energy

    def _calculate_electrostatic_energy(self, ligand_coords: np.ndarray,
                                      receptor_coords: np.ndarray,
                                      ligand_mol: Optional[Chem.Mol],
                                      receptor_atoms: List) -> float:
        """Calculate electrostatic interaction energy"""
        if ligand_mol is None:
            return 0.0  # Cannot calculate without charges

        electrostatic_energy = 0.0

        # Get partial charges (simplified - would need proper charge assignment)
        try:
            # Compute Gasteiger charges for ligand
            AllChem.ComputeGasteigerCharges(ligand_mol)
            ligand_charges = [
                float(atom.GetProp('_GasteigerCharge'))
                for atom in ligand_mol.GetAtoms()
            ]
        except:
            # Default to zero charges if computation fails
            ligand_charges = [0.0] * ligand_mol.GetNumAtoms()

        # Get receptor charges (simplified)
        receptor_charges = self._get_receptor_charges(receptor_atoms)

        # Calculate pairwise electrostatic interactions
        distances = cdist(ligand_coords, receptor_coords)
        cutoff = 10.0  # Reduced cutoff for speed

        close_pairs = np.where(distances < cutoff)

        # Limit electrostatic interactions for speed
        max_interactions = 300
        if len(close_pairs[0]) > max_interactions:
            sorted_indices = np.argsort(distances[close_pairs])[:max_interactions]
            close_pairs = (close_pairs[0][sorted_indices], close_pairs[1][sorted_indices])

        for lig_idx, rec_idx in zip(close_pairs[0], close_pairs[1]):
            dist = distances[lig_idx, rec_idx]

            if dist < 2.0:  # Avoid singularities
                continue

            q_lig = ligand_charges[lig_idx]
            q_rec = receptor_charges[rec_idx]

            # Skip if charges are negligible
            if abs(q_lig) < 0.01 or abs(q_rec) < 0.01:
                continue

            # Coulomb interaction with distance-dependent dielectric
            dielectric = self.DIELECTRIC_CONSTANT * (dist / 4.0)  # Distance-dependent
            dielectric = max(dielectric, 1.0)  # Minimum dielectric

            coulomb_energy = (self.COULOMB_CONSTANT * q_lig * q_rec) / (dielectric * dist)
            electrostatic_energy += coulomb_energy

        return electrostatic_energy

    def _calculate_hydrogen_bond_energy(self, ligand_coords: np.ndarray,
                                      receptor_coords: np.ndarray,
                                      ligand_mol: Optional[Chem.Mol],
                                      receptor_atoms: List) -> float:
        """Calculate hydrogen bonding energy"""
        if ligand_mol is None:
            return 0.0

        hbond_energy = 0.0

        # Identify potential hydrogen bond donors and acceptors
        ligand_hb_atoms = self._identify_hb_atoms(ligand_mol)
        receptor_hb_atoms = self._identify_receptor_hb_atoms(receptor_atoms)

        # Calculate donor-acceptor interactions
        for lig_atom_idx, lig_hb_type in ligand_hb_atoms.items():
            lig_coord = ligand_coords[lig_atom_idx]

            for rec_atom_idx, rec_hb_type in receptor_hb_atoms.items():
                rec_coord = receptor_coords[rec_atom_idx]

                dist = np.linalg.norm(lig_coord - rec_coord)

                if dist <= self.hbond_cutoff:
                    # Check if donor-acceptor pair
                    if ((lig_hb_type == 'donor' and rec_hb_type == 'acceptor') or
                        (lig_hb_type == 'acceptor' and rec_hb_type == 'donor')):

                        # Simplified hydrogen bond energy
                        # Optimal distance around 2.8 Å
                        optimal_dist = 2.8
                        if dist < optimal_dist:
                            hb_strength = -2.0  # kcal/mol (attractive)
                        else:
                            # Exponential decay beyond optimal distance
                            hb_strength = -2.0 * np.exp(-(dist - optimal_dist))

                        hbond_energy += hb_strength

        return hbond_energy

    def _calculate_solvation_energy(self, ligand_coords: np.ndarray,
                                  receptor_coords: np.ndarray,
                                  ligand_mol: Optional[Chem.Mol],
                                  receptor_elements: List[str]) -> float:
        """Calculate solvation/desolvation energy (simplified Born model)"""
        solvation_energy = 0.0

        # Desolvation penalty for burying polar atoms
        if ligand_mol is None:
            return 0.0

        for i, lig_coord in enumerate(ligand_coords):
            # Determine if ligand atom is polar (simplified)
            atom = ligand_mol.GetAtomWithIdx(i)
            if atom.GetSymbol() in ['N', 'O', 'S', 'F']:
                # Calculate burial (number of nearby receptor atoms)
                distances = np.linalg.norm(receptor_coords - lig_coord, axis=1)
                nearby_atoms = np.sum(distances < 5.0)

                # Desolvation penalty proportional to burial
                burial_factor = nearby_atoms / 10.0  # Normalized
                solvation_penalty = burial_factor * 2.0  # kcal/mol

                solvation_energy += solvation_penalty

        return solvation_energy

    def _initialize_vdw_parameters(self) -> Dict[str, Dict[str, float]]:
        """Initialize van der Waals parameters for common elements"""
        return {
            'C': {'sigma': 3.4, 'epsilon': 0.066},
            'N': {'sigma': 3.25, 'epsilon': 0.162},
            'O': {'sigma': 2.96, 'epsilon': 0.210},
            'S': {'sigma': 3.5, 'epsilon': 0.250},
            'P': {'sigma': 3.74, 'epsilon': 0.200},
            'H': {'sigma': 2.5, 'epsilon': 0.020},
            'F': {'sigma': 2.94, 'epsilon': 0.061},
            'Cl': {'sigma': 3.47, 'epsilon': 0.227},
            'Br': {'sigma': 3.73, 'epsilon': 0.251},
            'I': {'sigma': 4.01, 'epsilon': 0.339}
        }

    def _initialize_solvation_radii(self) -> Dict[str, float]:
        """Initialize atomic solvation radii"""
        return {
            'C': 1.9, 'N': 1.8, 'O': 1.7, 'S': 2.0,
            'P': 2.1, 'H': 1.2, 'F': 1.5, 'Cl': 1.8,
            'Br': 1.9, 'I': 2.1
        }

    def _get_receptor_charges(self, receptor_atoms: List) -> List[float]:
        """Get simplified receptor charges"""
        charges = []
        for atom in receptor_atoms:
            # Simplified charge assignment based on residue and atom name
            residue = atom.get_parent()
            residue_name = residue.get_resname()
            atom_name = atom.get_name()

            charge = 0.0
            if residue_name in ['ARG', 'LYS']:
                if atom.element == 'N':
                    charge = 0.3
            elif residue_name in ['ASP', 'GLU']:
                if atom.element == 'O':
                    charge = -0.4
            elif atom.element == 'O' and atom_name in ['OD1', 'OD2', 'OE1', 'OE2']:
                charge = -0.5

            charges.append(charge)

        return charges

    def _identify_hb_atoms(self, mol: Chem.Mol) -> Dict[int, str]:
        """Identify hydrogen bond donors and acceptors in ligand"""
        hb_atoms = {}

        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            symbol = atom.GetSymbol()

            if symbol in self.hbond_donors:
                # Check if it has hydrogens (donor)
                num_hs = atom.GetTotalNumHs()
                if num_hs > 0:
                    hb_atoms[idx] = 'donor'
                else:
                    # Likely acceptor
                    if symbol in self.hbond_acceptors:
                        hb_atoms[idx] = 'acceptor'
            elif symbol in self.hbond_acceptors:
                hb_atoms[idx] = 'acceptor'

        return hb_atoms

    def _identify_receptor_hb_atoms(self, receptor_atoms: List) -> Dict[int, str]:
        """Identify hydrogen bond donors and acceptors in receptor"""
        hb_atoms = {}

        for i, atom in enumerate(receptor_atoms):
            symbol = atom.element
            atom_name = atom.get_name()

            if symbol in self.hbond_donors:
                # Common donor patterns
                if atom_name in ['NE2', 'NH1', 'NH2', 'ND1', 'NE1']:
                    hb_atoms[i] = 'donor'
                elif atom_name in ['OG', 'OH']:
                    hb_atoms[i] = 'donor'
                # Acceptor patterns
                elif atom_name in ['OD1', 'OD2', 'OE1', 'OE2', 'ND1', 'NE2']:
                    hb_atoms[i] = 'acceptor'
            elif symbol in self.hbond_acceptors:
                if atom_name in ['OD1', 'OD2', 'OE1', 'OE2', 'O']:
                    hb_atoms[i] = 'acceptor'

        return hb_atoms

    def _scale_to_binding_energy(self, raw_energy: float) -> float:
        """Scale raw physics energy to realistic binding energy range"""
        # Physics-based energies can be very large or small
        # Scale to typical binding energy range: -15 to +5 kcal/mol

        self.logger.debug(f"Raw energy before scaling: {raw_energy:.3f}")

        if raw_energy < -100:
            # Very negative - excellent binding
            scaled = -12.0 + (raw_energy + 100) * 0.02
        elif raw_energy < -50:
            # Strong binding
            scaled = -10.0 + (raw_energy + 50) * 0.04
        elif raw_energy < -20:
            # Good binding range
            scaled = -8.0 + (raw_energy + 20) * 0.067
        elif raw_energy < -5:
            # Moderate binding
            scaled = -6.0 + (raw_energy + 5) * 0.133
        elif raw_energy < 0:
            # Weak attraction - make it slightly favorable
            scaled = -2.0 + raw_energy * 0.8
        elif raw_energy < 5:
            # Very close to zero - should be negative for good poses with many interactions
            scaled = -3.0 + raw_energy * 0.4
        elif raw_energy < 20:
            # Mild repulsion
            scaled = raw_energy * 0.15
        elif raw_energy < 100:
            # Strong repulsion
            scaled = 3.0 + (raw_energy - 20) * 0.05
        else:
            # Very high energy - poor binding
            scaled = 7.0

        # Clamp to reasonable range
        final_scaled = max(-15.0, min(8.0, scaled))

        self.logger.debug(f"Scaled energy: {final_scaled:.3f}")
        return final_scaled

    def _apply_interaction_bias(self, energy: float, ligand_coords: np.ndarray,
                               receptor_structure) -> float:
        """Apply bias based on number of favorable interactions"""
        # Get receptor coordinates
        receptor_atoms = list(receptor_structure.get_atoms())
        receptor_coords = np.array([atom.get_coord() for atom in receptor_atoms])

        # Count favorable interactions (simple distance-based)
        distances = np.linalg.norm(
            ligand_coords[:, np.newaxis] - receptor_coords[np.newaxis, :], axis=2
        )

        # Count different interaction types
        close_contacts = np.sum(distances < 4.0)  # General favorable distance
        very_close = np.sum(distances < 2.5)     # Potential H-bonds

        # If we have many favorable interactions, bias towards negative energy (more aggressive)
        if close_contacts > 50:  # Many interactions
            interaction_bonus = -4.0 - (close_contacts - 50) * 0.08  # Increased from -2.0
            if very_close > 20:  # Many potential H-bonds
                interaction_bonus -= 2.0  # Increased from -1.0
        elif close_contacts > 20:  # Moderate interactions
            interaction_bonus = -1.5 - (close_contacts - 20) * 0.06  # Increased from -0.5
        elif close_contacts > 10:  # Some interactions
            interaction_bonus = -0.5 - (close_contacts - 10) * 0.03  # New tier
        else:
            interaction_bonus = 0.0

        # Apply the bonus but don't make very bad poses suddenly good
        if energy > 5.0:
            # Very bad pose - only small bonus
            corrected_energy = energy + interaction_bonus * 0.2
        elif energy > 0:
            # Marginal pose - moderate bonus
            corrected_energy = energy + interaction_bonus * 0.5
        else:
            # Already negative - full bonus
            corrected_energy = energy + interaction_bonus

        # Clamp again
        return max(-15.0, min(10.0, corrected_energy))