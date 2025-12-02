"""
Advanced Precision Scoring Function

Implements comprehensive Precision WS scoring with:
- vdW: Van der Waals energy with reduced ionic charges
- Coulomb: Electrostatic energy with reduced ionic charges
- Lipo: Lipophilic/hydrophobic interactions from SiteMap
- HBond: Hydrogen bonding with neutral/charged differentiation
- Metal: Metal binding interactions
- Rewards: Special interaction rewards (π-cation, magic methyl, etc.)
- RotB: Rotatable bond entropy penalty
- Site: Polar interactions in active site
- Interaction energy decomposition for correlation analysis
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy.spatial.distance import cdist
from Bio.PDB import Structure
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
import math


class PrecisionScoring:
    """Advanced precision scoring function with interaction energy decomposition"""

    def __init__(self, **params):
        self.logger = logging.getLogger("pandadock.scoring.precision")

        # Precision scoring weights (calibrated values)
        self.weights = {
            'vdw': params.get('vdw_weight', 0.05),
            'coulomb': params.get('coulomb_weight', 0.15),
            'lipo': params.get('lipo_weight', 1.0),
            'hbond': params.get('hbond_weight', 1.0),
            'metal': params.get('metal_weight', 1.0),
            'rewards': params.get('rewards_weight', 1.0),
            'rotb': params.get('rotb_weight', 1.0),
            'site': params.get('site_weight', 1.0)
        }

        # Physical constants
        self.COULOMB_CONSTANT = 332.0637  # kcal·Å/(mol·e²)
        self.kT_298K = 0.593  # kcal/mol at 298K

        # VdW parameters (OPLS-like)
        self.vdw_params = self._initialize_vdw_parameters()

        # Hydrogen bond parameters
        self.hbond_optimal_dist = 2.8  # Å
        self.hbond_cutoff = 3.5  # Å

        # Metal coordination parameters
        self.metal_elements = {'MG', 'CA', 'ZN', 'FE', 'MN', 'CO', 'NI', 'CU'}
        self.metal_coordination_cutoff = 2.5  # Å

        # Hydrophobic enclosure parameters
        self.hydrophobic_elements = {'C'}
        self.hydrophobic_cutoff = 5.0  # Å

        # Interaction energy components storage
        self.interaction_components = {}

        self.logger.debug("Advanced precision scoring initialized with interaction energy decomposition")

    def calculate_binding_energy(self, ligand_coords: np.ndarray,
                                receptor_structure: Structure,
                                ligand_mol: Optional[Chem.Mol] = None) -> float:
        """
        Calculate total precision binding energy with interaction energy decomposition

        Returns both binding affinity and detailed interaction energies
        """
        # Reset interaction components
        self.interaction_components = {}

        # Get receptor information
        receptor_atoms = list(receptor_structure.get_atoms())
        receptor_coords = np.array([atom.get_coord() for atom in receptor_atoms])

        # Calculate all energy components
        vdw_energy = self._calculate_vdw_energy(ligand_coords, receptor_coords, receptor_atoms, ligand_mol)
        coulomb_energy = self._calculate_coulomb_energy(ligand_coords, receptor_coords, receptor_atoms, ligand_mol)
        lipo_energy = self._calculate_lipophilic_energy(ligand_coords, receptor_coords, receptor_atoms, ligand_mol)
        hbond_energy = self._calculate_hbond_energy(ligand_coords, receptor_coords, receptor_atoms, ligand_mol)
        metal_energy = self._calculate_metal_energy(ligand_coords, receptor_coords, receptor_atoms, ligand_mol)
        rewards_energy = self._calculate_rewards_energy(ligand_coords, receptor_coords, receptor_atoms, ligand_mol)
        rotb_penalty = self._calculate_rotatable_bond_penalty(ligand_mol)
        site_energy = self._calculate_site_energy(ligand_coords, receptor_coords, receptor_atoms, ligand_mol)

        # Store components for interaction energy analysis
        self.interaction_components = {
            'vdw': vdw_energy,
            'coulomb': coulomb_energy,
            'lipo': lipo_energy,
            'hbond': hbond_energy,
            'metal': metal_energy,
            'rewards': rewards_energy,
            'rotb': rotb_penalty,
            'site': site_energy
        }

        # Calculate weighted precision score
        precision_score = (
            self.weights['vdw'] * vdw_energy +
            self.weights['coulomb'] * coulomb_energy +
            self.weights['lipo'] * lipo_energy +
            self.weights['hbond'] * hbond_energy +
            self.weights['metal'] * metal_energy +
            self.weights['rewards'] * rewards_energy +
            self.weights['rotb'] * rotb_penalty +
            self.weights['site'] * site_energy
        )

        # Apply scaling to get realistic binding energies
        scaled_score = self._scale_precision_score(precision_score)

        # Apply interaction bias
        interaction_corrected_score = self._apply_interaction_bias(
            scaled_score, ligand_coords, receptor_structure
        )

        # Calculate interaction energy (1-100 scale)
        interaction_energy = self._calculate_interaction_energy()

        self.logger.debug(f"Precision Score Components: {self.interaction_components}")
        self.logger.debug(f"Raw Precision Score: {precision_score:.3f} kcal/mol")
        self.logger.debug(f"Scaled Score: {interaction_corrected_score:.3f} kcal/mol")
        self.logger.debug(f"Interaction Energy: {interaction_energy:.1f}")

        return interaction_corrected_score

    def get_interaction_energy(self) -> float:
        """Return interaction energy on 1-100 scale"""
        return self._calculate_interaction_energy()

    def get_energy_decomposition(self) -> Dict[str, float]:
        """Return detailed energy component breakdown"""
        return self.interaction_components.copy()

    def _calculate_interaction_energy(self) -> float:
        """
        Calculate interaction energy on 1-100 scale
        This correlates better with IC50 than binding affinity scores
        """
        if not self.interaction_components:
            return 0.0

        # Favorable interactions contribute positively to interaction energy
        favorable_interactions = (
            -self.interaction_components.get('hbond', 0.0) +  # More negative = better
            -self.interaction_components.get('lipo', 0.0) +
            -self.interaction_components.get('metal', 0.0) +
            -self.interaction_components.get('rewards', 0.0) +
            -min(0, self.interaction_components.get('vdw', 0.0)) * 0.1  # Only attractive vdW
        )

        # Unfavorable interactions reduce interaction energy
        unfavorable_interactions = (
            max(0, self.interaction_components.get('vdw', 0.0)) * 0.05 +  # Repulsive vdW
            self.interaction_components.get('rotb', 0.0) * 0.1 +
            max(0, self.interaction_components.get('coulomb', 0.0)) * 0.1  # Unfavorable electrostatics
        )

        # Scale to 1-100 range (typical Schrödinger interaction energy range)
        raw_score = favorable_interactions - unfavorable_interactions
        interaction_energy = max(1.0, min(100.0, 20.0 + raw_score * 5.0))

        return interaction_energy

    def _calculate_vdw_energy(self, ligand_coords: np.ndarray, receptor_coords: np.ndarray,
                            receptor_atoms: List, ligand_mol: Optional[Chem.Mol]) -> float:
        """Van der Waals energy with reduced ionic charges"""
        vdw_energy = 0.0
        distances = cdist(ligand_coords, receptor_coords)
        cutoff = 10.0

        close_pairs = np.where(distances < cutoff)

        for lig_idx, rec_idx in zip(close_pairs[0], close_pairs[1]):
            dist = distances[lig_idx, rec_idx]

            if dist < 1.8:  # Severe clash penalty
                vdw_energy += 100.0 * (1.8 - dist) ** 2
                continue

            # Get atom types
            rec_element = receptor_atoms[rec_idx].element
            lig_element = self._get_ligand_atom_element(ligand_mol, lig_idx) if ligand_mol else 'C'

            # VdW parameters
            sigma_rec = self.vdw_params.get(rec_element, {'sigma': 3.4})['sigma']
            epsilon_rec = self.vdw_params.get(rec_element, {'epsilon': 0.066})['epsilon']
            sigma_lig = self.vdw_params.get(lig_element, {'sigma': 3.4})['sigma']
            epsilon_lig = self.vdw_params.get(lig_element, {'epsilon': 0.066})['epsilon']

            # Combination rules
            sigma_ij = (sigma_rec + sigma_lig) / 2.0
            epsilon_ij = math.sqrt(epsilon_rec * epsilon_lig)

            # Lennard-Jones 12-6 potential
            if dist > sigma_ij * 0.8:  # Avoid numerical issues
                r6 = (sigma_ij / dist) ** 6
                r12 = r6 ** 2
                vdw_energy += 4 * epsilon_ij * (r12 - r6)

        return vdw_energy

    def _calculate_coulomb_energy(self, ligand_coords: np.ndarray, receptor_coords: np.ndarray,
                                receptor_atoms: List, ligand_mol: Optional[Chem.Mol]) -> float:
        """Electrostatic energy with reduced ionic charges"""
        if ligand_mol is None:
            return 0.0

        coulomb_energy = 0.0

        # Get partial charges
        try:
            AllChem.ComputeGasteigerCharges(ligand_mol)
            ligand_charges = []
            for atom in ligand_mol.GetAtoms():
                charge = float(atom.GetProp('_GasteigerCharge'))
                # Apply ionic charge reduction for formal charges
                formal_charge = atom.GetFormalCharge()
                if abs(formal_charge) > 0:
                    charge *= 0.75  # Reduce ionic charges
                ligand_charges.append(charge)
        except:
            ligand_charges = [0.0] * ligand_mol.GetNumAtoms()

        receptor_charges = self._get_receptor_charges(receptor_atoms, reduce_ionic=True)

        distances = cdist(ligand_coords, receptor_coords)
        cutoff = 12.0
        close_pairs = np.where(distances < cutoff)

        for lig_idx, rec_idx in zip(close_pairs[0], close_pairs[1]):
            dist = distances[lig_idx, rec_idx]

            if dist < 2.0:
                continue

            q_lig = ligand_charges[lig_idx]
            q_rec = receptor_charges[rec_idx]

            # Distance-dependent dielectric
            dielectric = 4.0 * (1 + dist / 4.0)  # Distance-dependent dielectric
            coulomb_energy += (self.COULOMB_CONSTANT * q_lig * q_rec) / (dielectric * dist)

        return coulomb_energy

    def _calculate_lipophilic_energy(self, ligand_coords: np.ndarray, receptor_coords: np.ndarray,
                                   receptor_atoms: List, ligand_mol: Optional[Chem.Mol]) -> float:
        """Lipophilic/hydrophobic interactions (SiteMap-style)"""
        lipo_energy = 0.0

        if ligand_mol is None:
            return 0.0

        # Identify hydrophobic atoms
        hydrophobic_lig_atoms = []
        for i, atom in enumerate(ligand_mol.GetAtoms()):
            if atom.GetSymbol() == 'C' and atom.GetTotalNumHs() > 0:
                hydrophobic_lig_atoms.append(i)

        hydrophobic_rec_atoms = []
        for i, atom in enumerate(receptor_atoms):
            if (atom.element == 'C' and
                atom.get_parent().get_resname() in ['PHE', 'TRP', 'TYR', 'LEU', 'ILE', 'VAL', 'MET', 'PRO']):
                hydrophobic_rec_atoms.append(i)

        # Calculate hydrophobic contacts
        for lig_idx in hydrophobic_lig_atoms:
            lig_coord = ligand_coords[lig_idx]

            for rec_idx in hydrophobic_rec_atoms:
                rec_coord = receptor_coords[rec_idx]
                dist = np.linalg.norm(lig_coord - rec_coord)

                if 3.0 < dist < 5.0:  # Optimal hydrophobic contact range
                    # Favorable hydrophobic interaction
                    contact_strength = -0.5 * (1.0 - (dist - 3.0) / 2.0)  # Linear decay
                    lipo_energy += contact_strength

        return lipo_energy

    def _calculate_hbond_energy(self, ligand_coords: np.ndarray, receptor_coords: np.ndarray,
                              receptor_atoms: List, ligand_mol: Optional[Chem.Mol]) -> float:
        """Hydrogen bonding energy with neutral/charged differentiation"""
        if ligand_mol is None:
            return 0.0

        hbond_energy = 0.0

        # Identify H-bond donors/acceptors in ligand
        lig_hb_atoms = self._identify_ligand_hb_atoms(ligand_mol)
        rec_hb_atoms = self._identify_receptor_hb_atoms(receptor_atoms)

        for lig_idx, lig_type in lig_hb_atoms.items():
            lig_coord = ligand_coords[lig_idx]

            for rec_idx, rec_type in rec_hb_atoms.items():
                if not ((lig_type == 'donor' and rec_type == 'acceptor') or
                       (lig_type == 'acceptor' and rec_type == 'donor')):
                    continue

                rec_coord = receptor_coords[rec_idx]
                dist = np.linalg.norm(lig_coord - rec_coord)

                if dist <= self.hbond_cutoff:
                    # Determine if neutral or charged H-bond
                    lig_atom = ligand_mol.GetAtomWithIdx(int(lig_idx))
                    rec_atom = receptor_atoms[rec_idx]

                    lig_charged = abs(lig_atom.GetFormalCharge()) > 0
                    rec_charged = self._is_charged_residue(rec_atom.get_parent().get_resname())

                    # Different strengths for different H-bond types
                    if lig_charged and rec_charged:
                        max_strength = -3.0  # Charged-charged H-bond
                    elif lig_charged or rec_charged:
                        max_strength = -2.0  # Neutral-charged H-bond
                    else:
                        max_strength = -1.5  # Neutral H-bond

                    # Distance-dependent strength
                    if dist <= self.hbond_optimal_dist:
                        strength = max_strength
                    else:
                        strength = max_strength * math.exp(-(dist - self.hbond_optimal_dist) / 0.5)

                    # Check for hydrophobic enclosure (additional reward)
                    if self._is_hydrophobically_enclosed(lig_coord, rec_coord, receptor_coords, receptor_atoms):
                        strength += -1.0  # Additional reward for enclosed H-bond

                    hbond_energy += strength

        return hbond_energy

    def _calculate_metal_energy(self, ligand_coords: np.ndarray, receptor_coords: np.ndarray,
                              receptor_atoms: List, ligand_mol: Optional[Chem.Mol]) -> float:
        """Metal coordination energy"""
        if ligand_mol is None:
            return 0.0

        metal_energy = 0.0

        # Find metal atoms in receptor
        metal_atoms = []
        for i, atom in enumerate(receptor_atoms):
            if atom.element in self.metal_elements:
                metal_atoms.append(i)

        if not metal_atoms:
            return 0.0

        # Find potential metal-coordinating atoms in ligand
        coordination_atoms = []
        for i, atom in enumerate(ligand_mol.GetAtoms()):
            if atom.GetSymbol() in ['O', 'N', 'S'] and atom.GetFormalCharge() <= 0:
                coordination_atoms.append(i)

        # Calculate metal coordination energy
        for lig_idx in coordination_atoms:
            lig_coord = ligand_coords[lig_idx]

            for metal_idx in metal_atoms:
                metal_coord = receptor_coords[metal_idx]
                dist = np.linalg.norm(lig_coord - metal_coord)

                if dist <= self.metal_coordination_cutoff:
                    # Strong favorable interaction for metal coordination
                    lig_atom = ligand_mol.GetAtomWithIdx(int(lig_idx))

                    # Anionic ligands preferred
                    if lig_atom.GetFormalCharge() < 0:
                        metal_energy += -4.0
                    else:
                        metal_energy += -2.0

        return metal_energy

    def _calculate_rewards_energy(self, ligand_coords: np.ndarray, receptor_coords: np.ndarray,
                                receptor_atoms: List, ligand_mol: Optional[Chem.Mol]) -> float:
        """Special interaction rewards (π-cation, magic methyl, etc.)"""
        if ligand_mol is None:
            return 0.0

        rewards_energy = 0.0

        # π-cation interactions
        rewards_energy += self._calculate_pi_cation_energy(ligand_coords, receptor_coords, receptor_atoms, ligand_mol)

        # Magic methyl rewards (displacement of water in small cavities)
        rewards_energy += self._calculate_magic_methyl_energy(ligand_coords, receptor_coords, receptor_atoms, ligand_mol)

        # Aromatic stacking
        rewards_energy += self._calculate_aromatic_stacking_energy(ligand_coords, receptor_coords, receptor_atoms, ligand_mol)

        return rewards_energy

    def _calculate_pi_cation_energy(self, ligand_coords: np.ndarray, receptor_coords: np.ndarray,
                                  receptor_atoms: List, ligand_mol: Chem.Mol) -> float:
        """π-cation interaction energy"""
        pi_cation_energy = 0.0

        # Find aromatic rings in ligand
        aromatic_atoms = set()
        for ring in ligand_mol.GetRingInfo().AtomRings():
            if len(ring) == 6:  # Assume 6-membered rings are aromatic
                aromatic_atoms.update(ring)

        # Find cationic residues in receptor
        cationic_atoms = []
        for i, atom in enumerate(receptor_atoms):
            residue = atom.get_parent().get_resname()
            if residue in ['ARG', 'LYS'] and atom.element == 'N':
                cationic_atoms.append(i)

        # Calculate π-cation interactions
        for ring in ligand_mol.GetRingInfo().AtomRings():
            if len(ring) != 6:
                continue

            # Ring center
            ring_coords = ligand_coords[list(ring)]
            ring_center = np.mean(ring_coords, axis=0)

            for cat_idx in cationic_atoms:
                cat_coord = receptor_coords[cat_idx]
                dist = np.linalg.norm(ring_center - cat_coord)

                if 3.0 < dist < 6.0:  # π-cation distance range
                    pi_cation_energy += -2.0 * (1.0 - (dist - 3.0) / 3.0)  # Distance-dependent

        return pi_cation_energy

    def _calculate_magic_methyl_energy(self, ligand_coords: np.ndarray, receptor_coords: np.ndarray,
                                     receptor_atoms: List, ligand_mol: Chem.Mol) -> float:
        """Magic methyl reward for water displacement in small cavities"""
        magic_methyl_energy = 0.0

        # Find methyl groups in ligand
        methyl_atoms = []
        for atom in ligand_mol.GetAtoms():
            if (atom.GetSymbol() == 'C' and
                atom.GetTotalNumHs() == 3 and
                atom.GetDegree() == 1):
                methyl_atoms.append(atom.GetIdx())

        # Check if methyl groups are in small, hydrophobic cavities
        for methyl_idx in methyl_atoms:
            methyl_coord = ligand_coords[methyl_idx]

            # Count nearby hydrophobic receptor atoms
            nearby_hydrophobic = 0
            nearby_polar = 0

            for i, rec_atom in enumerate(receptor_atoms):
                dist = np.linalg.norm(methyl_coord - receptor_coords[i])

                if 3.0 < dist < 5.0:
                    if rec_atom.element == 'C':
                        nearby_hydrophobic += 1
                    elif rec_atom.element in ['N', 'O']:
                        nearby_polar += 1

            # Magic methyl reward for hydrophobic enclosure
            if nearby_hydrophobic >= 4 and nearby_polar <= 1:
                magic_methyl_energy += -3.0

        return magic_methyl_energy

    def _calculate_aromatic_stacking_energy(self, ligand_coords: np.ndarray, receptor_coords: np.ndarray,
                                          receptor_atoms: List, ligand_mol: Chem.Mol) -> float:
        """Aromatic stacking interactions"""
        stacking_energy = 0.0

        # Find aromatic rings in receptor
        aromatic_residues = ['PHE', 'TRP', 'TYR']
        receptor_aromatic_atoms = {}

        for i, atom in enumerate(receptor_atoms):
            residue = atom.get_parent()
            if (residue.get_resname() in aromatic_residues and
                atom.element == 'C' and
                atom.get_name() in ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ']):
                res_id = residue.get_id()
                if res_id not in receptor_aromatic_atoms:
                    receptor_aromatic_atoms[res_id] = []
                receptor_aromatic_atoms[res_id].append(i)

        # Find aromatic rings in ligand
        for ring in ligand_mol.GetRingInfo().AtomRings():
            if len(ring) != 6:
                continue

            lig_ring_center = np.mean(ligand_coords[list(ring)], axis=0)

            # Check stacking with receptor aromatic rings
            for res_id, atom_indices in receptor_aromatic_atoms.items():
                if len(atom_indices) >= 6:  # Complete aromatic ring
                    rec_ring_center = np.mean(receptor_coords[atom_indices], axis=0)
                    dist = np.linalg.norm(lig_ring_center - rec_ring_center)

                    if 3.0 < dist < 5.0:  # Stacking distance range
                        stacking_energy += -1.5 * (1.0 - (dist - 3.0) / 2.0)

        return stacking_energy

    def _calculate_rotatable_bond_penalty(self, ligand_mol: Optional[Chem.Mol]) -> float:
        """Entropy penalty for rotatable bonds"""
        if ligand_mol is None:
            return 0.0

        # Count rotatable bonds
        num_rotatable = rdMolDescriptors.CalcNumRotatableBonds(ligand_mol)

        # Entropy penalty (approximate)
        entropy_penalty = num_rotatable * 1.0  # kcal/mol per rotatable bond

        return entropy_penalty

    def _calculate_site_energy(self, ligand_coords: np.ndarray, receptor_coords: np.ndarray,
                             receptor_atoms: List, ligand_mol: Optional[Chem.Mol]) -> float:
        """Polar interactions in active site"""
        if ligand_mol is None:
            return 0.0

        site_energy = 0.0

        # Reward polar atoms in hydrophobic regions
        for i, atom in enumerate(ligand_mol.GetAtoms()):
            if atom.GetSymbol() in ['N', 'O', 'S', 'F']:
                lig_coord = ligand_coords[i]

                # Count nearby hydrophobic vs polar receptor atoms
                nearby_hydrophobic = 0
                nearby_polar = 0

                for j, rec_atom in enumerate(receptor_atoms):
                    dist = np.linalg.norm(lig_coord - receptor_coords[j])

                    if dist < 5.0:
                        if rec_atom.element == 'C':
                            nearby_hydrophobic += 1
                        elif rec_atom.element in ['N', 'O']:
                            nearby_polar += 1

                # Reward polar atoms in hydrophobic environments
                if nearby_hydrophobic > nearby_polar * 2:
                    site_energy += -0.5

        return site_energy

    # Helper methods
    def _initialize_vdw_parameters(self) -> Dict[str, Dict[str, float]]:
        """Initialize OPLS-like van der Waals parameters"""
        return {
            'C': {'sigma': 3.50, 'epsilon': 0.066},
            'N': {'sigma': 3.25, 'epsilon': 0.170},
            'O': {'sigma': 2.96, 'epsilon': 0.210},
            'S': {'sigma': 3.55, 'epsilon': 0.250},
            'P': {'sigma': 3.74, 'epsilon': 0.200},
            'H': {'sigma': 2.50, 'epsilon': 0.030},
            'F': {'sigma': 2.94, 'epsilon': 0.061},
            'Cl': {'sigma': 3.47, 'epsilon': 0.227},
            'Br': {'sigma': 3.73, 'epsilon': 0.251},
            'I': {'sigma': 4.01, 'epsilon': 0.339},
            'MG': {'sigma': 1.18, 'epsilon': 0.875},
            'CA': {'sigma': 1.37, 'epsilon': 0.459},
            'ZN': {'sigma': 1.10, 'epsilon': 0.250},
            'FE': {'sigma': 1.30, 'epsilon': 0.400}
        }

    def _get_ligand_atom_element(self, ligand_mol: Chem.Mol, atom_idx: int) -> str:
        """Get element symbol for ligand atom"""
        return ligand_mol.GetAtomWithIdx(int(atom_idx)).GetSymbol()

    def _get_receptor_charges(self, receptor_atoms: List, reduce_ionic: bool = True) -> List[float]:
        """Get receptor partial charges with ionic charge reduction"""
        charges = []

        for atom in receptor_atoms:
            residue = atom.get_parent()
            residue_name = residue.get_resname()
            atom_name = atom.get_name()

            charge = 0.0

            # Standard amino acid charges
            if residue_name == 'ARG':
                if atom_name in ['NH1', 'NH2']:
                    charge = 0.8 if not reduce_ionic else 0.6
                elif atom_name == 'NE':
                    charge = 0.4 if not reduce_ionic else 0.3
            elif residue_name == 'LYS':
                if atom_name == 'NZ':
                    charge = 1.0 if not reduce_ionic else 0.75
            elif residue_name == 'ASP':
                if atom_name in ['OD1', 'OD2']:
                    charge = -0.8 if not reduce_ionic else -0.6
            elif residue_name == 'GLU':
                if atom_name in ['OE1', 'OE2']:
                    charge = -0.8 if not reduce_ionic else -0.6
            elif residue_name == 'HIS':
                if atom_name in ['ND1', 'NE2']:
                    charge = 0.4

            charges.append(charge)

        return charges

    def _identify_ligand_hb_atoms(self, ligand_mol: Chem.Mol) -> Dict[int, str]:
        """Identify hydrogen bond donors and acceptors in ligand"""
        hb_atoms = {}

        for atom in ligand_mol.GetAtoms():
            idx = atom.GetIdx()
            symbol = atom.GetSymbol()

            if symbol in ['N', 'O']:
                num_hs = atom.GetTotalNumHs()
                if num_hs > 0:
                    hb_atoms[idx] = 'donor'
                else:
                    # Check lone pairs for acceptors
                    valence = atom.GetTotalValence()
                    if (symbol == 'O' and valence <= 2) or (symbol == 'N' and valence <= 3):
                        hb_atoms[idx] = 'acceptor'
            elif symbol in ['S', 'F']:
                hb_atoms[idx] = 'acceptor'

        return hb_atoms

    def _identify_receptor_hb_atoms(self, receptor_atoms: List) -> Dict[int, str]:
        """Identify hydrogen bond donors and acceptors in receptor"""
        hb_atoms = {}

        for i, atom in enumerate(receptor_atoms):
            residue_name = atom.get_parent().get_resname()
            atom_name = atom.get_name()
            element = atom.element

            # Common H-bond patterns
            if atom_name in ['OG', 'OH', 'NE2'] and element in ['O', 'N']:
                hb_atoms[i] = 'donor'
            elif atom_name in ['OD1', 'OD2', 'OE1', 'OE2', 'ND1'] and element in ['O', 'N']:
                hb_atoms[i] = 'acceptor'
            elif residue_name == 'TRP' and atom_name == 'NE1':
                hb_atoms[i] = 'donor'
            elif element == 'O' and atom_name == 'O':  # Backbone carbonyl
                hb_atoms[i] = 'acceptor'

        return hb_atoms

    def _is_charged_residue(self, residue_name: str) -> bool:
        """Check if residue is charged"""
        return residue_name in ['ARG', 'LYS', 'ASP', 'GLU']

    def _is_hydrophobically_enclosed(self, coord1: np.ndarray, coord2: np.ndarray,
                                   receptor_coords: np.ndarray, receptor_atoms: List) -> bool:
        """Check if H-bond is hydrophobically enclosed"""
        midpoint = (coord1 + coord2) / 2.0

        hydrophobic_count = 0
        for i, rec_coord in enumerate(receptor_coords):
            if receptor_atoms[i].element == 'C':
                dist = np.linalg.norm(midpoint - rec_coord)
                if dist < 5.0:
                    hydrophobic_count += 1

        return hydrophobic_count >= 6

    def _scale_precision_score(self, raw_score: float) -> float:
        """Scale precision score to realistic binding energy range"""
        # Precision scores are typically structured differently
        # They may already be in a reasonable range but need adjustment

        if raw_score < -50:
            # Very favorable
            scaled = -10.0 + (raw_score + 50) * 0.06
        elif raw_score < -15:
            # Good binding
            scaled = -7.0 + (raw_score + 15) * 0.086
        elif raw_score < -5:
            # Moderate binding
            scaled = -4.0 + (raw_score + 5) * 0.3
        elif raw_score < 0:
            # Weak attraction
            scaled = -1.0 + raw_score * 0.6
        elif raw_score < 10:
            # Unfavorable
            scaled = raw_score * 0.3
        elif raw_score < 50:
            # Poor binding
            scaled = 3.0 + (raw_score - 10) * 0.075
        else:
            # Very poor
            scaled = 6.0

        # Return scaled value without artificial clamping
        # Let the interaction bias method handle final range checking
        return scaled

    def _apply_interaction_bias(self, energy: float, ligand_coords, receptor_structure) -> float:
        """Apply bias based on favorable interactions"""
        # Get receptor coordinates
        receptor_atoms = list(receptor_structure.get_atoms())
        receptor_coords = np.array([atom.get_coord() for atom in receptor_atoms])

        # Count favorable interactions
        distances = np.linalg.norm(
            ligand_coords[:, np.newaxis] - receptor_coords[np.newaxis, :], axis=2
        )

        close_contacts = int(np.sum(distances < 4.0))
        very_close = int(np.sum(distances < 2.5))

        # Apply interaction bonus (capped to prevent artificial over-rewarding)
        if close_contacts > 50:
            interaction_bonus = -1.8 - (close_contacts - 50) * 0.03
            if very_close > 20:
                interaction_bonus -= 0.7
        elif close_contacts > 20:
            interaction_bonus = -0.3 - (close_contacts - 20) * 0.02
        else:
            interaction_bonus = 0.0

        # Cap interaction bonus to prevent unrealistic scores
        # Max bonus: -5.0 kcal/mol (reasonable for excellent binding)
        interaction_bonus = max(interaction_bonus, -5.0)

        # Apply bonus
        if energy > 3.0:
            corrected_energy = energy + interaction_bonus * 0.25
        elif energy > 0:
            corrected_energy = energy + interaction_bonus * 0.5
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