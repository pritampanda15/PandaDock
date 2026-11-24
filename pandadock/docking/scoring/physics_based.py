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

        # Energy term weights calibrated onto the Vina / kcal/mol binding scale
        self.weights = {
            'vdw': params.get('vdw_weight', 0.28),
            'electrostatic': params.get('electrostatic_weight', 0.15),
            'hbond': params.get('hbond_weight', 0.587),
            'hydrophobic': params.get('hydrophobic_weight', 0.035),
            'solvation': params.get('solvation_weight', 0.08),
        }

        # Physical constants
        self.COULOMB_CONSTANT = 332.0637  # kcal·Å/(mol·e²)
        self.DIELECTRIC_CONSTANT = params.get('dielectric', 4.0)
        self.n_rot_weight = params.get('n_rot_weight', 0.05846)

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
        Calculate physics-based binding energy (kcal/mol) on the same scale as
        Vina/Vinardo/MM-GBSA:

          E = (w_vdw * E_vdW + w_elec * E_elec + w_hb * E_hbond
               + w_hydro * E_hydrophobic + w_solv * E_desolv) / (1 + w_rot * N_rot)

        Uses vectorized heavy-atom Lorentz-Berthelot LJ 12-6 (with actual ligand
        elements and soft-core clash capping), distance-dependent screened
        Coulomb, and Vina-consistent H-bond / hydrophobic chemistry.
        """
        from .vina_scoring import VinaScoring

        lig_coords_full = np.asarray(ligand_coords, dtype=np.float64)
        if len(lig_coords_full) == 0:
            return 0.0

        # Heavy atoms only (matches Vina & prevents unminimized H-clash noise)
        if ligand_mol is not None:
            heavy_idx = np.array(
                [a.GetIdx() for a in ligand_mol.GetAtoms() if a.GetAtomicNum() > 1 and a.GetIdx() < len(lig_coords_full)],
                dtype=np.int64,
            )
            if len(heavy_idx) == 0:
                heavy_idx = np.arange(min(len(lig_coords_full), ligand_mol.GetNumAtoms()), dtype=np.int64)
            lig_coords = lig_coords_full[heavy_idx]
            lig_elements = [ligand_mol.GetAtomWithIdx(int(i)).GetSymbol() for i in heavy_idx]
        else:
            heavy_idx = np.arange(len(lig_coords_full), dtype=np.int64)
            lig_coords = lig_coords_full
            lig_elements = ['C'] * len(lig_coords)

        # Prefilter receptor heavy atoms within 10 A bounding box of ligand
        all_rec_atoms = [
            a for a in receptor_structure.get_atoms()
            if (a.element or '').strip().upper() != 'H'
        ]
        if not all_rec_atoms:
            all_rec_atoms = list(receptor_structure.get_atoms())
        all_rec_coords = np.asarray([a.get_coord() for a in all_rec_atoms], dtype=np.float64)
        if len(all_rec_coords) == 0:
            return 0.0

        lo = lig_coords.min(axis=0) - 10.0
        hi = lig_coords.max(axis=0) + 10.0
        in_box = np.where(np.all((all_rec_coords >= lo) & (all_rec_coords <= hi), axis=1))[0]
        if len(in_box) == 0:
            return 0.0

        rec_atoms = [all_rec_atoms[int(i)] for i in in_box]
        rec_coords = all_rec_coords[in_box]
        rec_elements = [(a.element or 'C').strip().capitalize() for a in rec_atoms]

        distances = cdist(lig_coords, rec_coords)
        vina_helper = VinaScoring()

        # 1. Vectorized Lorentz-Berthelot LJ 12-6 with soft-core cap (8 A cutoff)
        lig_sig = np.array([self.vdw_params.get(e, {'sigma': 3.4})['sigma'] for e in lig_elements])
        lig_eps = np.array([self.vdw_params.get(e, {'epsilon': 0.086})['epsilon'] for e in lig_elements])
        rec_sig = np.array([self.vdw_params.get(e, {'sigma': 3.4})['sigma'] for e in rec_elements])
        rec_eps = np.array([self.vdw_params.get(e, {'epsilon': 0.086})['epsilon'] for e in rec_elements])

        m_vdw = distances < 8.0
        if np.any(m_vdw):
            li, ri = np.where(m_vdw)
            d = np.maximum(distances[li, ri], 1.0)
            sig_ij = 0.5 * (lig_sig[li] + rec_sig[ri])
            eps_ij = np.sqrt(lig_eps[li] * rec_eps[ri])
            ratio = np.minimum(sig_ij / d, 1.35)
            r6 = ratio ** 6
            vdw_e = float(np.sum(np.minimum(4.0 * eps_ij * (r6 * r6 - r6), 2.0)))
        else:
            vdw_e = 0.0

        # 2. Vectorized distance-dependent screened Coulomb (10 A cutoff)
        elec_e = 0.0
        if ligand_mol is not None:
            mol_copy = Chem.Mol(ligand_mol)
            try:
                AllChem.ComputeGasteigerCharges(mol_copy)
                raw_q = [float(a.GetProp('_GasteigerCharge')) for a in mol_copy.GetAtoms()]
            except Exception:
                raw_q = [0.0] * mol_copy.GetNumAtoms()
            lig_q = np.array([raw_q[int(i)] if np.isfinite(raw_q[int(i)]) else 0.0 for i in heavy_idx])
            rec_q = np.asarray(self._get_receptor_charges(rec_atoms), dtype=np.float64)
            m_el = (distances < 10.0) & (distances >= 2.0)
            if np.any(m_el):
                li, ri = np.where(m_el)
                d = distances[li, ri]
                diel = np.maximum(self.DIELECTRIC_CONSTANT * d, 4.0)
                elec_e = float(np.sum((self.COULOMB_CONSTANT * lig_q[li] * rec_q[ri]) / (diel * d)))

        # 3. Vina-consistent H-bonding and hydrophobic burial
        hbond_e = 0.0
        hydro_e = 0.0
        desolv_e = 0.0
        if ligand_mol is not None:
            lig_types = vina_helper._get_ligand_atom_types(ligand_mol)
            rec_types = vina_helper._get_receptor_atom_types(rec_atoms)
            lig_r = np.array([vina_helper._get_vdw_radius(lig_types[int(i)]) for i in heavy_idx])
            rec_r = np.array([vina_helper._get_vdw_radius(t) for t in rec_types])
            surf = distances - lig_r[:, None] - rec_r[None, :]

            lig_don, lig_acc = vina_helper._get_hbond_atoms_ligand(ligand_mol)
            rec_don, rec_acc = vina_helper._get_hbond_atoms_receptor(rec_atoms)
            idx_map = {int(orig): pos for pos, orig in enumerate(heavy_idx)}
            ld = np.array([idx_map[i] for i in lig_don if i in idx_map], dtype=np.int64)
            la = np.array([idx_map[i] for i in lig_acc if i in idx_map], dtype=np.int64)
            rd = np.array(rec_don, dtype=np.int64)
            ra = np.array(rec_acc, dtype=np.int64)

            def _hb_sum(a_idx, b_idx):
                if len(a_idx) == 0 or len(b_idx) == 0:
                    return 0.0
                s = surf[np.ix_(a_idx, b_idx)]
                contrib = np.clip((0.0 - s) / 0.7, 0.0, 1.0)
                return float(-np.sum(contrib))

            hbond_e = _hb_sum(ld, ra) + _hb_sum(la, rd)

            lig_h_all = vina_helper._is_hydrophobic_ligand(ligand_mol)
            lig_h = np.array([i for i, orig in enumerate(heavy_idx) if lig_h_all[int(orig)]], dtype=np.int64)
            rec_h = np.where(vina_helper._is_hydrophobic_receptor(rec_atoms))[0]
            if len(lig_h) > 0 and len(rec_h) > 0:
                s = surf[np.ix_(lig_h, rec_h)]
                hydro_e = float(-np.sum(np.clip((1.5 - s) / 1.0, 0.0, 1.0)))

            # Mild polar desolvation penalty balanced against H-bonds
            polar_idx = [p for p, e in enumerate(lig_elements) if e in ('N', 'O')]
            if polar_idx:
                buried = np.sum(distances[polar_idx] < 4.5, axis=1)
                desolv_e = float(np.sum(np.minimum(buried / 12.0, 1.0)))

        raw_total = (
            self.weights['vdw'] * vdw_e
            + self.weights['electrostatic'] * elec_e
            + self.weights['hbond'] * hbond_e
            + self.weights['hydrophobic'] * hydro_e
            + self.weights['solvation'] * desolv_e
        )

        n_rot = vina_helper._count_rotatable_bonds(ligand_mol) if ligand_mol is not None else 0
        final_energy = raw_total / (1.0 + self.n_rot_weight * n_rot)

        self.logger.debug(
            "PhysicsBasedScoring: vdw=%.3f elec=%.3f hb=%.3f hydro=%.3f desolv=%.3f -> %.3f kcal/mol",
            vdw_e, elec_e, hbond_e, hydro_e, desolv_e, final_energy
        )
        return float(final_energy)

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