"""
MM-GBSA Rescoring and Refinement

Implements molecular mechanics with generalized Born surface area (MM-GBSA) rescoring:
1. Molecular mechanics energy (bonded + non-bonded)
2. Generalized Born solvation energy
3. Surface area solvation correction
4. Entropy estimation (optional)
5. Binding free energy decomposition

This provides more accurate binding free energies than docking scores alone
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging
from scipy.optimize import minimize
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from Bio.PDB import Structure
import math

from ..core import Pose


class MMGBSARescoring:
    """MM-GBSA rescoring for improved binding affinity prediction"""

    def __init__(self, **params):
        self.logger = logging.getLogger("pandadock.refinement.mmgbsa")

        # MM-GBSA parameters (standard protein-interior dielectric=4.0 for rigid/docked poses)
        self.gb_model = params.get('gb_model', 'GB-OBC2')
        self.surface_tension = params.get('surface_tension', 0.0072)  # kcal/(mol·Å²)
        self.dielectric_interior = params.get('dielectric_interior', 4.0)
        self.dielectric_exterior = params.get('dielectric_exterior', 80.0)
        self.salt_concentration = params.get('salt_concentration', 0.15)  # M
        self.pocket_cutoff = params.get('pocket_cutoff', 12.0)  # Å around ligand
        self.vdw_clash_cap = params.get('vdw_clash_cap', 2.5)  # Soft-core cap (kcal/mol/pair)

        # Force field parameters (simplified AMBER/CHARMM-like)
        self.bond_force_constants = {'CC': 317, 'CN': 337, 'CO': 320, 'NN': 410, 'NO': 410}
        self.angle_force_constants = {'CCC': 63, 'CCN': 70, 'CNN': 70, 'COO': 80}
        self.torsion_barriers = {'CCCC': 1.4, 'NCCN': 2.0, 'OCCN': 1.8}

        # Atomic radii for GB calculations (Å)
        self.gb_radii = {
            'C': 1.70, 'N': 1.55, 'O': 1.50, 'S': 1.80, 'P': 1.85,
            'H': 1.20, 'F': 1.47, 'Cl': 1.77, 'Br': 1.90, 'I': 2.09
        }

        # van der Waals parameters
        self.vdw_radii = {
            'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80, 'P': 1.80,
            'H': 1.20, 'F': 1.47, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98
        }

        self.logger.info("MM-GBSA rescoring initialized with GB model: " + self.gb_model)

    def rescore_poses(self, poses: List[Pose], ligand_mol: Chem.Mol,
                     receptor_structure: Structure) -> List[Pose]:
        """Rescore poses using MM-GBSA"""
        rescored_poses = []

        for i, pose in enumerate(poses):
            try:
                self.logger.debug(f"MM-GBSA rescoring pose {i+1}/{len(poses)}")

                # Calculate MM-GBSA binding free energy
                binding_free_energy = self.calculate_binding_free_energy(
                    pose, ligand_mol, receptor_structure
                )

                # Create new pose with MM-GBSA energy
                rescored_pose = Pose(
                    coordinates=pose.coordinates,
                    center=pose.center,
                    rotation=pose.rotation,
                    energy=binding_free_energy,
                    conformer_id=pose.conformer_id
                )

                # Store energy decomposition
                if hasattr(self, '_last_energy_components'):
                    rescored_pose.mmgbsa_components = self._last_energy_components

                rescored_poses.append(rescored_pose)

            except Exception as e:
                self.logger.warning(f"MM-GBSA rescoring failed for pose {i+1}: {e}")
                rescored_poses.append(pose)  # Keep original if rescoring fails

        return rescored_poses

    def _extract_pocket(
        self, ligand_coords: np.ndarray, receptor_structure: Structure
    ) -> Tuple[np.ndarray, List]:
        """Return receptor heavy atoms within `self.pocket_cutoff` of the ligand."""
        atoms = [
            a for a in receptor_structure.get_atoms()
            if (a.element or '').strip().upper() != 'H'
        ]
        if not atoms:
            atoms = list(receptor_structure.get_atoms())
        coords = np.asarray([a.get_coord() for a in atoms], dtype=np.float64)
        if len(coords) == 0 or len(ligand_coords) == 0:
            return np.zeros((0, 3), dtype=np.float64), []

        # Fast bounding-box prefilter before exact distance check
        lo = ligand_coords.min(axis=0) - self.pocket_cutoff
        hi = ligand_coords.max(axis=0) + self.pocket_cutoff
        in_box = np.all((coords >= lo) & (coords <= hi), axis=1)
        box_idx = np.where(in_box)[0]
        if len(box_idx) == 0:
            return np.zeros((0, 3), dtype=np.float64), []

        from scipy.spatial.distance import cdist
        dists = cdist(ligand_coords, coords[box_idx])
        keep = np.min(dists, axis=0) <= self.pocket_cutoff
        sel = box_idx[keep]
        return coords[sel], [atoms[int(i)] for i in sel]

    @staticmethod
    def _get_ligand_charges(ligand_mol: Chem.Mol, indices: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute finite Gasteiger partial charges for ligand atoms."""
        mol = Chem.Mol(ligand_mol)
        try:
            AllChem.ComputeGasteigerCharges(mol)
            raw = [float(a.GetProp('_GasteigerCharge')) for a in mol.GetAtoms()]
        except Exception:
            raw = [0.0] * mol.GetNumAtoms()
        arr = np.asarray([c if np.isfinite(c) else 0.0 for c in raw], dtype=np.float64)
        return arr[indices] if indices is not None else arr

    def calculate_binding_free_energy(self, pose: Pose, ligand_mol: Chem.Mol,
                                    receptor_structure: Structure) -> float:
        """
        Calculate differential single-trajectory MM-GBSA binding free energy:

            ΔG_bind = ΔE_vdW + ΔE_elec + ΔE_strain + ΔG_GB + ΔG_SA + TΔS

        where ΔG_GB = G_GB(complex) - G_GB(pocket) - G_GB(ligand) and
        ΔG_SA = G_SA(complex) - G_SA(pocket) - G_SA(ligand), evaluated over
        heavy atoms within `pocket_cutoff` of the ligand in vectorized NumPy.
        """
        full_lig_coords = np.asarray(pose.coordinates, dtype=np.float64)
        heavy_idx = np.array(
            [a.GetIdx() for a in ligand_mol.GetAtoms() if a.GetAtomicNum() > 1 and a.GetIdx() < len(full_lig_coords)],
            dtype=np.int64,
        )
        if len(heavy_idx) == 0:
            heavy_idx = np.arange(min(len(full_lig_coords), ligand_mol.GetNumAtoms()), dtype=np.int64)
        lig_coords = full_lig_coords[heavy_idx]
        lig_elements = [ligand_mol.GetAtomWithIdx(int(i)).GetSymbol() for i in heavy_idx]
        lig_charges = self._get_ligand_charges(ligand_mol, heavy_idx)

        rec_coords, rec_atoms = self._extract_pocket(lig_coords, receptor_structure)
        rec_elements = [(a.element or 'C').strip().capitalize() for a in rec_atoms]
        rec_charges = np.asarray(self._get_receptor_charges(rec_atoms), dtype=np.float64)

        energy_components: Dict[str, float] = {}

        # 1. Intermolecular MM energy (soft-core vdW + screened Coulomb)
        vdw_e, elec_e = self._calculate_nonbonded_vectorized(
            lig_coords, lig_elements, lig_charges,
            rec_coords, rec_elements, rec_charges,
        )
        # Relative internal ligand strain vs input conformer (cancels when rigid/minimized)
        strain_e = self._calculate_relative_internal_strain(ligand_mol, full_lig_coords)
        mm_energy = vdw_e + elec_e + strain_e
        energy_components['vdW'] = float(vdw_e)
        energy_components['electrostatic'] = float(elec_e)
        energy_components['ligand_strain'] = float(strain_e)
        energy_components['MM_energy'] = float(mm_energy)

        # 2. Differential Generalized Born polar solvation: G_GB(C) - G_GB(R) - G_GB(L)
        gb_energy = self._calculate_gb_solvation_delta(
            lig_coords, lig_elements, lig_charges,
            rec_coords, rec_elements, rec_charges,
        )
        energy_components['GB_solvation'] = float(gb_energy)

        # 3. Differential nonpolar SASA burial: gamma * (SASA(C) - SASA(R) - SASA(L)) < 0
        sa_energy = self._calculate_surface_area_delta(
            lig_coords, lig_elements, rec_coords, rec_elements
        )
        energy_components['SA_solvation'] = float(sa_energy)

        # 4. Conformational entropy penalty
        entropy_energy = self._calculate_entropy_contribution(ligand_mol)
        energy_components['entropy'] = float(entropy_energy)

        total_energy = mm_energy + gb_energy + sa_energy + entropy_energy
        self._last_energy_components = energy_components

        self.logger.debug(f"MM-GBSA components: {energy_components}")
        self.logger.debug(f"Total MM-GBSA energy: {total_energy:.3f} kcal/mol")
        return float(total_energy)

    def _calculate_relative_internal_strain(
        self, ligand_mol: Chem.Mol, pose_coords: np.ndarray
    ) -> float:
        """Relative internal strain E_int(pose) - E_int(input) >= 0."""
        e_pose = (
            self._calculate_bond_energy(ligand_mol, pose_coords)
            + self._calculate_angle_energy(ligand_mol, pose_coords)
            + self._calculate_torsion_energy(ligand_mol, pose_coords)
        )
        if ligand_mol.GetNumConformers() > 0:
            ref_coords = np.asarray(ligand_mol.GetConformer().GetPositions(), dtype=np.float64)
            e_ref = (
                self._calculate_bond_energy(ligand_mol, ref_coords)
                + self._calculate_angle_energy(ligand_mol, ref_coords)
                + self._calculate_torsion_energy(ligand_mol, ref_coords)
            )
            return float(max(0.0, e_pose - e_ref))
        return 0.0

    def _calculate_mm_energy(self, ligand_coords: np.ndarray, receptor_coords: np.ndarray,
                           ligand_mol: Chem.Mol, receptor_atoms: List) -> float:
        """Calculate molecular mechanics energy from actual 3D pose coordinates."""
        mm_energy = 0.0

        # Internal ligand geometry energy evaluated on the pose coordinates
        mm_energy += self._calculate_bond_energy(ligand_mol, ligand_coords)
        mm_energy += self._calculate_angle_energy(ligand_mol, ligand_coords)
        mm_energy += self._calculate_torsion_energy(ligand_mol, ligand_coords)

        # Non-bonded interactions (van der Waals + electrostatic)
        mm_energy += self._calculate_nonbonded_energy(
            ligand_coords, receptor_coords, ligand_mol, receptor_atoms
        )

        return mm_energy

    def _calculate_bond_energy(
        self, ligand_mol: Chem.Mol, ligand_coords: Optional[np.ndarray] = None
    ) -> float:
        """Calculate harmonic bond stretching energy from 3D coordinates."""
        if ligand_coords is None:
            if ligand_mol.GetNumConformers() == 0:
                return 0.0
            ligand_coords = np.asarray(ligand_mol.GetConformer().GetPositions(), dtype=np.float64)

        bond_energy = 0.0
        for bond in ligand_mol.GetBonds():
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()
            i1, i2 = atom1.GetIdx(), atom2.GetIdx()
            if i1 >= len(ligand_coords) or i2 >= len(ligand_coords):
                continue

            bond_type = f"{atom1.GetSymbol()}{atom2.GetSymbol()}"
            if bond_type not in self.bond_force_constants:
                bond_type = bond_type[::-1]

            if bond_type in self.bond_force_constants:
                ideal_length = self._get_ideal_bond_length(atom1.GetSymbol(), atom2.GetSymbol())
                current_length = float(np.linalg.norm(ligand_coords[i1] - ligand_coords[i2]))
                force_constant = self.bond_force_constants[bond_type]
                bond_energy += force_constant * (current_length - ideal_length) ** 2

        return bond_energy

    def _calculate_angle_energy(
        self, ligand_mol: Chem.Mol, ligand_coords: Optional[np.ndarray] = None
    ) -> float:
        """Calculate harmonic angle bending energy from 3D coordinates."""
        if ligand_coords is None:
            if ligand_mol.GetNumConformers() == 0:
                return 0.0
            ligand_coords = np.asarray(ligand_mol.GetConformer().GetPositions(), dtype=np.float64)

        angle_energy = 0.0
        for atom in ligand_mol.GetAtoms():
            center_idx = atom.GetIdx()
            if center_idx >= len(ligand_coords):
                continue
            neighbors = [n for n in atom.GetNeighbors() if n.GetIdx() < len(ligand_coords)]
            if len(neighbors) < 2:
                continue

            hyb = atom.GetHybridization()
            if hyb == Chem.HybridizationType.SP:
                ideal_angle = 180.0
            elif hyb == Chem.HybridizationType.SP2:
                ideal_angle = 120.0
            else:
                ideal_angle = 109.5

            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    angle_type = f"{neighbors[i].GetSymbol()}{atom.GetSymbol()}{neighbors[j].GetSymbol()}"
                    if angle_type not in self.angle_force_constants:
                        angle_type = f"{neighbors[j].GetSymbol()}{atom.GetSymbol()}{neighbors[i].GetSymbol()}"
                    if angle_type in self.angle_force_constants:
                        v1 = ligand_coords[neighbors[i].GetIdx()] - ligand_coords[center_idx]
                        v2 = ligand_coords[neighbors[j].GetIdx()] - ligand_coords[center_idx]
                        n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
                        if n1 < 1e-6 or n2 < 1e-6:
                            continue
                        cos_theta = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
                        current_angle = math.degrees(math.acos(cos_theta))
                        force_constant = self.angle_force_constants[angle_type]
                        angle_deviation = math.radians(current_angle - ideal_angle)
                        angle_energy += force_constant * angle_deviation ** 2

        return angle_energy

    def _calculate_torsion_energy(
        self, ligand_mol: Chem.Mol, ligand_coords: Optional[np.ndarray] = None
    ) -> float:
        """Calculate dihedral torsional barrier energy from 3D coordinates."""
        if ligand_coords is None:
            if ligand_mol.GetNumConformers() == 0:
                return 0.0
            ligand_coords = np.asarray(ligand_mol.GetConformer().GetPositions(), dtype=np.float64)

        torsion_energy = 0.0
        for bond in ligand_mol.GetBonds():
            if not bond.IsInRing():
                atom1 = bond.GetBeginAtom()
                atom2 = bond.GetEndAtom()
                neighbors1 = [n for n in atom1.GetNeighbors() if n.GetIdx() != atom2.GetIdx()]
                neighbors2 = [n for n in atom2.GetNeighbors() if n.GetIdx() != atom1.GetIdx()]

                if neighbors1 and neighbors2:
                    i0, i1, i2, i3 = (
                        neighbors1[0].GetIdx(), atom1.GetIdx(),
                        atom2.GetIdx(), neighbors2[0].GetIdx()
                    )
                    if max(i0, i1, i2, i3) >= len(ligand_coords):
                        continue
                    torsion_type = f"{neighbors1[0].GetSymbol()}{atom1.GetSymbol()}{atom2.GetSymbol()}{neighbors2[0].GetSymbol()}"
                    if torsion_type not in self.torsion_barriers:
                        torsion_type = torsion_type[::-1]
                    if torsion_type in self.torsion_barriers:
                        b0 = ligand_coords[i1] - ligand_coords[i0]
                        b1 = ligand_coords[i2] - ligand_coords[i1]
                        b2 = ligand_coords[i3] - ligand_coords[i2]
                        b1_norm = float(np.linalg.norm(b1))
                        if b1_norm < 1e-6:
                            continue
                        b1_u = b1 / b1_norm
                        v = b0 - np.dot(b0, b1_u) * b1_u
                        w = b2 - np.dot(b2, b1_u) * b1_u
                        x = float(np.dot(v, w))
                        y = float(np.dot(np.cross(b1_u, v), w))
                        torsion_angle = math.atan2(y, x)
                        barrier = self.torsion_barriers[torsion_type]
                        torsion_energy += barrier * (1.0 + math.cos(3.0 * torsion_angle))

        return torsion_energy

    def _calculate_nonbonded_vectorized(
        self,
        lig_coords: np.ndarray,
        lig_elements: List[str],
        lig_charges: np.ndarray,
        rec_coords: np.ndarray,
        rec_elements: List[str],
        rec_charges: np.ndarray,
    ) -> Tuple[float, float]:
        """Vectorized intermolecular LJ 12-6 (with soft-core clash cap) + Coulomb."""
        if len(lig_coords) == 0 or len(rec_coords) == 0:
            return 0.0, 0.0

        from scipy.spatial.distance import cdist
        eps_table = {
            'C': 0.110, 'N': 0.170, 'O': 0.210, 'S': 0.250, 'P': 0.200,
            'H': 0.020, 'F': 0.061, 'Cl': 0.265, 'Br': 0.320, 'I': 0.400,
        }
        lig_r = np.array([self.vdw_radii.get(e, 1.70) for e in lig_elements], dtype=np.float64)
        rec_r = np.array([self.vdw_radii.get(e, 1.70) for e in rec_elements], dtype=np.float64)
        lig_eps = np.array([eps_table.get(e, 0.11) for e in lig_elements], dtype=np.float64)
        rec_eps = np.array([eps_table.get(e, 0.11) for e in rec_elements], dtype=np.float64)

        dists = cdist(lig_coords, rec_coords)
        mask = dists < self.pocket_cutoff
        if not np.any(mask):
            return 0.0, 0.0

        li, ri = np.where(mask)
        d = np.maximum(dists[li, ri], 1.0)
        # Standard Lorentz-Berthelot: r_min = r_i + r_j -> sigma = r_min / 2^(1/6)
        r_min = lig_r[li] + rec_r[ri]
        sigma = r_min / (2.0 ** (1.0 / 6.0))
        eps = np.sqrt(lig_eps[li] * rec_eps[ri])

        # Soft-core LJ capped smoothly so unminimized docked poses don't explode
        ratio = np.minimum(sigma / d, 1.35)
        r6 = ratio ** 6
        vdw_pair = np.minimum(4.0 * eps * (r6 * r6 - r6), self.vdw_clash_cap)
        vdw_energy = float(np.sum(vdw_pair))

        # Screened interior Coulomb
        coulomb_k = 332.0637
        q_prod = lig_charges[li] * rec_charges[ri]
        elec_pair = (coulomb_k * q_prod) / (self.dielectric_interior * np.maximum(d, 2.0))
        elec_energy = float(np.sum(elec_pair))
        return vdw_energy, elec_energy

    def _gb_energy_single(
        self, coords: np.ndarray, elements: List[str], charges: np.ndarray
    ) -> float:
        """Vectorized OBC Generalized Born solvation energy for a single state."""
        n = len(coords)
        if n == 0:
            return 0.0

        from scipy.spatial.distance import cdist
        radii = np.array([self.gb_radii.get(e, 1.70) for e in elements], dtype=np.float64)
        dists = cdist(coords, coords)
        np.fill_diagonal(dists, np.inf)

        # Pairwise volume-overlap proxy for effective Born radii (OBC2)
        cutoff_mat = radii[:, None] + radii[None, :] + 1.4
        close = dists < cutoff_mat
        safe_d = np.where(close, np.maximum(dists, 1.0), np.inf)
        integral = np.sum(np.where(close, (radii[None, :] ** 3) / (safe_d ** 3), 0.0), axis=1)
        psi = integral * radii / 16.0
        obc = np.tanh(0.8 * psi - 2.9 * psi * psi + 1.9 * (psi ** 3))
        born_r = radii / np.maximum(1.0 - np.clip(obc, -0.9, 0.9), 0.1)
        born_r = np.maximum(born_r, radii)

        prefactor = -0.5 * 332.0637 * (1.0 / self.dielectric_interior - 1.0 / self.dielectric_exterior)
        # Self + pairwise Still-Tempczyk matrix (factor 0.5 covers both self & symmetric off-diag)
        d2 = np.where(np.isfinite(dists), dists * dists, 0.0)
        ri_rj = born_r[:, None] * born_r[None, :]
        fgb = np.sqrt(d2 + ri_rj * np.exp(-d2 / (4.0 * ri_rj)))
        np.fill_diagonal(fgb, born_r)
        q_mat = charges[:, None] * charges[None, :]
        return float(prefactor * np.sum(q_mat / fgb))

    def _calculate_gb_solvation_delta(
        self,
        lig_coords: np.ndarray,
        lig_elements: List[str],
        lig_charges: np.ndarray,
        rec_coords: np.ndarray,
        rec_elements: List[str],
        rec_charges: np.ndarray,
    ) -> float:
        """Differential polar solvation ΔG_GB = G_GB(complex) - G_GB(pocket) - G_GB(ligand)."""
        if len(lig_coords) == 0 or len(rec_coords) == 0:
            return 0.0
        c_coords = np.vstack([lig_coords, rec_coords])
        c_elems = lig_elements + rec_elements
        c_charges = np.concatenate([lig_charges, rec_charges])

        g_complex = self._gb_energy_single(c_coords, c_elems, c_charges)
        g_pocket = self._gb_energy_single(rec_coords, rec_elements, rec_charges)
        g_ligand = self._gb_energy_single(lig_coords, lig_elements, lig_charges)
        return float(g_complex - g_pocket - g_ligand)

    def _sasa_single(self, coords: np.ndarray, elements: List[str]) -> float:
        """Vectorized neighbor-burial SASA approximation (Å²)."""
        if len(coords) == 0:
            return 0.0
        from scipy.spatial.distance import cdist
        probe = 1.4
        eff_r = np.array([self.vdw_radii.get(e, 1.70) + probe for e in elements], dtype=np.float64)
        dists = cdist(coords, coords)
        np.fill_diagonal(dists, np.inf)
        neighbors = np.sum(dists < (eff_r[:, None] + eff_r[None, :]), axis=1)
        max_sa = 4.0 * math.pi * (eff_r ** 2)
        burial = np.minimum(neighbors / 18.0, 1.0)
        return float(np.sum(max_sa * (1.0 - burial)))

    def _calculate_surface_area_delta(
        self,
        lig_coords: np.ndarray,
        lig_elements: List[str],
        rec_coords: np.ndarray,
        rec_elements: List[str],
    ) -> float:
        """Differential nonpolar solvation ΔG_SA = γ * (SASA(C) - SASA(R) - SASA(L)) <= 0."""
        if len(lig_coords) == 0 or len(rec_coords) == 0:
            return 0.0
        c_coords = np.vstack([lig_coords, rec_coords])
        c_elems = lig_elements + rec_elements
        delta_sasa = (
            self._sasa_single(c_coords, c_elems)
            - self._sasa_single(rec_coords, rec_elements)
            - self._sasa_single(lig_coords, lig_elements)
        )
        return float(self.surface_tension * delta_sasa)

    def _calculate_entropy_contribution(self, ligand_mol: Chem.Mol) -> float:
        """Conformational entropy penalty upon binding (~0.35 kcal/mol per rotatable bond)."""
        num_rotatable_bonds = Descriptors.NumRotatableBonds(ligand_mol)
        return float(num_rotatable_bonds * 0.35)

    def _get_ideal_bond_length(self, element1: str, element2: str) -> float:
        """Get ideal bond length between two elements"""
        bond_lengths = {
            ('C', 'C'): 1.54, ('C', 'N'): 1.47, ('C', 'O'): 1.43,
            ('C', 'S'): 1.81, ('N', 'N'): 1.45, ('N', 'O'): 1.40,
            ('O', 'O'): 1.48, ('C', 'H'): 1.09, ('N', 'H'): 1.01,
            ('O', 'H'): 0.96, ('S', 'H'): 1.34
        }

        bond_pair = tuple(sorted([element1, element2]))
        return bond_lengths.get(bond_pair, 1.5)  # Default bond length

    def _get_receptor_charges(self, receptor_atoms: List) -> List[float]:
        """Get receptor partial charges (simplified)"""
        charges = []

        for atom in receptor_atoms:
            residue_name = atom.get_parent().get_resname()
            atom_name = atom.get_name()
            element = atom.element

            charge = 0.0

            # Simplified charge assignment
            if residue_name == 'ARG':
                if atom_name in ['NH1', 'NH2']:
                    charge = 0.8
                elif atom_name == 'NE':
                    charge = 0.4
            elif residue_name == 'LYS':
                if atom_name == 'NZ':
                    charge = 1.0
            elif residue_name == 'ASP':
                if atom_name in ['OD1', 'OD2']:
                    charge = -0.8
            elif residue_name == 'GLU':
                if atom_name in ['OE1', 'OE2']:
                    charge = -0.8
            elif residue_name == 'HIS':
                if atom_name in ['ND1', 'NE2']:
                    charge = 0.4
            # Add more residue types as needed

            charges.append(charge)

        return charges