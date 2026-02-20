"""
Vina-Style Scoring Function for PandaDock

Implements AutoDock Vina-like scoring function with:
1. Gaussian steric interactions (gauss1, gauss2)
2. Repulsion term for steric clashes
3. Hydrophobic interactions
4. Hydrogen bonding

Reference: Trott & Olson, J Comput Chem 2010
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy.spatial.distance import cdist
from Bio.PDB import Structure
from rdkit import Chem
from rdkit.Chem import AllChem, Lipinski


class VinaScoring:
    """Vina-style empirical scoring function for molecular docking"""

    # Standard AutoDock Vina weights
    DEFAULT_WEIGHTS = {
        'gauss1': -0.035579,
        'gauss2': -0.005156,
        'repulsion': 0.840245,
        'hydrophobic': -0.035069,
        'hydrogen': -0.587439
    }

    # Van der Waals radii for common atoms (Å)
    VDW_RADII = {
        'C': 1.9, 'A': 1.9,  # A = aromatic carbon in Vina
        'N': 1.8, 'NA': 1.8,  # NA = H-bond acceptor nitrogen
        'O': 1.7, 'OA': 1.7,  # OA = H-bond acceptor oxygen
        'S': 2.0, 'SA': 2.0,
        'H': 1.0, 'HD': 1.0,  # HD = H-bond donor hydrogen
        'P': 2.1,
        'F': 1.5,
        'Cl': 1.8,
        'Br': 2.0,
        'I': 2.2,
        'Mg': 1.3,
        'Mn': 1.3,
        'Zn': 1.3,
        'Ca': 1.7,
        'Fe': 1.3
    }

    def __init__(self, **params):
        self.logger = logging.getLogger("pandadock.scoring.vina")

        # Allow custom weights
        self.weights = {
            'gauss1': params.get('gauss1_weight', self.DEFAULT_WEIGHTS['gauss1']),
            'gauss2': params.get('gauss2_weight', self.DEFAULT_WEIGHTS['gauss2']),
            'repulsion': params.get('repulsion_weight', self.DEFAULT_WEIGHTS['repulsion']),
            'hydrophobic': params.get('hydrophobic_weight', self.DEFAULT_WEIGHTS['hydrophobic']),
            'hydrogen': params.get('hydrogen_weight', self.DEFAULT_WEIGHTS['hydrogen'])
        }

        # Gaussian parameters
        self.gauss1_offset = 0.0
        self.gauss1_width = 0.5
        self.gauss2_offset = 3.0
        self.gauss2_width = 2.0

        # Repulsion cutoff
        self.repulsion_cutoff = 0.0

        # Hydrophobic cutoff distances
        self.hydrophobic_good = 0.5
        self.hydrophobic_bad = 1.5

        # Hydrogen bond parameters
        self.hbond_good = -0.7
        self.hbond_bad = 0.0

        # Distance cutoff for interactions
        self.cutoff = 8.0

    def calculate_binding_energy(self, ligand_coords: np.ndarray,
                                receptor_structure: Structure,
                                ligand_mol: Optional[Chem.Mol] = None) -> float:
        """
        Calculate Vina-style binding energy

        Args:
            ligand_coords: 3D coordinates of ligand atoms (N x 3)
            receptor_structure: BioPython receptor structure
            ligand_mol: RDKit molecule object

        Returns:
            Binding energy in kcal/mol
        """
        if ligand_mol is None or len(ligand_coords) == 0:
            return 0.0

        # Get receptor information
        receptor_atoms = list(receptor_structure.get_atoms())
        receptor_coords = np.array([atom.get_coord() for atom in receptor_atoms])

        # Get atom types
        ligand_types = self._get_ligand_atom_types(ligand_mol)
        receptor_types = self._get_receptor_atom_types(receptor_atoms)

        # Get radii
        ligand_radii = np.array([self._get_vdw_radius(t) for t in ligand_types])
        receptor_radii = np.array([self._get_vdw_radius(t) for t in receptor_types])

        # Calculate pairwise distances
        distances = cdist(ligand_coords, receptor_coords)

        # Calculate sum of radii for each pair
        radii_sum = ligand_radii[:, np.newaxis] + receptor_radii[np.newaxis, :]

        # Surface distance (d - r_i - r_j)
        surface_distances = distances - radii_sum

        # Calculate energy terms
        energy_components = {}

        # Gaussian 1 (narrow, close-range attraction)
        energy_components['gauss1'] = self._calculate_gauss(
            surface_distances, self.gauss1_offset, self.gauss1_width
        )

        # Gaussian 2 (broad, medium-range attraction)
        energy_components['gauss2'] = self._calculate_gauss(
            surface_distances, self.gauss2_offset, self.gauss2_width
        )

        # Repulsion (steric clashes)
        energy_components['repulsion'] = self._calculate_repulsion(surface_distances)

        # Hydrophobic interactions
        energy_components['hydrophobic'] = self._calculate_hydrophobic(
            surface_distances, ligand_types, receptor_types, ligand_mol, receptor_atoms
        )

        # Hydrogen bonding
        energy_components['hydrogen'] = self._calculate_hydrogen_bonds(
            surface_distances, ligand_types, receptor_types, ligand_mol, receptor_atoms
        )

        # Calculate weighted total
        total_energy = sum(
            self.weights[term] * energy
            for term, energy in energy_components.items()
        )

        # Apply Vina's N_rot correction for ligand flexibility
        n_rot = self._count_rotatable_bonds(ligand_mol)
        flexibility_penalty = 1 + 0.05846 * n_rot

        final_energy = total_energy / flexibility_penalty

        self.logger.debug(f"Vina scoring components: {energy_components}")
        self.logger.debug(f"N_rot: {n_rot}, flexibility_penalty: {flexibility_penalty:.3f}")
        self.logger.debug(f"Final energy: {final_energy:.3f} kcal/mol")

        return final_energy

    def _calculate_gauss(self, surface_distances: np.ndarray,
                        offset: float, width: float) -> float:
        """Calculate Gaussian interaction term"""
        # Only consider atoms within cutoff
        mask = surface_distances < self.cutoff
        if not np.any(mask):
            return 0.0

        d = surface_distances[mask] - offset
        gauss = np.exp(-(d / width) ** 2)

        return np.sum(gauss)

    def _calculate_repulsion(self, surface_distances: np.ndarray) -> float:
        """Calculate repulsion term for steric clashes"""
        # Only penalize when surface distance < 0 (atoms overlapping)
        clashing = surface_distances < self.repulsion_cutoff
        if not np.any(clashing):
            return 0.0

        d = surface_distances[clashing]
        repulsion = d ** 2

        return np.sum(repulsion)

    def _calculate_hydrophobic(self, surface_distances: np.ndarray,
                              ligand_types: List[str],
                              receptor_types: List[str],
                              ligand_mol: Chem.Mol,
                              receptor_atoms: List) -> float:
        """Calculate hydrophobic interaction term"""
        hydrophobic_energy = 0.0

        # Identify hydrophobic atoms
        ligand_hydrophobic = self._is_hydrophobic_ligand(ligand_mol)
        receptor_hydrophobic = self._is_hydrophobic_receptor(receptor_atoms)

        for i, is_lig_hydrophobic in enumerate(ligand_hydrophobic):
            if not is_lig_hydrophobic:
                continue
            for j, is_rec_hydrophobic in enumerate(receptor_hydrophobic):
                if not is_rec_hydrophobic:
                    continue

                d = surface_distances[i, j]

                if d >= self.cutoff:
                    continue

                # Piecewise linear function
                if d < self.hydrophobic_good:
                    contribution = 1.0
                elif d < self.hydrophobic_bad:
                    contribution = (self.hydrophobic_bad - d) / (self.hydrophobic_bad - self.hydrophobic_good)
                else:
                    contribution = 0.0

                hydrophobic_energy += contribution

        return hydrophobic_energy

    def _calculate_hydrogen_bonds(self, surface_distances: np.ndarray,
                                  ligand_types: List[str],
                                  receptor_types: List[str],
                                  ligand_mol: Chem.Mol,
                                  receptor_atoms: List) -> float:
        """Calculate hydrogen bonding term"""
        hbond_energy = 0.0

        # Identify H-bond donors/acceptors
        ligand_donors, ligand_acceptors = self._get_hbond_atoms_ligand(ligand_mol)
        receptor_donors, receptor_acceptors = self._get_hbond_atoms_receptor(receptor_atoms)

        # Donor-acceptor pairs
        for donor_idx in ligand_donors:
            for acceptor_idx in receptor_acceptors:
                d = surface_distances[donor_idx, acceptor_idx]
                hbond_energy += self._hbond_term(d)

        for acceptor_idx in ligand_acceptors:
            for donor_idx in receptor_donors:
                d = surface_distances[acceptor_idx, donor_idx]
                hbond_energy += self._hbond_term(d)

        return hbond_energy

    def _hbond_term(self, surface_distance: float) -> float:
        """Calculate hydrogen bond contribution"""
        if surface_distance >= self.cutoff:
            return 0.0

        # Piecewise linear function
        if surface_distance < self.hbond_good:
            return 1.0
        elif surface_distance < self.hbond_bad:
            return (self.hbond_bad - surface_distance) / (self.hbond_bad - self.hbond_good)
        else:
            return 0.0

    def _get_ligand_atom_types(self, mol: Chem.Mol) -> List[str]:
        """Assign Vina-like atom types to ligand atoms"""
        types = []
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()

            if symbol == 'C':
                # Aromatic carbon
                if atom.GetIsAromatic():
                    types.append('A')
                else:
                    types.append('C')
            elif symbol == 'N':
                # H-bond acceptor nitrogen
                if atom.GetTotalNumHs() > 0:
                    types.append('N')
                else:
                    types.append('NA')
            elif symbol == 'O':
                # H-bond acceptor oxygen
                if atom.GetTotalNumHs() > 0:
                    types.append('O')
                else:
                    types.append('OA')
            elif symbol == 'S':
                types.append('S')
            elif symbol == 'H':
                # Check if donor hydrogen
                neighbors = atom.GetNeighbors()
                if neighbors and neighbors[0].GetSymbol() in ['N', 'O']:
                    types.append('HD')
                else:
                    types.append('H')
            else:
                types.append(symbol)

        return types

    def _get_receptor_atom_types(self, receptor_atoms: List) -> List[str]:
        """Assign Vina-like atom types to receptor atoms"""
        types = []
        for atom in receptor_atoms:
            element = atom.element.strip()
            atom_name = atom.get_name()

            if element == 'C':
                # Check if likely aromatic (simplified)
                if atom_name in ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ']:
                    types.append('A')
                else:
                    types.append('C')
            elif element == 'N':
                # Backbone nitrogen or sidechain with H
                if atom_name in ['N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ']:
                    types.append('N')
                else:
                    types.append('NA')
            elif element == 'O':
                # Backbone carbonyl is acceptor, hydroxyl can be donor
                if atom_name in ['O', 'OD1', 'OD2', 'OE1', 'OE2']:
                    types.append('OA')
                else:
                    types.append('O')
            elif element == 'S':
                types.append('S')
            elif element == 'H':
                types.append('H')
            else:
                types.append(element if element else 'C')

        return types

    def _get_vdw_radius(self, atom_type: str) -> float:
        """Get van der Waals radius for atom type"""
        return self.VDW_RADII.get(atom_type, 1.7)

    def _is_hydrophobic_ligand(self, mol: Chem.Mol) -> List[bool]:
        """Identify hydrophobic atoms in ligand"""
        hydrophobic = []
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            # Carbon (non-polar) and sulfur are hydrophobic
            if symbol == 'C':
                # Check if attached to polar atoms
                neighbors = atom.GetNeighbors()
                is_polar = any(n.GetSymbol() in ['N', 'O', 'S', 'F', 'Cl', 'Br', 'I']
                              for n in neighbors)
                hydrophobic.append(not is_polar)
            elif symbol == 'S':
                hydrophobic.append(True)
            elif symbol == 'F':
                hydrophobic.append(True)
            else:
                hydrophobic.append(False)

        return hydrophobic

    def _is_hydrophobic_receptor(self, receptor_atoms: List) -> List[bool]:
        """Identify hydrophobic atoms in receptor"""
        hydrophobic = []
        hydrophobic_residues = {'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO'}

        for atom in receptor_atoms:
            element = atom.element.strip()
            residue = atom.get_parent()
            resname = residue.get_resname() if residue else ''

            # Carbon atoms in hydrophobic residues
            if element == 'C' and resname in hydrophobic_residues:
                hydrophobic.append(True)
            elif element == 'S':
                hydrophobic.append(True)
            else:
                hydrophobic.append(False)

        return hydrophobic

    def _get_hbond_atoms_ligand(self, mol: Chem.Mol) -> Tuple[List[int], List[int]]:
        """Get H-bond donor and acceptor indices in ligand"""
        donors = []
        acceptors = []

        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            symbol = atom.GetSymbol()

            if symbol in ['N', 'O']:
                # Acceptor if has lone pairs
                acceptors.append(idx)
                # Donor if has hydrogen
                if atom.GetTotalNumHs() > 0:
                    donors.append(idx)

        return donors, acceptors

    def _get_hbond_atoms_receptor(self, receptor_atoms: List) -> Tuple[List[int], List[int]]:
        """Get H-bond donor and acceptor indices in receptor"""
        donors = []
        acceptors = []

        for i, atom in enumerate(receptor_atoms):
            element = atom.element.strip()
            atom_name = atom.get_name()

            if element == 'N':
                # Backbone N and sidechain NH are donors
                if atom_name in ['N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ']:
                    donors.append(i)
                acceptors.append(i)
            elif element == 'O':
                acceptors.append(i)
                # Hydroxyl oxygens are also donors
                if atom_name in ['OG', 'OG1', 'OH']:
                    donors.append(i)

        return donors, acceptors

    def _count_rotatable_bonds(self, mol: Chem.Mol) -> int:
        """Count rotatable bonds for flexibility penalty"""
        try:
            return Lipinski.NumRotatableBonds(mol)
        except:
            return 0
