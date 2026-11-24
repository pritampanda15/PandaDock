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

    # Vinardo weights (Quiroga & Villarreal, PLOS ONE 2016)
    VINARDO_WEIGHTS = {
        'gauss1': -0.045,
        'gauss2': 0.0,
        'repulsion': 0.800,
        'hydrophobic': -0.035,
        'hydrogen': -0.600
    }

    # Standard amino acid carbons NOT covalently bonded to any heteroatom
    # (N, O, S). Follows AutoDock Vina's C_H typing: backbone C and CA are
    # never hydrophobic because they bond to N/O, whereas aliphatic/aromatic
    # carbons on polar sidechains (e.g. TYR ring, LYS/ARG/GLU methylene chains)
    # ARE hydrophobic.
    RECEPTOR_HYDROPHOBIC_CARBONS = {
        'ALA': frozenset({'CB'}),
        'ARG': frozenset({'CB', 'CG'}),
        'ASN': frozenset({'CB'}),
        'ASP': frozenset({'CB'}),
        'CYS': frozenset(),
        'GLN': frozenset({'CB', 'CG'}),
        'GLU': frozenset({'CB', 'CG'}),
        'GLY': frozenset(),
        'HIS': frozenset({'CB'}),
        'HID': frozenset({'CB'}),
        'HIE': frozenset({'CB'}),
        'HIP': frozenset({'CB'}),
        'ILE': frozenset({'CB', 'CG1', 'CG2', 'CD1'}),
        'LEU': frozenset({'CB', 'CG', 'CD1', 'CD2'}),
        'LYS': frozenset({'CB', 'CG', 'CD'}),
        'MET': frozenset({'CB'}),
        'PHE': frozenset({'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'}),
        'PRO': frozenset({'CB', 'CG'}),
        'SER': frozenset(),
        'THR': frozenset({'CG2'}),
        'TRP': frozenset({'CB', 'CG', 'CD2', 'CE3', 'CZ2', 'CZ3', 'CH2'}),
        'TYR': frozenset({'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2'}),
        'VAL': frozenset({'CB', 'CG1', 'CG2'}),
    }

    # Side-chain H-bond donor heavy atoms at physiological pH 7.4.
    # Backbone N is a donor for every standard residue EXCEPT PRO.
    RECEPTOR_SIDECHAIN_DONORS = {
        'ARG': frozenset({'NE', 'NH1', 'NH2'}),
        'ASN': frozenset({'ND2'}),
        'GLN': frozenset({'NE2'}),
        'HIS': frozenset({'ND1', 'NE2'}),
        'HID': frozenset({'ND1'}),
        'HIE': frozenset({'NE2'}),
        'HIP': frozenset({'ND1', 'NE2'}),
        'LYS': frozenset({'NZ'}),
        'SER': frozenset({'OG'}),
        'THR': frozenset({'OG1'}),
        'TRP': frozenset({'NE1'}),
        'TYR': frozenset({'OH'}),
    }

    # Side-chain H-bond acceptor heavy atoms with an available lone pair at pH 7.4.
    # Backbone O / OXT are acceptors for all standard residues. Amide N, LYS NZ,
    # ARG guanidinium, and TRP NE1 have no available lone pair and are NOT acceptors.
    RECEPTOR_SIDECHAIN_ACCEPTORS = {
        'ASN': frozenset({'OD1'}),
        'ASP': frozenset({'OD1', 'OD2'}),
        'GLN': frozenset({'OE1'}),
        'GLU': frozenset({'OE1', 'OE2'}),
        'HIS': frozenset({'ND1', 'NE2'}),
        'HID': frozenset({'NE2'}),
        'HIE': frozenset({'ND1'}),
        'HIP': frozenset(),
        'SER': frozenset({'OG'}),
        'THR': frozenset({'OG1'}),
        'TYR': frozenset({'OH'}),
    }

    STANDARD_RESIDUES = frozenset(RECEPTOR_HYDROPHOBIC_CARBONS.keys())

    # Van der Waals radii for common atoms (Å; metals follow AutoDock Vina XS_TYPE_Met_D = 1.2 Å)
    VDW_RADII = {
        'C': 1.9, 'A': 1.9,  # A = aromatic carbon in Vina
        'N': 1.8, 'NA': 1.8,  # NA = H-bond acceptor nitrogen
        'O': 1.7, 'OA': 1.7,  # OA = H-bond acceptor oxygen
        'S': 2.0, 'SA': 2.0,
        'H': 1.0, 'HD': 1.0,  # HD = H-bond donor hydrogen
        'P': 2.1,
        'F': 1.5,
        'Cl': 1.8, 'CL': 1.8,
        'Br': 2.0, 'BR': 2.0,
        'I': 2.2,
        'Mg': 1.2, 'MG': 1.2,
        'Mn': 1.2, 'MN': 1.2,
        'Zn': 1.2, 'ZN': 1.2,
        'Ca': 1.2, 'CA': 1.2,
        'Fe': 1.2, 'FE': 1.2,
        'Cu': 1.2, 'CU': 1.2,
        'Co': 1.2, 'CO': 1.2,
        'Ni': 1.2, 'NI': 1.2,
        'Cd': 1.2, 'CD': 1.2,
        'Hg': 1.2, 'HG': 1.2,
        'Pt': 1.3, 'PT': 1.3,
        'Pd': 1.3, 'PD': 1.3,
    }

    # Catalytic metal ions treated as H-bond donors (AutoDock Vina XS_TYPE_Met_D)
    METAL_DONOR_ELEMENTS = frozenset({
        'MG', 'MN', 'ZN', 'CA', 'FE', 'CU', 'CO', 'NI', 'CD', 'HG', 'PT', 'PD', 'MO',
    })

    def __init__(self, preset: str = 'vina', **params):
        self.logger = logging.getLogger("pandadock.scoring.vina")
        self.preset = preset.lower()

        if self.preset == 'vinardo':
            base_weights = self.VINARDO_WEIGHTS
            default_g1_width = 0.8
            default_hydro_good = 0.0
            default_hydro_bad = 2.5
            default_hbond_good = -0.6
        else:
            base_weights = self.DEFAULT_WEIGHTS
            default_g1_width = 0.5
            default_hydro_good = 0.5
            default_hydro_bad = 1.5
            default_hbond_good = -0.7

        # Allow custom weights
        self.weights = {
            'gauss1': params.get('gauss1_weight', base_weights['gauss1']),
            'gauss2': params.get('gauss2_weight', base_weights['gauss2']),
            'repulsion': params.get('repulsion_weight', base_weights['repulsion']),
            'hydrophobic': params.get('hydrophobic_weight', base_weights['hydrophobic']),
            'hydrogen': params.get('hydrogen_weight', base_weights['hydrogen'])
        }

        # Gaussian parameters
        self.gauss1_offset = params.get('gauss1_offset', 0.0)
        self.gauss1_width = params.get('gauss1_width', default_g1_width)
        self.gauss2_offset = params.get('gauss2_offset', 3.0)
        self.gauss2_width = params.get('gauss2_width', 2.0)

        # Repulsion cutoff
        self.repulsion_cutoff = params.get('repulsion_cutoff', 0.0)

        # Hydrophobic cutoff distances
        self.hydrophobic_good = params.get('hydrophobic_good', default_hydro_good)
        self.hydrophobic_bad = params.get('hydrophobic_bad', default_hydro_bad)

        # Hydrogen bond parameters
        self.hbond_good = params.get('hbond_good', default_hbond_good)
        self.hbond_bad = params.get('hbond_bad', 0.0)

        # Torsional entropy denominator weight (1 + n_rot_weight * n_rot)
        self.n_rot_weight = params.get('n_rot_weight', 0.05846)

        # Distance cutoff for interactions
        self.cutoff = params.get('cutoff', 8.0)

    @classmethod
    def vinardo(cls, **params) -> 'VinaScoring':
        """Construct a Vinardo scoring function (Quiroga & Villarreal, 2016)."""
        return cls(preset='vinardo', **params)

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
        flexibility_penalty = 1.0 + self.n_rot_weight * n_rot

        final_energy = self.normalize_energy(total_energy, n_rot)

        self.logger.debug(f"Vina scoring components: {energy_components}")
        self.logger.debug(f"N_rot: {n_rot}, flexibility_penalty: {flexibility_penalty:.3f}")
        self.logger.debug(f"Final energy: {final_energy:.3f} kcal/mol")

        return final_energy

    def normalize_energy(self, raw_energy: float, n_rot: int) -> float:
        """Apply Vina's torsional entropy denominator: E / (1 + w_rot * n_rot)."""
        return float(raw_energy) / (1.0 + self.n_rot_weight * max(0, int(n_rot)))

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
                # H-bond donor/acceptor nitrogen
                if atom.GetTotalNumHs(includeNeighbors=True) > 0:
                    types.append('N')
                else:
                    types.append('NA')
            elif symbol == 'O':
                # H-bond donor/acceptor oxygen
                if atom.GetTotalNumHs(includeNeighbors=True) > 0:
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
        """Identify hydrophobic atoms in ligand (AutoDock Vina C_H rule).

        In AutoDock Vina, a carbon is hydrophobic iff it is not covalently
        bonded to any heteroatom (N, O, S, P, F, Cl, Br, I).
        """
        hydrophobic = []
        polar_elements = {'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I'}
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C':
                is_polar = any(
                    n.GetSymbol() in polar_elements for n in atom.GetNeighbors()
                )
                hydrophobic.append(not is_polar)
            else:
                hydrophobic.append(False)

        return hydrophobic

    def _is_hydrophobic_receptor(self, receptor_atoms: List) -> List[bool]:
        """Identify hydrophobic atoms in receptor using residue-atom connectivity.

        Uses the standard 20-amino-acid heavy-atom table so backbone C and CA
        (which bond to N/O) are never hydrophobic, while nonpolar aliphatic and
        aromatic carbons on polar sidechains (e.g. TYR ring, LYS/ARG/GLU
        methylenes) are correctly classified as hydrophobic.
        """
        hydrophobic = []
        for atom in receptor_atoms:
            element = atom.element.strip().upper()
            if element != 'C':
                hydrophobic.append(False)
                continue

            atom_name = atom.get_name().strip().upper()
            residue = atom.get_parent()
            resname = residue.get_resname().strip().upper() if residue else ''

            if resname in self.RECEPTOR_HYDROPHOBIC_CARBONS:
                hydrophobic.append(atom_name in self.RECEPTOR_HYDROPHOBIC_CARBONS[resname])
            else:
                # Non-standard residue / cofactor: exclude backbone carbonyl/CA
                hydrophobic.append(atom_name not in {'C', 'CA'})

        return hydrophobic

    @staticmethod
    def _is_amide_or_conjugated_nitrogen(atom: Chem.Atom) -> bool:
        """True if nitrogen's lone pair is delocalized into an amide/urea/sulfonamide."""
        for nbr in atom.GetNeighbors():
            sym = nbr.GetSymbol()
            if sym in ('C', 'S', 'P'):
                for b in nbr.GetBonds():
                    other = b.GetOtherAtom(nbr)
                    if other.GetIdx() == atom.GetIdx():
                        continue
                    if (
                        b.GetBondType() == Chem.BondType.DOUBLE
                        and other.GetSymbol() in ('O', 'S')
                    ):
                        return True
        return False

    def _get_hbond_atoms_ligand(self, mol: Chem.Mol) -> Tuple[List[int], List[int]]:
        """Get H-bond donor and acceptor heavy-atom indices in ligand.

        Follows standard medicinal-chemistry / Vina rules:
        - Donors: N or O carrying at least one hydrogen (including protonated N+).
        - Acceptors: O with formal charge <= 0; N with formal charge <= 0 that
          has an available lone pair (excludes quaternary/protonated N+,
          amide/urea/sulfonamide N, and aromatic pyrrole/indole NH).
        """
        donors = []
        acceptors = []

        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            symbol = atom.GetSymbol()
            charge = atom.GetFormalCharge()
            n_hs = atom.GetTotalNumHs(includeNeighbors=True)

            if symbol == 'O':
                if n_hs > 0:
                    donors.append(idx)
                if charge <= 0:
                    acceptors.append(idx)
            elif symbol == 'N':
                if n_hs > 0:
                    donors.append(idx)
                # Must have formal charge <= 0 (excludes NH3+, NR4+, nitro N+)
                # and an un-delocalized lone pair (excludes pyrrole/indole NH and amides)
                if (
                    charge <= 0
                    and not (atom.GetIsAromatic() and n_hs > 0)
                    and not self._is_amide_or_conjugated_nitrogen(atom)
                ):
                    acceptors.append(idx)

        return donors, acceptors

    def _get_hbond_atoms_receptor(self, receptor_atoms: List) -> Tuple[List[int], List[int]]:
        """Get H-bond donor and acceptor heavy-atom indices in receptor.

        Uses standard amino acid chemistry at pH 7.4:
        - Backbone N is a donor for all residues except PRO, and NEVER an acceptor.
        - Backbone O / OXT are acceptors for all residues.
        - Sidechain donors and acceptors follow RECEPTOR_SIDECHAIN_DONORS /
          RECEPTOR_SIDECHAIN_ACCEPTORS (so LYS NZ, ARG guanidinium, TRP NE1,
          and ASN/GLN amides are donors only, not acceptors).
        """
        donors = []
        acceptors = []

        for i, atom in enumerate(receptor_atoms):
            element = atom.element.strip().upper()
            atom_name = atom.get_name().strip().upper()
            residue = atom.get_parent()
            resname = residue.get_resname().strip().upper() if residue else ''

            if resname in self.STANDARD_RESIDUES:
                if atom_name == 'N':
                    if resname != 'PRO':
                        donors.append(i)
                elif atom_name in ('O', 'OXT'):
                    acceptors.append(i)
                else:
                    if atom_name in self.RECEPTOR_SIDECHAIN_DONORS.get(resname, ()):
                        donors.append(i)
                    if atom_name in self.RECEPTOR_SIDECHAIN_ACCEPTORS.get(resname, ()):
                        acceptors.append(i)
            elif resname in ('HOH', 'WAT', 'DOD'):
                if element == 'O':
                    donors.append(i)
                    acceptors.append(i)
            else:
                # Non-standard residue, cofactor, or catalytic metal ion
                if element in self.METAL_DONOR_ELEMENTS:
                    # AutoDock Vina XS_TYPE_Met_D: metal cations act as H-bond donors
                    # so ligand N/O/S acceptors experience attractive coordination.
                    donors.append(i)
                elif element == 'O':
                    acceptors.append(i)
                    if atom_name in ('OG', 'OG1', 'OH', 'O'):
                        donors.append(i)
                elif element == 'N':
                    if atom_name in ('N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ'):
                        donors.append(i)
                    if atom_name in ('ND1', 'NE2', 'NA'):
                        acceptors.append(i)

        return donors, acceptors

    def _count_rotatable_bonds(self, mol: Chem.Mol) -> int:
        """Count rotatable bonds for flexibility penalty"""
        try:
            return Lipinski.NumRotatableBonds(mol)
        except:
            return 0
