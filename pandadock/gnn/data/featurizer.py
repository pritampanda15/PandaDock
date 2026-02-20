"""
Feature extraction for PandaDock-GNN.

Generates node (atom) and edge (interaction) features for the heterogeneous
graph neural network. Features are designed to be SE(3)-invariant where appropriate.

Node Features (55 dimensions):
- Element type one-hot (10)
- SYBYL atom type embedding (16)
- Partial charge (1)
- Hybridization state (4)
- Aromaticity flag (1)
- H-bond donor/acceptor (2)
- Ring membership (1)
- Protein-specific: residue type (20)

Edge Features (21 dimensions):
- Distance (1)
- Gaussian RBF distance encoding (16)
- Bond type one-hot (4)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from .mol2_parser import ParsedMolecule, Atom, is_protein_residue


# Element encoding
ELEMENTS = ['C', 'N', 'O', 'S', 'P', 'H', 'F', 'CL', 'BR', 'OTHER']
ELEMENT_TO_IDX = {e: i for i, e in enumerate(ELEMENTS)}

# SYBYL atom types
SYBYL_TYPES = [
    'C.3', 'C.2', 'C.1', 'C.ar', 'C.cat',
    'N.3', 'N.2', 'N.1', 'N.ar', 'N.am', 'N.pl3', 'N.4',
    'O.3', 'O.2', 'O.co2',
    'S.3', 'S.2', 'S.O', 'S.O2',
    'P.3',
    'H',
    'F', 'Cl', 'Br', 'I',
    'OTHER'
]
SYBYL_TO_IDX = {t: i for i, t in enumerate(SYBYL_TYPES)}

# Hybridization mapping
HYBRIDIZATION_MAP = {
    '1': 0,   # sp
    '2': 1,   # sp2
    '3': 2,   # sp3
    'ar': 3,  # aromatic
}

# Amino acid encoding (20 standard + unknown)
AMINO_ACIDS = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'UNK'
]
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# Bond types
BOND_TYPES = ['1', '2', '3', 'ar', 'am', 'OTHER']
BOND_TO_IDX = {b: i for i, b in enumerate(BOND_TYPES)}


class AtomFeaturizer:
    """
    Generate node features for atoms.

    Features are designed to capture chemical properties relevant for
    protein-ligand binding:
    - Element identity (electronegativity, size)
    - Hybridization (geometry, reactivity)
    - Charge (electrostatic interactions)
    - H-bond capability (polar interactions)
    - Aromaticity (pi-stacking)
    """

    def __init__(
        self,
        include_charge: bool = True,
        include_residue: bool = True,
        normalize_charge: bool = True
    ):
        """
        Initialize atom featurizer.

        Args:
            include_charge: Include partial charge as feature
            include_residue: Include residue type for protein atoms
            normalize_charge: Normalize charges to [-1, 1] range
        """
        self.include_charge = include_charge
        self.include_residue = include_residue
        self.normalize_charge = normalize_charge

        # Calculate feature dimensions
        self.element_dim = len(ELEMENTS)  # 10
        self.sybyl_dim = len(SYBYL_TYPES)  # 26 -> use embedding
        self.hybridization_dim = 4
        self.hbond_dim = 2
        self.other_dim = 2  # aromaticity, ring
        self.charge_dim = 1 if include_charge else 0
        self.residue_dim = len(AMINO_ACIDS) if include_residue else 0

        self.feature_dim = (
            self.element_dim +
            16 +  # SYBYL embedding (compressed)
            self.hybridization_dim +
            self.hbond_dim +
            self.other_dim +
            self.charge_dim +
            self.residue_dim
        )

    def featurize_atom(self, atom: Atom, is_protein: bool = False) -> np.ndarray:
        """
        Generate features for a single atom.

        Args:
            atom: Atom object from parser
            is_protein: Whether this atom belongs to protein

        Returns:
            Feature vector of shape (feature_dim,)
        """
        features = []

        # 1. Element one-hot (10 dims)
        element = atom.element.upper()
        if element == 'CL':
            element = 'CL'
        elif element == 'BR':
            element = 'BR'
        elif element not in ELEMENT_TO_IDX:
            element = 'OTHER'
        element_vec = np.zeros(self.element_dim, dtype=np.float32)
        element_vec[ELEMENT_TO_IDX.get(element, ELEMENT_TO_IDX['OTHER'])] = 1.0
        features.append(element_vec)

        # 2. SYBYL type embedding (16 dims via hashing)
        sybyl_vec = self._sybyl_embedding(atom.atom_type)
        features.append(sybyl_vec)

        # 3. Hybridization one-hot (4 dims)
        hyb_vec = np.zeros(self.hybridization_dim, dtype=np.float32)
        hyb = atom.hybridization
        if hyb in HYBRIDIZATION_MAP:
            hyb_vec[HYBRIDIZATION_MAP[hyb]] = 1.0
        else:
            hyb_vec[2] = 1.0  # Default to sp3
        features.append(hyb_vec)

        # 4. H-bond donor/acceptor (2 dims)
        hbond_vec = np.zeros(self.hbond_dim, dtype=np.float32)
        # Donor: N-H, O-H patterns
        if atom.element.upper() in ['N', 'O'] and 'H' not in atom.name:
            hbond_vec[1] = 1.0  # Acceptor
        if atom.element.upper() == 'N' and atom.atom_type in ['N.3', 'N.4', 'N.am', 'N.pl3']:
            hbond_vec[0] = 1.0  # Donor
        if atom.element.upper() == 'O' and atom.atom_type == 'O.3':
            hbond_vec[0] = 1.0  # Donor (hydroxyl)
        features.append(hbond_vec)

        # 5. Aromaticity and ring (2 dims)
        other_vec = np.zeros(self.other_dim, dtype=np.float32)
        if 'ar' in atom.atom_type.lower():
            other_vec[0] = 1.0  # Aromatic
        # Ring membership approximated from atom type
        if atom.atom_type in ['C.ar', 'N.ar', 'C.2', 'N.2']:
            other_vec[1] = 1.0  # Likely in ring
        features.append(other_vec)

        # 6. Partial charge (1 dim)
        if self.include_charge:
            charge = atom.charge
            if self.normalize_charge:
                charge = np.clip(charge, -1.0, 1.0)
            features.append(np.array([charge], dtype=np.float32))

        # 7. Residue type (21 dims) - only for protein atoms
        if self.include_residue:
            res_vec = np.zeros(len(AMINO_ACIDS), dtype=np.float32)
            if is_protein:
                res_name = atom.residue_name[:3].upper()
                res_idx = AA_TO_IDX.get(res_name, AA_TO_IDX['UNK'])
                res_vec[res_idx] = 1.0
            features.append(res_vec)

        return np.concatenate(features)

    def _sybyl_embedding(self, sybyl_type: str, dim: int = 16) -> np.ndarray:
        """
        Create a compressed embedding for SYBYL atom type.

        Uses a simple hash-based encoding to map ~26 types to 16 dims.
        """
        embedding = np.zeros(dim, dtype=np.float32)

        # Get index
        idx = SYBYL_TO_IDX.get(sybyl_type, SYBYL_TO_IDX['OTHER'])

        # Simple encoding: spread across multiple dimensions
        for i in range(3):
            pos = (idx * (i + 1)) % dim
            embedding[pos] = 1.0 / (i + 1)

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def featurize_molecule(
        self,
        molecule: ParsedMolecule,
        is_protein: bool = False
    ) -> np.ndarray:
        """
        Generate features for all atoms in a molecule.

        Args:
            molecule: ParsedMolecule object
            is_protein: Whether this is a protein

        Returns:
            Feature matrix of shape (num_atoms, feature_dim)
        """
        if not molecule.atoms:
            return np.zeros((0, self.feature_dim), dtype=np.float32)

        features = [
            self.featurize_atom(atom, is_protein)
            for atom in molecule.atoms
        ]
        return np.stack(features, axis=0)


class EdgeFeaturizer:
    """
    Generate edge features for atom pairs.

    Features encode spatial and chemical relationship between atoms:
    - Distance (continuous and discretized via RBF)
    - Bond type (for covalent bonds)
    - Interaction type (for non-covalent)
    """

    def __init__(
        self,
        rbf_min: float = 0.0,
        rbf_max: float = 10.0,
        rbf_dim: int = 16,
        include_bond_type: bool = True
    ):
        """
        Initialize edge featurizer.

        Args:
            rbf_min: Minimum distance for RBF centers
            rbf_max: Maximum distance for RBF centers
            rbf_dim: Number of RBF basis functions
            include_bond_type: Include bond type encoding
        """
        self.rbf_min = rbf_min
        self.rbf_max = rbf_max
        self.rbf_dim = rbf_dim
        self.include_bond_type = include_bond_type

        # Pre-compute RBF centers
        self.rbf_centers = np.linspace(rbf_min, rbf_max, rbf_dim)
        self.rbf_width = (rbf_max - rbf_min) / (rbf_dim - 1) if rbf_dim > 1 else 1.0

        # Feature dimensions
        self.distance_dim = 1
        self.bond_dim = len(BOND_TYPES) if include_bond_type else 0
        self.feature_dim = self.distance_dim + self.rbf_dim + self.bond_dim

    def gaussian_rbf(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute Gaussian RBF encoding of distances.

        Args:
            distances: Array of distances (N,)

        Returns:
            RBF encoding of shape (N, rbf_dim)
        """
        # Expand dimensions for broadcasting
        d = distances[:, np.newaxis]  # (N, 1)
        c = self.rbf_centers[np.newaxis, :]  # (1, rbf_dim)

        # Gaussian RBF: exp(-(d - c)^2 / (2 * sigma^2))
        rbf = np.exp(-0.5 * ((d - c) / self.rbf_width) ** 2)

        return rbf.astype(np.float32)

    def featurize_edges(
        self,
        coords1: np.ndarray,
        coords2: np.ndarray,
        edge_index: np.ndarray,
        bond_types: Optional[Dict[Tuple[int, int], str]] = None
    ) -> np.ndarray:
        """
        Generate features for edges.

        Args:
            coords1: Coordinates of source atoms (N1, 3)
            coords2: Coordinates of target atoms (N2, 3)
            edge_index: Edge indices (2, E), first row is source, second is target
            bond_types: Optional dict mapping (src, dst) to bond type

        Returns:
            Edge features of shape (E, feature_dim)
        """
        num_edges = edge_index.shape[1]
        if num_edges == 0:
            return np.zeros((0, self.feature_dim), dtype=np.float32)

        # Get source and target coordinates
        src_coords = coords1[edge_index[0]]  # (E, 3)
        dst_coords = coords2[edge_index[1]]  # (E, 3)

        # Compute distances
        distances = np.linalg.norm(src_coords - dst_coords, axis=1)  # (E,)

        features = []

        # 1. Raw distance (1 dim)
        features.append(distances[:, np.newaxis])

        # 2. RBF encoding (rbf_dim dims)
        rbf_features = self.gaussian_rbf(distances)
        features.append(rbf_features)

        # 3. Bond type (bond_dim dims)
        if self.include_bond_type:
            bond_features = np.zeros((num_edges, len(BOND_TYPES)), dtype=np.float32)
            if bond_types:
                for i in range(num_edges):
                    src, dst = edge_index[0, i], edge_index[1, i]
                    bt = bond_types.get((src, dst), bond_types.get((dst, src), None))
                    if bt:
                        idx = BOND_TO_IDX.get(bt, BOND_TO_IDX['OTHER'])
                        bond_features[i, idx] = 1.0
            features.append(bond_features)

        return np.concatenate(features, axis=1)

    def compute_pairwise_features(
        self,
        coords1: np.ndarray,
        coords2: np.ndarray,
        cutoff: float = 5.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute features for all pairs within cutoff distance.

        Args:
            coords1: Source coordinates (N1, 3)
            coords2: Target coordinates (N2, 3)
            cutoff: Distance cutoff

        Returns:
            edge_index: (2, E) indices of edges
            edge_features: (E, feature_dim) features
        """
        # Compute pairwise distances
        diff = coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :]  # (N1, N2, 3)
        dists = np.linalg.norm(diff, axis=2)  # (N1, N2)

        # Find pairs within cutoff
        src, dst = np.where(dists < cutoff)
        edge_index = np.stack([src, dst], axis=0)

        # Compute features
        edge_features = self.featurize_edges(coords1, coords2, edge_index)

        return edge_index, edge_features


@dataclass
class FeaturizationConfig:
    """Configuration for featurization."""
    # Atom features
    include_charge: bool = True
    include_residue: bool = True
    normalize_charge: bool = True

    # Edge features
    rbf_min: float = 0.0
    rbf_max: float = 10.0
    rbf_dim: int = 16
    include_bond_type: bool = True

    # Graph construction
    interaction_cutoff: float = 5.0
    intramolecular_cutoff: float = 3.0


def create_featurizers(
    config: Optional[FeaturizationConfig] = None
) -> Tuple[AtomFeaturizer, EdgeFeaturizer]:
    """
    Create atom and edge featurizers with given config.

    Args:
        config: Featurization configuration

    Returns:
        Tuple of (AtomFeaturizer, EdgeFeaturizer)
    """
    if config is None:
        config = FeaturizationConfig()

    atom_featurizer = AtomFeaturizer(
        include_charge=config.include_charge,
        include_residue=config.include_residue,
        normalize_charge=config.normalize_charge
    )

    edge_featurizer = EdgeFeaturizer(
        rbf_min=config.rbf_min,
        rbf_max=config.rbf_max,
        rbf_dim=config.rbf_dim,
        include_bond_type=config.include_bond_type
    )

    return atom_featurizer, edge_featurizer


if __name__ == "__main__":
    # Test featurizers
    from .mol2_parser import parse_mol2
    import sys

    if len(sys.argv) > 1:
        mol = parse_mol2(sys.argv[1])

        atom_feat = AtomFeaturizer()
        edge_feat = EdgeFeaturizer()

        node_features = atom_feat.featurize_molecule(mol)
        print(f"Node features shape: {node_features.shape}")
        print(f"Node feature dim: {atom_feat.feature_dim}")

        # Test edge features
        coords = mol.coordinates
        edge_index, edge_features = edge_feat.compute_pairwise_features(
            coords, coords, cutoff=5.0
        )
        print(f"Edge index shape: {edge_index.shape}")
        print(f"Edge features shape: {edge_features.shape}")
