"""
Heterogeneous graph construction for PandaDock-GNN.

Builds PyTorch Geometric HeteroData graphs from protein-ligand complexes.
The graph has two node types (protein, ligand) and multiple edge types
representing different interaction patterns.

Graph Structure:
- Node types: 'protein', 'ligand'
- Edge types:
  - ('protein', 'interacts', 'ligand'): protein -> ligand interactions
  - ('ligand', 'interacts', 'protein'): ligand -> protein interactions
  - ('protein', 'intra', 'protein'): intramolecular protein bonds
  - ('ligand', 'intra', 'ligand'): intramolecular ligand bonds
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, replace
from pathlib import Path
import logging

try:
    import torch
    from torch_geometric.data import HeteroData
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    HeteroData = None

from .mol2_parser import MOL2Parser, ParsedMolecule, Atom, Bond
from .featurizer import AtomFeaturizer, EdgeFeaturizer, FeaturizationConfig

logger = logging.getLogger("pandadock.gnn.data.graph_builder")



# Residue-aware SYBYL atom typing for standard amino acids.
#
# Mapping every element through a flat element->sp3 dictionary types every
# backbone carbonyl as C.3 rather than C.2, every aromatic ring carbon as C.3
# rather than C.ar, and every amide nitrogen as N.3 rather than N.am. The
# featurizer embeds the SYBYL type into 16 of its node features, so a receptor
# parsed from PDB presented the model with protein features quite unlike the
# MOL2 inputs it was trained on.
PROTEIN_BACKBONE_SYBYL = {
    "N": "N.am", "CA": "C.3", "C": "C.2", "O": "O.2", "OXT": "O.co2",
}

PROTEIN_SIDECHAIN_SYBYL = {
    "ARG": {"CB": "C.3", "CG": "C.3", "CD": "C.3", "NE": "N.pl3",
            "CZ": "C.cat", "NH1": "N.pl3", "NH2": "N.pl3"},
    "ASN": {"CB": "C.3", "CG": "C.2", "OD1": "O.2", "ND2": "N.am"},
    "ASP": {"CB": "C.3", "CG": "C.2", "OD1": "O.co2", "OD2": "O.co2"},
    "CYS": {"CB": "C.3", "SG": "S.3"},
    "GLN": {"CB": "C.3", "CG": "C.3", "CD": "C.2", "OE1": "O.2", "NE2": "N.am"},
    "GLU": {"CB": "C.3", "CG": "C.3", "CD": "C.2", "OE1": "O.co2", "OE2": "O.co2"},
    "HIS": {"CB": "C.3", "CG": "C.ar", "ND1": "N.ar", "CD2": "C.ar",
            "CE1": "C.ar", "NE2": "N.ar"},
    "ILE": {"CB": "C.3", "CG1": "C.3", "CG2": "C.3", "CD1": "C.3"},
    "LEU": {"CB": "C.3", "CG": "C.3", "CD1": "C.3", "CD2": "C.3"},
    "LYS": {"CB": "C.3", "CG": "C.3", "CD": "C.3", "CE": "C.3", "NZ": "N.4"},
    "MET": {"CB": "C.3", "CG": "C.3", "SD": "S.3", "CE": "C.3"},
    "PHE": {"CB": "C.3", "CG": "C.ar", "CD1": "C.ar", "CD2": "C.ar",
            "CE1": "C.ar", "CE2": "C.ar", "CZ": "C.ar"},
    "PRO": {"CB": "C.3", "CG": "C.3", "CD": "C.3"},
    "SER": {"CB": "C.3", "OG": "O.3"},
    "THR": {"CB": "C.3", "OG1": "O.3", "CG2": "C.3"},
    "TRP": {"CB": "C.3", "CG": "C.ar", "CD1": "C.ar", "CD2": "C.ar",
            "NE1": "N.ar", "CE2": "C.ar", "CE3": "C.ar", "CZ2": "C.ar",
            "CZ3": "C.ar", "CH2": "C.ar"},
    "TYR": {"CB": "C.3", "CG": "C.ar", "CD1": "C.ar", "CD2": "C.ar",
            "CE1": "C.ar", "CE2": "C.ar", "CZ": "C.ar", "OH": "O.3"},
    "VAL": {"CB": "C.3", "CG1": "C.3", "CG2": "C.3"},
    "ALA": {"CB": "C.3"},
}


def protein_sybyl_type(residue_name: str, atom_name: str, element: str) -> str:
    """SYBYL type for a protein atom, falling back to the sp3 form."""
    atom_name = (atom_name or "").strip()
    residue_name = (residue_name or "").strip().upper()

    sidechain = PROTEIN_SIDECHAIN_SYBYL.get(residue_name)
    if sidechain and atom_name in sidechain:
        return sidechain[atom_name]
    if atom_name in PROTEIN_BACKBONE_SYBYL:
        return PROTEIN_BACKBONE_SYBYL[atom_name]

    element = (element or "C").strip().upper()
    return {"C": "C.3", "N": "N.3", "O": "O.3", "S": "S.3", "P": "P.3",
            "H": "H", "F": "F", "CL": "Cl", "BR": "Br", "I": "I"}.get(
        element, f"{element.capitalize()}.3"
    )


def parse_pdb_file(pdb_path: str) -> ParsedMolecule:
    """
    Parse a PDB file into ParsedMolecule format.

    Args:
        pdb_path: Path to PDB file

    Returns:
        ParsedMolecule object
    """
    try:
        from Bio.PDB import PDBParser
    except ImportError:
        raise ImportError("BioPython required for PDB parsing. Install with: pip install biopython")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)

    atoms = []
    coords = []
    atom_id = 0

    # Element to SYBYL type mapping
    element_to_sybyl = {
        'C': 'C.3', 'N': 'N.3', 'O': 'O.3', 'S': 'S.3',
        'P': 'P.3', 'H': 'H', 'F': 'F', 'CL': 'Cl',
        'BR': 'Br', 'I': 'I', 'CA': 'Ca', 'MG': 'Mg',
        'ZN': 'Zn', 'FE': 'Fe', 'MN': 'Mn', 'CU': 'Cu'
    }

    for model in structure:
        for chain in model:
            for residue in chain:
                res_name = residue.get_resname()
                res_id = residue.get_id()[1]

                for atom in residue:
                    atom_id += 1
                    element = atom.element.upper() if atom.element else 'C'
                    sybyl_type = protein_sybyl_type(
                        residue.get_resname(), atom.get_name(), element
                    )

                    # Check for aromatic carbons in specific residues
                    if element == 'C' and res_name in ['PHE', 'TYR', 'TRP', 'HIS']:
                        if atom.name in ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'CH2', 'CZ2', 'CZ3', 'CE3']:
                            sybyl_type = 'C.ar'

                    atoms.append(Atom(
                        id=atom_id,
                        name=atom.name,
                        x=atom.coord[0],
                        y=atom.coord[1],
                        z=atom.coord[2],
                        atom_type=sybyl_type,
                        residue_id=res_id,
                        residue_name=res_name,
                        charge=0.0
                    ))
                    coords.append(atom.coord)

    mol = ParsedMolecule(
        name=Path(pdb_path).stem,
        atoms=atoms,
        bonds=[]  # PDB doesn't have explicit bonds
    )
    mol.num_atoms = len(atoms)
    return mol


def parse_sdf_file(sdf_path: str) -> ParsedMolecule:
    """
    Parse an SDF file into ParsedMolecule format.

    Args:
        sdf_path: Path to SDF file

    Returns:
        ParsedMolecule object
    """
    try:
        from rdkit import Chem
    except ImportError:
        raise ImportError("RDKit required for SDF parsing. Install via conda: conda install -c conda-forge rdkit")

    # Load molecule
    mol = Chem.SDMolSupplier(sdf_path, removeHs=False)[0]
    if mol is None:
        mol = Chem.MolFromMolFile(sdf_path, removeHs=False)

    if mol is None:
        raise ValueError(f"Could not parse SDF file: {sdf_path}")

    # Add hydrogens if needed
    mol = Chem.AddHs(mol, addCoords=True)

    # Get conformer
    if mol.GetNumConformers() == 0:
        from rdkit.Chem import AllChem
        AllChem.EmbedMolecule(mol, randomSeed=42)

    conf = mol.GetConformer()

    # Hybridization to SYBYL type mapping
    hybridization_to_sybyl = {
        Chem.HybridizationType.SP: '.1',
        Chem.HybridizationType.SP2: '.2',
        Chem.HybridizationType.SP3: '.3',
        Chem.HybridizationType.SP3D: '.3',
        Chem.HybridizationType.SP3D2: '.3',
    }

    atoms = []
    coords = []

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        element = atom.GetSymbol()
        pos = conf.GetAtomPosition(idx)

        # Determine SYBYL type
        hybridization = atom.GetHybridization()
        suffix = hybridization_to_sybyl.get(hybridization, '.3')

        if atom.GetIsAromatic():
            sybyl_type = f'{element}.ar'
        elif element == 'N' and atom.GetTotalNumHs() >= 2:
            sybyl_type = 'N.am'
        else:
            sybyl_type = f'{element}{suffix}'

        # Get partial charge if available
        try:
            charge = float(atom.GetProp('_GasteigerCharge'))
            if np.isnan(charge):
                charge = 0.0
        except:
            charge = 0.0

        atoms.append(Atom(
            id=idx + 1,
            name=f'{element}{idx + 1}',
            x=pos.x,
            y=pos.y,
            z=pos.z,
            atom_type=sybyl_type,
            residue_id=1,
            residue_name='LIG',
            charge=charge
        ))
        coords.append([pos.x, pos.y, pos.z])

    # Get bonds
    bond_type_map = {
        Chem.BondType.SINGLE: '1',
        Chem.BondType.DOUBLE: '2',
        Chem.BondType.TRIPLE: '3',
        Chem.BondType.AROMATIC: 'ar',
    }

    bonds = []
    for bond in mol.GetBonds():
        bt = bond_type_map.get(bond.GetBondType(), '1')
        bonds.append(Bond(
            id=bond.GetIdx() + 1,
            atom1_id=bond.GetBeginAtomIdx() + 1,
            atom2_id=bond.GetEndAtomIdx() + 1,
            bond_type=bt
        ))

    parsed_mol = ParsedMolecule(
        name=mol.GetProp('_Name') if mol.HasProp('_Name') else Path(sdf_path).stem,
        atoms=atoms,
        bonds=bonds
    )
    parsed_mol.num_atoms = len(atoms)
    parsed_mol.num_bonds = len(bonds)
    return parsed_mol


def parse_molecule_file(file_path: str) -> ParsedMolecule:
    """
    Parse a molecule file (MOL2, PDB, or SDF) into ParsedMolecule format.

    Args:
        file_path: Path to molecule file

    Returns:
        ParsedMolecule object
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == '.mol2':
        parser = MOL2Parser()
        return parser.parse(file_path)
    elif suffix == '.pdb':
        return parse_pdb_file(file_path)
    elif suffix in ['.sdf', '.mol']:
        return parse_sdf_file(file_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Supported: .mol2, .pdb, .sdf, .mol")



def extract_binding_site(protein: ParsedMolecule, centroid, radius: float = 10.0) -> ParsedMolecule:
    """
    Protein atoms within `radius` of a ligand centroid.

    The GNN is applied to a binding site, not a whole protein: its training data
    ships a precomputed site.mol2 per complex. Passing an entire receptor builds
    a graph one to two orders of magnitude larger and presents the model with
    something unlike anything it was trained on, as well as exhausting memory
    over a batch.
    """
    import numpy as _np

    centroid = _np.asarray(centroid, dtype=float)
    radius_sq = float(radius) ** 2

    site_atoms = [
        atom for atom in protein.atoms
        if (atom.x - centroid[0]) ** 2
        + (atom.y - centroid[1]) ** 2
        + (atom.z - centroid[2]) ** 2 <= radius_sq
    ]
    if not site_atoms:
        return protein

    site = ParsedMolecule(name=f"{protein.name}_site", atoms=site_atoms, bonds=[])
    site.num_atoms = len(site_atoms)
    return site


def drop_hydrogens(molecule: "ParsedMolecule") -> "ParsedMolecule":
    """
    Return the molecule without hydrogens, renumbering atoms.

    Bonds are dropped rather than remapped: the graph builder derives
    connectivity from distance, so it never reads them, and silently keeping
    bond indices that point at removed atoms would be worse than having none.
    """
    kept = [a for a in molecule.atoms if a.element.upper() not in ("H", "D")]
    if len(kept) == len(molecule.atoms):
        return molecule

    renumbered = []
    for i, atom in enumerate(kept):
        clone = replace(atom, id=i + 1)
        renumbered.append(clone)

    trimmed = ParsedMolecule(name=molecule.name, atoms=renumbered, bonds=[])
    trimmed.num_atoms = len(renumbered)
    return trimmed


@dataclass
class GraphConfig:
    """Configuration for graph construction."""
    # Distance cutoffs
    interaction_cutoff: float = 5.0  # Protein-ligand interactions
    intramolecular_cutoff: float = 2.5  # Covalent bonds

    # Options
    include_intramolecular: bool = True
    bidirectional_interactions: bool = True
    use_site_only: bool = True  # Use binding site instead of full protein
    # Radius of the sphere cut around the ligand centroid when a site has to be
    # derived from a whole protein. Must match whatever the model was trained
    # on; SAIR models were trained at 10 A.
    site_radius: float = 10.0
    # Drop hydrogens before building the graph. SAIR CIFs carry none, so models
    # trained on them are heavy-atom models and must be given heavy-atom input.
    # ULVSH and PDBbind models were trained with hydrogens present.
    strip_hydrogens: bool = False

    # Featurization
    featurization: Optional[FeaturizationConfig] = None


class HeterogeneousGraphBuilder:
    """
    Build heterogeneous graphs for protein-ligand complexes.

    Creates PyTorch Geometric HeteroData objects with:
    - Protein and ligand node features
    - Interaction edge features
    - Atomic coordinates (for equivariant networks)

    Example:
        builder = HeterogeneousGraphBuilder()
        graph = builder.build_from_files(
            protein_mol2="protein.mol2",
            ligand_mol2="ligand.mol2"
        )
        print(graph['protein'].x.shape)  # Node features
        print(graph['protein'].pos.shape)  # Coordinates
    """

    def __init__(self, config: Optional[GraphConfig] = None):
        """
        Initialize graph builder.

        Args:
            config: Graph construction configuration
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch and PyTorch Geometric required. "
                "Install with: pip install torch torch-geometric"
            )

        self.config = config or GraphConfig()
        feat_config = self.config.featurization or FeaturizationConfig()

        self.parser = MOL2Parser()
        self.atom_featurizer = AtomFeaturizer(
            include_charge=feat_config.include_charge,
            include_residue=feat_config.include_residue,
            normalize_charge=feat_config.normalize_charge
        )
        self.edge_featurizer = EdgeFeaturizer(
            rbf_min=feat_config.rbf_min,
            rbf_max=feat_config.rbf_max,
            rbf_dim=feat_config.rbf_dim,
            include_bond_type=feat_config.include_bond_type
        )

    def build_from_files(
        self,
        protein_file: str,
        ligand_file: str,
        site_file: Optional[str] = None
    ) -> HeteroData:
        """
        Build graph from molecule files (MOL2, PDB, or SDF).

        Args:
            protein_file: Path to protein file (MOL2 or PDB)
            ligand_file: Path to ligand file (MOL2, SDF, or PDB)
            site_file: Optional path to binding site file

        Returns:
            HeteroData graph
        """
        # Parse ligand (supports MOL2, SDF, PDB)
        ligand = parse_molecule_file(ligand_file)

        # Use site if provided and configured
        if site_file and self.config.use_site_only and Path(site_file).exists():
            protein = parse_molecule_file(site_file)
        else:
            protein = parse_molecule_file(protein_file)
            if self.config.use_site_only:
                # Without this the model receives an entire protein -- thousands
                # of atoms -- where it was trained on a few hundred around the
                # ligand. It returns numbers either way, which is what makes the
                # mismatch worth cutting here rather than leaving to the caller.
                centroid = np.array(
                    [[a.x, a.y, a.z] for a in ligand.atoms], dtype=float
                ).mean(axis=0)
                protein = extract_binding_site(
                    protein, centroid, radius=self.config.site_radius
                )

        if self.config.strip_hydrogens:
            protein = drop_hydrogens(protein)
            ligand = drop_hydrogens(ligand)

        return self.build_graph(protein, ligand)

    def build_graph(
        self,
        protein: ParsedMolecule,
        ligand: ParsedMolecule
    ) -> HeteroData:
        """
        Build heterogeneous graph from parsed molecules.

        Args:
            protein: Parsed protein/site molecule
            ligand: Parsed ligand molecule

        Returns:
            HeteroData graph with node features, positions, and edges
        """
        data = HeteroData()

        # 1. Add protein nodes
        protein_features = self.atom_featurizer.featurize_molecule(protein, is_protein=True)
        protein_coords = protein.coordinates

        data['protein'].x = torch.tensor(protein_features, dtype=torch.float32)
        data['protein'].pos = torch.tensor(protein_coords, dtype=torch.float32)
        data['protein'].num_nodes = len(protein.atoms)

        # 2. Add ligand nodes
        ligand_features = self.atom_featurizer.featurize_molecule(ligand, is_protein=False)
        ligand_coords = ligand.coordinates

        data['ligand'].x = torch.tensor(ligand_features, dtype=torch.float32)
        data['ligand'].pos = torch.tensor(ligand_coords, dtype=torch.float32)
        data['ligand'].num_nodes = len(ligand.atoms)

        # 3. Add protein-ligand interaction edges
        self._add_interaction_edges(data, protein_coords, ligand_coords)

        # 4. Add intramolecular edges (optional)
        if self.config.include_intramolecular:
            self._add_intramolecular_edges(data, protein, 'protein')
            self._add_intramolecular_edges(data, ligand, 'ligand')

        return data

    def _add_interaction_edges(
        self,
        data: HeteroData,
        protein_coords: np.ndarray,
        ligand_coords: np.ndarray
    ) -> None:
        """Add protein-ligand interaction edges based on distance cutoff."""
        cutoff = self.config.interaction_cutoff

        # Compute pairwise distances
        diff = protein_coords[:, np.newaxis, :] - ligand_coords[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)  # (N_protein, N_ligand)

        # Find pairs within cutoff
        protein_idx, ligand_idx = np.where(dists < cutoff)

        if len(protein_idx) == 0:
            # No interactions found, add dummy edge to avoid empty graph
            protein_idx = np.array([0])
            ligand_idx = np.array([0])

        # Create edge index
        edge_index = np.stack([protein_idx, ligand_idx], axis=0)

        # Compute edge features
        edge_features = self.edge_featurizer.featurize_edges(
            protein_coords, ligand_coords, edge_index
        )

        # Add forward edges: protein -> ligand
        data['protein', 'interacts', 'ligand'].edge_index = torch.tensor(
            edge_index, dtype=torch.long
        )
        data['protein', 'interacts', 'ligand'].edge_attr = torch.tensor(
            edge_features, dtype=torch.float32
        )

        # Add reverse edges: ligand -> protein
        if self.config.bidirectional_interactions:
            reverse_index = np.stack([ligand_idx, protein_idx], axis=0)
            data['ligand', 'interacts', 'protein'].edge_index = torch.tensor(
                reverse_index, dtype=torch.long
            )
            data['ligand', 'interacts', 'protein'].edge_attr = torch.tensor(
                edge_features, dtype=torch.float32
            )

    def _add_intramolecular_edges(
        self,
        data: HeteroData,
        molecule: ParsedMolecule,
        node_type: str
    ) -> None:
        """Add intramolecular (covalent) edges."""
        if not molecule.bonds:
            # Use distance-based if no bonds defined
            coords = molecule.coordinates
            if len(coords) == 0:
                return

            cutoff = self.config.intramolecular_cutoff
            diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)

            # Find pairs within cutoff (excluding self-loops)
            src, dst = np.where((dists < cutoff) & (dists > 0.01))
            edge_index = np.stack([src, dst], axis=0)
        else:
            # Use bond information
            bond_edges = []
            for bond in molecule.bonds:
                # Convert to 0-indexed
                src = bond.atom1_id - 1
                dst = bond.atom2_id - 1
                bond_edges.append([src, dst])
                bond_edges.append([dst, src])  # Bidirectional

            if not bond_edges:
                return

            edge_index = np.array(bond_edges, dtype=np.int64).T

        # Compute edge features
        coords = molecule.coordinates
        edge_features = self.edge_featurizer.featurize_edges(
            coords, coords, edge_index
        )

        # Add to graph
        data[node_type, 'intra', node_type].edge_index = torch.tensor(
            edge_index, dtype=torch.long
        )
        data[node_type, 'intra', node_type].edge_attr = torch.tensor(
            edge_features, dtype=torch.float32
        )

    def build_batch(
        self,
        complexes: List[Tuple[str, str, Optional[str]]]
    ) -> List[HeteroData]:
        """
        Build graphs for multiple complexes.

        Args:
            complexes: List of (protein_mol2, ligand_mol2, site_mol2) tuples

        Returns:
            List of HeteroData graphs
        """
        graphs = []
        for protein_path, ligand_path, site_path in complexes:
            try:
                graph = self.build_from_files(protein_path, ligand_path, site_path)
                graphs.append(graph)
            except Exception as e:
                print(f"Warning: Failed to build graph for {ligand_path}: {e}")
                continue
        return graphs


def collate_hetero_graphs(graphs: List[HeteroData]) -> HeteroData:
    """
    Collate multiple HeteroData graphs into a batch.

    Handles string metadata (compound_id, target) by collecting them as lists
    and excluding them from tensor batching.
    """
    from torch_geometric.data import Batch

    # Metadata keys that should not be batched as tensors
    metadata_keys = ['compound_id', 'target']

    # Collect metadata before batching
    metadata = {key: [] for key in metadata_keys}
    for graph in graphs:
        for key in metadata_keys:
            if hasattr(graph, key):
                metadata[key].append(getattr(graph, key))
                delattr(graph, key)

    # Batch the graphs (excluding metadata)
    batch = Batch.from_data_list(graphs)

    # Add metadata back as lists
    for key, values in metadata.items():
        if values:
            setattr(batch, key, values)

    return batch


class ComplexData:
    """
    Container for protein-ligand complex data.

    Stores graph along with metadata and labels.
    """

    def __init__(
        self,
        graph: HeteroData,
        compound_id: str,
        target: str,
        affinity: Optional[float] = None,
        active: Optional[bool] = None
    ):
        self.graph = graph
        self.compound_id = compound_id
        self.target = target
        self.affinity = affinity
        self.active = active

    def to_hetero_data(self) -> HeteroData:
        """Return HeteroData with labels attached."""
        data = self.graph

        if self.affinity is not None:
            data.y_affinity = torch.tensor([self.affinity], dtype=torch.float32)

        if self.active is not None:
            data.y_active = torch.tensor([float(self.active)], dtype=torch.float32)

        return data


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        protein_file = sys.argv[1]
        ligand_file = sys.argv[2]
        site_file = sys.argv[3] if len(sys.argv) > 3 else None

        builder = HeterogeneousGraphBuilder()
        graph = builder.build_from_files(protein_file, ligand_file, site_file)

        print("Graph structure:")
        print(f"  Protein nodes: {graph['protein'].num_nodes}")
        print(f"  Protein features: {graph['protein'].x.shape}")
        print(f"  Ligand nodes: {graph['ligand'].num_nodes}")
        print(f"  Ligand features: {graph['ligand'].x.shape}")
        print(f"  Interaction edges: {graph['protein', 'interacts', 'ligand'].edge_index.shape}")

        if ('protein', 'intra', 'protein') in graph.edge_types:
            print(f"  Protein intra edges: {graph['protein', 'intra', 'protein'].edge_index.shape}")
        if ('ligand', 'intra', 'ligand') in graph.edge_types:
            print(f"  Ligand intra edges: {graph['ligand', 'intra', 'ligand'].edge_index.shape}")
