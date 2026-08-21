"""
PDBbind Dataset Loader for PandaDock GNN

Loads protein-ligand complexes from PDBbind v2020 refined set.
"""

import os
import math
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np

# torch is an optional extra (`pip install pandadock[gnn]`), and this module is
# reached from `pandadock.gnn.data.__init__`, so importing it unguarded makes the
# whole subpackage unimportable in a base install rather than only the parts that
# need a tensor library.
try:
    import torch
    from torch.utils.data import Dataset
    from torch_geometric.data import HeteroData
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    Dataset = object
    HeteroData = None


@dataclass
class PDBbindCompound:
    """Information about a PDBbind complex."""
    pdb_id: str
    protein_pdb: str
    pocket_pdb: str
    ligand_mol2: str
    ligand_sdf: str
    pkd: float  # -log10(Kd) or -log10(Ki)
    resolution: float
    year: int


class PDBbindDataset(Dataset):
    """
    PDBbind dataset for training GNN scoring function.

    Uses:
    - pocket.pdb as the binding site (like site.mol2 in ULVSH)
    - ligand.mol2 for ligand structure
    - INDEX_refined_data.2020 for experimental pKd/pKi values
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform=None,
        min_pkd: float = 2.0,
        max_pkd: float = 12.0,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42
    ):
        """
        Args:
            root: Path to PDBbind folder containing PDBbind_v2020/
            split: 'train', 'val', 'test', or 'all'
            transform: Optional transform to apply
            min_pkd: Minimum pKd value to include
            max_pkd: Maximum pKd value to include
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            seed: Random seed for splitting
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.min_pkd = min_pkd
        self.max_pkd = max_pkd

        # Find PDBbind directory
        if (self.root / 'PDBbind_v2020').exists():
            self.pdbbind_dir = self.root / 'PDBbind_v2020'
        else:
            self.pdbbind_dir = self.root

        # Load compounds
        self.compounds = self._load_compounds()

        # Split dataset
        self.indices = self._create_split(train_ratio, val_ratio, seed)

        # Build graph builder
        self.graph_builder = None
        self._init_graph_builder()

        print(f"Loaded {len(self.compounds)} PDBbind compounds")
        print(f"Split '{split}': {len(self.indices)} compounds")

    def _load_compounds(self) -> List[PDBbindCompound]:
        """Load compound information from index file."""
        compounds = []

        # Parse index file
        index_file = self.pdbbind_dir / 'index' / 'INDEX_refined_data.2020'
        if not index_file.exists():
            index_file = self.pdbbind_dir / 'INDEX_refined_data.2020'

        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found: {index_file}")

        with open(index_file, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue

                parts = line.split()
                if len(parts) < 4:
                    continue

                pdb_id = parts[0]
                try:
                    resolution = float(parts[1])
                    year = int(parts[2])
                    pkd = float(parts[3])
                except ValueError:
                    continue

                # Filter by pKd range
                if pkd < self.min_pkd or pkd > self.max_pkd:
                    continue

                # Check files exist
                pdb_dir = self.pdbbind_dir / pdb_id
                if not pdb_dir.exists():
                    continue

                protein_pdb = pdb_dir / f"{pdb_id}_protein.pdb"
                pocket_pdb = pdb_dir / f"{pdb_id}_pocket.pdb"
                ligand_mol2 = pdb_dir / f"{pdb_id}_ligand.mol2"
                ligand_sdf = pdb_dir / f"{pdb_id}_ligand.sdf"

                if not all(f.exists() for f in [protein_pdb, pocket_pdb, ligand_mol2]):
                    continue

                compounds.append(PDBbindCompound(
                    pdb_id=pdb_id,
                    protein_pdb=str(protein_pdb),
                    pocket_pdb=str(pocket_pdb),
                    ligand_mol2=str(ligand_mol2),
                    ligand_sdf=str(ligand_sdf) if ligand_sdf.exists() else str(ligand_mol2),
                    pkd=pkd,
                    resolution=resolution,
                    year=year
                ))

        return compounds

    def _create_split(self, train_ratio: float, val_ratio: float, seed: int) -> List[int]:
        """Create train/val/test split."""
        n = len(self.compounds)
        indices = list(range(n))

        random.seed(seed)
        random.shuffle(indices)

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        if self.split == 'train':
            return indices[:train_end]
        elif self.split == 'val':
            return indices[train_end:val_end]
        elif self.split == 'test':
            return indices[val_end:]
        else:  # 'all'
            return indices

    def _init_graph_builder(self):
        """Initialize graph builder."""
        try:
            from .graph_builder import HeterogeneousGraphBuilder
            self.graph_builder = HeterogeneousGraphBuilder()
        except ImportError:
            pass

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        """Get a single graph."""
        compound = self.compounds[self.indices[idx]]

        if self.graph_builder is None:
            self._init_graph_builder()

        # Build graph using pocket (binding site) and ligand
        graph = self.graph_builder.build_from_files(
            protein_file=compound.pocket_pdb,  # Use pocket as binding site
            ligand_file=compound.ligand_mol2
        )

        # Add target values
        graph.y_affinity = torch.tensor([compound.pkd], dtype=torch.float32)
        graph.y_active = torch.tensor([float(compound.pkd >= 6.0)], dtype=torch.float32)
        graph.y_activity = graph.y_active  # Alias for BindingDB compatibility

        # Add metadata (for compatibility with ULVSH and BindingDB datasets)
        graph.compound_id = compound.pdb_id
        graph.complex_id = compound.pdb_id  # Alias for BindingDB compatibility
        graph.target = 'pdbbind'

        if self.transform:
            graph = self.transform(graph)

        return graph

    def get_compound_info(self, idx: int) -> PDBbindCompound:
        """Get compound information."""
        return self.compounds[self.indices[idx]]

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        pkd_values = [self.compounds[i].pkd for i in self.indices]
        return {
            'n_compounds': len(self.indices),
            'pkd_mean': np.mean(pkd_values),
            'pkd_std': np.std(pkd_values),
            'pkd_min': np.min(pkd_values),
            'pkd_max': np.max(pkd_values)
        }


class CombinedDataset(Dataset):
    """
    Combined dataset from multiple sources (ULVSH + PDBbind).

    Normalizes affinity values across datasets.
    """

    def __init__(
        self,
        datasets: List[Dataset],
        normalize: bool = True
    ):
        """
        Args:
            datasets: List of datasets to combine
            normalize: Whether to normalize affinity values
        """
        self.datasets = datasets
        self.normalize = normalize

        # Build index mapping
        self.index_map = []  # (dataset_idx, sample_idx)
        for ds_idx, ds in enumerate(datasets):
            for sample_idx in range(len(ds)):
                self.index_map.append((ds_idx, sample_idx))

        # Compute normalization parameters if needed
        if normalize:
            self._compute_normalization()

        print(f"Combined dataset: {len(self)} total samples from {len(datasets)} datasets")

    def _compute_normalization(self):
        """Compute mean and std for affinity normalization."""
        all_affinities = []

        for ds in self.datasets:
            if hasattr(ds, 'compounds') and hasattr(ds, 'indices'):
                # ULVSH or PDBbind style
                for idx in ds.indices:
                    compound = ds.compounds[idx]
                    if hasattr(compound, 'pec50'):
                        all_affinities.append(compound.pec50)
                    elif hasattr(compound, 'pkd'):
                        all_affinities.append(compound.pkd)

        if all_affinities:
            self.affinity_mean = np.mean(all_affinities)
            self.affinity_std = np.std(all_affinities)
        else:
            self.affinity_mean = 0.0
            self.affinity_std = 1.0

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int):
        ds_idx, sample_idx = self.index_map[idx]
        graph = self.datasets[ds_idx][sample_idx]

        # Normalize affinity if needed
        if self.normalize and hasattr(graph, 'y_affinity'):
            # Don't modify in place
            graph = graph.clone()
            # Keep original for evaluation
            graph.y_affinity_original = graph.y_affinity.clone()
            # Normalize
            graph.y_affinity = (graph.y_affinity - self.affinity_mean) / self.affinity_std

        return graph

    def get_normalization_params(self) -> Tuple[float, float]:
        """Return normalization parameters."""
        return self.affinity_mean, self.affinity_std


def create_combined_dataset(
    ulvsh_root: str = None,
    pdbbind_root: str = None,
    split: str = 'train',
    **kwargs
) -> Dataset:
    """
    Create a combined dataset from ULVSH and/or PDBbind.

    Args:
        ulvsh_root: Path to ULVSH dataset (optional)
        pdbbind_root: Path to PDBbind dataset (optional)
        split: 'train', 'val', 'test', or 'all'
        **kwargs: Additional arguments for datasets

    Returns:
        Combined dataset
    """
    datasets = []

    if ulvsh_root:
        from .dataset import ULVSHDataset
        ulvsh = ULVSHDataset(root=ulvsh_root, split=split, **kwargs)
        datasets.append(ulvsh)
        print(f"Added ULVSH: {len(ulvsh)} compounds")

    if pdbbind_root:
        pdbbind = PDBbindDataset(root=pdbbind_root, split=split, **kwargs)
        datasets.append(pdbbind)
        print(f"Added PDBbind: {len(pdbbind)} compounds")

    if len(datasets) == 0:
        raise ValueError("Must provide at least one dataset")
    elif len(datasets) == 1:
        return datasets[0]
    else:
        return CombinedDataset(datasets, normalize=True)
