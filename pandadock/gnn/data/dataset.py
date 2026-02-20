"""
ULVSH Dataset loader for PandaDock-GNN.

Loads the ULVSH (Ultra-Large Virtual Screening Hits) dataset containing
protein-ligand complexes with experimental binding affinity data.

Dataset Structure:
    ULVSH/
    ├── TARGET1/
    │   ├── raw/
    │   │   ├── vitro.tsv          # Experimental data
    │   │   ├── scores.tsv         # Computed scores
    │   │   └── ...
    │   └── minimized/
    │       ├── ZINC_ID1/
    │       │   ├── protein.mol2
    │       │   ├── ligand.mol2
    │       │   └── site.mol2
    │       └── ...
    └── TARGET2/
        └── ...
"""

import os
import math
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torch_geometric.data import HeteroData
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object
    HeteroData = None

from .graph_builder import HeterogeneousGraphBuilder, GraphConfig


@dataclass
class CompoundInfo:
    """Information about a single compound."""
    compound_id: str
    target: str
    ec50_um: float
    active: bool
    pec50: float  # -log10(EC50 in M)
    protein_mol2: str
    ligand_mol2: str
    site_mol2: str

    # Optional: pre-computed scores
    scores: Optional[Dict[str, float]] = None


def ec50_to_pec50(ec50_um: float) -> float:
    """
    Convert EC50 (micromolar) to pEC50 (-log10 molar).

    Args:
        ec50_um: EC50 in micromolar

    Returns:
        pEC50 value
    """
    if ec50_um <= 0:
        return 0.0
    # Convert μM to M: multiply by 1e-6
    ec50_m = ec50_um * 1e-6
    return -math.log10(ec50_m)


class ULVSHDataset(Dataset):
    """
    PyTorch Dataset for ULVSH protein-ligand complexes.

    Loads protein-ligand graphs with experimental binding affinity labels.
    Supports train/val/test splits and cross-target evaluation.

    Example:
        dataset = ULVSHDataset(
            root="/path/to/ULVSH",
            split="train",
            split_ratio=(0.8, 0.1, 0.1)
        )
        graph = dataset[0]
        print(graph.y_affinity)  # pEC50 value
    """

    # List of all targets in ULVSH
    TARGETS = [
        'ADRA2B', 'CASR', 'CNR1', 'CNR2', 'DRD3',
        'DRD4', 'MTR1A', 'ROCK1', 'SC6A4', 'SGMR2'
    ]

    def __init__(
        self,
        root: str,
        split: str = 'train',
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        split_type: str = 'random',  # 'random' or 'target'
        targets: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        graph_config: Optional[GraphConfig] = None,
        cache_graphs: bool = True,
        seed: int = 42
    ):
        """
        Initialize ULVSH dataset.

        Args:
            root: Path to ULVSH dataset root
            split: 'train', 'val', or 'test'
            split_ratio: Ratio for train/val/test split
            split_type: 'random' for random split, 'target' for leave-target-out
            targets: List of targets to include (None = all)
            transform: Transform to apply to each graph
            pre_transform: Transform to apply during preprocessing
            graph_config: Configuration for graph construction
            cache_graphs: Cache constructed graphs in memory
            seed: Random seed for splitting
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. Install with: pip install torch")

        self.root = Path(root)
        self.split = split
        self.split_ratio = split_ratio
        self.split_type = split_type
        self.targets = targets or self.TARGETS
        self.transform = transform
        self.pre_transform = pre_transform
        self.cache_graphs = cache_graphs
        self.seed = seed

        self.graph_builder = HeterogeneousGraphBuilder(graph_config)

        # Load compound information
        self.compounds = self._load_compounds()

        # Split data
        self._split_data()

        # Cache for graphs
        self._cache: Dict[int, HeteroData] = {}

    def _load_compounds(self) -> List[CompoundInfo]:
        """Load all compound information from dataset."""
        compounds = []

        for target in self.targets:
            target_dir = self.root / target

            if not target_dir.exists():
                print(f"Warning: Target directory not found: {target_dir}")
                continue

            # Load experimental data
            vitro_file = target_dir / "raw" / "vitro.tsv"
            if not vitro_file.exists():
                print(f"Warning: vitro.tsv not found for {target}")
                continue

            # Try tab separator first, then whitespace
            try:
                vitro_df = pd.read_csv(vitro_file, sep='\t')
                if len(vitro_df.columns) == 1:
                    # Tab separator didn't work, try whitespace
                    vitro_df = pd.read_csv(vitro_file, sep=r'\s+')
            except Exception:
                vitro_df = pd.read_csv(vitro_file, sep=r'\s+')

            # Load pre-computed scores (optional)
            scores_file = target_dir / "raw" / "scores.tsv"
            scores_df = None
            if scores_file.exists():
                try:
                    scores_df = pd.read_csv(scores_file, sep='\t')
                    scores_df = scores_df.set_index('ID')
                except Exception:
                    pass

            # Iterate over compounds
            minimized_dir = target_dir / "minimized"

            # Find the EC50 column (may have different names)
            ec50_col = None
            for col in vitro_df.columns:
                if 'EC50' in col or 'ec50' in col:
                    ec50_col = col
                    break

            for _, row in vitro_df.iterrows():
                compound_id = row['ID']

                # Handle EC50 value (may be 'n.d.' for not determined)
                try:
                    ec50 = float(row[ec50_col]) if ec50_col else 100.0
                except (ValueError, TypeError):
                    ec50 = 100.0  # Default to inactive threshold

                active = str(row['Active']).lower() == 'yes'

                # Check for minimized files
                compound_dir = minimized_dir / compound_id

                if not compound_dir.exists():
                    continue

                protein_mol2 = compound_dir / "protein.mol2"
                ligand_mol2 = compound_dir / "ligand.mol2"
                site_mol2 = compound_dir / "site.mol2"

                if not ligand_mol2.exists():
                    continue

                # Get pre-computed scores
                scores = None
                if scores_df is not None and compound_id in scores_df.index:
                    scores = scores_df.loc[compound_id].to_dict()

                compound = CompoundInfo(
                    compound_id=compound_id,
                    target=target,
                    ec50_um=ec50,
                    active=active,
                    pec50=ec50_to_pec50(ec50),
                    protein_mol2=str(protein_mol2),
                    ligand_mol2=str(ligand_mol2),
                    site_mol2=str(site_mol2),
                    scores=scores
                )
                compounds.append(compound)

        print(f"Loaded {len(compounds)} compounds from {len(self.targets)} targets")
        return compounds

    def _split_data(self) -> None:
        """Split data into train/val/test sets."""
        random.seed(self.seed)
        np.random.seed(self.seed)

        if self.split_type == 'target':
            # Leave-target-out split
            self._target_split()
        else:
            # Random split
            self._random_split()

    def _random_split(self) -> None:
        """Perform random train/val/test split."""
        indices = list(range(len(self.compounds)))
        random.shuffle(indices)

        n_total = len(indices)
        n_train = int(n_total * self.split_ratio[0])
        n_val = int(n_total * self.split_ratio[1])

        if self.split == 'train':
            self.indices = indices[:n_train]
        elif self.split == 'val':
            self.indices = indices[n_train:n_train + n_val]
        elif self.split == 'test':
            self.indices = indices[n_train + n_val:]
        else:
            self.indices = indices

        print(f"Split '{self.split}': {len(self.indices)} compounds")

    def _target_split(self) -> None:
        """Split by target for cross-target evaluation."""
        # Group compounds by target
        target_to_compounds = {}
        for i, compound in enumerate(self.compounds):
            if compound.target not in target_to_compounds:
                target_to_compounds[compound.target] = []
            target_to_compounds[compound.target].append(i)

        targets = list(target_to_compounds.keys())
        random.shuffle(targets)

        n_targets = len(targets)
        n_train = int(n_targets * self.split_ratio[0])
        n_val = int(n_targets * self.split_ratio[1])

        if self.split == 'train':
            split_targets = targets[:n_train]
        elif self.split == 'val':
            split_targets = targets[n_train:n_train + n_val]
        elif self.split == 'test':
            split_targets = targets[n_train + n_val:]
        else:
            split_targets = targets

        self.indices = []
        for target in split_targets:
            self.indices.extend(target_to_compounds[target])

        print(f"Split '{self.split}': {len(self.indices)} compounds from targets: {split_targets}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> HeteroData:
        """Get a single graph."""
        real_idx = self.indices[idx]

        # Check cache
        if self.cache_graphs and real_idx in self._cache:
            data = self._cache[real_idx]
        else:
            compound = self.compounds[real_idx]

            # Build graph
            data = self.graph_builder.build_from_files(
                protein_file=compound.protein_mol2,
                ligand_file=compound.ligand_mol2,
                site_file=compound.site_mol2
            )

            # Add labels
            data.y_affinity = torch.tensor([compound.pec50], dtype=torch.float32)
            data.y_active = torch.tensor([float(compound.active)], dtype=torch.float32)

            # Add metadata
            data.compound_id = compound.compound_id
            data.target = compound.target

            # Apply pre_transform
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            # Cache
            if self.cache_graphs:
                self._cache[real_idx] = data

        # Apply transform
        if self.transform is not None:
            data = self.transform(data)

        return data

    def get_compound_info(self, idx: int) -> CompoundInfo:
        """Get compound information by index."""
        return self.compounds[self.indices[idx]]

    def get_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0
    ) -> DataLoader:
        """
        Create a DataLoader for this dataset.

        Args:
            batch_size: Batch size
            shuffle: Shuffle data
            num_workers: Number of worker processes

        Returns:
            DataLoader
        """
        from torch_geometric.loader import DataLoader as PyGDataLoader

        return PyGDataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )

    @property
    def num_node_features(self) -> int:
        """Number of node features."""
        return self.graph_builder.atom_featurizer.feature_dim

    @property
    def num_edge_features(self) -> int:
        """Number of edge features."""
        return self.graph_builder.edge_featurizer.feature_dim

    def get_statistics(self) -> Dict[str, float]:
        """Compute dataset statistics."""
        pec50_values = [self.compounds[i].pec50 for i in self.indices]
        active_count = sum(1 for i in self.indices if self.compounds[i].active)

        return {
            'num_compounds': len(self.indices),
            'num_active': active_count,
            'num_inactive': len(self.indices) - active_count,
            'active_ratio': active_count / len(self.indices) if self.indices else 0,
            'pec50_mean': np.mean(pec50_values),
            'pec50_std': np.std(pec50_values),
            'pec50_min': np.min(pec50_values),
            'pec50_max': np.max(pec50_values),
        }


def create_dataloaders(
    root: str,
    batch_size: int = 32,
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    split_type: str = 'random',
    num_workers: int = 0,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        root: Path to ULVSH dataset
        batch_size: Batch size
        split_ratio: Train/val/test split ratio
        split_type: 'random' or 'target'
        num_workers: Number of dataloader workers
        seed: Random seed

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = ULVSHDataset(
        root=root,
        split='train',
        split_ratio=split_ratio,
        split_type=split_type,
        seed=seed
    )

    val_dataset = ULVSHDataset(
        root=root,
        split='val',
        split_ratio=split_ratio,
        split_type=split_type,
        seed=seed
    )

    test_dataset = ULVSHDataset(
        root=root,
        split='test',
        split_ratio=split_ratio,
        split_type=split_type,
        seed=seed
    )

    train_loader = train_dataset.get_dataloader(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = val_dataset.get_dataloader(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = test_dataset.get_dataloader(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        root = sys.argv[1]

        print("Loading ULVSH dataset...")
        dataset = ULVSHDataset(root=root, split='train')

        print(f"\nDataset statistics:")
        stats = dataset.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

        print(f"\nNode features: {dataset.num_node_features}")
        print(f"Edge features: {dataset.num_edge_features}")

        print("\nLoading first graph...")
        graph = dataset[0]
        print(f"  Protein nodes: {graph['protein'].num_nodes}")
        print(f"  Ligand nodes: {graph['ligand'].num_nodes}")
        print(f"  pEC50: {graph.y_affinity.item():.2f}")
        print(f"  Active: {bool(graph.y_active.item())}")
