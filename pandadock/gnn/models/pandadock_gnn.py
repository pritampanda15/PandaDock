"""
PandaDock-GNN: SE(3)-Equivariant Graph Neural Network for Molecular Docking.

Main model that combines:
- Node encoders for protein and ligand atoms
- SE(3)-equivariant message passing (EGNN)
- Attention-based hierarchical pooling
- Multi-task prediction heads (affinity + activity)

This is the core model for binding affinity prediction.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass
from pathlib import Path

from .layers import EGNNLayer, HeterogeneousEGNN, NodeEncoder
from .pooling import AttentionPooling, InteractionPooling, HierarchicalPooling


@dataclass
class ModelConfig:
    """Configuration for PandaDockGNN model."""
    # Node features (from featurizer)
    protein_input_dim: int = 56
    ligand_input_dim: int = 56
    edge_input_dim: int = 23

    # Hidden dimensions
    hidden_dim: int = 256
    num_layers: int = 6

    # Pooling
    num_heads: int = 8
    pool_type: str = 'hierarchical'  # 'hierarchical', 'attention', 'mean_max'

    # Prediction heads
    affinity_hidden: int = 256
    num_affinity_layers: int = 2

    # Regularization
    dropout: float = 0.1

    # Equivariance
    update_coords: bool = True

    # Multi-task
    predict_activity: bool = True
    predict_confidence: bool = False


class AffinityHead(nn.Module):
    """
    Prediction head for binding affinity (pEC50).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        layers = []
        dim = input_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout)
            ])
            dim = hidden_dim

        layers.append(nn.Linear(dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ActivityHead(nn.Module):
    """
    Prediction head for activity classification (active/inactive).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PandaDockGNN(nn.Module):
    """
    SE(3)-Equivariant GNN for Protein-Ligand Binding Affinity Prediction.

    Architecture:
    1. Node encoding (separate for protein and ligand)
    2. Heterogeneous EGNN message passing
    3. Hierarchical attention pooling
    4. Multi-task prediction heads

    Example:
        model = PandaDockGNN()
        predictions = model(hetero_data)
        affinity = predictions['affinity']
        activity = predictions['activity']
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize PandaDockGNN.

        Args:
            config: Model configuration
        """
        super().__init__()

        self.config = config or ModelConfig()

        # Node encoders
        self.protein_encoder = NodeEncoder(
            self.config.protein_input_dim,
            self.config.hidden_dim,
            self.config.dropout
        )
        self.ligand_encoder = NodeEncoder(
            self.config.ligand_input_dim,
            self.config.hidden_dim,
            self.config.dropout
        )

        # Edge encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(self.config.edge_input_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        )

        # Heterogeneous EGNN
        self.egnn = HeterogeneousEGNN(
            hidden_dim=self.config.hidden_dim,
            edge_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            update_coords=self.config.update_coords,
            dropout=self.config.dropout
        )

        # Pooling
        if self.config.pool_type == 'hierarchical':
            self.pooling = HierarchicalPooling(
                self.config.hidden_dim,
                self.config.num_heads,
                self.config.dropout
            )
            pool_output_dim = 3 * self.config.hidden_dim
        else:
            self.pooling = AttentionPooling(
                self.config.hidden_dim,
                self.config.num_heads,
                self.config.dropout
            )
            pool_output_dim = 2 * self.config.hidden_dim  # protein + ligand

        # Prediction heads
        self.affinity_head = AffinityHead(
            pool_output_dim,
            self.config.affinity_hidden,
            self.config.num_affinity_layers,
            self.config.dropout
        )

        if self.config.predict_activity:
            self.activity_head = ActivityHead(
                pool_output_dim,
                self.config.affinity_hidden // 2,
                self.config.dropout
            )

        if self.config.predict_confidence:
            self.confidence_head = ActivityHead(
                pool_output_dim,
                self.config.affinity_hidden // 2,
                self.config.dropout
            )

        # Store attention weights for interpretability
        self._last_attention: Optional[Dict[str, Any]] = None

    def forward(self, data) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            data: PyG HeteroData with:
                - data['protein'].x: Protein node features
                - data['protein'].pos: Protein coordinates
                - data['ligand'].x: Ligand node features
                - data['ligand'].pos: Ligand coordinates
                - data['protein', 'interacts', 'ligand'].edge_index
                - data['protein', 'interacts', 'ligand'].edge_attr

        Returns:
            Dictionary with:
                - 'affinity': Predicted pEC50 (B,)
                - 'activity': Predicted activity logits (B,) if enabled
                - 'confidence': Predicted confidence (B,) if enabled
        """
        # Extract node features and positions
        h_protein = data['protein'].x
        pos_protein = data['protein'].pos
        h_ligand = data['ligand'].x
        pos_ligand = data['ligand'].pos

        # Get batch indices if batched
        batch_protein = getattr(data['protein'], 'batch', None)
        batch_ligand = getattr(data['ligand'], 'batch', None)

        # Get interaction edges
        edge_index_inter = data['protein', 'interacts', 'ligand'].edge_index
        edge_attr_inter = data['protein', 'interacts', 'ligand'].edge_attr

        # Get intramolecular edges (if available)
        edge_index_intra_protein = None
        edge_attr_intra_protein = None
        edge_index_intra_ligand = None
        edge_attr_intra_ligand = None

        if ('protein', 'intra', 'protein') in data.edge_types:
            edge_index_intra_protein = data['protein', 'intra', 'protein'].edge_index
            edge_attr_intra_protein = data['protein', 'intra', 'protein'].edge_attr

        if ('ligand', 'intra', 'ligand') in data.edge_types:
            edge_index_intra_ligand = data['ligand', 'intra', 'ligand'].edge_index
            edge_attr_intra_ligand = data['ligand', 'intra', 'ligand'].edge_attr

        # 1. Encode nodes
        h_protein = self.protein_encoder(h_protein)
        h_ligand = self.ligand_encoder(h_ligand)

        # 2. Encode edges
        edge_attr_inter = self.edge_encoder(edge_attr_inter)

        if edge_attr_intra_protein is not None:
            edge_attr_intra_protein = self.edge_encoder(edge_attr_intra_protein)
        if edge_attr_intra_ligand is not None:
            edge_attr_intra_ligand = self.edge_encoder(edge_attr_intra_ligand)

        # 3. Message passing
        h_protein, h_ligand, pos_protein, pos_ligand = self.egnn(
            h_protein, h_ligand,
            pos_protein, pos_ligand,
            edge_index_inter, edge_attr_inter,
            edge_index_intra_protein, edge_attr_intra_protein,
            edge_index_intra_ligand, edge_attr_intra_ligand
        )

        # 4. Pooling
        if self.config.pool_type == 'hierarchical':
            g = self.pooling(
                h_protein, h_ligand, edge_index_inter,
                batch_protein, batch_ligand
            )
        else:
            # Simple: pool protein and ligand separately
            g_protein = self.pooling(h_protein, batch_protein)
            g_ligand = self.pooling(h_ligand, batch_ligand)
            g = torch.cat([g_protein, g_ligand], dim=-1)

        # 5. Predictions
        predictions = {}

        predictions['affinity'] = self.affinity_head(g)

        if self.config.predict_activity:
            predictions['activity'] = self.activity_head(g)

        if self.config.predict_confidence:
            predictions['confidence'] = self.confidence_head(g)

        return predictions

    def get_attention_weights(self) -> Optional[Dict[str, Any]]:
        """
        Get attention weights from pooling layers.

        Returns:
            Dictionary with attention weights for protein, ligand, and interactions
        """
        if not hasattr(self.pooling, 'protein_pool'):
            return None

        return {
            'protein': self.pooling.protein_pool.get_attention_weights(),
            'ligand': self.pooling.ligand_pool.get_attention_weights(),
            'interaction': self.pooling.interaction_pool.get_edge_attention()
        }

    def save(self, path: str) -> None:
        """Save model weights and config."""
        path = Path(path)
        torch.save({
            'config': self.config,
            'state_dict': self.state_dict()
        }, path)

    @classmethod
    def load(cls, path: str, map_location: str = 'cpu') -> 'PandaDockGNN':
        """Load model from checkpoint.

        Handles both model.save() format and trainer checkpoint format.
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)

        # Get config - handle both formats
        config = checkpoint.get('config')

        # Get state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            raise ValueError("Checkpoint missing state_dict")

        # Check if config is actually ModelConfig or TrainingConfig
        if config is None or not hasattr(config, 'protein_input_dim'):
            # Infer config from state dict shapes
            config = cls._infer_config_from_state_dict(state_dict)
            print(f"Inferred model config: hidden_dim={config.hidden_dim}, num_layers={config.num_layers}")

        model = cls(config=config)
        model.load_state_dict(state_dict)

        return model

    @staticmethod
    def _infer_config_from_state_dict(state_dict: dict) -> 'ModelConfig':
        """Infer ModelConfig from state dict shapes."""
        # Infer hidden_dim from encoder output
        if 'protein_encoder.encoder.2.weight' in state_dict:
            hidden_dim = state_dict['protein_encoder.encoder.2.weight'].shape[0]
        elif 'protein_encoder.encoder.0.weight' in state_dict:
            hidden_dim = state_dict['protein_encoder.encoder.0.weight'].shape[0]
        else:
            hidden_dim = 256  # default

        # Count EGNN layers
        num_layers = 0
        for key in state_dict.keys():
            if 'egnn.intra_layers' in key and 'edge_mlp.0.weight' in key:
                layer_idx = int(key.split('.')[2])
                num_layers = max(num_layers, layer_idx + 1)

        if num_layers == 0:
            num_layers = 6  # default

        # Check if activity head exists
        predict_activity = 'activity_head.net.0.weight' in state_dict

        return ModelConfig(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            predict_activity=predict_activity
        )

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(
    hidden_dim: int = 256,
    num_layers: int = 6,
    num_heads: int = 8,
    dropout: float = 0.1,
    **kwargs
) -> PandaDockGNN:
    """
    Factory function to create PandaDockGNN model.

    Args:
        hidden_dim: Hidden dimension
        num_layers: Number of EGNN layers
        num_heads: Number of attention heads
        dropout: Dropout probability
        **kwargs: Additional config options

    Returns:
        PandaDockGNN model
    """
    config = ModelConfig(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        **kwargs
    )
    return PandaDockGNN(config)


if __name__ == "__main__":
    # Test model
    print("Testing PandaDockGNN...")

    try:
        from torch_geometric.data import HeteroData

        # Create dummy data
        data = HeteroData()

        n_protein = 100
        n_ligand = 30
        n_edges = 200

        hidden_dim = 256
        node_dim = 56
        edge_dim = 21

        # Protein nodes
        data['protein'].x = torch.randn(n_protein, node_dim)
        data['protein'].pos = torch.randn(n_protein, 3)

        # Ligand nodes
        data['ligand'].x = torch.randn(n_ligand, node_dim)
        data['ligand'].pos = torch.randn(n_ligand, 3)

        # Interaction edges
        data['protein', 'interacts', 'ligand'].edge_index = torch.stack([
            torch.randint(0, n_protein, (n_edges,)),
            torch.randint(0, n_ligand, (n_edges,))
        ])
        data['protein', 'interacts', 'ligand'].edge_attr = torch.randn(n_edges, edge_dim)

        # Reverse edges
        data['ligand', 'interacts', 'protein'].edge_index = data['protein', 'interacts', 'ligand'].edge_index.flip(0)
        data['ligand', 'interacts', 'protein'].edge_attr = data['protein', 'interacts', 'ligand'].edge_attr

        # Create model
        config = ModelConfig(
            protein_input_dim=node_dim,
            ligand_input_dim=node_dim,
            edge_input_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=4,
            predict_activity=True
        )
        model = PandaDockGNN(config)

        print(f"Model parameters: {model.count_parameters():,}")

        # Forward pass
        model.eval()
        with torch.no_grad():
            predictions = model(data)

        print(f"Affinity prediction: {predictions['affinity'].shape}")
        print(f"Activity prediction: {predictions['activity'].shape}")

        # Test save/load
        model.save("/tmp/test_model.pt")
        loaded_model = PandaDockGNN.load("/tmp/test_model.pt")
        print("Save/load test passed!")

        print("\nAll tests passed!")

    except ImportError as e:
        print(f"PyTorch Geometric not installed: {e}")
        print("Install with: pip install torch-geometric")
