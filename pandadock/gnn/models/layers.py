"""
SE(3)-Equivariant Graph Neural Network Layers for PandaDock-GNN.

Implements E(n)-Equivariant Graph Neural Networks (EGNN) based on:
Satorras et al. "E(n) Equivariant Graph Neural Networks" (2021)

Key Properties:
- Rotation equivariant: Rotating inputs rotates outputs
- Translation invariant: Shifting coordinates doesn't change predictions
- Reflection equivariant (optional)

The EGNN updates both node features (invariant) and coordinates (equivariant).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union


class SiLU(nn.Module):
    """Sigmoid Linear Unit activation (also known as Swish)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class CoordinateEmbedding(nn.Module):
    """
    Embed pairwise distances using Gaussian radial basis functions.

    This creates SE(3)-invariant edge features from 3D coordinates.
    """

    def __init__(
        self,
        num_rbf: int = 16,
        d_min: float = 0.0,
        d_max: float = 10.0,
        trainable: bool = False
    ):
        super().__init__()
        self.num_rbf = num_rbf
        self.d_min = d_min
        self.d_max = d_max

        # Initialize RBF centers and widths
        centers = torch.linspace(d_min, d_max, num_rbf)
        widths = torch.ones(num_rbf) * (d_max - d_min) / (num_rbf - 1)

        if trainable:
            self.centers = nn.Parameter(centers)
            self.widths = nn.Parameter(widths)
        else:
            self.register_buffer('centers', centers)
            self.register_buffer('widths', widths)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Args:
            distances: (E,) pairwise distances

        Returns:
            (E, num_rbf) RBF embeddings
        """
        # Gaussian RBF: exp(-(d - c)^2 / (2 * w^2))
        d = distances.unsqueeze(-1)  # (E, 1)
        return torch.exp(-0.5 * ((d - self.centers) / self.widths) ** 2)


class EGNNLayer(nn.Module):
    """
    E(n)-Equivariant Graph Neural Network Layer.

    Updates node features h and coordinates x:
        m_ij = φ_e(h_i, h_j, ||x_i - x_j||², a_ij)
        x_i' = x_i + Σ_j (x_i - x_j) φ_x(m_ij)
        m_i = Σ_j m_ij
        h_i' = φ_h(h_i, m_i)

    where φ_e, φ_x, φ_h are MLPs.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int = 0,
        act_fn: str = 'silu',
        update_coords: bool = True,
        coords_weight: float = 1.0,
        attention: bool = True,
        normalize: bool = True,
        tanh: bool = True,
        dropout: float = 0.0
    ):
        """
        Args:
            hidden_dim: Dimension of node features
            edge_dim: Dimension of edge attributes (0 for none)
            act_fn: Activation function ('silu', 'relu', 'gelu')
            update_coords: Whether to update coordinates
            coords_weight: Scaling factor for coordinate updates
            attention: Use attention mechanism for messages
            normalize: Normalize coordinate updates
            tanh: Apply tanh to coordinate updates (for stability)
            dropout: Dropout probability
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.update_coords = update_coords
        self.coords_weight = coords_weight
        self.attention = attention
        self.normalize = normalize
        self.tanh = tanh

        # Activation
        if act_fn == 'silu':
            self.act = SiLU()
        elif act_fn == 'gelu':
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU()

        # Edge MLP: φ_e
        # Input: h_i (hidden) + h_j (hidden) + distance (1) + edge_attr (edge_dim)
        edge_input_dim = 2 * hidden_dim + 1 + edge_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            self.act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            self.act
        )

        # Coordinate MLP: φ_x (outputs scalar for each edge)
        if update_coords:
            self.coord_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                self.act,
                nn.Linear(hidden_dim, 1, bias=False)
            )
            # Initialize to small values for stability
            nn.init.xavier_uniform_(self.coord_mlp[-1].weight, gain=0.001)

        # Node MLP: φ_h
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            self.act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Attention (optional)
        if attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )

    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            h: Node features (N, hidden_dim)
            pos: Node coordinates (N, 3)
            edge_index: Edge indices (2, E)
            edge_attr: Edge attributes (E, edge_dim) or None

        Returns:
            Updated (h, pos)
        """
        src, dst = edge_index
        num_nodes = h.size(0)

        # Compute relative positions and distances
        rel_pos = pos[src] - pos[dst]  # (E, 3)
        dist_sq = (rel_pos ** 2).sum(dim=-1, keepdim=True)  # (E, 1)
        dist = torch.sqrt(dist_sq + 1e-8)  # (E, 1)

        # Build edge input
        edge_input = [h[src], h[dst], dist_sq]
        if edge_attr is not None:
            edge_input.append(edge_attr)
        edge_input = torch.cat(edge_input, dim=-1)

        # Edge messages
        m_ij = self.edge_mlp(edge_input)  # (E, hidden_dim)

        # Attention weighting (optional)
        if self.attention:
            att = self.att_mlp(m_ij)  # (E, 1)
            m_ij = m_ij * att

        # Update coordinates (equivariant)
        if self.update_coords:
            # Compute coordinate update weight
            coord_weight = self.coord_mlp(m_ij)  # (E, 1)

            if self.tanh:
                coord_weight = torch.tanh(coord_weight)

            # Normalize by distance (optional)
            if self.normalize:
                rel_pos_norm = rel_pos / (dist + 1e-8)
            else:
                rel_pos_norm = rel_pos

            # Weighted sum of relative positions
            coord_update = coord_weight * rel_pos_norm  # (E, 3)

            # Aggregate updates to each node
            pos_update = torch.zeros_like(pos)
            pos_update.scatter_add_(0, dst.unsqueeze(-1).expand_as(coord_update), coord_update.to(pos.dtype))

            # Update positions
            pos = pos + self.coords_weight * pos_update

        # Aggregate messages to nodes
        m_i = torch.zeros(num_nodes, self.hidden_dim, device=h.device, dtype=h.dtype)
        m_i.scatter_add_(0, dst.unsqueeze(-1).expand_as(m_ij), m_ij.to(h.dtype))

        # Update node features (invariant)
        h = h + self.node_mlp(torch.cat([h, m_i], dim=-1))

        return h, pos


class HeterogeneousEGNN(nn.Module):
    """
    Heterogeneous EGNN for protein-ligand graphs.

    Handles multiple node types (protein, ligand) and edge types
    (interactions, intramolecular) with separate parameters.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int = 0,
        num_layers: int = 6,
        act_fn: str = 'silu',
        update_coords: bool = True,
        dropout: float = 0.1,
        residual: bool = True
    ):
        """
        Args:
            hidden_dim: Dimension of node features
            edge_dim: Dimension of edge attributes
            num_layers: Number of EGNN layers
            act_fn: Activation function
            update_coords: Whether to update coordinates
            dropout: Dropout probability
            residual: Use residual connections between layers
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.residual = residual

        # EGNN layers for protein-ligand interactions
        self.interaction_layers = nn.ModuleList([
            EGNNLayer(
                hidden_dim=hidden_dim,
                edge_dim=edge_dim,
                act_fn=act_fn,
                update_coords=update_coords,
                attention=True,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Optional: separate layers for intramolecular edges
        self.intra_layers = nn.ModuleList([
            EGNNLayer(
                hidden_dim=hidden_dim,
                edge_dim=edge_dim,
                act_fn=act_fn,
                update_coords=False,  # Don't update coords for intra
                attention=False,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Layer normalization
        self.protein_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.ligand_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

    def forward(
        self,
        h_protein: torch.Tensor,
        h_ligand: torch.Tensor,
        pos_protein: torch.Tensor,
        pos_ligand: torch.Tensor,
        edge_index_inter: torch.Tensor,
        edge_attr_inter: Optional[torch.Tensor] = None,
        edge_index_intra_protein: Optional[torch.Tensor] = None,
        edge_attr_intra_protein: Optional[torch.Tensor] = None,
        edge_index_intra_ligand: Optional[torch.Tensor] = None,
        edge_attr_intra_ligand: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through heterogeneous EGNN.

        Args:
            h_protein: Protein node features (N_p, hidden_dim)
            h_ligand: Ligand node features (N_l, hidden_dim)
            pos_protein: Protein coordinates (N_p, 3)
            pos_ligand: Ligand coordinates (N_l, 3)
            edge_index_inter: Protein-ligand edges (2, E_inter)
            edge_attr_inter: Interaction edge features
            edge_index_intra_protein: Protein intra edges
            edge_attr_intra_protein: Protein intra edge features
            edge_index_intra_ligand: Ligand intra edges
            edge_attr_intra_ligand: Ligand intra edge features

        Returns:
            Updated (h_protein, h_ligand, pos_protein, pos_ligand)
        """
        n_protein = h_protein.size(0)
        n_ligand = h_ligand.size(0)

        for layer_idx in range(self.num_layers):
            # Store for residual
            h_protein_res = h_protein
            h_ligand_res = h_ligand

            # 1. Process intramolecular edges (optional)
            if edge_index_intra_protein is not None:
                h_protein, _ = self.intra_layers[layer_idx](
                    h_protein, pos_protein,
                    edge_index_intra_protein, edge_attr_intra_protein
                )

            if edge_index_intra_ligand is not None:
                h_ligand, _ = self.intra_layers[layer_idx](
                    h_ligand, pos_ligand,
                    edge_index_intra_ligand, edge_attr_intra_ligand
                )

            # 2. Process protein-ligand interactions
            # Combine into single graph for efficient processing
            h_combined = torch.cat([h_protein, h_ligand], dim=0)
            pos_combined = torch.cat([pos_protein, pos_ligand], dim=0)

            # Adjust edge indices for combined graph
            edge_index_combined = edge_index_inter.clone()
            edge_index_combined[1] = edge_index_combined[1] + n_protein

            # Apply EGNN layer
            h_combined, pos_combined = self.interaction_layers[layer_idx](
                h_combined, pos_combined,
                edge_index_combined, edge_attr_inter
            )

            # Split back
            h_protein = h_combined[:n_protein]
            h_ligand = h_combined[n_protein:]
            pos_protein = pos_combined[:n_protein]
            pos_ligand = pos_combined[n_protein:]

            # Residual connection
            if self.residual:
                h_protein = h_protein + h_protein_res
                h_ligand = h_ligand + h_ligand_res

            # Layer normalization
            h_protein = self.protein_norms[layer_idx](h_protein)
            h_ligand = self.ligand_norms[layer_idx](h_ligand)

        return h_protein, h_ligand, pos_protein, pos_ligand


class NodeEncoder(nn.Module):
    """
    Encode input node features to hidden dimension.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


if __name__ == "__main__":
    # Test EGNN layer
    print("Testing EGNN layer...")

    hidden_dim = 128
    num_nodes = 100
    num_edges = 500

    # Random inputs
    h = torch.randn(num_nodes, hidden_dim)
    pos = torch.randn(num_nodes, 3)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    # Create layer
    layer = EGNNLayer(hidden_dim=hidden_dim, edge_dim=0)

    # Forward pass
    h_out, pos_out = layer(h, pos, edge_index)

    print(f"Input h: {h.shape}, pos: {pos.shape}")
    print(f"Output h: {h_out.shape}, pos: {pos_out.shape}")

    # Test equivariance: rotating input should rotate output
    print("\nTesting rotation equivariance...")
    theta = torch.tensor(math.pi / 4)
    R = torch.tensor([
        [torch.cos(theta), -torch.sin(theta), 0],
        [torch.sin(theta), torch.cos(theta), 0],
        [0, 0, 1]
    ], dtype=torch.float32)

    pos_rotated = pos @ R.T
    h_out2, pos_out2 = layer(h, pos_rotated, edge_index)
    pos_out_rotated = pos_out @ R.T

    diff = (pos_out2 - pos_out_rotated).abs().mean()
    print(f"Rotation equivariance error: {diff:.6f}")
    print("PASS" if diff < 1e-4 else "FAIL")
