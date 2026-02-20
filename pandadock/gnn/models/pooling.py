"""
Attention-based Graph Pooling for PandaDock-GNN.

Implements attention mechanisms to aggregate node features into graph-level
representations. Attention weights provide interpretability by highlighting
important atoms and interactions.

Pooling Types:
- AttentionPooling: Attention-weighted sum of node features
- InteractionPooling: Pool over protein-ligand interaction edges
- SetTransformerPooling: Multi-head attention pooling (more expressive)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class AttentionPooling(nn.Module):
    """
    Attention-based graph pooling.

    Computes attention scores for each node and returns weighted sum.
    Attention weights can be used for interpretability.

    Pooling: g = Σ_i α_i * h_i, where α = softmax(score(h_i))
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            hidden_dim: Dimension of node features
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Attention scoring network
        self.score_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_heads)
        )

        # Value projection
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

        # Store attention weights for interpretability
        self._attention_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Pool node features to graph-level representation.

        Args:
            x: Node features (N, hidden_dim)
            batch: Batch assignment (N,) for batched graphs, None for single graph
            return_attention: Whether to return attention weights

        Returns:
            Graph representation (B, hidden_dim) or (1, hidden_dim) if no batch
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Compute attention scores
        scores = self.score_net(x)  # (N, num_heads)

        # Apply softmax per graph per head
        # Create attention mask for batched softmax
        num_graphs = batch.max().item() + 1

        # Compute values
        values = self.value_proj(x)  # (N, hidden_dim)
        values = values.view(-1, self.num_heads, self.head_dim)  # (N, num_heads, head_dim)

        # Pool per graph
        pooled = []
        attention_weights = []

        for g in range(num_graphs):
            mask = (batch == g)
            g_scores = scores[mask]  # (n_g, num_heads)
            g_values = values[mask]  # (n_g, num_heads, head_dim)

            # Softmax over nodes for each head
            g_attention = F.softmax(g_scores, dim=0)  # (n_g, num_heads)
            g_attention = self.dropout(g_attention)

            # Weighted sum: (num_heads, head_dim)
            g_pooled = torch.einsum('nh,nhd->hd', g_attention, g_values)
            g_pooled = g_pooled.view(-1)  # (hidden_dim,)

            pooled.append(g_pooled)
            attention_weights.append(g_attention.mean(dim=1))  # Average over heads

        pooled = torch.stack(pooled, dim=0)  # (B, hidden_dim)
        pooled = self.out_proj(pooled)

        # Store attention for interpretability
        self._attention_weights = attention_weights

        if return_attention:
            return pooled, attention_weights

        return pooled

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Return last computed attention weights."""
        return self._attention_weights


class InteractionPooling(nn.Module):
    """
    Pool over protein-ligand interaction edges.

    Captures pairwise interactions between protein and ligand atoms
    with attention-weighted aggregation. Useful for identifying
    binding hotspots.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            hidden_dim: Dimension of node features
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Edge attention network
        # Input: protein_h + ligand_h
        self.attention_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_heads)
        )

        # Edge value network
        self.value_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.dropout = nn.Dropout(dropout)

        # Store attention for interpretability
        self._edge_attention: Optional[torch.Tensor] = None

    def forward(
        self,
        h_protein: torch.Tensor,
        h_ligand: torch.Tensor,
        edge_index: torch.Tensor,
        batch_protein: Optional[torch.Tensor] = None,
        batch_ligand: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool over interaction edges.

        Args:
            h_protein: Protein node features (N_p, hidden_dim)
            h_ligand: Ligand node features (N_l, hidden_dim)
            edge_index: Protein-ligand edges (2, E), [protein_idx, ligand_idx]
            batch_protein: Batch assignment for protein nodes
            batch_ligand: Batch assignment for ligand nodes

        Returns:
            Interaction representation (B, hidden_dim)
        """
        src, dst = edge_index  # protein -> ligand

        # Get edge features
        edge_h = torch.cat([h_protein[src], h_ligand[dst]], dim=-1)  # (E, 2*hidden_dim)

        # Compute attention scores and values
        attention_logits = self.attention_net(edge_h)  # (E, num_heads)
        values = self.value_net(edge_h)  # (E, hidden_dim)

        # Determine batch for each edge
        if batch_protein is not None:
            edge_batch = batch_protein[src]
            num_graphs = batch_protein.max().item() + 1
        else:
            edge_batch = torch.zeros(edge_index.size(1), dtype=torch.long, device=h_protein.device)
            num_graphs = 1

        # Pool per graph with attention
        pooled = []
        edge_attentions = []

        for g in range(num_graphs):
            mask = (edge_batch == g)
            if mask.sum() == 0:
                # No edges for this graph
                pooled.append(torch.zeros(self.hidden_dim, device=h_protein.device))
                edge_attentions.append(torch.tensor([]))
                continue

            g_logits = attention_logits[mask]  # (n_e, num_heads)
            g_values = values[mask]  # (n_e, hidden_dim)

            # Softmax over edges
            g_attention = F.softmax(g_logits.mean(dim=1), dim=0)  # (n_e,)
            g_attention = self.dropout(g_attention)

            # Weighted sum
            g_pooled = (g_attention.unsqueeze(-1) * g_values).sum(dim=0)

            pooled.append(g_pooled)
            edge_attentions.append(g_attention)

        pooled = torch.stack(pooled, dim=0)  # (B, hidden_dim)

        # Store for interpretability
        self._edge_attention = edge_attentions

        return pooled

    def get_edge_attention(self) -> Optional[torch.Tensor]:
        """Return last computed edge attention weights."""
        return self._edge_attention


class GlobalMeanMaxPooling(nn.Module):
    """
    Simple global mean and max pooling concatenated.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Output is 2 * hidden_dim (mean + max)
        self.output_dim = 2 * hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Node features (N, hidden_dim)
            batch: Batch assignment (N,)

        Returns:
            (B, 2*hidden_dim)
        """
        if batch is None:
            mean_pool = x.mean(dim=0, keepdim=True)
            max_pool = x.max(dim=0, keepdim=True)[0]
            return torch.cat([mean_pool, max_pool], dim=-1)

        from torch_geometric.nn import global_mean_pool, global_max_pool

        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)

        return torch.cat([mean_pool, max_pool], dim=-1)


class SetTransformerPooling(nn.Module):
    """
    Set Transformer-based pooling (more expressive than simple attention).

    Uses multi-head self-attention followed by pooling by multi-head attention.
    Based on: Lee et al. "Set Transformer" (2019)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        num_seeds: int = 1,
        dropout: float = 0.1
    ):
        """
        Args:
            hidden_dim: Dimension of node features
            num_heads: Number of attention heads
            num_seeds: Number of seed vectors for pooling
            dropout: Dropout probability
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_seeds = num_seeds

        # Learnable seed vectors
        self.seeds = nn.Parameter(torch.randn(num_seeds, hidden_dim))

        # Multi-head attention layers
        self.mab = nn.MultiheadAttention(
            hidden_dim, num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Node features (N, hidden_dim)
            batch: Batch assignment (N,)

        Returns:
            (B, num_seeds * hidden_dim)
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        num_graphs = batch.max().item() + 1

        outputs = []
        for g in range(num_graphs):
            mask = (batch == g)
            g_x = x[mask].unsqueeze(0)  # (1, n_g, hidden_dim)

            # Expand seeds for this graph
            seeds = self.seeds.unsqueeze(0)  # (1, num_seeds, hidden_dim)

            # Multi-head attention: query=seeds, key/value=nodes
            attn_out, _ = self.mab(seeds, g_x, g_x)

            # Residual + LayerNorm
            attn_out = self.layer_norm(seeds + attn_out)

            # FFN
            ffn_out = self.ffn(attn_out)
            out = self.layer_norm2(attn_out + ffn_out)

            out = out.view(-1)  # (num_seeds * hidden_dim,)
            outputs.append(out)

        return torch.stack(outputs, dim=0)


class HierarchicalPooling(nn.Module):
    """
    Hierarchical pooling that combines protein, ligand, and interaction poolings.

    Output: [protein_pool, ligand_pool, interaction_pool]
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.protein_pool = AttentionPooling(hidden_dim, num_heads, dropout)
        self.ligand_pool = AttentionPooling(hidden_dim, num_heads, dropout)
        self.interaction_pool = InteractionPooling(hidden_dim, num_heads, dropout)

        # Output dimension is 3 * hidden_dim
        self.output_dim = 3 * hidden_dim

    def forward(
        self,
        h_protein: torch.Tensor,
        h_ligand: torch.Tensor,
        edge_index: torch.Tensor,
        batch_protein: Optional[torch.Tensor] = None,
        batch_ligand: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            h_protein: Protein node features
            h_ligand: Ligand node features
            edge_index: Protein-ligand interaction edges
            batch_protein: Batch assignment for protein
            batch_ligand: Batch assignment for ligand

        Returns:
            Combined representation (B, 3*hidden_dim)
        """
        # Pool protein
        g_protein = self.protein_pool(h_protein, batch_protein)

        # Pool ligand
        g_ligand = self.ligand_pool(h_ligand, batch_ligand)

        # Pool interactions
        g_inter = self.interaction_pool(
            h_protein, h_ligand, edge_index,
            batch_protein, batch_ligand
        )

        # Concatenate
        return torch.cat([g_protein, g_ligand, g_inter], dim=-1)


if __name__ == "__main__":
    # Test pooling layers
    print("Testing pooling layers...")

    hidden_dim = 128
    num_nodes = 50

    x = torch.randn(num_nodes, hidden_dim)
    batch = torch.randint(0, 4, (num_nodes,))

    # Test AttentionPooling
    pool = AttentionPooling(hidden_dim)
    out = pool(x, batch)
    print(f"AttentionPooling: input {x.shape} -> output {out.shape}")

    # Test InteractionPooling
    n_protein = 30
    n_ligand = 20
    n_edges = 100

    h_protein = torch.randn(n_protein, hidden_dim)
    h_ligand = torch.randn(n_ligand, hidden_dim)
    edge_index = torch.stack([
        torch.randint(0, n_protein, (n_edges,)),
        torch.randint(0, n_ligand, (n_edges,))
    ])

    inter_pool = InteractionPooling(hidden_dim)
    out = inter_pool(h_protein, h_ligand, edge_index)
    print(f"InteractionPooling: edges {n_edges} -> output {out.shape}")

    # Test HierarchicalPooling
    hier_pool = HierarchicalPooling(hidden_dim)
    out = hier_pool(h_protein, h_ligand, edge_index)
    print(f"HierarchicalPooling: output {out.shape} (expected {3*hidden_dim})")
