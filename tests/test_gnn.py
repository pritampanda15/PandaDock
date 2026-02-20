"""
Test suite for PandaDock-GNN module.

Run with: pytest tests/test_gnn.py -v
"""

import pytest
import numpy as np


# Skip all tests if torch/torch-geometric not available
torch = pytest.importorskip("torch")
torch_geometric = pytest.importorskip("torch_geometric")


class TestMOL2Parser:
    """Test MOL2 file parsing."""

    def test_parse_string(self):
        """Test parsing MOL2 from string."""
        from pandadock.gnn.data.mol2_parser import MOL2Parser

        sample_mol2 = """@<TRIPOS>MOLECULE
test_molecule
5 4 0 0 0
SMALL
NO_CHARGES

@<TRIPOS>ATOM
1 C1 0.0 0.0 0.0 C.3 1 LIG 0.0
2 C2 1.5 0.0 0.0 C.3 1 LIG 0.0
3 N1 2.5 1.0 0.0 N.3 1 LIG 0.0
4 O1 0.0 1.5 0.0 O.3 1 LIG 0.0
5 H1 -0.5 -0.5 0.0 H 1 LIG 0.0
@<TRIPOS>BOND
1 1 2 1
2 2 3 1
3 1 4 1
4 1 5 1
"""
        parser = MOL2Parser()
        mol = parser.parse_string(sample_mol2)

        assert mol.name == "test_molecule"
        assert mol.num_atoms == 5
        assert mol.num_bonds == 4
        assert mol.coordinates.shape == (5, 3)


class TestFeaturizer:
    """Test atom and edge featurization."""

    def test_atom_featurizer(self):
        """Test atom feature extraction."""
        from pandadock.gnn.data.mol2_parser import MOL2Parser
        from pandadock.gnn.data.featurizer import AtomFeaturizer

        sample_mol2 = """@<TRIPOS>MOLECULE
test
3 2 0 0 0
SMALL
NO_CHARGES

@<TRIPOS>ATOM
1 C1 0.0 0.0 0.0 C.ar 1 LIG 0.0
2 N1 1.5 0.0 0.0 N.am 1 LIG -0.5
3 O1 0.0 1.5 0.0 O.3 1 LIG 0.2
@<TRIPOS>BOND
1 1 2 ar
2 1 3 1
"""
        parser = MOL2Parser()
        mol = parser.parse_string(sample_mol2)

        atom_feat = AtomFeaturizer()
        features = atom_feat.featurize_molecule(mol)

        assert features.shape[0] == 3
        assert features.shape[1] == atom_feat.feature_dim

    def test_edge_featurizer(self):
        """Test edge feature computation."""
        from pandadock.gnn.data.featurizer import EdgeFeaturizer

        edge_feat = EdgeFeaturizer()
        coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)

        edge_index, edge_features = edge_feat.compute_pairwise_features(
            coords, coords, cutoff=5.0
        )

        assert edge_index.shape[0] == 2
        assert edge_features.shape[1] == edge_feat.feature_dim


class TestEGNNLayer:
    """Test EGNN equivariant layers."""

    def test_forward_pass(self):
        """Test EGNN layer forward pass."""
        from pandadock.gnn.models.layers import EGNNLayer

        torch.manual_seed(42)

        hidden_dim = 64
        num_nodes = 20
        num_edges = 50

        layer = EGNNLayer(hidden_dim=hidden_dim)

        h = torch.randn(num_nodes, hidden_dim)
        pos = torch.randn(num_nodes, 3)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        h_out, pos_out = layer(h, pos, edge_index)

        assert h_out.shape == h.shape
        assert pos_out.shape == pos.shape

    def test_rotation_equivariance(self):
        """Test that EGNN is rotation equivariant."""
        import math
        from pandadock.gnn.models.layers import EGNNLayer

        torch.manual_seed(42)

        layer = EGNNLayer(hidden_dim=64)

        h = torch.randn(20, 64)
        pos = torch.randn(20, 3)
        edge_index = torch.randint(0, 20, (2, 50))

        # Forward pass
        h_out, pos_out = layer(h.clone(), pos.clone(), edge_index)

        # Apply rotation
        theta = torch.tensor(math.pi / 3)
        R = torch.tensor([
            [torch.cos(theta), -torch.sin(theta), 0],
            [torch.sin(theta), torch.cos(theta), 0],
            [0, 0, 1]
        ], dtype=torch.float32)

        pos_rotated = pos @ R.T
        h_out2, pos_out2 = layer(h.clone(), pos_rotated, edge_index)

        # Rotated output should match rotation of output
        pos_out_rotated = pos_out @ R.T

        diff = (pos_out2 - pos_out_rotated).abs().mean().item()
        assert diff < 1e-4, f"Rotation equivariance error: {diff}"


class TestPooling:
    """Test attention pooling."""

    def test_hierarchical_pooling(self):
        """Test hierarchical pooling."""
        from pandadock.gnn.models.pooling import HierarchicalPooling

        torch.manual_seed(42)

        hidden_dim = 128
        n_protein = 50
        n_ligand = 20
        n_edges = 100

        h_protein = torch.randn(n_protein, hidden_dim)
        h_ligand = torch.randn(n_ligand, hidden_dim)
        edge_index = torch.stack([
            torch.randint(0, n_protein, (n_edges,)),
            torch.randint(0, n_ligand, (n_edges,))
        ])

        pool = HierarchicalPooling(hidden_dim)
        output = pool(h_protein, h_ligand, edge_index)

        expected_dim = 3 * hidden_dim
        assert output.shape == (1, expected_dim)


class TestPandaDockGNN:
    """Test full model."""

    def test_forward_pass(self):
        """Test model forward pass."""
        from torch_geometric.data import HeteroData
        from pandadock.gnn.models.pandadock_gnn import PandaDockGNN, ModelConfig

        torch.manual_seed(42)

        n_protein = 100
        n_ligand = 30
        n_edges = 200
        node_dim = 56
        edge_dim = 23

        data = HeteroData()
        data['protein'].x = torch.randn(n_protein, node_dim)
        data['protein'].pos = torch.randn(n_protein, 3)
        data['ligand'].x = torch.randn(n_ligand, node_dim)
        data['ligand'].pos = torch.randn(n_ligand, 3)

        edge_index = torch.stack([
            torch.randint(0, n_protein, (n_edges,)),
            torch.randint(0, n_ligand, (n_edges,))
        ])
        data['protein', 'interacts', 'ligand'].edge_index = edge_index
        data['protein', 'interacts', 'ligand'].edge_attr = torch.randn(n_edges, edge_dim)
        data['ligand', 'interacts', 'protein'].edge_index = edge_index.flip(0)
        data['ligand', 'interacts', 'protein'].edge_attr = torch.randn(n_edges, edge_dim)

        config = ModelConfig(
            protein_input_dim=node_dim,
            ligand_input_dim=node_dim,
            edge_input_dim=edge_dim,
            hidden_dim=128,
            num_layers=3,
            predict_activity=True
        )
        model = PandaDockGNN(config)

        model.eval()
        with torch.no_grad():
            predictions = model(data)

        assert 'affinity' in predictions
        assert 'activity' in predictions
        assert predictions['affinity'].shape == (1,)


class TestLossFunctions:
    """Test loss functions."""

    def test_multi_task_loss(self):
        """Test multi-task loss computation."""
        from pandadock.gnn.training.losses import MultiTaskLoss

        torch.manual_seed(42)
        batch_size = 16

        predictions = {
            'affinity': torch.randn(batch_size),
            'activity': torch.randn(batch_size)
        }
        targets = {
            'affinity': torch.randn(batch_size) + 6,
            'activity': (torch.rand(batch_size) > 0.5).float()
        }

        loss_fn = MultiTaskLoss(w_affinity=1.0, w_activity=0.5)
        losses = loss_fn(predictions, targets)

        assert 'total' in losses
        assert 'affinity' in losses
        assert 'activity' in losses
        assert losses['total'].item() > 0


class TestMetrics:
    """Test evaluation metrics."""

    def test_correlation_metrics(self):
        """Test Pearson R and Spearman rho."""
        from pandadock.gnn.training.metrics import pearson_r, spearman_rho, rmse

        np.random.seed(42)

        y_true = np.random.randn(100) + 6
        y_pred = y_true + np.random.randn(100) * 0.5

        r = pearson_r(y_true, y_pred)
        rho = spearman_rho(y_true, y_pred)
        error = rmse(y_true, y_pred)

        assert 0.8 < r < 1.0
        assert 0.8 < rho < 1.0
        assert error < 1.0
