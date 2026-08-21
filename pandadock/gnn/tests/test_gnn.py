"""
Test suite for PandaDock-GNN module.

Run with: python -m pytest pandadock/gnn/tests/test_gnn.py -v
Or:       python pandadock/gnn/tests/test_gnn.py
"""

import sys
import numpy as np
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def test_mol2_parser():
    """Test MOL2 file parsing."""
    print("\n[TEST] MOL2 Parser...")

    from pandadock.gnn.data.mol2_parser import MOL2Parser, ParsedMolecule

    # Test with sample content
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

    assert mol.name == "test_molecule", f"Expected 'test_molecule', got '{mol.name}'"
    assert mol.num_atoms == 5, f"Expected 5 atoms, got {mol.num_atoms}"
    assert mol.num_bonds == 4, f"Expected 4 bonds, got {mol.num_bonds}"
    assert mol.coordinates.shape == (5, 3), f"Expected (5,3) coords, got {mol.coordinates.shape}"

    print("  PASSED: MOL2 parser works correctly")


def test_featurizer():
    """Test atom and edge featurization."""
    print("\n[TEST] Featurizer...")

    from pandadock.gnn.data.mol2_parser import MOL2Parser
    from pandadock.gnn.data.featurizer import AtomFeaturizer, EdgeFeaturizer

    # Create sample molecule
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

    # Test atom featurizer
    atom_feat = AtomFeaturizer()
    features = atom_feat.featurize_molecule(mol)

    assert features.shape[0] == 3, f"Expected 3 atoms, got {features.shape[0]}"
    assert features.shape[1] == atom_feat.feature_dim, f"Feature dim mismatch"

    # Test edge featurizer
    edge_feat = EdgeFeaturizer()
    coords = mol.coordinates
    edge_index, edge_features = edge_feat.compute_pairwise_features(
        coords, coords, cutoff=5.0
    )

    assert edge_index.shape[0] == 2, "Edge index should have 2 rows"
    assert edge_features.shape[1] == edge_feat.feature_dim, "Edge feature dim mismatch"

    print(f"  Node features: {features.shape}")
    print(f"  Edge features: {edge_features.shape}")
    print("  PASSED: Featurizer works correctly")


def test_egnn_layer():
    """Test EGNN layer equivariance."""
    print("\n[TEST] EGNN Layer Equivariance...")

    import torch
    import math
    from pandadock.gnn.models.layers import EGNNLayer

    torch.manual_seed(42)

    hidden_dim = 64
    num_nodes = 20
    num_edges = 50

    # Create layer
    layer = EGNNLayer(hidden_dim=hidden_dim)

    # Random inputs
    h = torch.randn(num_nodes, hidden_dim)
    pos = torch.randn(num_nodes, 3)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    # Forward pass
    h_out, pos_out = layer(h, pos, edge_index)

    # Test rotation equivariance
    theta = torch.tensor(math.pi / 3)  # 60 degrees
    R = torch.tensor([
        [torch.cos(theta), -torch.sin(theta), 0],
        [torch.sin(theta), torch.cos(theta), 0],
        [0, 0, 1]
    ], dtype=torch.float32)

    pos_rotated = pos @ R.T
    h_out2, pos_out2 = layer(h, pos_rotated, edge_index)

    # Rotated output should match rotation of output
    pos_out_rotated = pos_out @ R.T

    diff = (pos_out2 - pos_out_rotated).abs().mean().item()

    assert diff < 1e-4, f"Rotation equivariance error: {diff}"

    print(f"  Rotation equivariance error: {diff:.6f}")
    print("  PASSED: EGNN layer is equivariant")


def test_pooling():
    """Test attention pooling."""
    print("\n[TEST] Attention Pooling...")

    import torch
    from pandadock.gnn.models.pooling import AttentionPooling, HierarchicalPooling

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

    # Test hierarchical pooling
    pool = HierarchicalPooling(hidden_dim)
    output = pool(h_protein, h_ligand, edge_index)

    expected_dim = 3 * hidden_dim
    assert output.shape == (1, expected_dim), f"Expected (1, {expected_dim}), got {output.shape}"

    print(f"  Output shape: {output.shape}")
    print("  PASSED: Pooling works correctly")


def test_model_forward():
    """Test PandaDockGNN forward pass."""
    print("\n[TEST] PandaDockGNN Forward Pass...")

    import torch
    from torch_geometric.data import HeteroData
    from pandadock.gnn.models.pandadock_gnn import PandaDockGNN, ModelConfig

    torch.manual_seed(42)

    # Create dummy data
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

    # Create model
    config = ModelConfig(
        protein_input_dim=node_dim,
        ligand_input_dim=node_dim,
        edge_input_dim=edge_dim,
        hidden_dim=128,
        num_layers=3,
        predict_activity=True
    )
    model = PandaDockGNN(config)

    # Forward pass
    model.eval()
    with torch.no_grad():
        predictions = model(data)

    assert 'affinity' in predictions, "Missing affinity prediction"
    assert 'activity' in predictions, "Missing activity prediction"
    assert predictions['affinity'].shape == (1,), f"Expected (1,), got {predictions['affinity'].shape}"

    print(f"  Model parameters: {model.count_parameters():,}")
    print(f"  Affinity prediction: {predictions['affinity'].item():.4f}")
    print(f"  Activity prediction: {predictions['activity'].item():.4f}")
    print("  PASSED: Model forward pass works")


def test_loss_functions():
    """Test loss functions."""
    print("\n[TEST] Loss Functions...")

    import torch
    from pandadock.gnn.training.losses import MultiTaskLoss, AffinityLoss, ActivityLoss

    torch.manual_seed(42)
    batch_size = 16

    # Create predictions and targets
    predictions = {
        'affinity': torch.randn(batch_size),
        'activity': torch.randn(batch_size)
    }
    targets = {
        'affinity': torch.randn(batch_size) + 6,
        'activity': (torch.rand(batch_size) > 0.5).float()
    }

    # Test multi-task loss
    loss_fn = MultiTaskLoss(w_affinity=1.0, w_activity=0.5)
    losses = loss_fn(predictions, targets)

    assert 'total' in losses, "Missing total loss"
    assert 'affinity' in losses, "Missing affinity loss"
    assert 'activity' in losses, "Missing activity loss"
    assert losses['total'].item() > 0, "Total loss should be positive"

    print(f"  Total loss: {losses['total'].item():.4f}")
    print(f"  Affinity loss: {losses['affinity'].item():.4f}")
    print(f"  Activity loss: {losses['activity'].item():.4f}")
    print("  PASSED: Loss functions work correctly")


def test_metrics():
    """Test evaluation metrics."""
    print("\n[TEST] Metrics...")

    import numpy as np
    from pandadock.gnn.training.metrics import (
        pearson_r, spearman_rho, rmse, mae,
        auc_roc, compute_affinity_metrics
    )

    np.random.seed(42)

    # Create correlated predictions
    y_true = np.random.randn(100) + 6
    y_pred = y_true + np.random.randn(100) * 0.5

    r = pearson_r(y_true, y_pred)
    rho = spearman_rho(y_true, y_pred)
    error = rmse(y_true, y_pred)

    assert 0.8 < r < 1.0, f"Expected high correlation, got {r}"
    assert error < 1.0, f"Expected low RMSE, got {error}"

    # Compute all metrics
    metrics = compute_affinity_metrics(y_true, y_pred)

    print(f"  Pearson R: {metrics['pearson_r']:.4f}")
    print(f"  Spearman rho: {metrics['spearman_rho']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print("  PASSED: Metrics work correctly")


def test_ulvsh_dataset():
    """Test ULVSH dataset loading (if data available)."""
    print("\n[TEST] ULVSH Dataset...")

    ulvsh_path = Path("/mnt/cce9630f-8b3b-4312-932d-ff04311ba514/SSD/PandaDock/ULVSH")

    if not ulvsh_path.exists():
        print("  SKIPPED: ULVSH dataset not found")
        return

    from pandadock.gnn.data.dataset import ULVSHDataset

    # Load small subset
    dataset = ULVSHDataset(
        root=str(ulvsh_path),
        split='train',
        cache_graphs=False
    )

    print(f"  Total compounds: {len(dataset)}")

    if len(dataset) > 0:
        # Load first sample
        sample = dataset[0]

        print(f"  Protein nodes: {sample['protein'].num_nodes}")
        print(f"  Ligand nodes: {sample['ligand'].num_nodes}")
        print(f"  pEC50: {sample.y_affinity.item():.3f}")
        print("  PASSED: Dataset loads correctly")



def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("PandaDock-GNN Test Suite")
    print("=" * 60)

    tests = [
        ("MOL2 Parser", test_mol2_parser),
        ("Featurizer", test_featurizer),
        ("EGNN Layer", test_egnn_layer),
        ("Pooling", test_pooling),
        ("Model Forward", test_model_forward),
        ("Loss Functions", test_loss_functions),
        ("Metrics", test_metrics),
        ("ULVSH Dataset", test_ulvsh_dataset),
    ]

    results = []

    for name, test_fn in tests:
        try:
            # A test passes by returning without raising, matching pytest.
            test_fn()
            results.append((name, "PASSED"))
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append((name, f"ERROR: {e}"))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, r in results if r == "PASSED")
    total = len(results)

    for name, result in results:
        status = "✓" if result == "PASSED" else "✗"
        print(f"  {status} {name}: {result}")

    print(f"\nTotal: {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    try:
        import torch
        import torch_geometric
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install torch torch-geometric")
        sys.exit(1)
