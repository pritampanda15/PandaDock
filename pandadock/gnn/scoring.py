"""
GNN-based scoring function for PandaDock integration.

Provides a scoring interface compatible with PandaDock's DockingEngine,
allowing the trained GNN model to be used for rescoring docked poses.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any, Tuple

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class GNNScoring:
    """
    GNN-based scoring function for protein-ligand binding.

    Integrates the trained PandaDockGNN model as a scoring function
    that can be used with DockingEngine or standalone.

    Example:
        scorer = GNNScoring(model_path='model.pt')
        energy = scorer.calculate_binding_energy(
            ligand_coords, receptor_structure
        )
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'auto',
        use_attention: bool = True,
        site_radius: float = 10.0,
        strip_hydrogens: bool = False
    ):
        """
        Initialize GNN scoring function.

        Args:
            model_path: Path to trained model checkpoint
            device: Device to use ('auto', 'cuda', 'cpu')
            use_attention: Store attention weights for interpretability
            site_radius: When no binding site file is supplied, the protein is
                cut to this radius around the ligand centroid. Must match the
                radius the model was trained at.
            strip_hydrogens: Drop hydrogens before scoring. Required for models
                trained on SAIR, whose CIFs carry no hydrogens; wrong for the
                ULVSH and PDBbind models, which were trained with them present.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. Install with: pip install torch")

        self.use_attention = use_attention
        self.site_radius = site_radius
        self.strip_hydrogens = strip_hydrogens

        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Graph builder for inference
        self.graph_builder = None
        self._last_attention = None

        # Load model
        self.model = None
        if model_path is not None:
            self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        """Load trained model from checkpoint."""
        from .models.pandadock_gnn import PandaDockGNN

        self.model = PandaDockGNN.load(model_path, map_location=str(self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        # Initialize graph builder
        from .data.graph_builder import GraphConfig, HeterogeneousGraphBuilder
        self.graph_builder = HeterogeneousGraphBuilder(
            GraphConfig(
                site_radius=self.site_radius,
                strip_hydrogens=self.strip_hydrogens,
            )
        )

    def calculate_binding_energy(
        self,
        ligand_coords: np.ndarray,
        receptor_structure: Any,
        ligand_mol: Optional[Any] = None,
        **kwargs
    ) -> float:
        """
        Calculate binding energy using the GNN model.

        This method is compatible with PandaDock's scoring interface.

        Args:
            ligand_coords: Ligand atomic coordinates (N, 3)
            receptor_structure: Receptor structure (BioPython Structure or path)
            ligand_mol: Optional RDKit molecule object
            **kwargs: Additional arguments (ignored)

        Returns:
            Predicted binding energy (lower = better binding)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Build graph from pose
        graph = self._build_graph_from_pose(
            ligand_coords, receptor_structure, ligand_mol
        )

        # Predict
        with torch.no_grad():
            graph = graph.to(self.device)
            predictions = self.model(graph)

        # Convert pEC50 to energy-like score
        # Higher pEC50 = better binding = more negative energy
        pec50 = predictions['affinity'].item()

        # Approximate RT*ln(K) conversion at 300K
        # ΔG = -RT * ln(K) = -RT * 2.303 * pK
        # At 300K: RT ≈ 0.593 kcal/mol
        energy = -0.593 * 2.303 * pec50

        # Store attention if requested
        if self.use_attention:
            self._last_attention = self.model.get_attention_weights()

        return energy

    def predict_affinity(
        self,
        protein_mol2: str,
        ligand_mol2: str,
        site_mol2: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Predict binding affinity from MOL2 files.

        Args:
            protein_mol2: Path to protein MOL2 file
            ligand_mol2: Path to ligand MOL2 file
            site_mol2: Optional path to binding site MOL2

        Returns:
            Dict with 'pec50', 'activity_prob', and 'energy'
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Build graph
        graph = self.graph_builder.build_from_files(
            protein_mol2, ligand_mol2, site_mol2
        )

        # Predict
        with torch.no_grad():
            graph = graph.to(self.device)
            predictions = self.model(graph)

        pec50 = predictions['affinity'].item()

        result = {
            'pec50': pec50,
            'energy': -0.593 * 2.303 * pec50
        }

        if 'activity' in predictions:
            activity_logit = predictions['activity'].item()
            result['activity_prob'] = 1.0 / (1.0 + np.exp(-activity_logit))
            result['active'] = result['activity_prob'] > 0.5

        return result

    def predict_from_graph(self, graph) -> Dict[str, float]:
        """
        Predict binding affinity from a pre-built graph.

        Args:
            graph: HeteroData graph with protein and ligand nodes

        Returns:
            Dict with 'pec50', 'activity_prob', and 'energy'
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Predict
        with torch.no_grad():
            graph = graph.to(self.device)
            predictions = self.model(graph)

        pec50 = predictions['affinity'].item()

        result = {
            'pec50': pec50,
            'energy': -0.593 * 2.303 * pec50
        }

        if 'activity' in predictions:
            activity_logit = predictions['activity'].item()
            result['activity_prob'] = 1.0 / (1.0 + np.exp(-activity_logit))
            result['active'] = result['activity_prob'] > 0.5

        return result

    def predict_batch(
        self,
        complexes: list
    ) -> list:
        """
        Predict for multiple complexes.

        Args:
            complexes: List of (protein_mol2, ligand_mol2, site_mol2) tuples

        Returns:
            List of prediction dicts
        """
        results = []
        for protein, ligand, site in complexes:
            try:
                result = self.predict_affinity(protein, ligand, site)
                results.append(result)
            except Exception as e:
                results.append({'error': str(e)})
        return results

    def _build_graph_from_pose(
        self,
        ligand_coords: np.ndarray,
        receptor_structure: Any,
        ligand_mol: Optional[Any] = None
    ):
        """Build graph from pose coordinates."""
        # This is a simplified version - in production, you'd need to
        # properly handle the receptor structure and ligand molecule

        from torch_geometric.data import HeteroData
        from .data.featurizer import AtomFeaturizer, EdgeFeaturizer

        # For now, return a dummy graph if real data isn't available
        # In production, you'd properly featurize the inputs
        raise NotImplementedError(
            "Direct pose scoring requires proper structure handling. "
            "Use predict_affinity() with MOL2 files instead."
        )

    def get_attention_weights(self) -> Optional[Dict[str, Any]]:
        """
        Get attention weights from last prediction.

        Returns:
            Dict with 'protein', 'ligand', 'interaction' attention weights
        """
        return self._last_attention

    def get_binding_hotspots(
        self,
        protein_mol2: str,
        ligand_mol2: str,
        site_mol2: Optional[str] = None,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Identify binding hotspots from attention weights.

        Args:
            protein_mol2: Path to protein MOL2
            ligand_mol2: Path to ligand MOL2
            site_mol2: Optional binding site MOL2
            top_k: Number of top interactions to return

        Returns:
            Dict with hotspot information
        """
        # Make prediction to compute attention
        _ = self.predict_affinity(protein_mol2, ligand_mol2, site_mol2)

        attention = self.get_attention_weights()
        if attention is None:
            return {'error': 'No attention weights available'}

        # Parse molecules for atom information
        from .data.mol2_parser import MOL2Parser
        parser = MOL2Parser()

        protein = parser.parse(site_mol2 if site_mol2 else protein_mol2)
        ligand = parser.parse(ligand_mol2)

        # Get interaction attention
        inter_attention = attention.get('interaction', [])
        if not inter_attention or len(inter_attention) == 0:
            return {'error': 'No interaction attention'}

        # Note: This is simplified - real implementation would map
        # attention weights back to specific residue-atom pairs
        return {
            'protein_atoms': len(protein.atoms),
            'ligand_atoms': len(ligand.atoms),
            'attention_available': True,
            'message': 'Use visualization tools for detailed hotspot analysis'
        }


def create_gnn_scorer(model_path: str, device: str = 'auto') -> GNNScoring:
    """
    Factory function to create GNN scorer.

    Args:
        model_path: Path to trained model
        device: Device to use

    Returns:
        GNNScoring instance
    """
    return GNNScoring(model_path=model_path, device=device)


if __name__ == "__main__":
    print("GNNScoring module loaded successfully")
    print("Usage: scorer = GNNScoring(model_path='model.pt')")
