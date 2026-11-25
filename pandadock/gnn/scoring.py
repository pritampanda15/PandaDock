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
        site_radius: Optional[float] = None,
        strip_hydrogens: Optional[bool] = None
    ):
        """
        Initialize GNN scoring function.

        Args:
            model_path: Path to trained model checkpoint
            device: Device to use ('auto', 'cuda', 'cpu')
            use_attention: Store attention weights for interpretability
            site_radius: When no binding site file is supplied, the protein is
                cut to this radius around the ligand centroid. Defaults to the
                checkpoint's stored metadata if present, otherwise 10.0 A.
            strip_hydrogens: Drop hydrogens before scoring. Required for models
                trained on SAIR, whose CIFs carry no hydrogens; wrong for the
                ULVSH and PDBbind models, which were trained with them present.
                Defaults to the checkpoint's stored metadata if present, else False.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. Install with: pip install torch")

        self.use_attention = use_attention
        self._explicit_site_radius = site_radius
        self._explicit_strip_hydrogens = strip_hydrogens
        self.site_radius = 10.0 if site_radius is None else float(site_radius)
        self.strip_hydrogens = False if strip_hydrogens is None else bool(strip_hydrogens)

        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Graph builder for inference
        self.graph_builder = None
        self._last_attention = None
        self._receptor_cache: Dict[int, Any] = {}

        # Load model
        self.model = None
        if model_path is not None:
            self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        """Load trained model from checkpoint and apply stored preprocessing metadata."""
        from .models.pandadock_gnn import PandaDockGNN

        self.model = PandaDockGNN.load(model_path, map_location=str(self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        meta = getattr(self.model, 'metadata', {}) or {}
        if self._explicit_site_radius is None and 'site_radius' in meta:
            self.site_radius = float(meta['site_radius'])
        if self._explicit_strip_hydrogens is None and 'strip_hydrogens' in meta:
            self.strip_hydrogens = bool(meta['strip_hydrogens'])

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
            result['activity_prob'] = float(1.0 / (1.0 + np.exp(-activity_logit)))
            result['active'] = bool(result['activity_prob'] > 0.5)

        if 'pose_rmsd' in predictions:
            result['predicted_rmsd_A'] = float(predictions['pose_rmsd'].item())

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
            result['activity_prob'] = float(1.0 / (1.0 + np.exp(-activity_logit)))
            result['active'] = bool(result['activity_prob'] > 0.5)

        if 'pose_rmsd' in predictions:
            result['predicted_rmsd_A'] = float(predictions['pose_rmsd'].item())

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

    def _receptor_to_parsed(self, receptor_structure: Any):
        """Convert a file path, ParsedMolecule, or BioPython Structure to ParsedMolecule."""
        from .data.mol2_parser import Atom, ParsedMolecule
        from .data.graph_builder import parse_molecule_file, protein_sybyl_type

        if isinstance(receptor_structure, ParsedMolecule):
            return receptor_structure

        if isinstance(receptor_structure, (str, Path)):
            return parse_molecule_file(str(receptor_structure))

        cache_key = id(receptor_structure)
        if cache_key in self._receptor_cache:
            return self._receptor_cache[cache_key]

        atoms = []
        atom_id = 0
        for model in receptor_structure:
            for chain in model:
                for residue in chain:
                    res_name = residue.get_resname()
                    res_id = residue.get_id()[1]
                    for atom in residue:
                        atom_id += 1
                        element = atom.element.strip().upper() if atom.element else 'C'
                        sybyl_type = protein_sybyl_type(res_name, atom.get_name(), element)
                        coord = atom.get_coord()
                        atoms.append(Atom(
                            id=atom_id,
                            name=atom.get_name(),
                            x=float(coord[0]),
                            y=float(coord[1]),
                            z=float(coord[2]),
                            atom_type=sybyl_type,
                            residue_id=int(res_id),
                            residue_name=res_name,
                            charge=0.0,
                        ))
            break  # first model only

        parsed = ParsedMolecule(name="receptor", atoms=atoms, bonds=[])
        parsed.num_atoms = len(atoms)
        if len(self._receptor_cache) >= 8:
            self._receptor_cache.clear()
        self._receptor_cache[cache_key] = parsed
        return parsed

    @staticmethod
    def _rdkit_to_sybyl(atom) -> str:
        """Map an RDKit atom to a SYBYL atom type."""
        from rdkit import Chem

        sym = atom.GetSymbol()
        hyb = atom.GetHybridization()
        if sym == 'C':
            if atom.GetIsAromatic():
                return 'C.ar'
            if hyb == Chem.HybridizationType.SP2:
                return 'C.2'
            if hyb == Chem.HybridizationType.SP:
                return 'C.1'
            return 'C.3'
        if sym == 'N':
            if atom.GetIsAromatic():
                return 'N.ar'
            if atom.GetFormalCharge() > 0:
                return 'N.4'
            if hyb == Chem.HybridizationType.SP2:
                return 'N.2'
            if hyb == Chem.HybridizationType.SP:
                return 'N.1'
            return 'N.3'
        if sym == 'O':
            if hyb == Chem.HybridizationType.SP2:
                return 'O.2'
            return 'O.3'
        if sym == 'S':
            if hyb == Chem.HybridizationType.SP2:
                return 'S.2'
            return 'S.3'
        if sym == 'P':
            return 'P.3'
        return sym

    def _ligand_to_parsed(self, ligand_coords: np.ndarray, ligand_mol: Optional[Any]):
        """Convert ligand coordinates + optional RDKit Mol or ParsedMolecule to ParsedMolecule."""
        from .data.mol2_parser import Atom, Bond, ParsedMolecule

        coords = np.asarray(ligand_coords, dtype=np.float64)
        if isinstance(ligand_mol, ParsedMolecule):
            atoms = [
                Atom(
                    id=a.id, name=a.name,
                    x=float(coords[i, 0]), y=float(coords[i, 1]), z=float(coords[i, 2]),
                    atom_type=a.atom_type, residue_id=a.residue_id,
                    residue_name=a.residue_name, charge=a.charge,
                )
                for i, a in enumerate(ligand_mol.atoms)
            ]
            parsed = ParsedMolecule(name=ligand_mol.name, atoms=atoms, bonds=list(ligand_mol.bonds))
            parsed.num_atoms = len(atoms)
            parsed.num_bonds = len(parsed.bonds)
            return parsed

        if ligand_mol is not None and hasattr(ligand_mol, 'GetAtoms'):
            atoms = []
            for i, atom in enumerate(ligand_mol.GetAtoms()):
                if i >= len(coords):
                    break
                charge = 0.0
                if atom.HasProp('_GasteigerCharge'):
                    try:
                        c = float(atom.GetProp('_GasteigerCharge'))
                        if np.isfinite(c):
                            charge = c
                    except ValueError:
                        pass
                atoms.append(Atom(
                    id=i + 1,
                    name=f"{atom.GetSymbol()}{i + 1}",
                    x=float(coords[i, 0]),
                    y=float(coords[i, 1]),
                    z=float(coords[i, 2]),
                    atom_type=self._rdkit_to_sybyl(atom),
                    residue_id=1,
                    residue_name="LIG",
                    charge=charge,
                ))
            bonds = []
            for b_idx, bond in enumerate(ligand_mol.GetBonds()):
                a1 = bond.GetBeginAtomIdx()
                a2 = bond.GetEndAtomIdx()
                if a1 < len(atoms) and a2 < len(atoms):
                    btype = "ar" if bond.GetIsAromatic() else str(int(round(bond.GetBondTypeAsDouble())))
                    bonds.append(Bond(id=b_idx + 1, atom1_id=a1 + 1, atom2_id=a2 + 1, bond_type=btype))
            name = ligand_mol.GetProp('_Name') if ligand_mol.HasProp('_Name') else 'ligand'
            parsed = ParsedMolecule(name=name, atoms=atoms, bonds=bonds)
            parsed.num_atoms = len(atoms)
            parsed.num_bonds = len(bonds)
            return parsed

        raise ValueError(
            "Direct pose scoring requires `ligand_mol` (an RDKit Mol or ParsedMolecule) "
            "so ligand atom types and bonds can be constructed."
        )

    def _build_graph_from_pose(
        self,
        ligand_coords: np.ndarray,
        receptor_structure: Any,
        ligand_mol: Optional[Any] = None
    ):
        """Build a binding-site HeteroData graph directly from pose coordinates."""
        from .data.graph_builder import drop_hydrogens, extract_binding_site

        protein = self._receptor_to_parsed(receptor_structure)
        ligand = self._ligand_to_parsed(ligand_coords, ligand_mol)
        centroid = np.asarray(ligand_coords, dtype=np.float64).mean(axis=0)
        site = extract_binding_site(protein, centroid, radius=self.site_radius)
        if self.strip_hydrogens:
            site = drop_hydrogens(site)
            ligand = drop_hydrogens(ligand)
        return self.graph_builder.build_graph(site, ligand)

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
        Identify binding hotspots from attention weights, mapped back to residues and atoms.

        Args:
            protein_mol2: Path to protein file (MOL2/PDB)
            ligand_mol2: Path to ligand file (MOL2/SDF/PDB)
            site_mol2: Optional path to binding site file
            top_k: Number of top residues / atoms to return

        Returns:
            Dict with ranked `top_protein_residues`, `top_ligand_atoms`, and raw counts
        """
        from .data.graph_builder import drop_hydrogens, extract_binding_site, parse_molecule_file

        protein = parse_molecule_file(site_mol2 if site_mol2 else protein_mol2)
        ligand = parse_molecule_file(ligand_mol2)

        if site_mol2 is None and self.site_radius is not None:
            lig_xyz = ligand.coordinates
            if len(lig_xyz) > 0:
                protein = extract_binding_site(protein, lig_xyz.mean(axis=0), radius=self.site_radius)

        if self.strip_hydrogens:
            protein = drop_hydrogens(protein)
            ligand = drop_hydrogens(ligand)

        graph = self.graph_builder.build_graph(protein, ligand)
        prediction = self.predict_from_graph(graph)
        if self.use_attention:
            self._last_attention = self.model.get_attention_weights()

        attention = self.get_attention_weights()
        if attention is None:
            return {'error': 'No attention weights available', 'prediction': prediction}

        def _to_1d(value):
            if value is None:
                return None
            if isinstance(value, (list, tuple)):
                if not value:
                    return None
                arrays = []
                for item in value:
                    a = item.detach().cpu().numpy() if hasattr(item, 'detach') else np.asarray(item)
                    if a.ndim > 1:
                        a = a.mean(axis=-1)
                    arrays.append(a.reshape(-1))
                return np.mean(np.stack(arrays, axis=0), axis=0)
            arr = value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)
            if arr.ndim > 1:
                arr = arr.mean(axis=-1)
            return arr.reshape(-1)

        prot_scores = _to_1d(attention.get('protein'))
        lig_scores = _to_1d(attention.get('ligand'))

        residue_totals: Dict[Tuple[str, int], float] = {}
        if prot_scores is not None:
            for atom, score in zip(protein.atoms, prot_scores):
                key = (atom.residue_name, int(atom.residue_id))
                residue_totals[key] = residue_totals.get(key, 0.0) + float(score)

        top_residues = [
            {'residue_name': res, 'residue_id': rid, 'attention': val}
            for (res, rid), val in sorted(
                residue_totals.items(), key=lambda kv: kv[1], reverse=True
            )[:top_k]
        ]

        top_ligand_atoms = []
        if lig_scores is not None:
            ranked_lig = sorted(
                zip(ligand.atoms, lig_scores), key=lambda pair: float(pair[1]), reverse=True
            )[:top_k]
            top_ligand_atoms = [
                {
                    'atom_id': int(a.id),
                    'atom_name': a.name,
                    'atom_type': a.atom_type,
                    'attention': float(s),
                }
                for a, s in ranked_lig
            ]

        return {
            'prediction': prediction,
            'protein_atoms': len(protein.atoms),
            'ligand_atoms': len(ligand.atoms),
            'attention_available': True,
            'top_protein_residues': top_residues,
            'top_ligand_atoms': top_ligand_atoms,
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
