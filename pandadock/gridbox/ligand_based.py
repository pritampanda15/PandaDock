import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from rdkit import Chem
from rdkit.Chem import rdMolAlign, AllChem
import logging
from .core import GridBox, GridBoxGenerator


class LigandBasedGridBoxGenerator(GridBoxGenerator):
    """Generate grid boxes based on reference ligands or pharmacophores"""

    def __init__(self):
        super().__init__()

    def generate(self, receptor_file: str, reference_ligand: str = None,
                 pharmacophore_features: List[str] = None, similarity_threshold: float = 0.7,
                 spacing: float = 0.375, **kwargs) -> List[GridBox]:
        """Generate grid boxes based on ligand similarity or pharmacophore matching"""

        if reference_ligand:
            return self._generate_from_reference_ligand(
                receptor_file, reference_ligand, similarity_threshold, spacing, **kwargs
            )
        elif pharmacophore_features:
            return self._generate_from_pharmacophore(
                receptor_file, pharmacophore_features, spacing, **kwargs
            )
        else:
            raise ValueError("Either reference_ligand or pharmacophore_features must be provided")

    def _generate_from_reference_ligand(self, receptor_file: str, reference_ligand: str,
                                      similarity_threshold: float, spacing: float, **kwargs) -> List[GridBox]:
        """Generate grid box based on reference ligand structure"""

        # Load reference ligand
        ref_mol = self._load_ligand(reference_ligand)
        if ref_mol is None:
            raise ValueError(f"Could not load reference ligand: {reference_ligand}")

        # Get ligand center and dimensions
        conf = ref_mol.GetConformer()
        coords = np.array([conf.GetAtomPosition(i) for i in range(ref_mol.GetNumAtoms())])

        center = np.mean(coords, axis=0)
        min_coords = np.min(coords, axis=0)
        max_coords = np.max(coords, axis=0)

        # Add padding around ligand
        padding = kwargs.get('padding', 8.0)
        dimensions = max_coords - min_coords + 2 * padding

        # Ensure reasonable dimensions for docking (15-30 Å)
        min_dim = 15.0
        max_dim = 30.0
        dimensions = np.maximum(dimensions, [min_dim, min_dim, min_dim])
        dimensions = np.minimum(dimensions, [max_dim, max_dim, max_dim])

        # Calculate ligand properties for scoring
        ligand_props = self._calculate_ligand_properties(ref_mol)

        grid_box = GridBox(
            center=tuple(center),
            dimensions=tuple(dimensions),
            spacing=spacing,
            receptor_file=receptor_file,
            method_used="reference_ligand",
            confidence_score=0.9,
            binding_site_info={
                'reference_ligand': reference_ligand,
                'ligand_properties': ligand_props,
                'padding_used': padding,
                'similarity_threshold': similarity_threshold
            }
        )

        return [grid_box]

    def _generate_from_pharmacophore(self, receptor_file: str, pharmacophore_features: List[str],
                                   spacing: float, **kwargs) -> List[GridBox]:
        """Generate grid boxes based on pharmacophore features"""

        structure = self._parse_receptor(receptor_file)
        atoms = list(structure.get_atoms())

        # Find regions that match pharmacophore features
        matching_sites = self._find_pharmacophore_sites(atoms, pharmacophore_features)

        grid_boxes = []
        for i, site in enumerate(matching_sites):
            if site['score'] >= kwargs.get('min_score', 0.5):
                grid_box = GridBox(
                    center=tuple(site['center']),
                    dimensions=tuple(site['dimensions']),
                    spacing=spacing,
                    receptor_file=receptor_file,
                    method_used="pharmacophore_matching",
                    confidence_score=site['score'],
                    binding_site_info={
                        'pharmacophore_features': pharmacophore_features,
                        'matched_features': site['matched_features'],
                        'site_rank': i + 1,
                        'feature_density': site['feature_density']
                    }
                )
                grid_boxes.append(grid_box)

        return grid_boxes

    def _load_ligand(self, ligand_file: str) -> Optional[Chem.Mol]:
        """Load ligand molecule from file"""
        try:
            if ligand_file.endswith('.sdf'):
                mol = Chem.SDMolSupplier(ligand_file)[0]
            elif ligand_file.endswith('.mol2'):
                mol = Chem.MolFromMol2File(ligand_file)
            elif ligand_file.endswith('.pdb'):
                mol = Chem.MolFromPDBFile(ligand_file)
            else:
                # Try to guess format
                mol = Chem.MolFromMolFile(ligand_file)

            if mol is not None:
                # Preserve original coordinates - don't add hydrogens or re-embed
                # We only need the heavy atom positions for grid box calculation
                pass

            return mol

        except Exception as e:
            self.logger.error(f"Error loading ligand {ligand_file}: {e}")
            return None

    def _calculate_ligand_properties(self, mol: Chem.Mol) -> Dict[str, Any]:
        """Calculate ligand properties for binding site characterization"""

        from rdkit.Chem import Descriptors, Lipinski

        props = {
            'molecular_weight': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'num_hbd': Descriptors.NumHDonors(mol),
            'num_hba': Descriptors.NumHAcceptors(mol),
            'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'tpsa': Descriptors.TPSA(mol),
            'num_rings': Descriptors.RingCount(mol),
            'num_aromatic_rings': Descriptors.NumAromaticRings(mol),
            'num_atoms': mol.GetNumAtoms(),
            'num_heavy_atoms': mol.GetNumHeavyAtoms()
        }

        # Lipinski's Rule of Five compliance
        props['lipinski_violations'] = (
            (props['molecular_weight'] > 500) +
            (props['logp'] > 5) +
            (props['num_hbd'] > 5) +
            (props['num_hba'] > 10)
        )

        # Simple druggability assessment
        props['drug_like'] = props['lipinski_violations'] <= 1

        return props

    def _find_pharmacophore_sites(self, atoms: List, features: List[str]) -> List[Dict[str, Any]]:
        """Find protein sites that match pharmacophore features"""

        # Define pharmacophore feature patterns
        feature_patterns = {
            'hydrophobic': self._find_hydrophobic_regions,
            'donor': self._find_hbond_donors,
            'acceptor': self._find_hbond_acceptors,
            'aromatic': self._find_aromatic_regions,
            'positive': self._find_positive_regions,
            'negative': self._find_negative_regions,
            'metal': self._find_metal_binding_sites
        }

        # Find all feature sites
        all_feature_sites = {}
        for feature in features:
            if feature in feature_patterns:
                sites = feature_patterns[feature](atoms)
                all_feature_sites[feature] = sites

        # Combine and score sites based on feature co-occurrence
        combined_sites = self._combine_pharmacophore_sites(all_feature_sites, features)

        return combined_sites

    def _find_hydrophobic_regions(self, atoms: List) -> List[Dict[str, Any]]:
        """Find hydrophobic regions in protein"""

        hydrophobic_residues = {'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO'}
        hydrophobic_atoms = []

        for atom in atoms:
            try:
                residue = atom.get_parent()
                if residue.get_resname() in hydrophobic_residues:
                    if atom.element in ['C'] and atom.get_name() not in ['C', 'CA', 'N', 'O']:
                        hydrophobic_atoms.append(atom)
            except:
                continue

        return self._cluster_atoms_to_sites(hydrophobic_atoms, 'hydrophobic')

    def _find_hbond_donors(self, atoms: List) -> List[Dict[str, Any]]:
        """Find hydrogen bond donor sites"""

        donor_atoms = []
        for atom in atoms:
            try:
                if atom.element in ['N', 'O'] and atom.get_name() in ['NE2', 'NH1', 'NH2', 'ND1', 'NE1', 'OG', 'OH']:
                    donor_atoms.append(atom)
            except:
                continue

        return self._cluster_atoms_to_sites(donor_atoms, 'donor')

    def _find_hbond_acceptors(self, atoms: List) -> List[Dict[str, Any]]:
        """Find hydrogen bond acceptor sites"""

        acceptor_atoms = []
        for atom in atoms:
            try:
                if atom.element in ['O', 'N'] and atom.get_name() in ['OD1', 'OD2', 'OE1', 'OE2', 'ND1', 'NE2']:
                    acceptor_atoms.append(atom)
            except:
                continue

        return self._cluster_atoms_to_sites(acceptor_atoms, 'acceptor')

    def _find_aromatic_regions(self, atoms: List) -> List[Dict[str, Any]]:
        """Find aromatic regions in protein"""

        aromatic_residues = {'PHE', 'TYR', 'TRP', 'HIS'}
        aromatic_atoms = []

        for atom in atoms:
            try:
                residue = atom.get_parent()
                if residue.get_resname() in aromatic_residues:
                    if atom.get_name() in ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'CH2', 'CZ2', 'CZ3']:
                        aromatic_atoms.append(atom)
            except:
                continue

        return self._cluster_atoms_to_sites(aromatic_atoms, 'aromatic')

    def _find_positive_regions(self, atoms: List) -> List[Dict[str, Any]]:
        """Find positively charged regions"""

        positive_atoms = []
        for atom in atoms:
            try:
                residue = atom.get_parent()
                if residue.get_resname() in ['ARG', 'LYS', 'HIS'] and atom.element == 'N':
                    positive_atoms.append(atom)
            except:
                continue

        return self._cluster_atoms_to_sites(positive_atoms, 'positive')

    def _find_negative_regions(self, atoms: List) -> List[Dict[str, Any]]:
        """Find negatively charged regions"""

        negative_atoms = []
        for atom in atoms:
            try:
                residue = atom.get_parent()
                if residue.get_resname() in ['ASP', 'GLU'] and atom.element == 'O':
                    negative_atoms.append(atom)
            except:
                continue

        return self._cluster_atoms_to_sites(negative_atoms, 'negative')

    def _find_metal_binding_sites(self, atoms: List) -> List[Dict[str, Any]]:
        """Find potential metal binding sites"""

        metal_binding_atoms = []
        for atom in atoms:
            try:
                if atom.element in ['N', 'O', 'S'] and atom.get_name() in ['NE2', 'ND1', 'OD1', 'OD2', 'OE1', 'OE2', 'SG']:
                    metal_binding_atoms.append(atom)
            except:
                continue

        return self._cluster_atoms_to_sites(metal_binding_atoms, 'metal')

    def _cluster_atoms_to_sites(self, atoms: List, feature_type: str) -> List[Dict[str, Any]]:
        """Cluster atoms into sites based on spatial proximity"""

        if not atoms:
            return []

        coords = np.array([atom.get_coord() for atom in atoms])

        # Simple clustering based on distance
        sites = []
        used_atoms = set()

        for i, atom in enumerate(atoms):
            if atom in used_atoms:
                continue

            # Find nearby atoms
            distances = np.linalg.norm(coords - coords[i], axis=1)
            nearby_indices = np.where(distances <= 8.0)[0]  # 8 Angstrom cutoff

            if len(nearby_indices) >= 2:  # Minimum cluster size
                cluster_atoms = [atoms[j] for j in nearby_indices]
                cluster_coords = coords[nearby_indices]

                center = np.mean(cluster_coords, axis=0)
                site = {
                    'center': center,
                    'atoms': cluster_atoms,
                    'feature_type': feature_type,
                    'size': len(cluster_atoms)
                }
                sites.append(site)

                # Mark atoms as used
                for j in nearby_indices:
                    used_atoms.add(atoms[j])

        return sites

    def _combine_pharmacophore_sites(self, feature_sites: Dict[str, List], required_features: List[str]) -> List[Dict[str, Any]]:
        """Combine individual feature sites into pharmacophore-matching sites"""

        combined_sites = []

        # For each combination of feature sites, check if they're close enough to form a binding site
        if len(required_features) == 1:
            # Single feature pharmacophore
            feature = required_features[0]
            for site in feature_sites.get(feature, []):
                dimensions = np.array([15.0, 15.0, 15.0])  # Default size
                combined_sites.append({
                    'center': site['center'],
                    'dimensions': dimensions,
                    'matched_features': [feature],
                    'feature_density': 1.0,
                    'score': 0.7
                })

        else:
            # Multi-feature pharmacophore
            # This is a simplified approach - in practice, you'd want more sophisticated matching
            for feature1 in required_features:
                for site1 in feature_sites.get(feature1, []):
                    matched_features = [feature1]
                    all_coords = [site1['center']]

                    for feature2 in required_features:
                        if feature2 == feature1:
                            continue

                        for site2 in feature_sites.get(feature2, []):
                            distance = np.linalg.norm(site1['center'] - site2['center'])
                            if 5.0 <= distance <= 20.0:  # Reasonable pharmacophore distance
                                matched_features.append(feature2)
                                all_coords.append(site2['center'])
                                break

                    if len(matched_features) >= max(2, len(required_features) // 2):
                        center = np.mean(all_coords, axis=0)
                        coords_array = np.array(all_coords)
                        min_coords = np.min(coords_array, axis=0)
                        max_coords = np.max(coords_array, axis=0)
                        dimensions = max_coords - min_coords + 10.0  # Add padding

                        score = len(matched_features) / len(required_features)

                        combined_sites.append({
                            'center': center,
                            'dimensions': dimensions,
                            'matched_features': matched_features,
                            'feature_density': len(matched_features) / np.prod(dimensions) * 1000,
                            'score': score
                        })

        # Remove duplicate sites (those with centers too close together)
        filtered_sites = []
        for site in combined_sites:
            is_duplicate = False
            for existing_site in filtered_sites:
                distance = np.linalg.norm(np.array(site['center']) - np.array(existing_site['center']))
                if distance < 10.0:  # Sites closer than 10 Å are considered duplicates
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered_sites.append(site)

        # Sort by score
        filtered_sites.sort(key=lambda x: x['score'], reverse=True)

        return filtered_sites