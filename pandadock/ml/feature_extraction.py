"""
Feature extraction for ML-enhanced docking

Extracts protein, ligand, and interaction features for training ML models.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from Bio.PDB import PDBParser, NeighborSearch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.rdMolAlign import AlignMol
import torch

class ProteinFeatureExtractor:
    """Extract features from protein binding sites"""

    def __init__(self, grid_resolution: float = 1.0, pocket_radius: float = 10.0):
        self.grid_resolution = grid_resolution
        self.pocket_radius = pocket_radius
        self.logger = logging.getLogger(__name__)

        # Amino acid properties
        self.aa_properties = {
            'ALA': [0, 0, 0, 0],  # [hydrophobic, aromatic, positive, negative]
            'ARG': [0, 0, 1, 0],
            'ASN': [0, 0, 0, 0],
            'ASP': [0, 0, 0, 1],
            'CYS': [0, 0, 0, 0],
            'GLN': [0, 0, 0, 0],
            'GLU': [0, 0, 0, 1],
            'GLY': [0, 0, 0, 0],
            'HIS': [0, 1, 1, 0],
            'ILE': [1, 0, 0, 0],
            'LEU': [1, 0, 0, 0],
            'LYS': [0, 0, 1, 0],
            'MET': [1, 0, 0, 0],
            'PHE': [1, 1, 0, 0],
            'PRO': [0, 0, 0, 0],
            'SER': [0, 0, 0, 0],
            'THR': [0, 0, 0, 0],
            'TRP': [1, 1, 0, 0],
            'TYR': [1, 1, 0, 0],
            'VAL': [1, 0, 0, 0]
        }

    def extract_from_complex(self, pdb_file: str, ligand_center: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Extract protein features from a PDB complex"""
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("protein", pdb_file)

            # If ligand center not provided, try to detect it
            if ligand_center is None:
                ligand_center = self._detect_ligand_center(structure)

            # Extract pocket features
            pocket_features = self._extract_pocket_features(structure, ligand_center)

            # Extract volumetric features
            volumetric_features = self._extract_volumetric_features(structure, ligand_center)

            # Extract sequence features
            sequence_features = self._extract_sequence_features(structure, ligand_center)

            return {
                'pocket_features': pocket_features,
                'volumetric_features': volumetric_features,
                'sequence_features': sequence_features,
                'pocket_center': ligand_center
            }

        except Exception as e:
            self.logger.error(f"Error extracting protein features from {pdb_file}: {e}")
            raise

    def _detect_ligand_center(self, structure) -> np.ndarray:
        """Detect ligand center from HETATM records"""
        ligand_coords = []

        for model in structure:
            for chain in model:
                for residue in chain:
                    # Skip standard amino acids and water
                    if residue.get_resname() in self.aa_properties or residue.get_resname() == 'HOH':
                        continue

                    for atom in residue:
                        if atom.get_name().startswith('H'):  # Skip hydrogens
                            continue
                        ligand_coords.append(atom.get_coord())

        if ligand_coords:
            return np.mean(ligand_coords, axis=0)
        else:
            # Fallback: use protein center
            protein_coords = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.get_resname() in self.aa_properties:
                            for atom in residue:
                                protein_coords.append(atom.get_coord())
            return np.mean(protein_coords, axis=0)

    def _extract_pocket_features(self, structure, center: np.ndarray) -> np.ndarray:
        """Extract pocket-specific features"""
        features = []

        # Find residues within pocket radius
        pocket_residues = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.get_resname() not in self.aa_properties:
                        continue

                    # Get CA atom distance to center
                    if 'CA' in residue:
                        ca_coord = residue['CA'].get_coord()
                        distance = np.linalg.norm(ca_coord - center)

                        if distance <= self.pocket_radius:
                            pocket_residues.append((residue, distance))

        # Sort by distance
        pocket_residues.sort(key=lambda x: x[1])

        # Extract features for top 20 closest residues
        max_residues = 20
        for i in range(max_residues):
            if i < len(pocket_residues):
                residue, distance = pocket_residues[i]
                resname = residue.get_resname()

                # Residue type features
                res_features = self.aa_properties.get(resname, [0, 0, 0, 0])

                # Distance feature (normalized)
                dist_feature = [distance / self.pocket_radius]

                # Combine features
                features.extend(res_features + dist_feature)
            else:
                # Padding for missing residues
                features.extend([0, 0, 0, 0, 1.0])  # Distance = 1.0 for padding

        return np.array(features, dtype=np.float32)

    def _extract_volumetric_features(self, structure, center: np.ndarray) -> np.ndarray:
        """Extract 3D volumetric features"""
        # Define grid dimensions
        grid_size = int(2 * self.pocket_radius / self.grid_resolution)
        grid = np.zeros((grid_size, grid_size, grid_size, 4), dtype=np.float32)

        # Grid origin
        origin = center - self.pocket_radius

        # Fill grid with atom features
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.get_resname() not in self.aa_properties:
                        continue

                    for atom in residue:
                        coord = atom.get_coord()

                        # Convert to grid coordinates
                        grid_coord = ((coord - origin) / self.grid_resolution).astype(int)

                        # Check bounds
                        if np.all(grid_coord >= 0) and np.all(grid_coord < grid_size):
                            x, y, z = grid_coord

                            # Atom type features
                            element = atom.get_name()[0]
                            if element == 'C':
                                grid[x, y, z, 0] = 1.0  # Carbon
                            elif element == 'N':
                                grid[x, y, z, 1] = 1.0  # Nitrogen
                            elif element == 'O':
                                grid[x, y, z, 2] = 1.0  # Oxygen
                            elif element == 'S':
                                grid[x, y, z, 3] = 1.0  # Sulfur

        return grid

    def _extract_sequence_features(self, structure, center: np.ndarray) -> np.ndarray:
        """Extract sequence-based features"""
        # Get pocket sequence
        pocket_residues = []

        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.get_resname() not in self.aa_properties:
                        continue

                    if 'CA' in residue:
                        ca_coord = residue['CA'].get_coord()
                        distance = np.linalg.norm(ca_coord - center)

                        if distance <= self.pocket_radius:
                            pocket_residues.append((residue.get_resname(), distance))

        # Sort by distance and create sequence
        pocket_residues.sort(key=lambda x: x[1])
        sequence = [res[0] for res in pocket_residues[:50]]  # Max 50 residues

        # Convert to features
        sequence_features = []
        for i in range(50):  # Fixed length
            if i < len(sequence):
                resname = sequence[i]
                sequence_features.extend(self.aa_properties.get(resname, [0, 0, 0, 0]))
            else:
                sequence_features.extend([0, 0, 0, 0])  # Padding

        return np.array(sequence_features, dtype=np.float32)


class LigandFeatureExtractor:
    """Extract features from ligand molecules"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def extract_from_mol(self, mol: Chem.Mol) -> Dict[str, np.ndarray]:
        """Extract ligand features from RDKit molecule"""
        try:
            # Molecular descriptors
            descriptors = self._extract_descriptors(mol)

            # Fingerprints
            fingerprints = self._extract_fingerprints(mol)

            # 3D conformer features
            conformer_features = self._extract_conformer_features(mol)

            # Atom features
            atom_features = self._extract_atom_features(mol)

            return {
                'descriptors': descriptors,
                'fingerprints': fingerprints,
                'conformer_features': conformer_features,
                'atom_features': atom_features
            }

        except Exception as e:
            self.logger.error(f"Error extracting ligand features: {e}")
            raise

    def extract_from_complex(self, pdb_file: str) -> Dict[str, np.ndarray]:
        """Extract ligand features from PDB complex"""
        try:
            # Try to extract ligand from PDB
            mol = self._extract_ligand_from_pdb(pdb_file)
            if mol is not None:
                return self.extract_from_mol(mol)
            else:
                # Return empty features
                return {
                    'descriptors': np.zeros(50, dtype=np.float32),
                    'fingerprints': np.zeros(2048, dtype=np.float32),
                    'conformer_features': np.zeros(20, dtype=np.float32),
                    'atom_features': np.zeros((50, 8), dtype=np.float32)
                }

        except Exception as e:
            self.logger.error(f"Error extracting ligand from {pdb_file}: {e}")
            raise

    def _extract_ligand_from_pdb(self, pdb_file: str) -> Optional[Chem.Mol]:
        """Extract ligand molecule from PDB file"""
        try:
            # Read PDB and find ligand residues
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("complex", pdb_file)

            # Standard amino acids to exclude
            standard_residues = set(self.aa_properties.keys()) if hasattr(self, 'aa_properties') else {
                'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
            }

            for model in structure:
                for chain in model:
                    for residue in chain:
                        resname = residue.get_resname()

                        # Skip standard amino acids and water
                        if resname in standard_residues or resname in ['HOH', 'WAT']:
                            continue

                        # Try to convert residue to RDKit molecule
                        # This is a simplified approach - in practice, you might need
                        # more sophisticated ligand extraction
                        mol = Chem.MolFromPDBBlock(str(residue))
                        if mol is not None:
                            return mol

            return None

        except Exception as e:
            self.logger.error(f"Error extracting ligand from PDB: {e}")
            return None

    def _extract_descriptors(self, mol: Chem.Mol) -> np.ndarray:
        """Extract molecular descriptors"""
        descriptors = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumSaturatedRings(mol),
            Descriptors.NumHeteroatoms(mol),
            Descriptors.BertzCT(mol),
            rdMolDescriptors.CalcNumLipinskiHBD(mol),
            rdMolDescriptors.CalcNumLipinskiHBA(mol),
            rdMolDescriptors.CalcNumRings(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcNumAliphaticRings(mol)
        ]

        # Pad to fixed size
        while len(descriptors) < 50:
            descriptors.append(0.0)

        return np.array(descriptors[:50], dtype=np.float32)

    def _extract_fingerprints(self, mol: Chem.Mol) -> np.ndarray:
        """Extract molecular fingerprints"""
        # Morgan fingerprint (ECFP4)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        return np.array(fp, dtype=np.float32)

    def _extract_conformer_features(self, mol: Chem.Mol) -> np.ndarray:
        """Extract 3D conformer features"""
        features = []

        if mol.GetNumConformers() > 0:
            conf = mol.GetConformer(0)

            # Calculate basic 3D properties
            coords = []
            for i in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                coords.append([pos.x, pos.y, pos.z])

            coords = np.array(coords)

            # Center of mass
            com = np.mean(coords, axis=0)

            # Principal moments of inertia
            centered_coords = coords - com
            inertia_tensor = np.dot(centered_coords.T, centered_coords)
            eigenvals = np.linalg.eigvals(inertia_tensor)
            eigenvals.sort()

            # Radius of gyration
            rg = np.sqrt(np.mean(np.sum(centered_coords**2, axis=1)))

            # Asphericity
            asphericity = eigenvals[2] - 0.5 * (eigenvals[0] + eigenvals[1])

            features = [
                com[0], com[1], com[2],  # Center of mass
                eigenvals[0], eigenvals[1], eigenvals[2],  # Principal moments
                rg,  # Radius of gyration
                asphericity,  # Asphericity
                np.max(coords[:, 0]) - np.min(coords[:, 0]),  # X span
                np.max(coords[:, 1]) - np.min(coords[:, 1]),  # Y span
                np.max(coords[:, 2]) - np.min(coords[:, 2])   # Z span
            ]
        else:
            features = [0.0] * 11

        # Pad to fixed size
        while len(features) < 20:
            features.append(0.0)

        return np.array(features[:20], dtype=np.float32)

    def _extract_atom_features(self, mol: Chem.Mol) -> np.ndarray:
        """Extract per-atom features"""
        max_atoms = 50
        atom_features = np.zeros((max_atoms, 8), dtype=np.float32)

        for i, atom in enumerate(mol.GetAtoms()):
            if i >= max_atoms:
                break

            # Atomic number (one-hot for common elements)
            atomic_num = atom.GetAtomicNum()
            atom_type = [0, 0, 0, 0, 0]  # C, N, O, S, Other
            if atomic_num == 6:
                atom_type[0] = 1  # Carbon
            elif atomic_num == 7:
                atom_type[1] = 1  # Nitrogen
            elif atomic_num == 8:
                atom_type[2] = 1  # Oxygen
            elif atomic_num == 16:
                atom_type[3] = 1  # Sulfur
            else:
                atom_type[4] = 1  # Other

            # Other features
            features = atom_type + [
                atom.GetDegree(),
                atom.GetTotalNumHs(),
                int(atom.GetIsAromatic())
            ]

            atom_features[i] = features[:8]

        return atom_features


class ComplexFeatureExtractor:
    """Extract interaction features from protein-ligand complexes"""

    def __init__(self, grid_resolution: float = 1.0):
        self.grid_resolution = grid_resolution
        self.protein_extractor = ProteinFeatureExtractor(grid_resolution=grid_resolution)
        self.ligand_extractor = LigandFeatureExtractor()
        self.logger = logging.getLogger(__name__)

    def extract_from_complex(self, pdb_file: str) -> Dict[str, np.ndarray]:
        """Extract interaction features from protein-ligand complex"""
        try:
            # Extract protein and ligand features separately
            protein_features = self.protein_extractor.extract_from_complex(pdb_file)
            ligand_features = self.ligand_extractor.extract_from_complex(pdb_file)

            # Extract interaction-specific features
            interaction_features = self._extract_interaction_features(pdb_file, protein_features['pocket_center'])

            return {
                'protein': protein_features,
                'ligand': ligand_features,
                'interactions': interaction_features
            }

        except Exception as e:
            self.logger.error(f"Error extracting complex features from {pdb_file}: {e}")
            raise

    def _extract_interaction_features(self, pdb_file: str, pocket_center: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract protein-ligand interaction features"""
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("complex", pdb_file)

            # Separate protein and ligand atoms
            protein_atoms = []
            ligand_atoms = []

            for model in structure:
                for chain in model:
                    for residue in chain:
                        resname = residue.get_resname()

                        for atom in residue:
                            coord = atom.get_coord()
                            element = atom.get_name()[0]

                            if resname in self.protein_extractor.aa_properties:
                                protein_atoms.append((coord, element, atom))
                            elif resname not in ['HOH', 'WAT']:
                                ligand_atoms.append((coord, element, atom))

            # Calculate interaction features
            interaction_features = self._calculate_interactions(protein_atoms, ligand_atoms, pocket_center)

            return interaction_features

        except Exception as e:
            self.logger.error(f"Error extracting interactions: {e}")
            return {'contact_map': np.zeros((20, 20), dtype=np.float32)}

    def _calculate_interactions(self, protein_atoms: List, ligand_atoms: List, center: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate various interaction features"""
        features = {}

        # Contact map (simplified)
        contact_map = np.zeros((20, 20), dtype=np.float32)

        # Find close contacts
        contacts = []
        for i, (p_coord, p_elem, p_atom) in enumerate(protein_atoms[:20]):
            for j, (l_coord, l_elem, l_atom) in enumerate(ligand_atoms[:20]):
                distance = np.linalg.norm(p_coord - l_coord)

                if distance < 5.0:  # Contact threshold
                    contact_map[i, j] = 1.0 / (1.0 + distance)  # Distance-weighted contact
                    contacts.append((distance, p_elem, l_elem))

        features['contact_map'] = contact_map

        # Interaction type counts
        interaction_counts = {
            'hydrophobic': 0,
            'polar': 0,
            'charged': 0,
            'aromatic': 0
        }

        for distance, p_elem, l_elem in contacts:
            # Simplified interaction classification
            if p_elem == 'C' and l_elem == 'C':
                interaction_counts['hydrophobic'] += 1
            elif p_elem in ['N', 'O'] or l_elem in ['N', 'O']:
                interaction_counts['polar'] += 1

        features['interaction_counts'] = np.array(list(interaction_counts.values()), dtype=np.float32)

        return features