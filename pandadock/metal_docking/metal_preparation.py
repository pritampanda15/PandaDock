"""
Metal Ligand and Metalloprotein Preparation

Handles preparation of metal-containing ligands and metalloproteins
following Glide protocols for zero-order bonds and coordination geometry.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from Bio.PDB import PDBParser, Structure, Atom
import logging

class MetalLigandPreparator:
    """Prepares metal-containing ligands for docking"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Common metal elements
        self.metal_elements = {
            'Li', 'Na', 'K', 'Mg', 'Ca', 'Ba', 'Al', 'Ti', 'V', 'Cr', 'Mn',
            'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Mo', 'Pd', 'Ag', 'Cd', 'Pt',
            'Au', 'Hg', 'Pb', 'La', 'Gd'
        }

        # Typical coordination numbers for metals
        self.typical_coordination = {
            'Li': [4, 6], 'Na': [4, 6], 'K': [6, 8], 'Mg': [4, 6], 'Ca': [6, 8],
            'Ba': [8, 12], 'Al': [4, 6], 'Ti': [6], 'V': [6], 'Cr': [6],
            'Mn': [4, 6], 'Fe': [4, 6], 'Co': [4, 6], 'Ni': [4, 6], 'Cu': [4, 6],
            'Zn': [4, 6], 'Mo': [6], 'Pd': [4], 'Ag': [2, 4], 'Cd': [4, 6],
            'Pt': [4, 6], 'Au': [2, 4], 'Hg': [2, 4], 'Pb': [6, 8],
            'La': [8, 9], 'Gd': [8, 9]
        }

    def identify_metal_atoms(self, mol: Chem.Mol) -> List[int]:
        """
        Identify metal atoms in a molecule

        Args:
            mol: RDKit molecule

        Returns:
            List of metal atom indices
        """
        metal_indices = []

        for atom in mol.GetAtoms():
            if atom.GetSymbol() in self.metal_elements:
                metal_indices.append(atom.GetIdx())

        return metal_indices

    def identify_coordinating_atoms(self, mol: Chem.Mol, metal_idx: int,
                                  max_distance: float = 3.0) -> List[int]:
        """
        Identify atoms coordinating to a metal center

        Args:
            mol: RDKit molecule
            metal_idx: Index of metal atom
            max_distance: Maximum coordination distance in Angstroms

        Returns:
            List of coordinating atom indices
        """
        if not mol.GetNumConformers():
            # Generate 3D coordinates if not present
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)

        coordinating_atoms = []
        conf = mol.GetConformer()
        metal_pos = conf.GetAtomPosition(metal_idx)

        for atom in mol.GetAtoms():
            if atom.GetIdx() == metal_idx:
                continue

            atom_pos = conf.GetAtomPosition(atom.GetIdx())
            distance = metal_pos.Distance(atom_pos)

            # Check for potential coordinating atoms (N, O, S, P, halogens)
            if (distance <= max_distance and
                atom.GetSymbol() in ['N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']):
                coordinating_atoms.append(atom.GetIdx())

        return coordinating_atoms

    def convert_to_zero_order_bonds(self, mol: Chem.Mol, metal_idx: int,
                                   coordinating_atoms: List[int]) -> Chem.Mol:
        """
        Convert covalent metal-ligand bonds to zero-order bonds

        Args:
            mol: RDKit molecule
            metal_idx: Index of metal atom
            coordinating_atoms: List of coordinating atom indices

        Returns:
            Modified molecule with zero-order bonds
        """
        # Create editable molecule
        editable_mol = Chem.EditableMol(mol)

        # Remove existing bonds to metal
        bonds_to_remove = []
        for bond in mol.GetBonds():
            if (bond.GetBeginAtomIdx() == metal_idx and
                bond.GetEndAtomIdx() in coordinating_atoms) or \
               (bond.GetEndAtomIdx() == metal_idx and
                bond.GetBeginAtomIdx() in coordinating_atoms):
                bonds_to_remove.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))

        # Remove bonds (in reverse order to maintain indices)
        for begin_idx, end_idx in reversed(bonds_to_remove):
            editable_mol.RemoveBond(begin_idx, end_idx)

        # Add zero-order bonds
        for coord_idx in coordinating_atoms:
            editable_mol.AddBond(metal_idx, coord_idx, Chem.BondType.ZERO)

        try:
            new_mol = editable_mol.GetMol()
            Chem.SanitizeMol(new_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                           Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            return new_mol
        except:
            self.logger.warning("Could not create zero-order bonds, returning original molecule")
            return mol

    def adjust_formal_charges(self, mol: Chem.Mol, metal_idx: int,
                            coordinating_atoms: List[int]) -> Chem.Mol:
        """
        Adjust formal charges on metal and coordinating atoms

        Args:
            mol: RDKit molecule
            metal_idx: Index of metal atom
            coordinating_atoms: List of coordinating atom indices

        Returns:
            Molecule with adjusted formal charges
        """
        editable_mol = Chem.EditableMol(mol)
        new_mol = editable_mol.GetMol()

        metal_atom = new_mol.GetAtomWithIdx(metal_idx)
        metal_symbol = metal_atom.GetSymbol()

        # Set typical formal charges for common metals
        typical_charges = {
            'Li': 1, 'Na': 1, 'K': 1,
            'Mg': 2, 'Ca': 2, 'Ba': 2,
            'Al': 3, 'Zn': 2, 'Cu': 2, 'Fe': 2,
            'Ni': 2, 'Co': 2, 'Mn': 2
        }

        if metal_symbol in typical_charges:
            metal_atom.SetFormalCharge(typical_charges[metal_symbol])

        # Adjust coordinating atom charges to balance overall charge
        for coord_idx in coordinating_atoms:
            coord_atom = new_mol.GetAtomWithIdx(coord_idx)
            if coord_atom.GetSymbol() in ['O', 'S']:
                # Negative charge on oxygen/sulfur when coordinating
                coord_atom.SetFormalCharge(-1)

        return new_mol

    def validate_coordination_geometry(self, mol: Chem.Mol, metal_idx: int,
                                     coordinating_atoms: List[int]) -> bool:
        """
        Validate that coordination geometry is reasonable

        Args:
            mol: RDKit molecule
            metal_idx: Index of metal atom
            coordinating_atoms: List of coordinating atom indices

        Returns:
            True if geometry is reasonable
        """
        if not mol.GetNumConformers():
            return False

        metal_symbol = mol.GetAtomWithIdx(metal_idx).GetSymbol()
        coordination_number = len(coordinating_atoms)

        # Check if coordination number is typical for this metal
        if metal_symbol in self.typical_coordination:
            if coordination_number not in self.typical_coordination[metal_symbol]:
                self.logger.warning(f"Unusual coordination number {coordination_number} for {metal_symbol}")

        # Check bond angles and distances
        conf = mol.GetConformer()
        metal_pos = conf.GetAtomPosition(metal_idx)

        # Validate distances
        for coord_idx in coordinating_atoms:
            coord_pos = conf.GetAtomPosition(coord_idx)
            distance = metal_pos.Distance(coord_pos)

            if distance < 1.5 or distance > 3.5:  # Reasonable coordination distance range
                self.logger.warning(f"Unusual coordination distance: {distance:.2f} Å")
                return False

        return True

    def prepare_metal_ligand(self, mol: Chem.Mol, validate_geometry: bool = True) -> Chem.Mol:
        """
        Prepare metal-containing ligand following Glide protocol

        Args:
            mol: RDKit molecule containing metal
            validate_geometry: Whether to validate coordination geometry

        Returns:
            Prepared molecule with zero-order bonds and adjusted charges
        """
        metal_indices = self.identify_metal_atoms(mol)

        if not metal_indices:
            self.logger.info("No metal atoms found in ligand")
            return mol

        if len(metal_indices) > 1:
            self.logger.warning(f"Multiple metals found ({len(metal_indices)}), treating as single molecule")

        prepared_mol = mol

        for metal_idx in metal_indices:
            self.logger.info(f"Processing metal atom {metal_idx}: {mol.GetAtomWithIdx(metal_idx).GetSymbol()}")

            # Identify coordinating atoms
            coordinating_atoms = self.identify_coordinating_atoms(prepared_mol, metal_idx)

            if not coordinating_atoms:
                self.logger.warning(f"No coordinating atoms found for metal {metal_idx}")
                continue

            self.logger.info(f"Found {len(coordinating_atoms)} coordinating atoms")

            # Validate geometry if requested
            if validate_geometry:
                if not self.validate_coordination_geometry(prepared_mol, metal_idx, coordinating_atoms):
                    self.logger.warning("Coordination geometry validation failed")

            # Convert to zero-order bonds
            prepared_mol = self.convert_to_zero_order_bonds(prepared_mol, metal_idx, coordinating_atoms)

            # Adjust formal charges
            prepared_mol = self.adjust_formal_charges(prepared_mol, metal_idx, coordinating_atoms)

        return prepared_mol


class MetalloproteinPreparator:
    """Prepares metalloproteins for metal-aware docking"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Common metal-coordinating residues
        self.coordinating_residues = {
            'HIS': ['ND1', 'NE2'],     # Histidine
            'CYS': ['SG'],             # Cysteine
            'MET': ['SD'],             # Methionine
            'ASP': ['OD1', 'OD2'],     # Aspartic acid
            'GLU': ['OE1', 'OE2'],     # Glutamic acid
            'TYR': ['OH'],             # Tyrosine
            'SER': ['OG'],             # Serine
            'THR': ['OG1'],            # Threonine
            'ASN': ['OD1'],            # Asparagine
            'GLN': ['OE1'],            # Glutamine
        }

    def identify_metal_sites(self, structure: Structure.Structure,
                           metal_distance_cutoff: float = 3.0) -> List[Dict]:
        """
        Identify metal binding sites in protein structure

        Args:
            structure: BioPython structure object
            metal_distance_cutoff: Maximum distance to consider coordination

        Returns:
            List of metal site dictionaries
        """
        metal_sites = []

        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        if atom.element in ['MG', 'ZN', 'FE', 'CA', 'MN', 'CU', 'NI', 'CO']:
                            # Found metal atom
                            metal_info = {
                                'atom': atom,
                                'residue': residue,
                                'chain': chain,
                                'element': atom.element,
                                'coordinating_atoms': []
                            }

                            # Find coordinating atoms
                            for other_model in structure:
                                for other_chain in other_model:
                                    for other_residue in other_chain:
                                        if other_residue == residue:
                                            continue

                                        residue_name = other_residue.get_resname()
                                        if residue_name in self.coordinating_residues:
                                            for coord_atom_name in self.coordinating_residues[residue_name]:
                                                if coord_atom_name in other_residue:
                                                    coord_atom = other_residue[coord_atom_name]
                                                    distance = atom - coord_atom

                                                    if distance <= metal_distance_cutoff:
                                                        metal_info['coordinating_atoms'].append({
                                                            'atom': coord_atom,
                                                            'residue': other_residue,
                                                            'distance': distance
                                                        })

                            metal_sites.append(metal_info)

        return metal_sites

    def prepare_metalloprotein(self, pdb_file: str) -> Dict:
        """
        Prepare metalloprotein structure for metal-aware docking

        Args:
            pdb_file: Path to PDB file

        Returns:
            Dictionary with metal site information
        """
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_file)

        metal_sites = self.identify_metal_sites(structure)

        if not metal_sites:
            self.logger.info("No metal sites identified in protein")
            return {'metal_sites': []}

        self.logger.info(f"Identified {len(metal_sites)} metal sites")

        # Analyze each metal site
        prepared_sites = []
        for site in metal_sites:
            site_info = {
                'metal_element': site['element'],
                'metal_coords': site['atom'].get_coord(),
                'coordination_number': len(site['coordinating_atoms']),
                'coordinating_residues': [],
                'net_charge': self._calculate_site_charge(site)
            }

            for coord_info in site['coordinating_atoms']:
                site_info['coordinating_residues'].append({
                    'residue_name': coord_info['residue'].get_resname(),
                    'residue_id': coord_info['residue'].get_id(),
                    'atom_name': coord_info['atom'].get_name(),
                    'distance': coord_info['distance']
                })

            prepared_sites.append(site_info)

        return {
            'metal_sites': prepared_sites,
            'structure_file': pdb_file
        }

    def _calculate_site_charge(self, site: Dict) -> float:
        """Calculate net charge at metal site"""
        # Simplified charge calculation
        metal_charges = {'MG': 2, 'ZN': 2, 'FE': 2, 'CA': 2, 'MN': 2, 'CU': 2, 'NI': 2, 'CO': 2}

        net_charge = metal_charges.get(site['element'], 2)  # Default +2

        # Subtract coordinating atom contributions
        for coord_info in site['coordinating_atoms']:
            residue_name = coord_info['residue'].get_resname()
            if residue_name in ['ASP', 'GLU']:
                net_charge -= 1  # Negative charge contribution
            elif residue_name == 'CYS':
                net_charge -= 0.5  # Partial negative charge

        return net_charge