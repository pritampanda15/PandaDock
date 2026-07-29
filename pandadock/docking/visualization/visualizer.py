"""
PDB File Generation and Molecular Visualization

Creates professional PDB outputs:
- Complex PDB files (protein + ligand)
- Individual pose PDB files
- Proper atom naming and formatting
"""

import numpy as np
from pathlib import Path
from typing import Optional
import logging
from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain, Residue, Atom
from ..core import Pose


class LigandHETATMPDBIO(PDBIO):
    """Custom PDB writer that outputs ligands as HETATM records"""

    def __init__(self, ligand_chain_ids=None, ligand_residue_names=None):
        super().__init__()
        self.ligand_chain_ids = set(ligand_chain_ids or [])
        self.ligand_residue_names = set(ligand_residue_names or [])

    def _get_atom_line(self, atom, hetfield, segid, atom_number, resname, resseq, icode, chain_id, element="  "):
        """Override to use HETATM for ligand atoms"""
        coord = atom.get_coord()
        bfactor = atom.get_bfactor()
        occupancy = atom.get_occupancy()

        # Determine if this should be HETATM
        is_ligand = (chain_id in self.ligand_chain_ids or
                    resname in self.ligand_residue_names)
        record_type = "HETATM" if is_ligand else "ATOM  "

        atom_name = atom.get_name()
        if len(atom_name) < 4:
            atom_name = f" {atom_name}"

        # Extract proper element symbol from atom
        if hasattr(atom, 'element') and atom.element.strip() and atom.element.strip() != 'X':
            element_symbol = atom.element.strip()
        else:
            # Extract from atom name (handle names like 'C01', 'N03', 'O12', etc.)
            clean_name = atom_name.strip()
            if clean_name:
                # Extract the first character, which should be the element
                first_char = clean_name[0]
                if first_char.isalpha():
                    element_symbol = first_char
                else:
                    element_symbol = 'C'  # Default fallback
            else:
                element_symbol = 'C'

        line = f"{record_type}{atom_number:5d} {atom_name:4s} {resname:3s} {chain_id:1s}{resseq:4d}{icode:1s}   {coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}{occupancy:6.2f}{bfactor:6.2f}          {element_symbol:>2s}  \n"
        return line




class DockingVisualizer:
    """Professional molecular visualization and PDB generation"""

    def __init__(self):
        self.logger = logging.getLogger("pandadock.visualization.visualizer")

    @staticmethod
    def _ligand_element_and_name(rdkit_atom, index):
        """
        PDB element symbol and atom name for a ligand atom.

        BioPython asserts that element symbols are upper case, matching the PDB
        specification. RDKit returns them capitalised ("Br", "Cl"), so passing
        the symbol through unchanged raised AssertionError for every two-letter
        element and no PDB file was written at all. Chlorine and bromine are
        common in drug-like ligands, so this silently affected a large share of
        real runs.
        """
        element = rdkit_atom.GetSymbol().upper() if rdkit_atom is not None else "C"
        name = f"{element}{index + 1}"
        return element, name[:4]

    def _build_ligand_residue(self, coords, ligand_mol, residue_name="LIG"):
        """Build a BioPython residue holding the ligand pose."""
        residue = Residue.Residue((" ", 1, " "), residue_name, " ")

        for i, coord in enumerate(coords):
            rdkit_atom = None
            if ligand_mol is not None and i < ligand_mol.GetNumAtoms():
                rdkit_atom = ligand_mol.GetAtomWithIdx(i)
            element, atom_name = self._ligand_element_and_name(rdkit_atom, i)

            residue.add(
                Atom.Atom(
                    name=atom_name,
                    coord=coord,
                    bfactor=20.0,
                    occupancy=1.0,
                    altloc=" ",
                    fullname=atom_name.ljust(4),
                    serial_number=i + 1,
                    element=element,
                )
            )
        return residue

    def save_complex_pdb(self, receptor_file: str, pose: Pose, output_file: Path, ligand_mol=None):
        """
        Save protein-ligand complex as PDB file with proper atom types and hydrogens

        Args:
            receptor_file: Original receptor PDB file
            pose: Ligand pose to include
            output_file: Output PDB file path
            ligand_mol: RDKit molecule for proper atom types
        """
        try:
            # Load receptor structure
            parser = PDBParser(QUIET=True)
            receptor_structure = parser.get_structure('receptor', receptor_file)

            # Create new structure for complex
            complex_structure = Structure.Structure('complex')
            complex_model = Model.Model(0)
            complex_structure.add(complex_model)

            # Add receptor chains
            for chain in receptor_structure.get_chains():
                complex_model.add(chain.copy())

            # Add ligand as new chain
            ligand_chain = Chain.Chain('L')
            ligand_residue = self._build_ligand_residue(pose.coordinates, ligand_mol)

            ligand_chain.add(ligand_residue)
            complex_model.add(ligand_chain)

            # Save PDB with ligands as HETATM records
            io = LigandHETATMPDBIO(ligand_chain_ids=['L'], ligand_residue_names=['LIG'])
            io.set_structure(complex_structure)
            io.save(str(output_file))

            self.logger.debug(f"Complex PDB saved with proper atoms: {output_file}")

        except Exception as e:
            self.logger.error(f"Error saving complex PDB: {e}")

    def save_pose_pdb(self, pose: Pose, output_file: Path, ligand_mol=None):
        """
        Save individual ligand pose as PDB file with proper atom types

        Args:
            pose: Ligand pose
            output_file: Output PDB file path
            ligand_mol: RDKit molecule for proper atom types
        """
        try:
            # Create structure for pose
            pose_structure = Structure.Structure('pose')
            pose_model = Model.Model(0)
            pose_chain = Chain.Chain('A')
            pose_residue = self._build_ligand_residue(pose.coordinates, ligand_mol)

            pose_chain.add(pose_residue)
            pose_model.add(pose_chain)
            pose_structure.add(pose_model)

            # Save PDB with ligand as HETATM records
            io = LigandHETATMPDBIO(ligand_chain_ids=['A'], ligand_residue_names=['LIG'])
            io.set_structure(pose_structure)
            io.save(str(output_file))

            self.logger.debug(f"Pose PDB saved: {output_file}")

        except Exception as e:
            self.logger.error(f"Error saving pose PDB: {e}")