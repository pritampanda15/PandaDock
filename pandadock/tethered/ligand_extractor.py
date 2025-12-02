#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ligand Extractor for Tethered Docking

Extracts ligands from PDB complex files and prepares them for tethered docking
validation experiments. Handles multiple file formats and provides clean
separation of protein and ligand components.

Author: Pritam Kumar Panda @ Stanford University
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from Bio.PDB import PDBParser, PDBIO, Select, Structure, Model, Chain, Residue, Atom
import numpy as np


class LigandExtractor:
    """Extract ligands from PDB complex files for tethered docking"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def extract_ligand_from_complex(
        self,
        complex_file: str,
        ligand_id: str,
        chain_id: Optional[str] = None,
        output_dir: str = ".",
        output_format: str = "sdf"
    ) -> Dict[str, str]:
        """
        Extract ligand from PDB complex file

        Args:
            complex_file: Path to PDB complex file
            ligand_id: Ligand residue name (e.g., 'LIG', 'ATP', 'HEM')
            chain_id: Specific chain containing ligand (optional)
            output_dir: Output directory for extracted files
            output_format: Output format for ligand ('sdf', 'mol2', 'pdb')

        Returns:
            Dictionary of extracted file paths
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Parse the complex structure
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('complex', complex_file)

            # Find ligand residues
            ligand_residues = self._find_ligand_residues(structure, ligand_id, chain_id)

            if not ligand_residues:
                raise ValueError(f"No ligand '{ligand_id}' found in complex file")

            self.logger.info(f"Found {len(ligand_residues)} ligand residue(s): {ligand_id}")

            # Extract components
            extracted_files = {}

            # 1. Extract receptor (protein without ligand)
            receptor_file = output_path / f"receptor_{Path(complex_file).stem}.pdb"
            self._extract_receptor(structure, ligand_residues, receptor_file)
            extracted_files['receptor'] = str(receptor_file)

            # 2. Extract ligand in specified format
            ligand_file = output_path / f"ligand_{Path(complex_file).stem}.{output_format}"
            self._extract_ligand(ligand_residues, ligand_file, output_format)
            extracted_files['ligand'] = str(ligand_file)

            # 3. Extract reference pose (ligand coordinates as PDB)
            reference_file = output_path / f"reference_pose_{Path(complex_file).stem}.pdb"
            self._extract_reference_pose(ligand_residues, reference_file)
            extracted_files['reference_pose'] = str(reference_file)

            # 4. Generate extraction summary
            summary_file = output_path / f"extraction_summary_{Path(complex_file).stem}.json"
            self._generate_extraction_summary(
                complex_file, ligand_id, chain_id, ligand_residues,
                extracted_files, summary_file
            )
            extracted_files['summary'] = str(summary_file)

            self.logger.info(f"Ligand extraction completed successfully")
            return extracted_files

        except Exception as e:
            self.logger.error(f"Ligand extraction failed: {e}")
            raise

    def _find_ligand_residues(
        self,
        structure: Structure.Structure,
        ligand_id: str,
        chain_id: Optional[str] = None
    ) -> List[Residue.Residue]:
        """Find all residues matching the ligand ID"""
        ligand_residues = []

        for model in structure:
            for chain in model:
                # Filter by chain if specified
                if chain_id and chain.get_id() != chain_id:
                    continue

                for residue in chain:
                    resname = residue.get_resname().strip()
                    if resname == ligand_id:
                        ligand_residues.append(residue)
                        self.logger.debug(f"Found ligand: {resname} in chain {chain.get_id()}")

        return ligand_residues

    def _extract_receptor(
        self,
        structure: Structure.Structure,
        ligand_residues: List[Residue.Residue],
        output_file: Path
    ):
        """Extract receptor by removing ligand residues"""

        class ReceptorSelector(Select):
            def __init__(self, ligand_residues):
                self.ligand_residues_ids = {
                    (res.get_parent().get_id(), res.get_id()) for res in ligand_residues
                }

            def accept_residue(self, residue):
                chain_id = residue.get_parent().get_id()
                res_id = residue.get_id()
                return (chain_id, res_id) not in self.ligand_residues_ids

        # Save receptor structure without ligands
        io = PDBIO()
        io.set_structure(structure)
        io.save(str(output_file), ReceptorSelector(ligand_residues))

        self.logger.info(f"Receptor extracted: {output_file}")

    def _extract_ligand(
        self,
        ligand_residues: List[Residue.Residue],
        output_file: Path,
        output_format: str
    ):
        """Extract ligand structure in specified format"""

        if output_format.lower() == 'pdb':
            self._extract_ligand_pdb(ligand_residues, output_file)
        elif output_format.lower() in ['sdf', 'mol']:
            self._extract_ligand_sdf(ligand_residues, output_file)
        elif output_format.lower() == 'mol2':
            self._extract_ligand_mol2(ligand_residues, output_file)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def _extract_ligand_pdb(self, ligand_residues: List[Residue.Residue], output_file: Path):
        """Extract ligand as PDB format"""

        # Create new structure with only ligand residues
        ligand_structure = Structure.Structure('ligand')
        ligand_model = Model.Model(0)
        ligand_structure.add(ligand_model)

        for i, residue in enumerate(ligand_residues):
            # Create new chain for each ligand (or group by original chain)
            chain_id = residue.get_parent().get_id()

            # Get or create chain
            try:
                ligand_chain = ligand_model[chain_id]
            except KeyError:
                ligand_chain = Chain.Chain(chain_id)
                ligand_model.add(ligand_chain)

            # Add residue copy
            ligand_chain.add(residue.copy())

        # Save ligand structure
        io = PDBIO()
        io.set_structure(ligand_structure)
        io.save(str(output_file))

        self.logger.info(f"Ligand extracted as PDB: {output_file}")

    def _extract_ligand_sdf(self, ligand_residues: List[Residue.Residue], output_file: Path):
        """Extract ligand as SDF format (simplified)"""

        # For now, create a simple SDF file with coordinates
        # In a full implementation, you'd use RDKit or similar
        with open(output_file, 'w') as f:
            f.write(f"Ligand extracted from PDB\n")
            f.write(f"Generated by PandaDock Tethered Module\n")
            f.write(f"\n")

            # Count atoms
            total_atoms = sum(len(list(res.get_atoms())) for res in ligand_residues)
            f.write(f"{total_atoms:3d}{0:3d}  0  0  0  0  0  0  0  0999 V2000\n")

            # Write atom coordinates
            for residue in ligand_residues:
                for atom in residue:
                    coord = atom.get_coord()
                    element = atom.element.strip() if hasattr(atom, 'element') and atom.element.strip() else atom.get_name()[0]
                    f.write(f"{coord[0]:10.4f}{coord[1]:10.4f}{coord[2]:10.4f} {element:<3s} 0  0  0  0  0  0  0  0  0  0  0  0\n")

            f.write("M  END\n")
            f.write("$$$$\n")

        self.logger.info(f"Ligand extracted as SDF: {output_file}")

    def _extract_ligand_mol2(self, ligand_residues: List[Residue.Residue], output_file: Path):
        """Extract ligand as MOL2 format (simplified)"""

        with open(output_file, 'w') as f:
            f.write("@<TRIPOS>MOLECULE\n")
            f.write("Ligand\n")

            # Count atoms and bonds (simplified - no bond detection)
            total_atoms = sum(len(list(res.get_atoms())) for res in ligand_residues)
            f.write(f"{total_atoms} 0 1 0 0\n")
            f.write("SMALL\n")
            f.write("NO_CHARGES\n")
            f.write("\n")

            f.write("@<TRIPOS>ATOM\n")
            atom_id = 1
            for residue in ligand_residues:
                for atom in residue:
                    coord = atom.get_coord()
                    element = atom.element.strip() if hasattr(atom, 'element') and atom.element.strip() else atom.get_name()[0]
                    f.write(f"{atom_id:7d} {atom.get_name():<8s} {coord[0]:9.4f} {coord[1]:9.4f} {coord[2]:9.4f} {element:<5s} 1 {residue.get_resname()} 0.0000\n")
                    atom_id += 1

        self.logger.info(f"Ligand extracted as MOL2: {output_file}")

    def _extract_reference_pose(self, ligand_residues: List[Residue.Residue], output_file: Path):
        """Extract reference pose as PDB for constraint definition"""

        # Same as PDB extraction but specifically for reference
        self._extract_ligand_pdb(ligand_residues, output_file)
        self.logger.info(f"Reference pose extracted: {output_file}")

    def _generate_extraction_summary(
        self,
        complex_file: str,
        ligand_id: str,
        chain_id: Optional[str],
        ligand_residues: List[Residue.Residue],
        extracted_files: Dict[str, str],
        summary_file: Path
    ):
        """Generate extraction summary report"""

        import json

        # Calculate ligand properties
        total_atoms = sum(len(list(res.get_atoms())) for res in ligand_residues)

        # Get ligand center of mass
        all_coords = []
        for residue in ligand_residues:
            for atom in residue:
                all_coords.append(atom.get_coord())

        center_of_mass = np.mean(all_coords, axis=0) if all_coords else [0, 0, 0]

        summary = {
            "extraction_info": {
                "complex_file": str(complex_file),
                "ligand_id": ligand_id,
                "chain_filter": chain_id,
                "extraction_timestamp": str(Path().cwd())
            },
            "ligand_properties": {
                "num_residues": len(ligand_residues),
                "total_atoms": total_atoms,
                "center_of_mass": center_of_mass.tolist(),
                "chains_found": list(set(res.get_parent().get_id() for res in ligand_residues))
            },
            "extracted_files": extracted_files,
            "residue_details": [
                {
                    "chain": res.get_parent().get_id(),
                    "residue_id": res.get_id(),
                    "resname": res.get_resname(),
                    "num_atoms": len(list(res.get_atoms()))
                }
                for res in ligand_residues
            ]
        }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Extraction summary saved: {summary_file}")

    def list_available_ligands(self, complex_file: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        List all potential ligands in a PDB complex file

        Returns:
            Dictionary mapping chain IDs to ligand information
        """
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('complex', complex_file)

            ligands_by_chain = {}

            for model in structure:
                for chain in model:
                    chain_id = chain.get_id()
                    chain_ligands = []

                    for residue in chain:
                        resname = residue.get_resname().strip()

                        # Skip standard amino acids and nucleotides
                        standard_residues = {
                            'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
                            'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
                            'THR', 'TRP', 'TYR', 'VAL', 'MSE',  # Amino acids
                            'A', 'T', 'G', 'C', 'U', 'DA', 'DT', 'DG', 'DC',  # Nucleotides
                            'HOH', 'WAT'  # Water
                        }

                        if resname not in standard_residues:
                            ligand_info = {
                                'resname': resname,
                                'residue_id': residue.get_id(),
                                'num_atoms': len(list(residue.get_atoms())),
                                'center': np.mean([atom.get_coord() for atom in residue], axis=0).tolist()
                            }
                            chain_ligands.append(ligand_info)

                    if chain_ligands:
                        ligands_by_chain[chain_id] = chain_ligands

            return ligands_by_chain

        except Exception as e:
            self.logger.error(f"Error listing ligands: {e}")
            return {}