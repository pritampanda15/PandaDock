import os
import logging
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, rdForceFieldHelpers
from rdkit.Chem.rdMolAlign import AlignMol
import numpy as np


class LigandPreprocessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.processed_info = {}

    def process(self, input_file, output_dir, add_hydrogens=True, minimize=True):
        """
        Process ligand file (SDF/MOL2/PDB) with hydrogen addition and minimization.
        Keeps original format.
        """
        input_path = Path(input_file)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine output filename with same extension
        output_file = output_dir / f"ligand_processed{input_path.suffix}"

        self.processed_info = {
            'input_file': str(input_path),
            'output_file': str(output_file),
            'format': input_path.suffix.lower(),
            'original_atoms': 0,
            'final_atoms': 0,
            'hydrogens_added': 0,
            'minimized': False,
            'energy_before': None,
            'energy_after': None,
            'conformers_generated': 0
        }

        try:
            # Read ligand molecule
            mol = self._read_ligand(input_path)
            if mol is None:
                raise ValueError(f"Could not read ligand from {input_path}")

            original_atom_count = mol.GetNumAtoms()
            self.processed_info['original_atoms'] = original_atom_count

            # Add hydrogens if requested
            if add_hydrogens:
                mol = self._add_hydrogens(mol)

            # Minimize structure if requested
            if minimize:
                mol = self._minimize_ligand(mol)

            # Write output in original format
            self._write_ligand(mol, output_file, input_path.suffix.lower())

            # Update final atom count
            final_atom_count = mol.GetNumAtoms()
            self.processed_info['final_atoms'] = final_atom_count

            return self.processed_info

        except Exception as e:
            self.logger.error(f"Error processing ligand: {e}")
            raise

    def _read_ligand(self, file_path):
        """Read ligand from various formats"""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        try:
            if extension == '.sdf':
                supplier = Chem.SDMolSupplier(str(file_path))
                mol = next(supplier)
            elif extension == '.mol2':
                mol = Chem.MolFromMol2File(str(file_path))
            elif extension == '.pdb':
                mol = Chem.MolFromPDBFile(str(file_path))
            else:
                raise ValueError(f"Unsupported file format: {extension}")

            if mol is None:
                return None

            # Sanitize molecule
            Chem.SanitizeMol(mol)
            return mol

        except Exception as e:
            self.logger.error(f"Error reading ligand: {e}")
            return None

    def _add_hydrogens(self, mol):
        """Add hydrogens to the ligand"""
        try:
            # Count hydrogens before
            h_before = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'H')

            # Add hydrogens with coordinates
            mol_with_h = Chem.AddHs(mol, addCoords=True)

            # If coordinates weren't added properly, embed the molecule
            if mol_with_h.GetNumConformers() == 0:
                AllChem.EmbedMolecule(mol_with_h, randomSeed=42)
            else:
                # Check if hydrogen coordinates are valid
                conf = mol_with_h.GetConformer()
                has_zero_coords = False
                for i, atom in enumerate(mol_with_h.GetAtoms()):
                    if atom.GetSymbol() == 'H':
                        pos = conf.GetAtomPosition(i)
                        if abs(pos.x) < 0.001 and abs(pos.y) < 0.001 and abs(pos.z) < 0.001:
                            has_zero_coords = True
                            break

                if has_zero_coords:
                    self.logger.warning("Hydrogen coordinates at origin, re-embedding molecule")
                    AllChem.EmbedMolecule(mol_with_h, randomSeed=42)

            # Count hydrogens after
            h_after = sum(1 for atom in mol_with_h.GetAtoms() if atom.GetSymbol() == 'H')

            self.processed_info['hydrogens_added'] = h_after - h_before

            return mol_with_h

        except Exception as e:
            self.logger.warning(f"Error adding hydrogens: {e}")
            return mol

    def _minimize_ligand(self, mol):
        """Minimize ligand structure using RDKit force fields"""
        try:
            # Generate 3D coordinates if not present
            if mol.GetNumConformers() == 0:
                result = AllChem.EmbedMolecule(mol, randomSeed=42)
                if result != 0:
                    self.logger.warning("Failed to embed molecule, skipping minimization")
                    return mol
                self.processed_info['conformers_generated'] = 1

            # Validate coordinates before minimization
            conf = mol.GetConformer()
            total_atoms = mol.GetNumAtoms()
            zero_coords = 0
            for i in range(total_atoms):
                pos = conf.GetAtomPosition(i)
                if abs(pos.x) < 0.001 and abs(pos.y) < 0.001 and abs(pos.z) < 0.001:
                    zero_coords += 1

            if zero_coords > total_atoms * 0.1:  # More than 10% at origin
                self.logger.warning(f"Too many atoms at origin ({zero_coords}/{total_atoms}), re-embedding")
                AllChem.EmbedMolecule(mol, randomSeed=123)

            # Try UFF first, then MMFF
            ff = None
            force_field_name = ""
            try:
                ff = rdForceFieldHelpers.UFFGetMoleculeForceField(mol)
                force_field_name = "UFF"
                if ff is None:
                    ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol)
                    force_field_name = "MMFF"
            except:
                try:
                    ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol)
                    force_field_name = "MMFF"
                except:
                    self.logger.warning("Could not create force field for minimization")
                    return mol

            if ff is not None:
                # Get energy before minimization
                energy_before = ff.CalcEnergy()

                # Check for reasonable energy values
                if abs(energy_before) > 1e6:
                    self.logger.warning(f"Unreasonable energy ({energy_before:.2f}), skipping minimization")
                    return mol

                self.processed_info['energy_before'] = energy_before

                # Minimize with more conservative settings
                ff.Initialize()
                converged = ff.Minimize(maxIts=500, forceTol=1e-4, energyTol=1e-6)

                # Get energy after minimization
                energy_after = ff.CalcEnergy()

                # Validate final energy
                if abs(energy_after) > 1e6:
                    self.logger.warning(f"Unreasonable final energy ({energy_after:.2f})")
                    return mol

                self.processed_info['energy_after'] = energy_after
                self.processed_info['minimized'] = True

                self.logger.info(f"Minimization ({force_field_name}) converged: {converged}")
                self.logger.info(f"Energy change: {energy_before:.2f} -> {energy_after:.2f}")

            return mol

        except Exception as e:
            self.logger.warning(f"Error minimizing ligand: {e}")
            return mol

    def _write_ligand(self, mol, output_file, format_ext):
        """Write ligand to file in original format"""
        try:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            if format_ext == '.sdf':
                writer = Chem.SDWriter(str(output_file))
                writer.write(mol)
                writer.close()
            elif format_ext == '.mol2':
                Chem.MolToMol2File(mol, str(output_file))
            elif format_ext == '.pdb':
                Chem.MolToPDBFile(mol, str(output_file))
            else:
                # Default to SDF if format not supported for writing
                sdf_file = output_file.with_suffix('.sdf')
                writer = Chem.SDWriter(str(sdf_file))
                writer.write(mol)
                writer.close()
                self.processed_info['output_file'] = str(sdf_file)

        except Exception as e:
            self.logger.error(f"Error writing ligand: {e}")
            raise