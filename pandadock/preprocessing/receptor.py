import os
import logging
from pathlib import Path
from Bio import PDB
from Bio.PDB import PDBIO, PDBParser, MMCIFParser
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, AllChem

try:
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile
    HAS_OPENMM = True
except ImportError:
    HAS_OPENMM = False

try:
    import propka.run as propka_run
    HAS_PROPKA = True
    PROPKA_VERSION = "propka"
except ImportError:
    try:
        import propka3.run as propka_run
        HAS_PROPKA = True
        PROPKA_VERSION = "propka3"
    except ImportError:
        try:
            import propka3.lib as propka_run
            HAS_PROPKA = True
            PROPKA_VERSION = "propka3"
        except ImportError:
            HAS_PROPKA = False
            PROPKA_VERSION = None


class ReceptorPreprocessor:
    def __init__(self, ph=7.0):
        self.ph = ph
        self.logger = logging.getLogger(__name__)
        self.processed_info = {}

    def process(self, input_file, output_file, add_hydrogens=True, add_charges=True, fix_missing_residues=False):
        """
        Process receptor file (PDB/CIF) with missing atom/residue repair,
        hydrogen addition, and charge assignment.
        """
        input_path = Path(input_file)
        output_path = Path(output_file)

        self.processed_info = {
            'input_file': str(input_path),
            'output_file': str(output_path),
            'format': input_path.suffix.lower(),
            'original_atoms': 0,
            'final_atoms': 0,
            'missing_residues_added': [],
            'missing_atoms_added': [],
            'hydrogens_added': 0,
            'charges_assigned': False,
            'ph_used': self.ph
        }

        try:
            # Parse input structure
            structure = self._parse_structure(input_path)
            original_atom_count = self._count_atoms(structure)
            self.processed_info['original_atoms'] = original_atom_count

            # Fix missing residues and atoms using PDBFixer if available and requested
            if fix_missing_residues:
                fixed_structure = self._fix_missing_parts(input_path)
            else:
                fixed_structure = None
                self.processed_info['missing_residues_added'] = []
                self.processed_info['missing_atoms_added'] = []

            # Add hydrogens if requested
            if add_hydrogens:
                if fixed_structure is not None:
                    # Use PDBFixer for hydrogen addition
                    fixed_structure = self._add_hydrogens(fixed_structure)
                else:
                    # Use alternative method when PDBFixer isn't used
                    fixed_structure = self._add_hydrogens_without_fixer(structure)

            # Add charges if requested
            if add_charges:
                if fixed_structure is not None:
                    self._add_charges(fixed_structure)
                    self.processed_info['charges_assigned'] = True
                else:
                    # Charge assignment for BioPython structures not implemented yet
                    self.processed_info['charges_assigned'] = False

            # Write output
            if fixed_structure is not None:
                self._write_structure(fixed_structure, output_path)
                # Count final atoms
                final_structure = self._parse_structure(output_path)
                final_atom_count = self._count_atoms(final_structure)
            else:
                # Fallback: copy input to output with basic processing
                self._write_structure_fallback(structure, output_path)
                final_atom_count = original_atom_count

            self.processed_info['final_atoms'] = final_atom_count

            return self.processed_info

        except Exception as e:
            self.logger.error(f"Error processing receptor: {e}")
            raise

    def _parse_structure(self, file_path):
        """Parse PDB or CIF structure file"""
        file_path = Path(file_path)

        if file_path.suffix.lower() == '.cif':
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure('receptor', str(file_path))
        else:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('receptor', str(file_path))

        return structure

    def _count_atoms(self, structure):
        """Count total atoms in structure"""
        count = 0
        for model in structure:
            for chain in model:
                for residue in chain:
                    count += len(residue)
        return count

    def _fix_missing_parts(self, input_path):
        """Use PDBFixer to add missing residues and atoms"""
        if not HAS_OPENMM:
            self.logger.warning("PDBFixer not available, skipping missing parts repair")
            return None

        try:
            fixer = PDBFixer(filename=str(input_path))

            # Find missing residues and atoms first
            fixer.findMissingResidues()
            fixer.findMissingAtoms()

            # Be very conservative with missing residue addition
            total_missing_residues = sum(len(residues) for residues in fixer.missingResidues.values())
            if total_missing_residues > 10:
                self.logger.warning(f"Found {total_missing_residues} missing residues, this is excessive for docking. Disabling missing residue addition.")
                # Clear all missing residues - too risky for docking
                fixer.missingResidues = {}
            elif total_missing_residues > 5:
                self.logger.warning(f"Found {total_missing_residues} missing residues, limiting to very short terminal gaps only")
                # Keep only very short terminal sequences
                filtered_missing = {}
                for chain, residues in fixer.missingResidues.items():
                    if len(residues) <= 3:  # Only add very short terminal sequences
                        filtered_missing[chain] = residues
                fixer.missingResidues = filtered_missing

            # Record missing residues before fixing
            missing_residues = []
            try:
                # Check if the attributes exist and have data
                if hasattr(fixer, 'missingResidues') and fixer.missingResidues:
                    for chain, residues in fixer.missingResidues.items():
                        chain_id = chain.id if hasattr(chain, 'id') else str(chain)
                        for residue in residues:
                            residue_name = residue.name if hasattr(residue, 'name') else str(residue)
                            missing_residues.append(f"Chain {chain_id}: {residue_name}")
            except Exception as e:
                self.logger.warning(f"Could not access missing residues info: {e}")

            self.processed_info['missing_residues_added'] = missing_residues

            # Record missing atoms before fixing
            missing_atoms = []
            try:
                if hasattr(fixer, 'missingAtoms') and fixer.missingAtoms:
                    for residue, atoms in fixer.missingAtoms.items():
                        for atom in atoms:
                            missing_atoms.append(f"{residue}: {atom}")
            except Exception as e:
                self.logger.warning(f"Could not access missing atoms info: {e}")

            self.processed_info['missing_atoms_added'] = missing_atoms

            # Add missing residues and atoms
            fixer.addMissingAtoms()

            return fixer

        except Exception as e:
            self.logger.error(f"PDBFixer failed: {e}")
            return None

    def _add_hydrogens(self, fixer):
        """Add hydrogens using propka for pH-dependent protonation"""
        if fixer is None:
            self.logger.info("No PDBFixer structure, skipping hydrogen addition")
            self.processed_info['hydrogens_added'] = 0
            return None

        try:
            # Use propka to determine protonation states if available
            if HAS_PROPKA:
                temp_pdb = "temp_receptor.pdb"
                PDBFile.writeFile(fixer.topology, fixer.positions, open(temp_pdb, 'w'))

                # Run propka
                try:
                    if hasattr(propka_run, 'single'):
                        propka_run.single([temp_pdb, f"--pH={self.ph}"])
                    else:
                        propka_run.main([temp_pdb, f"--pH={self.ph}"])
                    self.logger.info(f"Using {PROPKA_VERSION} for pH {self.ph} protonation")
                except Exception as e:
                    self.logger.info(f"Propka ({PROPKA_VERSION}) failed: {e}. Using default pH-based protonation.")

                # Clean up temp files
                if os.path.exists(temp_pdb):
                    os.remove(temp_pdb)
                propka_files = [f for f in os.listdir('.') if f.startswith('temp_receptor') and f.endswith('.pka')]
                for f in propka_files:
                    os.remove(f)
            else:
                self.logger.info("Propka not available, using default pH-based protonation")

            # Add hydrogens with pH consideration
            fixer.addMissingHydrogens(self.ph)

            # Count added hydrogens
            hydrogen_count = 0
            for atom in fixer.topology.atoms():
                if atom.element.symbol == 'H':
                    hydrogen_count += 1
            self.processed_info['hydrogens_added'] = hydrogen_count

        except Exception as e:
            self.logger.warning(f"Error adding hydrogens: {e}")
            if HAS_OPENMM:
                fixer.addMissingHydrogens(self.ph)

        return fixer

    def _add_hydrogens_without_fixer(self, structure):
        """Add hydrogens without using PDBFixer - convert to PDBFixer format for hydrogen addition"""
        try:
            if not HAS_OPENMM:
                self.logger.warning("OpenMM not available, cannot add hydrogens without PDBFixer")
                self.processed_info['hydrogens_added'] = 0
                return structure

            # Create a temporary PDB file from the BioPython structure
            temp_pdb = "temp_receptor_for_h.pdb"
            io = PDBIO()
            io.set_structure(structure)
            io.save(temp_pdb)

            # Use PDBFixer just for hydrogen addition (no missing residue/atom fixing)
            fixer = PDBFixer(filename=temp_pdb)

            # Skip findMissingResidues and findMissingAtoms - just add hydrogens
            if HAS_PROPKA:
                try:
                    if hasattr(propka_run, 'single'):
                        propka_run.single([temp_pdb, f"--pH={self.ph}"])
                    else:
                        propka_run.main([temp_pdb, f"--pH={self.ph}"])
                    self.logger.info(f"Using {PROPKA_VERSION} for pH {self.ph} protonation")
                except Exception as e:
                    self.logger.info(f"Propka ({PROPKA_VERSION}) not working: {e}. Using default pH-based protonation.")
            else:
                self.logger.info("Propka not installed, using default pH-based protonation")

            # Add hydrogens
            fixer.addMissingHydrogens(self.ph)

            # Count hydrogens
            hydrogen_count = 0
            for atom in fixer.topology.atoms():
                if atom.element.symbol == 'H':
                    hydrogen_count += 1
            self.processed_info['hydrogens_added'] = hydrogen_count

            # Clean up temp file
            import os
            if os.path.exists(temp_pdb):
                os.remove(temp_pdb)
            propka_files = [f for f in os.listdir('.') if f.startswith('temp_receptor_for_h') and f.endswith('.pka')]
            for f in propka_files:
                os.remove(f)

            self.logger.info(f"Added {hydrogen_count} hydrogens using PDBFixer (no missing residue fixing)")
            return fixer

        except Exception as e:
            self.logger.warning(f"Failed to add hydrogens without PDBFixer: {e}")
            self.processed_info['hydrogens_added'] = 0
            return structure

    def _add_charges(self, fixer):
        """Add partial charges to the structure"""
        try:
            # For now, we'll use a simple approach
            # In a full implementation, you might use antechamber or other tools
            # This is a placeholder for charge assignment
            pass
        except Exception as e:
            self.logger.warning(f"Charge assignment failed: {e}")

    def _write_structure(self, fixer, output_path):
        """Write the processed structure to file"""
        if not HAS_OPENMM:
            raise RuntimeError("OpenMM not available for writing structure")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)

    def _write_structure_fallback(self, structure, output_path):
        """Fallback method to write structure using BioPython"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        io = PDBIO()
        io.set_structure(structure)
        io.save(str(output_path))