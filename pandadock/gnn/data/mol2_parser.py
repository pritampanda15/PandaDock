"""
MOL2 file parser for PandaDock-GNN.

Parses TRIPOS MOL2 format files to extract atomic coordinates, types, bonds,
and charges for both protein and ligand molecules.

Reference:
    TRIPOS MOL2 format specification
    http://chemyang.ccnu.edu.cn/ccb/server/AIMMS/mol2.pdf
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Atom:
    """Represents a single atom from MOL2 file."""
    id: int
    name: str
    x: float
    y: float
    z: float
    atom_type: str  # SYBYL atom type (e.g., C.3, N.am, O.co2)
    residue_id: int = 1
    residue_name: str = "LIG"
    charge: float = 0.0
    flags: str = ""

    @property
    def element(self) -> str:
        """Extract element symbol from SYBYL type."""
        return self.atom_type.split('.')[0].upper()

    @property
    def hybridization(self) -> str:
        """Extract hybridization from SYBYL type."""
        parts = self.atom_type.split('.')
        return parts[1] if len(parts) > 1 else ""

    @property
    def coords(self) -> np.ndarray:
        """Return coordinates as numpy array."""
        return np.array([self.x, self.y, self.z], dtype=np.float32)


@dataclass
class Bond:
    """Represents a bond from MOL2 file."""
    id: int
    atom1_id: int
    atom2_id: int
    bond_type: str  # 1, 2, 3, ar, am, etc.

    @property
    def order(self) -> int:
        """Convert bond type to numeric order."""
        type_map = {'1': 1, '2': 2, '3': 3, 'ar': 1.5, 'am': 1, 'du': 0, 'un': 0, 'nc': 0}
        return type_map.get(self.bond_type.lower(), 1)


@dataclass
class Residue:
    """Represents a residue/substructure from MOL2 file."""
    id: int
    name: str
    root_atom: int
    residue_type: str = "GROUP"
    chain: str = "A"


@dataclass
class ParsedMolecule:
    """Container for parsed MOL2 data."""
    name: str
    atoms: List[Atom] = field(default_factory=list)
    bonds: List[Bond] = field(default_factory=list)
    residues: List[Residue] = field(default_factory=list)
    mol_type: str = "SMALL"
    charge_type: str = "NO_CHARGES"
    num_atoms: int = 0
    num_bonds: int = 0

    @property
    def coordinates(self) -> np.ndarray:
        """Return all atomic coordinates as (N, 3) array."""
        if not self.atoms:
            return np.zeros((0, 3), dtype=np.float32)
        return np.array([[a.x, a.y, a.z] for a in self.atoms], dtype=np.float32)

    @property
    def atom_types(self) -> List[str]:
        """Return list of SYBYL atom types."""
        return [a.atom_type for a in self.atoms]

    @property
    def elements(self) -> List[str]:
        """Return list of element symbols."""
        return [a.element for a in self.atoms]

    @property
    def charges(self) -> np.ndarray:
        """Return array of partial charges."""
        return np.array([a.charge for a in self.atoms], dtype=np.float32)

    @property
    def bond_indices(self) -> np.ndarray:
        """Return bond connectivity as (2, num_bonds) array (0-indexed)."""
        if not self.bonds:
            return np.zeros((2, 0), dtype=np.int64)
        # Convert to 0-indexed
        edges = np.array([[b.atom1_id - 1, b.atom2_id - 1] for b in self.bonds], dtype=np.int64)
        return edges.T

    def get_center(self) -> np.ndarray:
        """Return geometric center of molecule."""
        coords = self.coordinates
        if len(coords) == 0:
            return np.zeros(3, dtype=np.float32)
        return coords.mean(axis=0)


class MOL2Parser:
    """
    Parser for TRIPOS MOL2 molecular structure files.

    Supports:
    - Protein structures with residue information
    - Small molecule ligands
    - Multiple molecules in single file

    Example:
        parser = MOL2Parser()
        mol = parser.parse("ligand.mol2")
        print(f"Atoms: {mol.num_atoms}, Bonds: {mol.num_bonds}")
        print(f"Coordinates shape: {mol.coordinates.shape}")
    """

    # Section markers
    MOLECULE = "@<TRIPOS>MOLECULE"
    ATOM = "@<TRIPOS>ATOM"
    BOND = "@<TRIPOS>BOND"
    SUBSTRUCTURE = "@<TRIPOS>SUBSTRUCTURE"

    # SYBYL atom types for element mapping
    SYBYL_ELEMENTS = {
        'C': ['C.3', 'C.2', 'C.1', 'C.ar', 'C.cat'],
        'N': ['N.3', 'N.2', 'N.1', 'N.ar', 'N.am', 'N.pl3', 'N.4'],
        'O': ['O.3', 'O.2', 'O.co2', 'O.spc', 'O.t3p'],
        'S': ['S.3', 'S.2', 'S.O', 'S.O2'],
        'P': ['P.3'],
        'H': ['H', 'H.spc', 'H.t3p'],
        'F': ['F'],
        'Cl': ['Cl'],
        'Br': ['Br'],
        'I': ['I'],
    }

    def __init__(self, verbose: bool = False):
        """
        Initialize MOL2 parser.

        Args:
            verbose: Print parsing details
        """
        self.verbose = verbose

    def parse(self, file_path: str) -> ParsedMolecule:
        """
        Parse a MOL2 file and return structured data.

        Args:
            file_path: Path to MOL2 file

        Returns:
            ParsedMolecule with atoms, bonds, and metadata
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"MOL2 file not found: {file_path}")

        with open(path, 'r') as f:
            content = f.read()

        return self._parse_content(content)

    def parse_string(self, content: str) -> ParsedMolecule:
        """Parse MOL2 content from string."""
        return self._parse_content(content)

    def _parse_content(self, content: str) -> ParsedMolecule:
        """Internal parsing logic."""
        lines = content.split('\n')

        molecule = ParsedMolecule(name="Unknown")
        current_section = None

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Check for section markers
            if line.startswith("@<TRIPOS>"):
                current_section = line
                i += 1
                continue

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                i += 1
                continue

            # Parse based on current section
            if current_section == self.MOLECULE:
                molecule, i = self._parse_molecule_section(lines, i, molecule)
                current_section = None
            elif current_section == self.ATOM:
                atom = self._parse_atom_line(line)
                if atom:
                    molecule.atoms.append(atom)
                i += 1
            elif current_section == self.BOND:
                bond = self._parse_bond_line(line)
                if bond:
                    molecule.bonds.append(bond)
                i += 1
            elif current_section == self.SUBSTRUCTURE:
                residue = self._parse_substructure_line(line)
                if residue:
                    molecule.residues.append(residue)
                i += 1
            else:
                i += 1

        # Update counts
        molecule.num_atoms = len(molecule.atoms)
        molecule.num_bonds = len(molecule.bonds)

        if self.verbose:
            print(f"Parsed: {molecule.name}")
            print(f"  Atoms: {molecule.num_atoms}")
            print(f"  Bonds: {molecule.num_bonds}")
            print(f"  Residues: {len(molecule.residues)}")

        return molecule

    def _parse_molecule_section(
        self,
        lines: List[str],
        start_idx: int,
        molecule: ParsedMolecule
    ) -> Tuple[ParsedMolecule, int]:
        """Parse MOLECULE section."""
        i = start_idx

        # Line 1: molecule name
        if i < len(lines):
            molecule.name = lines[i].strip()
            i += 1

        # Line 2: num_atoms num_bonds num_subst num_feat num_sets
        if i < len(lines):
            parts = lines[i].split()
            if parts:
                molecule.num_atoms = int(parts[0]) if len(parts) > 0 else 0
                molecule.num_bonds = int(parts[1]) if len(parts) > 1 else 0
            i += 1

        # Line 3: mol_type
        if i < len(lines):
            molecule.mol_type = lines[i].strip()
            i += 1

        # Line 4: charge_type
        if i < len(lines):
            molecule.charge_type = lines[i].strip()
            i += 1

        return molecule, i

    def _parse_atom_line(self, line: str) -> Optional[Atom]:
        """
        Parse a single ATOM line.

        Format:
        atom_id atom_name x y z atom_type [subst_id [subst_name [charge [status_bit]]]]
        """
        parts = line.split()
        if len(parts) < 6:
            return None

        try:
            atom = Atom(
                id=int(parts[0]),
                name=parts[1],
                x=float(parts[2]),
                y=float(parts[3]),
                z=float(parts[4]),
                atom_type=parts[5],
                residue_id=int(parts[6]) if len(parts) > 6 else 1,
                residue_name=parts[7] if len(parts) > 7 else "LIG",
                charge=float(parts[8]) if len(parts) > 8 else 0.0,
                flags=parts[9] if len(parts) > 9 else ""
            )
            return atom
        except (ValueError, IndexError) as e:
            if self.verbose:
                print(f"Warning: Could not parse atom line: {line}")
            return None

    def _parse_bond_line(self, line: str) -> Optional[Bond]:
        """
        Parse a single BOND line.

        Format:
        bond_id origin_atom_id target_atom_id bond_type [status_bits]
        """
        parts = line.split()
        if len(parts) < 4:
            return None

        try:
            bond = Bond(
                id=int(parts[0]),
                atom1_id=int(parts[1]),
                atom2_id=int(parts[2]),
                bond_type=parts[3]
            )
            return bond
        except (ValueError, IndexError):
            if self.verbose:
                print(f"Warning: Could not parse bond line: {line}")
            return None

    def _parse_substructure_line(self, line: str) -> Optional[Residue]:
        """
        Parse a single SUBSTRUCTURE line.

        Format:
        subst_id subst_name root_atom [subst_type [dict_type [chain [sub_type [inter_bonds [status [comment]]]]]]]
        """
        parts = line.split()
        if len(parts) < 3:
            return None

        try:
            residue = Residue(
                id=int(parts[0]),
                name=parts[1],
                root_atom=int(parts[2]),
                residue_type=parts[3] if len(parts) > 3 else "GROUP",
                chain=parts[5] if len(parts) > 5 else "A"
            )
            return residue
        except (ValueError, IndexError):
            if self.verbose:
                print(f"Warning: Could not parse substructure line: {line}")
            return None

    def parse_multi(self, file_path: str) -> List[ParsedMolecule]:
        """
        Parse MOL2 file containing multiple molecules.

        Args:
            file_path: Path to MOL2 file

        Returns:
            List of ParsedMolecule objects
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"MOL2 file not found: {file_path}")

        with open(path, 'r') as f:
            content = f.read()

        # Split by MOLECULE markers
        molecules = []
        mol_sections = content.split(self.MOLECULE)

        for i, section in enumerate(mol_sections[1:], 1):  # Skip first empty split
            mol_content = self.MOLECULE + section
            try:
                mol = self._parse_content(mol_content)
                molecules.append(mol)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not parse molecule {i}: {e}")

        return molecules


def parse_mol2(file_path: str, verbose: bool = False) -> ParsedMolecule:
    """
    Convenience function to parse a MOL2 file.

    Args:
        file_path: Path to MOL2 file
        verbose: Print parsing details

    Returns:
        ParsedMolecule with atoms, bonds, and metadata
    """
    parser = MOL2Parser(verbose=verbose)
    return parser.parse(file_path)


# Standard amino acid residue names
AMINO_ACIDS = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    # Common variants
    'HIE', 'HID', 'HIP', 'CYX', 'ASH', 'GLH', 'LYN'
}


def is_protein_residue(residue_name: str) -> bool:
    """Check if residue name is a standard amino acid."""
    # Strip any numbers from residue name (e.g., "ALA123" -> "ALA")
    clean_name = ''.join(c for c in residue_name if c.isalpha()).upper()
    return clean_name[:3] in AMINO_ACIDS


if __name__ == "__main__":
    # Test parsing
    import sys
    if len(sys.argv) > 1:
        mol = parse_mol2(sys.argv[1], verbose=True)
        print(f"\nCoordinates shape: {mol.coordinates.shape}")
        print(f"Elements: {set(mol.elements)}")
        print(f"Center: {mol.get_center()}")
