import numpy as np
from pathlib import Path
import os
import sys

class Ligand:
    """Class representing a small molecule ligand."""
    
    def __init__(self, mol_file=None):
        """
        Initialize a ligand object.
        
        Parameters:
        -----------
        mol_file : str
            Path to MOL/SDF file containing ligand structure
        """
        self.atoms = []
        self.bonds = []
        self.xyz = np.empty((0, 3))  # Initialize as empty array with correct shape
        self.rotatable_bonds = []
        self.conformers = []
        
        if mol_file:
            self.load_molecule(mol_file)
    
    def load_molecule(self, mol_file):
        """
        Load ligand structure from MOL/SDF file.
        
        Parameters:
        -----------
        mol_file : str
            Path to MOL/SDF file
        """
        mol_path = Path(mol_file)
        if not mol_path.exists():
            raise FileNotFoundError(f"Molecule file not found: {mol_file}")
        
        # Try multiple loading methods
        successful = False
        
        # Method 1: Try with RDKit
        try:
            successful = self._load_with_rdkit(mol_path)
        except Exception as e:
            print(f"Warning: RDKit loading failed: {e}")
        
        # Method 2: Try with OpenBabel if RDKit failed
        if not successful:
            try:
                successful = self._load_with_openbabel(mol_path)
            except Exception as e:
                print(f"Warning: OpenBabel loading failed: {e}")
        
        # Method 3: Fall back to basic MOL parser
        if not successful:
            try:
                successful = self._parse_mol_file(mol_path)
            except Exception as e:
                print(f"Warning: Basic MOL parser failed: {e}")
        
        # Final check
        if not successful or len(self.atoms) == 0 or len(self.xyz) == 0:
            raise ValueError(f"Failed to load molecule from {mol_file} using any available method")
        
        print(f"Successfully loaded ligand with {len(self.atoms)} atoms and {len(self.bonds)} bonds")
        print(f"Identified {len(self.rotatable_bonds)} rotatable bonds")
            
    def _load_with_rdkit(self, mol_path):
        """Load molecule using RDKit."""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            
            print("Attempting to load with RDKit...")
            
            # Determine file format from extension
            ext = mol_path.suffix.lower()
            if ext == '.mol2':
                mol = Chem.MolFromMol2File(str(mol_path))
            elif ext == '.sdf':
                supplier = Chem.SDMolSupplier(str(mol_path))
                mol = next(supplier) if supplier else None
            else:
                mol = Chem.MolFromMolFile(str(mol_path))
            
            if mol is None:
                print(f"RDKit could not parse the molecule file")
                return False
            
            # Get atom coordinates
            try:
                conformer = mol.GetConformer()
                self.xyz = np.array([conformer.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
            except:
                print("Failed to get coordinates from conformer")
                return False
            
            if len(self.xyz) == 0:
                print(f"No atom coordinates found in molecule file")
                return False
            
            # Get atom information
            for atom in mol.GetAtoms():
                pos = conformer.GetAtomPosition(atom.GetIdx())
                self.atoms.append({
                    'idx': atom.GetIdx(),
                    'symbol': atom.GetSymbol(),
                    'formal_charge': atom.GetFormalCharge(),
                    'coords': np.array([pos.x, pos.y, pos.z])
                })
            
            # Get bond information
            for bond in mol.GetBonds():
                self.bonds.append({
                    'begin_atom_idx': bond.GetBeginAtomIdx(),
                    'end_atom_idx': bond.GetEndAtomIdx(),
                    'bond_type': bond.GetBondType(),
                    'is_rotatable': bond.GetBondTypeAsDouble() == 1 and not bond.IsInRing()
                })
                
                # Track rotatable bonds for conformer generation
                if bond.GetBondTypeAsDouble() == 1 and not bond.IsInRing():
                    self.rotatable_bonds.append(bond.GetIdx())
            
            return True
            
        except ImportError:
            print("RDKit not available")
            return False
    
    def _load_with_openbabel(self, mol_path):
        """Load molecule using OpenBabel."""
        try:
            import openbabel
            from openbabel import pybel
            
            print("Attempting to load with OpenBabel...")
            
            # Determine file format from extension
            ext = mol_path.suffix.lower()[1:]  # Remove the dot
            if ext == 'mol2':
                fmt = 'mol2'
            elif ext == 'sdf':
                fmt = 'sdf'
            else:
                fmt = 'mol'
                
            # Read the molecule
            mols = list(pybel.readfile(fmt, str(mol_path)))
            if not mols:
                print("No molecules found in file with OpenBabel")
                return False
                
            mol = mols[0]
            
            # Get coordinates
            if mol.dim < 3:
                print(f"Molecule has {mol.dim}D coordinates, not 3D")
                # Try to make 3D coordinates
                mol.make3D()
            
            # Extract atom data
            atom_coords = []
            for i, atom in enumerate(mol.atoms):
                coords = atom.coords
                atom_coords.append(coords)
                
                self.atoms.append({
                    'idx': i,
                    'symbol': atom.type.split()[0] if hasattr(atom, 'type') else atom.OBAtom.GetAtomicNum(),
                    'coords': np.array(coords)
                })
            
            self.xyz = np.array(atom_coords)
            
            # Extract bond data
            for i, bond in enumerate(openbabel.OBMolBondIter(mol.OBMol)):
                begin_idx = bond.GetBeginAtomIdx() - 1  # OB uses 1-based indexing
                end_idx = bond.GetEndAtomIdx() - 1
                bond_order = bond.GetBondOrder()
                
                self.bonds.append({
                    'begin_atom_idx': begin_idx,
                    'end_atom_idx': end_idx,
                    'bond_type': bond_order,
                    'is_rotatable': bond_order == 1 and not bond.IsInRing()
                })
                
                # Track rotatable bonds
                if bond_order == 1 and not bond.IsInRing():
                    self.rotatable_bonds.append(i)
            
            return True
            
        except ImportError:
            print("OpenBabel not available")
            return False
        
        except Exception as e:
            print(f"Error in OpenBabel loading: {e}")
            return False
    
    def _parse_mol_file(self, mol_path):
        """Simple parser for MOL files without external libraries."""
        print("Attempting to parse MOL file directly...")
        
        try:
            with open(mol_path, 'r') as f:
                lines = f.readlines()
            
            if len(lines) < 4:
                print("MOL file too short, must have at least 4 lines")
                return False
                
            # Parse atom and bond counts (line 4 in MOL format)
            counts_line = lines[3].strip()
            if len(counts_line) < 6:
                print(f"Line 4 too short: '{counts_line}'")
                return False
                
            try:
                atom_count = int(counts_line[0:3].strip())
                bond_count = int(counts_line[3:6].strip())
            except ValueError:
                print(f"Failed to parse atom/bond counts from '{counts_line}'")
                return False
                
            if atom_count == 0:
                print("No atoms declared in MOL file")
                return False
                
            if 4 + atom_count > len(lines):
                print(f"MOL file has {len(lines)} lines, but needs at least {4 + atom_count} for {atom_count} atoms")
                return False
            
            # Parse atoms
            atom_coords = []
            for i in range(atom_count):
                try:
                    atom_line = lines[4+i]
                    x = float(atom_line[0:10].strip())
                    y = float(atom_line[10:20].strip())
                    z = float(atom_line[20:30].strip())
                    symbol = atom_line[31:34].strip()
                    
                    self.atoms.append({
                        'idx': i,
                        'symbol': symbol,
                        'coords': np.array([x, y, z])
                    })
                    atom_coords.append([x, y, z])
                except Exception as e:
                    print(f"Error parsing atom {i+1}: {e}")
                    return False
            
            self.xyz = np.array(atom_coords)
            
            # Parse bonds (if any and if we have enough lines)
            if bond_count > 0 and 4 + atom_count + bond_count <= len(lines):
                for i in range(bond_count):
                    try:
                        bond_line = lines[4+atom_count+i]
                        atom1 = int(bond_line[0:3].strip()) - 1  # MOL indices start at 1
                        atom2 = int(bond_line[3:6].strip()) - 1
                        bond_type = int(bond_line[6:9].strip())
                        
                        self.bonds.append({
                            'begin_atom_idx': atom1,
                            'end_atom_idx': atom2,
                            'bond_type': bond_type,
                            'is_rotatable': bond_type == 1  # Simple heuristic for rotatability
                        })
                        
                        # Track rotatable bonds (simple heuristic - single bonds)
                        if bond_type == 1:
                            self.rotatable_bonds.append(i)
                    except Exception as e:
                        print(f"Error parsing bond {i+1}: {e}")
                        # Continue parsing other bonds
            
            # Check final validation
            if len(self.atoms) > 0 and len(self.xyz) > 0:
                return True
            return False
            
        except Exception as e:
            print(f"Error in MOL file parsing: {e}")
            return False
    
    def translate(self, vector):
        """
        Translate ligand by a vector.
        
        Parameters:
        -----------
        vector : array-like
            Translation vector [x, y, z]
        """
        # Check if we have coordinates
        if self.xyz is None or len(self.xyz) == 0:
            print("Warning: Cannot translate, no coordinates present")
            return
            
        vector = np.array(vector)
        self.xyz += vector
        for atom in self.atoms:
            atom['coords'] += vector
    
    def rotate(self, rotation_matrix):
        """
        Rotate ligand using a rotation matrix.
        
        Parameters:
        -----------
        rotation_matrix : array-like
            3x3 rotation matrix
        """
        # Check if we have coordinates
        if self.xyz is None or len(self.xyz) == 0:
            print("Warning: Cannot rotate, no coordinates present")
            return
            
        rotation_matrix = np.array(rotation_matrix)
        self.xyz = np.dot(self.xyz, rotation_matrix.T)
        for atom in self.atoms:
            atom['coords'] = np.dot(atom['coords'], rotation_matrix.T)