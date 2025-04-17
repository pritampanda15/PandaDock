# ligand.py
import numpy as np
from pathlib import Path

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
        self.xyz = None
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
        
        try:
            # Use RDKit for robust molecule parsing if available
            from rdkit import Chem
            mol = Chem.MolFromMolFile(str(mol_path))
            if mol is None:
                raise ValueError(f"Failed to parse molecule file: {mol_file}")
            
            # Get atom coordinates
            conformer = mol.GetConformer()
            self.xyz = np.array([conformer.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
            
            # Get atom information
            for atom in mol.GetAtoms():
                self.atoms.append({
                    'idx': atom.GetIdx(),
                    'symbol': atom.GetSymbol(),
                    'formal_charge': atom.GetFormalCharge(),
                    'coords': self.xyz[atom.GetIdx()]
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
            
            print(f"Loaded ligand with {len(self.atoms)} atoms and {len(self.bonds)} bonds")
            print(f"Identified {len(self.rotatable_bonds)} rotatable bonds")
            
        except ImportError:
            # Fallback to simple MOL file parsing if RDKit is not available
            print("Warning: RDKit not available, using simplified MOL parser")
            self._parse_mol_file(mol_path)
    
    def _parse_mol_file(self, mol_path):
        """Simple parser for MOL files without RDKit."""
        with open(mol_path, 'r') as f:
            lines = f.readlines()
        
        # Parse atom and bond counts (line 4 in MOL format)
        counts_line = lines[3].strip()
        atom_count = int(counts_line[0:3])
        bond_count = int(counts_line[3:6])
        
        # Parse atoms (starts at line 5)
        atom_coords = []
        for i in range(atom_count):
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
        
        self.xyz = np.array(atom_coords)
        
        # Parse bonds
        for i in range(bond_count):
            bond_line = lines[4+atom_count+i]
            atom1 = int(bond_line[0:3].strip()) - 1  # MOL indices start at 1
            atom2 = int(bond_line[3:6].strip()) - 1
            bond_type = int(bond_line[6:9].strip())
            
            self.bonds.append({
                'begin_atom_idx': atom1,
                'end_atom_idx': atom2,
                'bond_type': bond_type
            })
    
    def generate_conformers(self, n_conformers=10):
        """
        Generate ligand conformers by rotating bonds using RDKit's ETKDG algorithm.
        
        Parameters:
        -----------
        n_conformers : int
            Number of conformers to generate
        
        Returns:
        --------
        list
            List of conformers as numpy arrays
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            
            # Convert the ligand to an RDKit molecule
            mol = Chem.MolFromMolBlock(self.to_mol_block())
            if mol is None:
                raise ValueError("Failed to convert ligand to RDKit molecule")
            
            # Add hydrogens to the molecule
            mol = Chem.AddHs(mol)
            
            # Generate conformers using the ETKDG algorithm
            params = AllChem.ETKDGv3()  # Use the latest version of ETKDG
            params.numThreads = 0  # Use all available threads
            params.maxAttempts = 1000  # Increase attempts for better success rate
            params.randomSeed = 42  # Set a seed for reproducibility
            params.pruneRmsThresh = 0.5  # Prune conformers with RMSD < 0.5 Å
            
            print(f"Generating {n_conformers} conformers using ETKDG...")
            AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)
            
            # Extract conformers as numpy arrays
            conformers = []
            for conf_id in range(mol.GetNumConformers()):
                conformer = mol.GetConformer(conf_id)
                coords = np.array([list(conformer.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
                conformers.append(coords)
            
            self.conformers = conformers
            print(f"Generated {len(conformers)} conformers")
            return conformers
        
        except ImportError:
            print("RDKit is required for conformer generation")
            return []
        except Exception as e:
            print(f"Error during conformer generation: {e}")
            return []
    
    def translate(self, vector):
        """
        Translate ligand by a vector.
        
        Parameters:
        -----------
        vector : array-like
            Translation vector [x, y, z]
        """
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
        rotation_matrix = np.array(rotation_matrix)
        self.xyz = np.dot(self.xyz, rotation_matrix.T)
        for atom in self.atoms:
            atom['coords'] = np.dot(atom['coords'], rotation_matrix.T)


