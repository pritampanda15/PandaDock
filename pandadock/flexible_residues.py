# flexible_residues.py
import numpy as np
from copy import deepcopy
from scipy.spatial.transform import Rotation
from typing import List, Dict, Tuple

class FlexibleResidue:
    """Enhanced class representing a flexible residue with proper rotamer handling."""
    
    def __init__(self, residue_id: str, atoms: List[Dict], rotatable_bonds: List[Tuple[int, int]], 
                 rotamer_library: List[np.ndarray] = None):
        """
        Initialize a flexible residue with rotamer support.
        
        Parameters:
        -----------
        residue_id : str
            Residue identifier (e.g., "A_42_SER")
        atoms : list
            List of atom dictionaries with 'coords', 'atom_name', 'element', etc.
        rotatable_bonds : list
            List of rotatable bonds as (atom1_idx, atom2_idx) tuples
        rotamer_library : list, optional
            List of rotamer conformations as coordinate arrays
        """
        self.residue_id = residue_id
        self.atoms = deepcopy(atoms)
        self.rotatable_bonds = rotatable_bonds
        self.original_coords = np.array([atom['coords'] for atom in atoms])
        self.current_rotamer_idx = -1  # -1 indicates original conformation
        
        # Initialize rotamer library
        self.rotamer_library = rotamer_library or []
        if rotamer_library:
            self._validate_rotamers()
        
        # Build molecular graph
        self._build_molecular_graph()
        
    def _build_molecular_graph(self):
        """Build molecular graph for rotation propagation."""
        self.graph = {i: set() for i in range(len(self.atoms))}
        
        # Create bonds from atom connectivity
        for i, atom in enumerate(self.atoms):
            if 'bonds' in atom:
                for bonded_atom in atom['bonds']:
                    self.graph[i].add(bonded_atom)
                    self.graph[bonded_atom].add(i)
    
    def _validate_rotamers(self):
        """Validate rotamer library dimensions."""
        for rotamer in self.rotamer_library:
            if rotamer.shape != self.original_coords.shape:
                raise ValueError("Rotamer coordinates must match original atom count")
    
    def apply_rotamer(self, rotamer_idx: int):
        """Apply a rotamer from the library."""
        if 0 <= rotamer_idx < len(self.rotamer_library):
            for i, coords in enumerate(self.rotamer_library[rotamer_idx]):
                self.atoms[i]['coords'] = coords.copy()
            self.current_rotamer_idx = rotamer_idx
    
    def rotate_bond(self, bond_idx: int, angle: float):
        """
        Rotate a bond by the specified angle with proper side chain handling.
        
        Parameters:
        -----------
        bond_idx : int
            Index of the bond to rotate
        angle : float
            Rotation angle in radians
        """
        if bond_idx >= len(self.rotatable_bonds):
            return
            
        atom1_idx, atom2_idx = self.rotatable_bonds[bond_idx]
        self._rotate_atoms_around_bond(atom1_idx, atom2_idx, angle)
        self.current_rotamer_idx = -1  # Mark as custom conformation
    
    def _rotate_atoms_around_bond(self, fixed_atom_idx: int, pivot_atom_idx: int, angle: float):
        """Rotate atoms around a bond using proper graph traversal."""
        # Get rotation axis and center
        center = self.atoms[pivot_atom_idx]['coords']
        axis = center - self.atoms[fixed_atom_idx]['coords']
        axis = axis / np.linalg.norm(axis)
        
        # Find all atoms that should rotate (BFS)
        rotating_atoms = self._find_rotating_atoms(fixed_atom_idx, pivot_atom_idx)
        
        # Apply rotation
        rotation = Rotation.from_rotvec(axis * angle)
        for atom_idx in rotating_atoms:
            self.atoms[atom_idx]['coords'] = rotation.apply(
                self.atoms[atom_idx]['coords'] - center
            ) + center
    
    def _find_rotating_atoms(self, fixed_atom_idx: int, pivot_atom_idx: int) -> List[int]:
        """Find all atoms that should rotate when a bond is rotated."""
        visited = set()
        queue = [pivot_atom_idx]
        rotating_atoms = []
        
        while queue:
            current = queue.pop(0)
            if current not in visited and current != fixed_atom_idx:
                visited.add(current)
                rotating_atoms.append(current)
                queue.extend(self.graph[current] - visited)
        
        return rotating_atoms
    
    def reset_to_original(self):
        """Reset residue to original conformation."""
        for i, coords in enumerate(self.original_coords):
            self.atoms[i]['coords'] = coords.copy()
        self.current_rotamer_idx = -1
    
    def get_coords(self) -> np.ndarray:
        """Get current coordinates of all atoms."""
        return np.array([atom['coords'] for atom in self.atoms])