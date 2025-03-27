# flexible_residues.py
import numpy as np
from copy import deepcopy
from scipy.spatial.transform import Rotation

class FlexibleResidue:
    """Class representing a flexible residue in a protein."""
    
    def __init__(self, residue_id, atoms, rotatable_bonds, original_positions=None):
        """
        Initialize a flexible residue.
        
        Parameters:
        -----------
        residue_id : str
            Residue identifier (e.g., "A_42")
        atoms : list
            List of atom dictionaries for this residue
        rotatable_bonds : list
            List of rotatable bonds as (atom1_idx, atom2_idx) tuples
        original_positions : array, optional
            Original atom positions to enable reset
        """
        self.residue_id = residue_id
        self.atoms = atoms
        self.rotatable_bonds = rotatable_bonds
        self.original_positions = original_positions or np.array([atom['coords'] for atom in atoms])
        
        # Create atom adjacency lists for rotation propagation
        self.adjacency_list = self._build_adjacency_list()
    
    def _build_adjacency_list(self):
        """Build atom adjacency list from bonds."""
        adj_list = {i: [] for i in range(len(self.atoms))}
        
        # Create bonds from atoms if needed
        bonds = []
        for i, atom in enumerate(self.atoms):
            if 'bonds' in atom:
                for bonded_atom in atom['bonds']:
                    bonds.append((i, bonded_atom))
        
        # Add all bonds to adjacency list
        for a1, a2 in bonds:
            adj_list[a1].append(a2)
            adj_list[a2].append(a1)
            
        return adj_list
    
    def rotate_bond(self, bond_idx, angle):
        """
        Rotate a bond by the specified angle.
        
        Parameters:
        -----------
        bond_idx : int
            Index of the bond to rotate
        angle : float
            Rotation angle in radians
        """
        if bond_idx >= len(self.rotatable_bonds):
            return
            
        # Get the bond atoms
        atom1_idx, atom2_idx = self.rotatable_bonds[bond_idx]
        
        # Get coordinates
        atom1_coords = self.atoms[atom1_idx]['coords']
        atom2_coords = self.atoms[atom2_idx]['coords']
        
        # Define rotation axis (the bond vector)
        axis = atom2_coords - atom1_coords
        axis = axis / np.linalg.norm(axis)
        
        # Create rotation object
        rotation = Rotation.from_rotvec(axis * angle)
        
        # Determine which atoms to rotate (those connected to atom2 excluding atom1)
        atoms_to_rotate = self._get_atoms_to_rotate(atom1_idx, atom2_idx)
        
        # Rotate the selected atoms
        for idx in atoms_to_rotate:
            # Translate to origin (relative to atom1)
            v = self.atoms[idx]['coords'] - atom1_coords
            
            # Apply rotation
            v_rotated = rotation.apply(v)
            
            # Translate back
            self.atoms[idx]['coords'] = atom1_coords + v_rotated
    
    def _get_atoms_to_rotate(self, fixed_atom_idx, pivot_atom_idx):
        """Find atoms that should rotate when a bond is rotated."""
        # Use BFS to find all connected atoms from pivot, excluding the path through fixed
        visited = {fixed_atom_idx}
        queue = [pivot_atom_idx]
        result = []
        
        while queue:
            atom_idx = queue.pop(0)
            if atom_idx not in visited:
                visited.add(atom_idx)
                result.append(atom_idx)
                queue.extend([a for a in self.adjacency_list[atom_idx] if a not in visited])
        
        return result
    
    def reset_to_original(self):
        """Reset residue to original conformation."""
        for i, pos in enumerate(self.original_positions):
            self.atoms[i]['coords'] = pos.copy()