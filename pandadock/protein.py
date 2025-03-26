# protein.py
import numpy as np
from pathlib import Path

class Protein:
    """Class representing a protein structure."""
    
    def __init__(self, pdb_file=None):
        """
        Initialize a protein object.
        
        Parameters:
        -----------
        pdb_file : str
            Path to PDB file containing protein structure
        """
        self.atoms = []
        self.residues = {}
        self.active_site = None
        self.xyz = None
        
        if pdb_file:
            self.load_pdb(pdb_file)
    
    def load_pdb(self, pdb_file):
        """
        Load protein structure from PDB file.
        
        Parameters:
        -----------
        pdb_file : str
            Path to PDB file
        """
        pdb_path = Path(pdb_file)
        if not pdb_path.exists():
            raise FileNotFoundError(f"PDB file not found: {pdb_file}")
        
        with open(pdb_path, 'r') as f:
            atom_coords = []
            for line in f:
                if line.startswith("ATOM"):
                    # Parse PDB ATOM record
                    atom_name = line[12:16].strip()
                    residue_name = line[17:20].strip()
                    chain_id = line[21]
                    residue_id = int(line[22:26])
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    
                    # Store atom information
                    atom = {
                        'name': atom_name,
                        'residue_name': residue_name,
                        'chain_id': chain_id,
                        'residue_id': residue_id,
                        'coords': np.array([x, y, z])
                    }
                    self.atoms.append(atom)
                    atom_coords.append([x, y, z])
                    
                    # Organize by residue
                    res_key = f"{chain_id}_{residue_id}"
                    if res_key not in self.residues:
                        self.residues[res_key] = []
                    self.residues[res_key].append(atom)
            
            self.xyz = np.array(atom_coords)
            print(f"Loaded protein with {len(self.atoms)} atoms and {len(self.residues)} residues")
    
    def define_active_site(self, center, radius=10.0):
        """
        Define the active site of the protein.
        
        Parameters:
        -----------
        center : tuple or list
            (x, y, z) coordinates of active site center
        radius : float
            Radius of active site in Angstroms
        """
        self.active_site = {
            'center': np.array(center),
            'radius': radius
        }
        
        # Find atoms within the active site
        active_atoms = []
        active_residues = set()
        
        for i, atom in enumerate(self.atoms):
            distance = np.linalg.norm(self.xyz[i] - self.active_site['center'])
            if distance <= radius:
                active_atoms.append(atom)
                res_key = f"{atom['chain_id']}_{atom['residue_id']}"
                active_residues.add(res_key)
        
        self.active_site['atoms'] = active_atoms
        self.active_site['residues'] = list(active_residues)
        print(f"Defined active site with {len(active_atoms)} atoms and {len(active_residues)} residues")
    
    def detect_pockets(self):
        """
        Simple algorithm to detect potential binding pockets.
        
        Returns:
        --------
        list
            List of potential binding pockets as (center, radius) tuples
        """
        # This is a placeholder for a more sophisticated algorithm
        # A real implementation would use methods like fpocket, LIGSITE, etc.
        # For now, we'll just return a simple cluster-based approach
        
        from sklearn.cluster import DBSCAN
        
        # Only consider surface atoms (this is a simplification)
        # Real pocket detection would be more sophisticated
        pockets = []
        
        try:
            # Apply DBSCAN clustering to find dense regions
            clustering = DBSCAN(eps=3.5, min_samples=5).fit(self.xyz)
            labels = clustering.labels_
            
            # Get cluster centers and sizes
            unique_labels = set(labels)
            for label in unique_labels:
                if label == -1:  # Skip noise points
                    continue
                    
                cluster_points = self.xyz[labels == label]
                center = np.mean(cluster_points, axis=0)
                radius = np.max(np.linalg.norm(cluster_points - center, axis=1))
                
                pockets.append({
                    'center': center,
                    'radius': radius,
                    'size': len(cluster_points)
                })
            
            # Sort pockets by size (larger is likely more important)
            pockets = sorted(pockets, key=lambda x: x['size'], reverse=True)
            
        except Exception as e:
            print(f"Error in pocket detection: {e}")
            print("Make sure scikit-learn is installed")
        
        return pockets


