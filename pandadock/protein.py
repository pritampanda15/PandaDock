# protein.py
import numpy as np
from pathlib import Path
from .pockets import OptimizedCastpDetector

class Protein:
    """Class representing a protein structure."""
    
    def __init__(self, pdb_file=None, grid_center=None, grid_size=None):
        """
        Initialize a protein object.
        
        Parameters:
        -----------
        pdb_file : str
            Path to PDB file containing protein structure
        grid_center : tuple or list, optional
            (x, y, z) coordinates of the grid center
        grid_size : float, optional
            Size of the grid for docking
        """
        self.atoms = []
        self.residues = {}
        self.active_site = None
        self.xyz = None
        self.grid_center = grid_center
        self.grid_size = grid_size
        
        if pdb_file:
            self.load_pdb(pdb_file)
        if grid_center and grid_size:
            self.set_grid(grid_center, grid_size)
        self.flexible_residues = []
        self.flexible_residue_ids = []
        self.flexible_residue_indices = []
        self.flexible_residue_bonds = []
    
        if grid_center is not None:
                self.active_site = self._select_atoms_near(grid_center, radius=10.0)
        else:
                self.detect_pockets()
    
    def _select_atoms_near(self, center, radius=10.0):
        """Select atoms within a radius of the specified center point."""
        center = np.array(center)
        nearby_atoms = [
            atom for atom in self.atoms
            if np.linalg.norm(np.array(atom['coords']) - center) <= radius
        ]
        return {'atoms': nearby_atoms}
    

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
        Advanced algorithm to detect potential binding pockets.
        Returns:
        --------
        list
            List of potential binding pockets as dictionaries with center, radius, and other properties.
        """
        print("Detecting binding pockets...")
        detector = OptimizedCastpDetector(probe_radius=1.4, grid_spacing=1.0)  # Increased spacing for better performance
        pockets = detector.detect_pockets(self)
        
        if not pockets:
            print("No pockets detected. Using protein center as fallback.")
            center = np.mean(self.xyz, axis=0)
            radius = 10.0
            self.define_active_site(center, radius)
            return []
        
        print(f"Detected {len(pockets)} potential binding pockets.")
        
        # Add the pockets to the protein for visualization or further use
        for i, pocket in enumerate(pockets):
            print(f"Pocket {i+1}: volume = {pocket['volume']:.1f} Å³, {len(pocket['residues'])} residues")
            
        # Use the largest pocket as the default active site
        if pockets:
            self.define_active_site(pockets[0]['center'], np.power(3*pockets[0]['volume']/(4*np.pi), 1/3))
            
        return pockets
    # def detect_pockets(self):
    #     """
    #     Advanced algorithm to detect potential binding pockets using geometric and physicochemical properties.

    #     Returns:
    #     --------
    #     list
    #         List of potential binding pockets as dictionaries with center, radius, and other properties.
    #     """
    #     from scipy.spatial import ConvexHull, distance
    #     from sklearn.cluster import DBSCAN

    #     pockets = []

    #     try:
    #         # Step 1: Identify surface atoms
    #         print("Detecting surface atoms...")
    #         surface_atoms = self._get_surface_atoms()
    #         if len(surface_atoms) == 0:
    #             print("No surface atoms detected. Pocket detection failed.")
    #             # Fallback to protein centroid
    #             center = np.mean(self.xyz, axis=0)
    #             radius = 10.0
    #             self.define_active_site(center, radius)
    #             return []

    #         print(f"Detected {len(surface_atoms)} surface atoms.")

    #         # Step 2: Cluster surface atoms
    #         print("Clustering surface atoms to identify pockets...")
    #         clustering = DBSCAN(eps=4.0, min_samples=5).fit(surface_atoms)
    #         labels = clustering.labels_

    #         unique_labels = set(labels)
    #         for label in unique_labels:
    #             if label == -1:
    #                 continue
    #             cluster_points = surface_atoms[labels == label]
    #             if len(cluster_points) < 10:
    #                 continue

    #             center = np.mean(cluster_points, axis=0)
    #             radius = np.max(np.linalg.norm(cluster_points - center, axis=1))

    #             hull = ConvexHull(cluster_points)
    #             pockets.append({
    #                 'center': center,
    #                 'radius': radius,
    #                 'size': len(cluster_points),
    #                 'hull_volume': hull.volume,
    #                 'hull_area': hull.area
    #             })

    #         pockets = sorted(pockets, key=lambda x: (x['size'], x['hull_volume']), reverse=True)
    #         print(f"Detected {len(pockets)} potential binding pockets.")

    #     except Exception as e:
    #         print(f"Error in pocket detection: {e}")
    #         print("Ensure required libraries (scipy, sklearn) are installed.")

    #     return pockets

    # def detect_pockets(self):
    #     """
    #     Advanced algorithm to detect potential binding pockets.
    #     Returns:
    #     --------
    #     list
    #         List of potential binding pockets as dictionaries with center, radius, and other properties.
    #     """
    #     detector = CastpLikeDetector(probe_radius=1.4, grid_spacing=0.8)
    #     pockets = detector.detect_pockets(self)
        
    #     if not pockets:
    #         print("No pockets detected. Using protein center as fallback.")
    #         center = np.mean(self.xyz, axis=0)
    #         radius = 10.0
    #         self.define_active_site(center, radius)
    #         return []
        
    #     print(f"Detected {len(pockets)} potential binding pockets.")
    #     return pockets

    def _get_surface_atoms(self, probe_radius=1.4):
        """
        Identify surface atoms using a distance-based approach or SASA calculation.
        
        Parameters:
        -----------
        probe_radius : float
            Radius of the probe used to define the surface (default: 1.4 Å for water).
        
        Returns:
        --------
        np.ndarray
            Array of coordinates of surface atoms.
        """
        from scipy.spatial import KDTree

        surface_atoms = []
        kdtree = KDTree(self.xyz)

        for i, atom in enumerate(self.xyz):
            # Find neighbors within the probe radius
            neighbors = kdtree.query_ball_point(atom, r=probe_radius + 1.5)
            if len(neighbors) < 4:  # Surface atoms typically have fewer neighbors
                surface_atoms.append(atom)

        return np.array(surface_atoms)

    # Add to protein.py

    def define_flexible_residues(self, flexible_residue_ids, max_rotatable_bonds=3):
        """
        Define which residues are flexible.
        
        Parameters:
        -----------
        flexible_residue_ids : list
            List of residue IDs to make flexible
        max_rotatable_bonds : int
            Maximum number of rotatable bonds per residue
        """
        from .flexible_residues import FlexibleResidue
        
        self.flexible_residues = []
        
        print(f"Setting up {len(flexible_residue_ids)} flexible residues")
        
        for res_id in flexible_residue_ids:
            if res_id in self.residues:
                residue_atoms = self.residues[res_id]
                
                # Find rotatable bonds in this residue
                rotatable_bonds = self._find_rotatable_bonds(residue_atoms, max_rotatable_bonds)
                
                # Create flexible residue object
                flex_residue = FlexibleResidue(
                    residue_id=res_id,
                    atoms=residue_atoms,
                    rotatable_bonds=rotatable_bonds
                )
                
                self.flexible_residues.append(flex_residue)
                print(f"  Added flexible residue {res_id} with {len(rotatable_bonds)} rotatable bonds")
            else:
                print(f"  Warning: Residue {res_id} not found in protein")
                
        print(f"Defined {len(self.flexible_residues)} flexible residues with "
            f"total {sum(len(r.rotatable_bonds) for r in self.flexible_residues)} rotatable bonds")

    def _find_rotatable_bonds(self, residue_atoms, max_bonds):
        """
        Find rotatable bonds in a residue based on detailed chemistry rules.
        
        Parameters:
        -----------
        residue_atoms : list
            List of atoms in the residue
        max_bonds : int
            Maximum number of rotatable bonds to return
        
        Returns:
        --------
        list
            List of (atom1_idx, atom2_idx) tuples representing rotatable bonds
        """
        rotatable_bonds = []
        
        # Create atom indices mapping
        atom_indices = {atom['name']: i for i, atom in enumerate(residue_atoms)}
        
        # Get residue name
        residue_name = residue_atoms[0]['residue_name'] if residue_atoms else 'UNK'
        
        # Common backbone-to-sidechain bond (usually rotatable in all amino acids)
        if 'CA' in atom_indices and 'CB' in atom_indices:
            rotatable_bonds.append((atom_indices['CA'], atom_indices['CB']))
        
        # Common bonds found in multiple amino acids
        if 'CB' in atom_indices and 'CG' in atom_indices:
            rotatable_bonds.append((atom_indices['CB'], atom_indices['CG']))
        
        if 'CG' in atom_indices and 'CD' in atom_indices:
            rotatable_bonds.append((atom_indices['CG'], atom_indices['CD']))
        
        # Amino acid specific rotatable bonds
        if residue_name == 'ARG':  # Arginine - many rotatable bonds
            if 'CD' in atom_indices and 'NE' in atom_indices:
                rotatable_bonds.append((atom_indices['CD'], atom_indices['NE']))
            if 'NE' in atom_indices and 'CZ' in atom_indices:
                rotatable_bonds.append((atom_indices['NE'], atom_indices['CZ']))
        
        elif residue_name == 'LYS':  # Lysine - long flexible chain
            if 'CD' in atom_indices and 'CE' in atom_indices:
                rotatable_bonds.append((atom_indices['CD'], atom_indices['CE']))
            if 'CE' in atom_indices and 'NZ' in atom_indices:
                rotatable_bonds.append((atom_indices['CE'], atom_indices['NZ']))
        
        elif residue_name == 'MET':  # Methionine
            if 'CG' in atom_indices and 'SD' in atom_indices:
                rotatable_bonds.append((atom_indices['CG'], atom_indices['SD']))
            if 'SD' in atom_indices and 'CE' in atom_indices:
                rotatable_bonds.append((atom_indices['SD'], atom_indices['CE']))
        
        elif residue_name == 'GLU' or residue_name == 'GLN':  # Glutamic acid or Glutamine
            if 'CD' in atom_indices and 'OE1' in atom_indices:
                rotatable_bonds.append((atom_indices['CD'], atom_indices['OE1']))
        
        elif residue_name == 'ASP' or residue_name == 'ASN':  # Aspartic acid or Asparagine
            if 'CG' in atom_indices and 'OD1' in atom_indices:
                rotatable_bonds.append((atom_indices['CG'], atom_indices['OD1']))
        
        elif residue_name == 'PHE' or residue_name == 'TYR' or residue_name == 'TRP':  # Aromatic amino acids
            # Ring rotation around CG-CB bond is already covered above
            if residue_name == 'TYR' and 'CZ' in atom_indices and 'OH' in atom_indices:
                rotatable_bonds.append((atom_indices['CZ'], atom_indices['OH']))
        
        elif residue_name == 'SER' or residue_name == 'THR':  # Hydroxyl amino acids
            if 'CB' in atom_indices and 'OG' in atom_indices:
                rotatable_bonds.append((atom_indices['CB'], atom_indices['OG']))
            if residue_name == 'THR' and 'CB' in atom_indices and 'CG2' in atom_indices:
                rotatable_bonds.append((atom_indices['CB'], atom_indices['CG2']))
        
        elif residue_name == 'CYS':  # Cysteine
            if 'CB' in atom_indices and 'SG' in atom_indices:
                rotatable_bonds.append((atom_indices['CB'], atom_indices['SG']))
        
        elif residue_name == 'LEU' or residue_name == 'ILE':  # Branched amino acids
            if 'CG' in atom_indices and 'CD1' in atom_indices:
                rotatable_bonds.append((atom_indices['CG'], atom_indices['CD1']))
            if 'CG' in atom_indices and 'CD2' in atom_indices:
                rotatable_bonds.append((atom_indices['CG'], atom_indices['CD2']))
            if residue_name == 'ILE' and 'CB' in atom_indices and 'CG2' in atom_indices:
                rotatable_bonds.append((atom_indices['CB'], atom_indices['CG2']))
        
        elif residue_name == 'VAL':  # Valine
            if 'CB' in atom_indices and 'CG1' in atom_indices:
                rotatable_bonds.append((atom_indices['CB'], atom_indices['CG1']))
            if 'CB' in atom_indices and 'CG2' in atom_indices:
                rotatable_bonds.append((atom_indices['CB'], atom_indices['CG2']))
        
        elif residue_name == 'HIS':  # Histidine
            # Imidazole ring rotation around CB-CG bond is already covered above
            pass
        
        # Avoid adding bonds for PRO (Proline) and GLY (Glycine) - no meaningful rotatable bonds
        
        # Remove duplicate bonds (if any)
        unique_bonds = []
        for bond in rotatable_bonds:
            if bond not in unique_bonds and (bond[1], bond[0]) not in unique_bonds:
                unique_bonds.append(bond)
        
        # Sort bonds by atom indices for consistency
        sorted_bonds = [(min(a1, a2), max(a1, a2)) for a1, a2 in unique_bonds]
        
        # Limit to maximum number of bonds
        return sorted_bonds[:max_bonds]
