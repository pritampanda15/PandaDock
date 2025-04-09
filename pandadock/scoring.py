# scoring.py
import numpy as np
from scipy.spatial.distance import cdist

class ScoringFunction:
    """Base class for scoring functions."""
    
    def __init__(self):
        # Atomic parameters
        self.vdw_radii = {
            'H': 1.2, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8,
            'P': 1.8, 'F': 1.47, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98
        }
        self.atom_depths = {
            'H': 0.1, 'C': 0.2, 'N': 0.15, 'O': 0.15, 'S': 0.2,
            'P': 0.2, 'F': 0.15, 'Cl': 0.2, 'Br': 0.2, 'I': 0.2
        }
    
    def score(self, protein, ligand):
        """
        Calculate binding score between protein and ligand.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        ligand : Ligand
            Ligand object
        
        Returns:
        --------
        float
            Binding score (lower is better)
        """
        raise NotImplementedError("Subclasses must implement this method")


class VdwScoringFunction(ScoringFunction):
    """Simple van der Waals scoring function."""
    
    def score(self, protein, ligand):
        """Calculate van der Waals interaction score."""
        # If active site is defined, only consider those atoms
        if protein.active_site and 'atoms' in protein.active_site:
            protein_atoms = protein.active_site['atoms']
        else:
            protein_atoms = protein.atoms
        
        score = 0.0
        for p_atom in protein_atoms:
            p_coords = p_atom['coords']
            p_symbol = p_atom.get('element', p_atom.get('name', 'C'))[0]
            
            for l_atom in ligand.atoms:
                l_coords = l_atom['coords']
                l_symbol = l_atom.get('symbol', 'C')
                
                # Calculate distance
                distance = np.linalg.norm(p_coords - l_coords)
                
                # Get VDW parameters
                p_radius = self.vdw_radii.get(p_symbol, 1.7)
                l_radius = self.vdw_radii.get(l_symbol, 1.7)
                optimal_dist = p_radius + l_radius
                
                # Simple Lennard-Jones-like potential
                if distance < 0.1:  # Avoid division by zero
                    score += 1000
                else:
                    # Simplified 6-12 potential
                    ratio = optimal_dist / distance
                    vdw_energy = (ratio**12 - 2 * ratio**6)
                    score += vdw_energy
        
        return score


class HBondScoringFunction(ScoringFunction):
    """Hydrogen bond scoring function."""
    
    def __init__(self):
        super().__init__()
        # Define H-bond donors and acceptors by atom type
        self.donors = ['N', 'O']  # Simplified; should include -NH, -OH groups
        self.acceptors = ['N', 'O']  # Simplified
    
    def score(self, protein, ligand):
        """Calculate hydrogen bond score."""
        # Get protein atoms
        if protein.active_site and 'atoms' in protein.active_site:
            protein_atoms = protein.active_site['atoms']
        else:
            protein_atoms = protein.atoms
        
        score = 0.0
        for p_atom in protein_atoms:
            p_coords = p_atom['coords']
            p_name = p_atom.get('name', '')[0]
            
            for l_atom in ligand.atoms:
                l_coords = l_atom['coords']
                l_symbol = l_atom.get('symbol', '')
                
                # Calculate distance
                distance = np.linalg.norm(p_coords - l_coords)
                
                # Check if H-bond is possible
                if distance < 3.5:  # Maximum H-bond distance
                    # Protein donor - Ligand acceptor
                    if p_name in self.donors and l_symbol in self.acceptors:
                        # Ideal H-bond distance ~2.8-3.0 Å
                        hbond_score = 1.0 - abs(distance - 2.9) / 1.0
                        if hbond_score > 0:
                            score -= hbond_score * 5.0  # H-bonds are favorable
                    
                    # Ligand donor - Protein acceptor
                    if l_symbol in self.donors and p_name in self.acceptors:
                        hbond_score = 1.0 - abs(distance - 2.9) / 1.0
                        if hbond_score > 0:
                            score -= hbond_score * 5.0
        
        return score


class CompositeScoringFunction(ScoringFunction):
    """Composite scoring function combining multiple terms."""
    
    def __init__(self):
        super().__init__()
        self.vdw_scorer = VdwScoringFunction()
        self.hbond_scorer = HBondScoringFunction()
        
        # Weights for different scoring terms
        self.weights = {
            'vdw': 1.0,
            'hbond': 2.0,
            'clash': 10.0
        }
    
    def score(self, protein, ligand):
        """Calculate composite score."""
        # Calculate individual terms
        vdw_score = self.vdw_scorer.score(protein, ligand)
        hbond_score = self.hbond_scorer.score(protein, ligand)
        
        # Calculate clash score (severe steric clashes)
        clash_score = self._calculate_clashes(protein, ligand)
        
        # Combine scores
        total_score = (
            self.weights['vdw'] * vdw_score +
            self.weights['hbond'] * hbond_score +
            self.weights['clash'] * clash_score
        )
        
        return total_score
    
    def _calculate_clashes(self, protein, ligand):
        """Calculate severe steric clashes."""
        # Get protein atoms
        if protein.active_site and 'atoms' in protein.active_site:
            protein_atoms = protein.active_site['atoms']
        else:
            protein_atoms = protein.atoms
        
        clash_score = 0.0
        for p_atom in protein_atoms:
            p_coords = p_atom['coords']
            p_symbol = p_atom.get('element', p_atom.get('name', 'C'))[0]
            
            for l_atom in ligand.atoms:
                l_coords = l_atom['coords']
                l_symbol = l_atom.get('symbol', 'C')
                
                # Calculate distance
                distance = np.linalg.norm(p_coords - l_coords)
                
                # Get VDW parameters
                p_radius = self.vdw_radii.get(p_symbol, 1.7)
                l_radius = self.vdw_radii.get(l_symbol, 1.7)
                min_allowed = (p_radius + l_radius) * 0.7  # Allow some overlap
                
                # Penalize severe clashes
                if distance < min_allowed:
                    clash_factor = (min_allowed - distance) / min_allowed
                    clash_score += clash_factor ** 2  # Quadratic penalty
        
        return clash_score


class EnhancedScoringFunction(CompositeScoringFunction):
    """
    Enhanced scoring function with additional interaction terms for more reliable docking.
    Incorporates electrostatics, desolvation, and hydrophobic effects.
    """
    
    def __init__(self):
        super().__init__()
        
        # Atom type parameters
        self.atom_charges = {
            'H': 0.0, 'C': 0.0, 'N': -0.5, 'O': -0.5, 'S': -0.2,
            'P': 0.5, 'F': -0.25, 'Cl': -0.1, 'Br': -0.1, 'I': -0.1
        }
        
        # Solvation parameters
        self.atom_solvation = {
            'H': 0.0, 'C': 0.4, 'N': -0.2, 'O': -0.3, 'S': 0.6,
            'P': -0.3, 'F': 0.1, 'Cl': 0.5, 'Br': 0.5, 'I': 0.5
        }
        
        # Hydrophobic atom types
        self.hydrophobic_types = ['C', 'S', 'Cl', 'Br', 'I']
        
        # Distance cutoffs
        self.elec_cutoff = 12.0  # Å for electrostatics
        self.desolv_cutoff = 8.0  # Å for desolvation
        self.hydrophobic_cutoff = 4.5  # Å for hydrophobic interactions
        
        # Adjust weights for different terms
        self.weights = {
            'vdw': 1.0,
            'hbond': 2.5,
            'elec': 1.5,
            'desolv': 1.0,
            'hydrophobic': 1.0,
            'clash': 10.0,
            'entropy': 0.5
        }
    
    def score(self, protein, ligand):
        """Calculate enhanced composite score."""
        # Calculate base terms
        vdw_score = self.vdw_scorer.score(protein, ligand)
        hbond_score = self.hbond_scorer.score(protein, ligand)
        clash_score = self._calculate_clashes(protein, ligand)
        
        # Calculate additional terms
        elec_score = self._calculate_electrostatics(protein, ligand)
        desolv_score = self._calculate_desolvation(protein, ligand)
        hydrophobic_score = self._calculate_hydrophobic(protein, ligand)
        entropy_score = self._calculate_entropy(ligand)
        
        # Combine scores
        total_score = (
            self.weights['vdw'] * vdw_score +
            self.weights['hbond'] * hbond_score +
            self.weights['elec'] * elec_score +
            self.weights['desolv'] * desolv_score +
            self.weights['hydrophobic'] * hydrophobic_score +
            self.weights['clash'] * clash_score +
            self.weights['entropy'] * entropy_score
        )
        
        return total_score
    
    def _calculate_electrostatics(self, protein, ligand):
        """
        Calculate electrostatic interactions using a distance-dependent dielectric.
        
        Returns:
        --------
        float
            Electrostatic interaction score
        """
        # Get protein atoms
        if protein.active_site and 'atoms' in protein.active_site:
            protein_atoms = protein.active_site['atoms']
        else:
            protein_atoms = protein.atoms
        
        elec_score = 0.0
        
        for p_atom in protein_atoms:
            p_coords = p_atom['coords']
            p_symbol = p_atom.get('element', p_atom.get('name', 'C'))[0]
            p_charge = self.atom_charges.get(p_symbol, 0.0)
            
            for l_atom in ligand.atoms:
                l_coords = l_atom['coords']
                l_symbol = l_atom.get('symbol', 'C')
                l_charge = self.atom_charges.get(l_symbol, 0.0)
                
                # Skip if both charges are zero
                if abs(p_charge * l_charge) < 1e-6:
                    continue
                
                # Calculate distance
                distance = np.linalg.norm(p_coords - l_coords)
                
                # Skip if beyond cutoff
                if distance > self.elec_cutoff:
                    continue
                
                # Distance-dependent dielectric constant
                dielectric = 4.0 * distance
                
                # Coulomb's law with distance-dependent dielectric
                if distance < 0.1:  # Avoid division by zero
                    elec_energy = 0.0
                else:
                    elec_energy = 332.0 * p_charge * l_charge / (dielectric * distance)
                
                elec_score += elec_energy
        
        return elec_score
    
    def _calculate_desolvation(self, protein, ligand):
        """
        Calculate desolvation effects using atomic solvation parameters.
        
        Returns:
        --------
        float
            Desolvation score
        """
        # Get protein atoms
        if protein.active_site and 'atoms' in protein.active_site:
            protein_atoms = protein.active_site['atoms']
        else:
            protein_atoms = protein.atoms
        
        desolv_score = 0.0
        
        for p_atom in protein_atoms:
            p_coords = p_atom['coords']
            p_symbol = p_atom.get('element', p_atom.get('name', 'C'))[0]
            p_solvation = self.atom_solvation.get(p_symbol, 0.0)
            
            for l_atom in ligand.atoms:
                l_coords = l_atom['coords']
                l_symbol = l_atom.get('symbol', 'C')
                l_solvation = self.atom_solvation.get(l_symbol, 0.0)
                
                # Calculate distance
                distance = np.linalg.norm(p_coords - l_coords)
                
                # Skip if beyond cutoff
                if distance > self.desolv_cutoff:
                    continue
                
                # Gaussian-like desolvation effect
                desolv_energy = p_solvation * l_solvation * np.exp(-(distance**2) / 7.5)
                desolv_score += desolv_energy
        
        return desolv_score
    
    def _calculate_hydrophobic(self, protein, ligand):
        """
        Calculate hydrophobic interactions.
        
        Returns:
        --------
        float
            Hydrophobic interaction score
        """
        # Get protein atoms
        if protein.active_site and 'atoms' in protein.active_site:
            protein_atoms = protein.active_site['atoms']
        else:
            protein_atoms = protein.atoms
        
        hydrophobic_score = 0.0
        
        # Identify hydrophobic atoms
        p_hydrophobic = [atom for atom in protein_atoms 
                        if atom.get('element', atom.get('name', ''))[0] in self.hydrophobic_types]
        
        l_hydrophobic = [atom for atom in ligand.atoms 
                        if atom.get('symbol', '') in self.hydrophobic_types]
        
        for p_atom in p_hydrophobic:
            p_coords = p_atom['coords']
            
            for l_atom in l_hydrophobic:
                l_coords = l_atom['coords']
                
                # Calculate distance
                distance = np.linalg.norm(p_coords - l_coords)
                
                # Skip if beyond cutoff
                if distance > self.hydrophobic_cutoff:
                    continue
                
                # Linear hydrophobic interaction term
                if distance < 0.5:  # Avoid unrealistic close contacts
                    contact_score = 0.0
                else:
                    # Stronger interaction for closer contact
                    contact_score = (self.hydrophobic_cutoff - distance) / self.hydrophobic_cutoff
                
                hydrophobic_score -= contact_score  # Negative since it's favorable
        
        return hydrophobic_score
    
    def _calculate_entropy(self, ligand):
        """
        Estimate entropy penalty based on ligand flexibility.
        
        Returns:
        --------
        float
            Entropy score
        """
        # Simple entropy term based on rotatable bonds
        if hasattr(ligand, 'rotatable_bonds'):
            n_rotatable = len(ligand.rotatable_bonds)
            entropy_penalty = 0.5 * n_rotatable
        else:
            # Fallback if rotatable bonds not defined
            entropy_penalty = 0.0
        
        return entropy_penalty

# Add this class after your existing scoring function classes
class TetheredScoringFunction:
    """
    A scoring function wrapper that adds a penalty term based on RMSD from a reference position.
    """
    
    def __init__(self, base_scoring_function, reference_ligand, weight=10.0, max_penalty=100.0):
        self.base_scoring_function = base_scoring_function
        self.reference_coordinates = reference_ligand.xyz.copy()
        self.weight = weight
        self.max_penalty = max_penalty
        
    def score(self, protein, ligand):
        # Get the base score
        base_score = self.base_scoring_function.score(protein, ligand)
        
        # Calculate RMSD from reference
        rmsd = self._calculate_rmsd(ligand.xyz)
        
        # Apply RMSD penalty, capped at max_penalty
        rmsd_penalty = min(self.weight * rmsd, self.max_penalty)
        
        # Return combined score
        return base_score + rmsd_penalty
    
    def _calculate_rmsd(self, coordinates):
        # Ensure same number of atoms
        if len(coordinates) != len(self.reference_coordinates):
            raise ValueError(f"Coordinate mismatch: reference has {len(self.reference_coordinates)} atoms, but current pose has {len(coordinates)} atoms")
        
        # Calculate squared differences
        squared_diff = np.sum((coordinates - self.reference_coordinates) ** 2, axis=1)
        
        # Return RMSD
        return np.sqrt(np.mean(squared_diff))
    
class PhysicsBasedScoringFunction(EnhancedScoringFunction):
    """
    Physics-based scoring function using calibrated energy terms and parameters.
    Implements a comprehensive free energy model for protein-ligand binding.
    """
    
    def __init__(self):
        super().__init__()
        
        # Override with physics-based calibrated weights
        self.weights = {
            'vdw': 0.1662,           # Van der Waals weight
            'hbond': 0.1209,         # Hydrogen bond weight
            'elec': 0.1406,          # Electrostatic weight 
            'desolv': 0.1322,        # Desolvation weight
            'entropy': 0.2983,       # Torsional entropy weight
            'clash': 1.0             # Keep clash detection
        }
        
        # Extended atom type parameters for more precise interactions
        self.atom_type_map = {
            # Carbon types (more specific than just 'C')
            'C.3': 'C',    # sp3 carbon
            'C.2': 'A',    # sp2 carbon
            'C.ar': 'A',   # aromatic carbon
            'C.1': 'C',    # sp carbon
            
            # Nitrogen types
            'N.3': 'N',    # sp3 nitrogen
            'N.2': 'NA',   # sp2 nitrogen
            'N.ar': 'NA',  # aromatic nitrogen
            'N.1': 'N',    # sp nitrogen
            'N.am': 'NA',  # amide nitrogen
            
            # Oxygen types
            'O.3': 'OA',   # sp3 oxygen (hydroxyl, ether)
            'O.2': 'O',    # sp2 oxygen (carbonyl)
            'O.co2': 'OA', # carboxylate oxygen
            
            # Sulfur types
            'S.3': 'SA',   # sp3 sulfur
            'S.2': 'S',    # sp2 sulfur
            
            # Default element mappings (fallback)
            'C': 'C',    # Non-polar carbon
            'N': 'N',    # Nitrogen
            'O': 'O',    # Oxygen
            'S': 'S',    # Sulfur
            'P': 'P',    # Phosphorus
            'F': 'F',    # Fluorine
            'Cl': 'Cl',  # Chlorine
            'Br': 'Br',  # Bromine
            'I': 'I',    # Iodine
            'H': 'H',    # Hydrogen
        }
        
        # VDW parameters (r_eq and epsilon)
        self.vdw_params = {
            'C': {'r_eq': 4.00, 'epsilon': 0.150},
            'A': {'r_eq': 4.00, 'epsilon': 0.150},  # Aromatic carbon
            'N': {'r_eq': 3.50, 'epsilon': 0.160},
            'NA': {'r_eq': 3.50, 'epsilon': 0.160}, # H-bond acceptor N
            'O': {'r_eq': 3.20, 'epsilon': 0.200},
            'OA': {'r_eq': 3.20, 'epsilon': 0.200}, # H-bond acceptor O
            'S': {'r_eq': 4.00, 'epsilon': 0.200},
            'SA': {'r_eq': 4.00, 'epsilon': 0.200}, # H-bond acceptor S
            'H': {'r_eq': 2.00, 'epsilon': 0.020},
            'F': {'r_eq': 3.09, 'epsilon': 0.080},
            'Cl': {'r_eq': 3.90, 'epsilon': 0.276},
            'Br': {'r_eq': 4.33, 'epsilon': 0.389},
            'I': {'r_eq': 4.72, 'epsilon': 0.550},
            'P': {'r_eq': 4.20, 'epsilon': 0.200},
        }
        
        # H-bond parameters (12-10 potential)
        self.hbond_params = {
            # Donor-Acceptor type pairs (r_eq in Å, epsilon in kcal/mol)
            'O-O': {'r_eq': 1.90, 'epsilon': 5.0},
            'O-N': {'r_eq': 1.90, 'epsilon': 5.0},
            'N-O': {'r_eq': 1.90, 'epsilon': 5.0},
            'N-N': {'r_eq': 1.90, 'epsilon': 5.0},
            'O-S': {'r_eq': 2.50, 'epsilon': 1.0},
            'N-S': {'r_eq': 2.50, 'epsilon': 1.0},
        }
        
        # Define H-bond donors and acceptors more precisely
        self.hbond_donor_types = {'N', 'NA', 'O', 'OA', 'N.3', 'N.am', 'O.3'}
        self.hbond_acceptor_types = {'O', 'OA', 'N', 'NA', 'SA', 'O.2', 'O.3', 'N.2'}
        
        # Update hydrophobic types
        self.hydrophobic_types = ['C', 'A', 'F', 'Cl', 'Br', 'I', 'C.3', 'C.2', 'C.ar']
        
        # Atomic solvation parameters (kcal/mol·Å²)
        self.atom_solvation_params = {
            'C': 12.77,
            'A': 12.77,  # Aromatic carbon
            'N': -17.40,
            'NA': -17.40,
            'O': -17.40,
            'OA': -17.40,
            'S': -8.31,
            'SA': -8.31,
            'H': 0.0,
            'F': -6.60,
            'Cl': -0.72,
            'Br': -0.85,
            'I': -0.88,
            'P': -6.70,
        }
        
        # Atom volume parameters (Å³)
        self.atom_volume_params = {
            'C': 33.51,
            'A': 33.51,
            'N': 22.45, 
            'NA': 22.45,
            'O': 17.07,
            'OA': 17.07,
            'S': 33.51,
            'SA': 33.51,
            'H': 0.0,
            'F': 15.45,
            'Cl': 35.82,
            'Br': 42.56,
            'I': 55.06,
            'P': 38.80,
        }
        
        # Constants for desolvation
        self.solpar = 0.01097   # kcal/mol per Å²
        self.solvation_k = 3.5  # Å, solvation radius
        
        # Distance cutoffs
        self.vdw_cutoff = 8.0  # Å for van der Waals
        self.hbond_cutoff = 4.0  # Å for hydrogen bonds
        self.elec_cutoff = 20.0  # Å for electrostatics
        self.desolv_cutoff = 8.0  # Å for desolvation
        
    def _get_atom_type(self, atom, default='C'):
        """Determine the atom type for an atom based on available information."""
        # Try to get atom type from atom data
        atom_type = atom.get('type', None)
        if atom_type and atom_type in self.atom_type_map:
            return self.atom_type_map[atom_type]
        
        # Fall back to element
        element = None
        if 'element' in atom:
            element = atom['element']
        elif 'name' in atom:
            element = atom['name'][0]
        elif 'symbol' in atom:
            element = atom['symbol']
            
        if element:
            # Convert to uppercase for consistency
            element = element.upper()
            if element in self.atom_type_map:
                return self.atom_type_map[element]
        
        # Default fallback
        return default
    
    def score(self, protein, ligand):
        """Calculate comprehensive physics-based binding score."""
        # Get protein and ligand atoms
        if protein.active_site and 'atoms' in protein.active_site:
            protein_atoms = protein.active_site['atoms']
        else:
            protein_atoms = protein.atoms
            
        ligand_atoms = ligand.atoms
        
        # Calculate all energy terms
        vdw_energy = self._calculate_vdw_physics(protein_atoms, ligand_atoms)
        hbond_energy = self._calculate_hbond_physics(protein_atoms, ligand_atoms, protein, ligand)
        elec_energy = self._calculate_electrostatics_physics(protein_atoms, ligand_atoms)
        desolv_energy = self._calculate_desolvation_physics(protein_atoms, ligand_atoms)
        entropy_energy = self._calculate_entropy(ligand)
        
        # Combine scores with calibrated weights
        total_score = (
            self.weights['vdw'] * vdw_energy +
            self.weights['hbond'] * hbond_energy +
            self.weights['elec'] * elec_energy +
            self.weights['desolv'] * desolv_energy +
            self.weights['entropy'] * entropy_energy
        )
        
        return total_score
    
    def _calculate_vdw_physics(self, protein_atoms, ligand_atoms):
        """
        Calculate van der Waals energy using a modified 12-6 Lennard-Jones potential
        with atom-specific parameters and smoothing function for close contacts.
        """
        vdw_energy = 0.0
        
        for p_atom in protein_atoms:
            p_type = self._get_atom_type(p_atom)
            p_coords = p_atom['coords']
            p_params = self.vdw_params.get(p_type, self.vdw_params['C'])
            
            for l_atom in ligand_atoms:
                l_type = self._get_atom_type(l_atom)
                l_coords = l_atom['coords']
                l_params = self.vdw_params.get(l_type, self.vdw_params['C'])
                
                # Calculate distance
                distance = np.linalg.norm(p_coords - l_coords)
                
                # Skip if beyond cutoff
                if distance > self.vdw_cutoff:
                    continue
                
                # Calculate combined parameters
                r_eq = (p_params['r_eq'] + l_params['r_eq']) / 2.0  # Arithmetic mean
                epsilon = np.sqrt(p_params['epsilon'] * l_params['epsilon'])  # Geometric mean
                
                # Prevent division by zero
                if distance < 0.1:
                    distance = 0.1
                
                # Calculate ratio for efficiency
                ratio = r_eq / distance
                
                # Use modified potential with smoothing at close distances
                if distance >= 0.5 * r_eq:
                    # Regular 12-6 Lennard-Jones
                    vdw_term = epsilon * ((ratio**12) - 2.0 * (ratio**6))
                else:
                    # Smoothed function for very close distances
                    smoothed_ratio = 0.5 * r_eq / distance
                    vdw_term = epsilon * ((smoothed_ratio**12) - 2.0 * (smoothed_ratio**6))
                
                vdw_energy += vdw_term
        
        return vdw_energy
    
    def _calculate_hbond_physics(self, protein_atoms, ligand_atoms, protein, ligand):
        """
        Calculate hydrogen bonding using a 12-10 potential with angular dependence.
        """
        hbond_energy = 0.0
        
        # Check for protein donor - ligand acceptor pairs
        for p_atom in protein_atoms:
            p_type = self._get_atom_type(p_atom)
            p_coords = p_atom['coords']
            p_element = p_atom.get('element', p_atom.get('name', 'C'))[0].upper()
            
            for l_atom in ligand_atoms:
                l_type = self._get_atom_type(l_atom)
                l_coords = l_atom['coords']
                l_element = l_atom.get('symbol', 'C').upper()
                
                # Calculate distance
                distance = np.linalg.norm(p_coords - l_coords)
                
                # Skip if beyond cutoff
                if distance > self.hbond_cutoff:
                    continue
                
                # Protein donor - Ligand acceptor
                if p_element in self.hbond_donor_types and l_element in self.hbond_acceptor_types:
                    # Get H-bond parameters
                    hb_key = f"{p_element}-{l_element}"
                    
                    # Look up parameters or use defaults
                    if hb_key in self.hbond_params:
                        params = self.hbond_params[hb_key]
                    else:
                        # Default parameters for this pair
                        params = {'r_eq': 1.9, 'epsilon': 3.0}
                    
                    # Get parameters
                    r_eq = params['r_eq']
                    epsilon = params['epsilon']
                    
                    # 12-10 potential
                    if distance < 0.1:
                        distance = 0.1
                    
                    # Calculate 12-10 energy term (ideal at r_eq)
                    ratio = r_eq / distance
                    hbond_term = epsilon * ((ratio**12) - 2.0 * (ratio**10))
                    
                    # Apply angular factor (simplified)
                    angle_factor = self._calculate_hbond_angle_factor(p_atom, l_atom, protein, ligand)
                    hbond_energy += hbond_term * angle_factor
                
                # Ligand donor - Protein acceptor
                if l_element in self.hbond_donor_types and p_element in self.hbond_acceptor_types:
                    # Similar calculation as above with reversed roles
                    hb_key = f"{l_element}-{p_element}"
                    
                    if hb_key in self.hbond_params:
                        params = self.hbond_params[hb_key]
                    else:
                        params = {'r_eq': 1.9, 'epsilon': 3.0}
                    
                    r_eq = params['r_eq']
                    epsilon = params['epsilon']
                    
                    if distance < 0.1:
                        distance = 0.1
                    
                    ratio = r_eq / distance
                    hbond_term = epsilon * ((ratio**12) - 2.0 * (ratio**10))
                    
                    angle_factor = self._calculate_hbond_angle_factor(l_atom, p_atom, ligand, protein)
                    hbond_energy += hbond_term * angle_factor
        
        return hbond_energy
    
    def _calculate_hbond_angle_factor(self, donor_atom, acceptor_atom, donor_mol, acceptor_mol):
        """
        Calculate angular dependency factor for hydrogen bond.
        Returns a value between 0 (poor geometry) and 1 (ideal geometry).
        """
        try:
            # Get coordinates
            donor_coords = donor_atom['coords']
            acceptor_coords = acceptor_atom['coords']
            
            # Calculate basic vector
            d_a_vector = acceptor_coords - donor_coords
            d_a_distance = np.linalg.norm(d_a_vector)
            if d_a_distance < 0.1:
                return 0.0  # Atoms are too close
            
            d_a_vector = d_a_vector / d_a_distance
            
            # For simplicity, we'll use a default angle factor
            # In a real implementation, you'd use bonding information to calculate precise angles
            return 0.5  # Default 50% effectiveness
            
        except Exception as e:
            return 0.25  # Fallback if calculation fails
    
    def _calculate_electrostatics_physics(self, protein_atoms, ligand_atoms):
        """
        Calculate electrostatic interactions using Coulomb's law with
        distance-dependent dielectric.
        """
        elec_energy = 0.0
        coulomb_constant = 332.0  # Convert to kcal/mol
        
        for p_atom in protein_atoms:
            p_coords = p_atom['coords']
            p_element = p_atom.get('element', p_atom.get('name', 'C'))[0].upper()
            p_charge = self.atom_charges.get(p_element, 0.0)
            
            # Skip atoms with zero charge
            if abs(p_charge) < 1e-6:
                continue
            
            for l_atom in ligand_atoms:
                l_coords = l_atom['coords']
                l_element = l_atom.get('symbol', 'C').upper()
                l_charge = self.atom_charges.get(l_element, 0.0)
                
                # Skip atoms with zero charge
                if abs(l_charge) < 1e-6:
                    continue
                
                # Calculate distance
                distance = np.linalg.norm(p_coords - l_coords)
                
                # Skip if beyond cutoff
                if distance > self.elec_cutoff:
                    continue
                
                # Prevent division by zero
                if distance < 0.1:
                    distance = 0.1
                
                # Calculate distance-dependent dielectric
                # Uses ε(r) = 4r model
                dielectric = 4.0 * distance
                
                # Calculate Coulomb energy
                elec_term = coulomb_constant * p_charge * l_charge / (dielectric * distance)
                elec_energy += elec_term
        
        return elec_energy
    
    def _calculate_desolvation_physics(self, protein_atoms, ligand_atoms):
        """
        Calculate desolvation energy using atomic solvation and volume parameters.
        """
        desolv_energy = 0.0
        sigma = self.solvation_k  # Solvation radius in Å
        sigma_squared_2 = 2.0 * sigma * sigma  # Pre-calculate
        
        for p_atom in protein_atoms:
            p_coords = p_atom['coords']
            p_type = self._get_atom_type(p_atom)
            p_solv = self.atom_solvation_params.get(p_type, 0.0)
            p_vol = self.atom_volume_params.get(p_type, 0.0)
            
            for l_atom in ligand_atoms:
                l_coords = l_atom['coords']
                l_type = self._get_atom_type(l_atom)
                l_solv = self.atom_solvation_params.get(l_type, 0.0)
                l_vol = self.atom_volume_params.get(l_type, 0.0)
                
                # Calculate distance
                distance = np.linalg.norm(p_coords - l_coords)
                
                # Skip if beyond cutoff
                if distance > self.desolv_cutoff:
                    continue
                
                # Calculate exponential term
                exp_term = np.exp(-(distance*distance) / sigma_squared_2)
                
                # Calculate desolvation energy (volume-weighted)
                desolv_term = (self.solpar * p_solv * l_vol + 
                              self.solpar * l_solv * p_vol) * exp_term
                
                desolv_energy += desolv_term
        
        return desolv_energy
    
        # Forward any other methods to the base scoring function
    def __getattr__(self, name):
        return getattr(self.base_scoring_function, name)