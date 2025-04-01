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
