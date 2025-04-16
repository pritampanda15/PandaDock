# scoring_physics_improved.py
import numpy as np
from scipy.spatial.distance import cdist

class ScoringFunction:
    """Base class for scoring functions with physics-based calculations."""
    
    def __init__(self):
        # Basic atomic parameters
        self.vdw_radii = {
            'H': 1.2, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8,
            'P': 1.8, 'F': 1.47, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98
        }
        self.atom_depths = {
            'H': 0.1, 'C': 0.2, 'N': 0.15, 'O': 0.15, 'S': 0.2,
            'P': 0.2, 'F': 0.15, 'Cl': 0.2, 'Br': 0.2, 'I': 0.2
        }
        
        # Physics-based parameters
        # Adjusted atom charges for better electrostatics
        self.atom_charges = {
            'H': 0.0, 'C': 0.0, 'N': -0.4, 'O': -0.4, 'S': -0.15,
            'P': 0.4, 'F': -0.2, 'Cl': -0.08, 'Br': -0.08, 'I': -0.08
        }
        
        # Recalibrated solvation parameters (reduced magnitude)
        self.atom_solvation = {
            'H': 0.0, 'C': 0.4, 'N': -1.5, 'O': -1.5, 'S': 0.4,
            'P': -0.2, 'F': 0.1, 'Cl': 0.3, 'Br': 0.3, 'I': 0.3
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
        
        # Hydrophobic atom types
        self.hydrophobic_types = ['C', 'A', 'F', 'Cl', 'Br', 'I', 'C.3', 'C.2', 'C.ar']
        
        # Recalibrated atomic solvation parameters (kcal/mol·Å²)
        # Significantly reduced magnitudes to prevent solvation from dominating
        self.atom_solvation_params = {
            'C': 0.4,    # Reduced from 12.77
            'A': 0.4,    # Reduced from 12.77
            'N': -1.5,   # Less negative from -17.40
            'NA': -1.5,  # Less negative
            'O': -1.5,   # Less negative from -17.40
            'OA': -1.5,  # Less negative
            'S': -0.8,   # Less negative from -8.31
            'SA': -0.8,  # Less negative
            'H': 0.0,    # Unchanged
            'F': -0.5,   # Less negative from -6.60
            'Cl': -0.1,  # Slightly adjusted from -0.72
            'Br': -0.1,  # Slightly adjusted from -0.85
            'I': -0.1,   # Slightly adjusted from -0.88
            'P': -0.7,   # Less negative from -6.70
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
        self.solpar = 0.005   # Reduced from 0.01097 kcal/mol per Å²
        self.solvation_k = 3.5  # Å, solvation radius
        
        # Distance cutoffs
        self.vdw_cutoff = 8.0  # Å for van der Waals
        self.hbond_cutoff = 4.0  # Å for hydrogen bonds
        self.elec_cutoff = 20.0  # Å for electrostatics
        self.desolv_cutoff = 8.0  # Å for desolvation
        self.hydrophobic_cutoff = 4.5  # Å for hydrophobic interactions
    
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
                
                # Use modified potential with smoother transition for close distances
                if distance >= 0.7 * r_eq:  # Increased from 0.5 for smoother behavior
                    # Regular 12-6 Lennard-Jones
                    vdw_term = epsilon * ((ratio**12) - 2.0 * (ratio**6))
                    vdw_term = min(max(vdw_term, -50.0), 50.0)  # Clip extreme values
                else:
                    # Linear repulsion function for very close distances
                    # This prevents explosion of energy at close contacts
                    rep_factor = 50.0 * (0.7 * r_eq - distance) / (0.7 * r_eq)
                    vdw_term = min(rep_factor, 50.0)  # Cap at 50.0
                
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
                    
                    # 12-10 potential with smoother distance dependence
                    if distance < 0.1:
                        distance = 0.1
                    
                    # Calculate distance from optimal H-bond length
                    dist_diff = abs(distance - r_eq)
                    
                    # Gaussian-like function with optimal value at r_eq
                    # This is smoother than the 12-10 potential and better represents H-bond energetics
                    if dist_diff <= 0.8:  # H-bonds only contribute significantly within ~0.8Å of optimal
                        hbond_term = -epsilon * np.exp(-(dist_diff**2) / 0.3)  
                    else:
                        hbond_term = 0.0  # Negligible contribution beyond cutoff
                    
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
                    
                    # Gaussian-like function
                    dist_diff = abs(distance - r_eq)
                    
                    if dist_diff <= 0.8:
                        hbond_term = -epsilon * np.exp(-(dist_diff**2) / 0.3)
                    else:
                        hbond_term = 0.0
                    
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
            return 0.7  # Increased from 0.5 for stronger H-bond contributions
            
        except Exception as e:
            return 0.3  # Increased from 0.25 for fallback
    
    def _calculate_electrostatics_physics(self, protein_atoms, ligand_atoms):
        """
        Calculate electrostatic interactions using Coulomb's law with
        improved distance-dependent dielectric model.
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
                
                # Calculate improved distance-dependent dielectric
                # Increased from 4.0 * distance to 6.0 * distance to reduce electrostatic penalties
                dielectric = 6.0 * distance
                
                # Calculate Coulomb energy with Debye-Hückel-like screening
                # This adds a distance-dependent screening term that attenuates long-range interactions
                screening_factor = np.exp(-distance / 10.0)  # 10Å screening length
                elec_term = coulomb_constant * p_charge * l_charge * screening_factor / (dielectric * distance)
                
                # Scale down extreme electrostatic interactions
                elec_term = np.sign(elec_term) * min(abs(elec_term), 10.0)
                
                elec_energy += elec_term
        
        return elec_energy
    
    def _calculate_desolvation_physics(self, protein_atoms, ligand_atoms):
        """
        Calculate desolvation energy using recalibrated atomic solvation and volume parameters.
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
                
                # Apply a scaling factor to further reduce extreme desolvation values
                desolv_term = np.sign(desolv_term) * min(abs(desolv_term), 5.0)
                
                desolv_energy += desolv_term
        
        return desolv_energy
    
    def _calculate_hydrophobic_physics(self, protein_atoms, ligand_atoms):
        """
        Calculate hydrophobic interactions using physics-based approach.
        """
        hydrophobic_score = 0.0
        
        # Identify hydrophobic atoms
        p_hydrophobic = [atom for atom in protein_atoms 
                        if self._get_atom_type(atom) in self.hydrophobic_types]
        
        l_hydrophobic = [atom for atom in ligand_atoms 
                        if self._get_atom_type(atom) in self.hydrophobic_types]
        
        for p_atom in p_hydrophobic:
            p_coords = p_atom['coords']
            
            for l_atom in l_hydrophobic:
                l_coords = l_atom['coords']
                
                # Calculate distance
                distance = np.linalg.norm(p_coords - l_coords)
                
                # Skip if beyond cutoff
                if distance > self.hydrophobic_cutoff:
                    continue
                
                # Linear hydrophobic interaction term with smoother transition
                if distance < 0.5:  # Avoid unrealistic close contacts
                    contact_score = 0.0
                else:
                    # Stronger interaction for closer contact with smoothing
                    direct_factor = (self.hydrophobic_cutoff - distance) / self.hydrophobic_cutoff
                    # Apply sigmoidal scaling for smoother transition
                    contact_score = direct_factor / (1.0 + np.exp(-(direct_factor*10 - 5)))
                
                hydrophobic_score -= contact_score  # Negative since it's favorable
        
        return hydrophobic_score
    
    def _calculate_entropy(self, ligand):
        """
        Estimate entropy penalty based on ligand flexibility using physics-based approach.
        """
        # Detailed entropy term based on rotatable bonds and molecule size
        if hasattr(ligand, 'rotatable_bonds'):
            n_rotatable = len(ligand.rotatable_bonds)
            # Scale based on number of atoms (larger molecules have more internal modes)
            n_atoms = len(ligand.atoms)
            # Physics-based relationship between conformational entropy and rotatable bonds
            entropy_penalty = 0.5 * n_rotatable * (1.0 + 0.05 * n_atoms)
        else:
            # Rough estimate based on number of atoms if rotatable bonds not defined
            n_atoms = len(ligand.atoms)
            entropy_penalty = 0.1 * n_atoms
        
        return entropy_penalty


class VdwScoringFunction(ScoringFunction):
    """Simple van der Waals scoring function using physics-based approach."""
    
    def score(self, protein, ligand):
        """Calculate physics-based van der Waals interaction score."""
        # If active site is defined, only consider those atoms
        if protein.active_site and 'atoms' in protein.active_site:
            protein_atoms = protein.active_site['atoms']
        else:
            protein_atoms = protein.atoms
        
        # Use the physics-based VDW calculation
        vdw_score = self._calculate_vdw_physics(protein_atoms, ligand.atoms)
        
        return vdw_score


class HBondScoringFunction(ScoringFunction):
    """Hydrogen bond scoring function using physics-based approach."""
    
    def score(self, protein, ligand):
        """Calculate physics-based hydrogen bond score."""
        # Get protein atoms
        if protein.active_site and 'atoms' in protein.active_site:
            protein_atoms = protein.active_site['atoms']
        else:
            protein_atoms = protein.atoms
        
        # Use the physics-based H-bond calculation
        hbond_score = self._calculate_hbond_physics(protein_atoms, ligand.atoms, protein, ligand)
        
        return hbond_score


class CompositeScoringFunction(ScoringFunction):
    """Composite scoring function using physics-based approaches."""
    
    def __init__(self):
        super().__init__()
        
        # Calibrated weights
        self.weights = {
            'vdw': 1.0,
            'hbond': 2.0,
            'clash': 10.0
        }
    
    def score(self, protein, ligand):
        """Calculate composite score using physics-based methods."""
        # Get active site atoms
        if protein.active_site and 'atoms' in protein.active_site:
            protein_atoms = protein.active_site['atoms']
        else:
            protein_atoms = protein.atoms
        
        # Calculate individual terms using physics-based methods
        vdw_score = self._calculate_vdw_physics(protein_atoms, ligand.atoms)
        hbond_score = self._calculate_hbond_physics(protein_atoms, ligand.atoms, protein, ligand)
        clash_score = self._calculate_clashes_physics(protein_atoms, ligand.atoms)
        
        # Combine scores
        total_score = (
            self.weights['vdw'] * vdw_score +
            self.weights['hbond'] * hbond_score +
            self.weights['clash'] * clash_score
        )
        
        return total_score
    
    def _calculate_clashes_physics(self, protein_atoms, ligand_atoms):
        """Calculate severe steric clashes using physics-based approach."""
        clash_score = 0.0
        
        for p_atom in protein_atoms:
            p_type = self._get_atom_type(p_atom)
            p_coords = p_atom['coords']
            p_radius = self.vdw_params.get(p_type, self.vdw_params['C'])['r_eq'] / 2.0
            
            for l_atom in ligand_atoms:
                l_type = self._get_atom_type(l_atom)
                l_coords = l_atom['coords']
                l_radius = self.vdw_params.get(l_type, self.vdw_params['C'])['r_eq'] / 2.0
                
                # Calculate distance
                distance = np.linalg.norm(p_coords - l_coords)
                
                # Calculate minimum allowed distance with physics-based parameters
                min_allowed = (p_radius + l_radius) * 0.7  # Allow some overlap
                
                # Penalize severe clashes with smoother transition
                if distance < min_allowed:
                    # Use exponential repulsion for more realistic physics
                    clash_factor = np.exp((min_allowed - distance) / min_allowed) - 1.0
                    clash_score += clash_factor ** 2  # Quadratic penalty
        
        return clash_score


class EnhancedScoringFunction(CompositeScoringFunction):
    """
    Enhanced scoring function with all physics-based interaction terms.
    """
    
    def __init__(self):
        super().__init__()
        
        # Improved physics-based calibrated weights for better balance
        self.weights = {
            'vdw': 0.3,           # Increased from 0.1662
            'hbond': 0.2,         # Increased from 0.1209
            'elec': 0.2,          # Increased from 0.1406
            'desolv': 0.05,       # Decreased from 0.1322 to reduce domination
            'hydrophobic': 0.2,   # Increased from 0.1418  
            'clash': 1.0,         # Kept the same
            'entropy': 0.25       # Slightly decreased from 0.2983
        }
    
    def score(self, protein, ligand):
        """Calculate full physics-based composite score."""
        # Get active site atoms
        if protein.active_site and 'atoms' in protein.active_site:
            protein_atoms = protein.active_site['atoms']
        else:
            protein_atoms = protein.atoms
        
        # Calculate all energy terms using improved physics-based methods
        vdw_score = self._calculate_vdw_physics(protein_atoms, ligand.atoms)
        hbond_score = self._calculate_hbond_physics(protein_atoms, ligand.atoms, protein, ligand)
        elec_score = self._calculate_electrostatics_physics(protein_atoms, ligand.atoms)
        desolv_score = self._calculate_desolvation_physics(protein_atoms, ligand.atoms)
        hydrophobic_score = self._calculate_hydrophobic_physics(protein_atoms, ligand.atoms)
        clash_score = self._calculate_clashes_physics(protein_atoms, ligand.atoms)
        entropy_score = self._calculate_entropy(ligand)
        
        # Combine scores with improved physics-based weights
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


class TetheredScoringFunction:
    """
    A scoring function wrapper that adds a penalty term based on RMSD from a reference position.
    Uses physics-based scoring for the base calculation.
    """
    
    def __init__(self, base_scoring_function, reference_ligand, weight=10.0, max_penalty=100.0):
        self.base_scoring_function = base_scoring_function
        self.reference_coordinates = reference_ligand.xyz.copy()
        self.weight = weight
        self.max_penalty = max_penalty
        
    def score(self, protein, ligand):
        # Get the base score using physics-based methods
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
    
    # Forward methods to base scoring function
    def __getattr__(self, name):
        return getattr(self.base_scoring_function, name)