# unified_scoring.py

import numpy as np
from scipy.spatial.distance import cdist
import time
import warnings
from pathlib import Path
import os
import sys
from typing import List, Dict, Any
from typing import TYPE_CHECKING
from pandadock.physics import PhysicsBasedScoring, PhysicsBasedScoringFunction


if TYPE_CHECKING:
    from pandadock.protein import Protein
    from pandadock.ligand import Ligand
    from pandadock.utils import get_logger

class ScoringFunction:
    """Base class for all scoring functions with shared parameters and utility methods."""
    
    def __init__(self):
        # Basic atomic parameters
        self.vdw_radii = {
            'H': 1.2, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8,
            'P': 1.8, 'F': 1.47, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98
        }
        
        # Physics-based parameters
        self.atom_charges = {
            'H': 0.0, 'C': 0.0, 'N': -0.4, 'O': -0.4, 'S': -0.15,
            'P': 0.4, 'F': -0.2, 'Cl': -0.08, 'Br': -0.08, 'I': -0.08
        }
        
        # Recalibrated solvation parameters
        self.atom_solvation = {
            'H': 0.0, 'C': 0.4, 'N': -1.5, 'O': -1.5, 'S': 0.4,
            'P': -0.2, 'F': 0.1, 'Cl': 0.3, 'Br': 0.3, 'I': 0.3
        }
        
        # Extended atom type parameters
        self.atom_type_map = {
            # Carbon types
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
        
        # Van der Waals well depth parameters
        self.vdw_well_depth = {
            'C': 0.1094,
            'N': 0.0784,
            'O': 0.2100,
            'S': 0.2500,
            'P': 0.2000,
            'F': 0.0610,
            'Cl': 0.2650,
            'Br': 0.3200,
            'I': 0.4000,
            'H': 0.0157
        }
        
        # H-bond parameters
        self.hbond_params = {
            'O-O': {'r_eq': 1.90, 'epsilon': 5.0},
            'O-N': {'r_eq': 1.90, 'epsilon': 5.0},
            'N-O': {'r_eq': 1.90, 'epsilon': 5.0},
            'N-N': {'r_eq': 1.90, 'epsilon': 5.0},
            'O-S': {'r_eq': 2.50, 'epsilon': 1.0},
            'N-S': {'r_eq': 2.50, 'epsilon': 1.0},
        }
        
        # H-bond donor/acceptor types
        self.hbond_donor_types = {'N', 'NA', 'O', 'OA', 'N.3', 'N.am', 'O.3'}
        self.hbond_acceptor_types = {'O', 'OA', 'N', 'NA', 'SA', 'O.2', 'O.3', 'N.2'}
        
        # Hydrophobic atom types
        self.hydrophobic_types = ['C', 'A', 'F', 'Cl', 'Br', 'I', 'C.3', 'C.2', 'C.ar']
        
        # Atomic solvation parameters
        self.atom_solvation_params = {
            'C': 0.4,
            'A': 0.4,
            'N': -1.5,
            'NA': -1.5,
            'O': -1.5,
            'OA': -1.5,
            'S': -0.8,
            'SA': -0.8,
            'H': 0.0,
            'F': -0.5,
            'Cl': -0.1,
            'Br': -0.1,
            'I': -0.1,
            'P': -0.7,
        }
        
        # Atom volume parameters
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
        self.solpar = 0.005
        self.solvation_k = 3.5
        
        # Distance cutoffs
        self.vdw_cutoff = 8.0
        self.hbond_cutoff = 4.0
        self.elec_cutoff = 20.0
        self.desolv_cutoff = 8.0
        self.hydrophobic_cutoff = 4.5
        
        # Default weights for composite scoring
        self.weights = {
            'vdw': 1.0,
            'hbond': 1.0,
            'elec': 1.0,
            'desolv': 1.0,
            'hydrophobic': 1.0,
            'clash': 1.0,
            'entropy': 0.25,
        }
        
        # Debug flag for detailed output
        self.verbose = False
    
    def score(self, protein: 'Protein', ligand: 'Ligand') -> float:
        """
        Calculate binding score between protein and ligand.
        
        Parameters:
        -----------
        protein : Protein
        ligand : Ligand
        
        Returns:
        --------
        float
            Binding score (lower is better)
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def _get_atom_type(self, atom, default='C'):
        """Determine the atom type for an atom based on available information."""
        # Handle case where atom is a list
        if isinstance(atom, list):
            return [self._get_atom_type(a, default) for a in atom]
        
        # Try to get atom type from atom data
        if isinstance(atom, dict):
            atom_type = atom.get('type', None)
        else:
            raise TypeError(f"Expected a dictionary or list for atom, but got {type(atom)}")
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
    
    def _get_protein_atoms(self, protein):
        """Get active site atoms if defined, otherwise all protein atoms."""
        if isinstance(protein, list):
            # Flatten the list of protein atoms
            atoms = []
            for p in protein:
                if hasattr(p, 'active_site') and p.active_site and 'atoms' in p.active_site:
                    atoms.extend(p.active_site['atoms'])
                else:
                    atoms.extend(getattr(p, 'atoms', []))
            return atoms
        if hasattr(protein, 'active_site') and protein.active_site and 'atoms' in protein.active_site:
            return protein.active_site['atoms']
        else:
            return getattr(protein, 'atoms', [])
        
    def _get_ligand_atoms(self, ligand):
        """Get ligand atoms."""
        return ligand.atoms
    
    def _estimate_pose_restriction(self, ligand, protein=None):
        """
        Estimate pose-specific conformational restriction.
        Returns a factor between 0 (fully restricted) and 1 (fully flexible).
        """
        if not protein or not hasattr(protein, 'active_site') or not protein.active_site.get('atoms'):
            return 0.5  # Fallback if no protein info

        ligand_coords = np.array([atom['coords'] for atom in ligand.atoms if 'coords' in atom])
        protein_coords = np.array([atom['coords'] for atom in protein.active_site['atoms'] if 'coords' in atom])

        # Safety checks
        if len(protein_coords) == 0 or len(ligand_coords) == 0:
            print("Warning: Empty protein or ligand coordinates for pose restriction. Skipping entropy.")
            return 0.5

        ligand_coords = np.array(ligand_coords)
        protein_coords = np.array(protein_coords)

        if ligand_coords.ndim != 2 or protein_coords.ndim != 2 or protein_coords.shape[1] != 3 or ligand_coords.shape[1] != 3:
            print("Warning: Malformed coordinates for pose restriction. Skipping entropy.")
            return 0.5

        # Compute pairwise distances
        from scipy.spatial import cKDTree
        kdtree = cKDTree(protein_coords)
        close_contacts = kdtree.query_ball_point(ligand_coords, r=4.0)  # 4Å cutoff

        buried_atoms = sum(1 for contacts in close_contacts if len(contacts) > 0)
        burial_fraction = buried_atoms / len(ligand.atoms)

        # Heuristic: more burial → more restriction
        flexibility_factor = 1.0 - burial_fraction  # 0 = buried, 1 = exposed

        # Clamp to [0.1, 1.0] for numerical stability
        return max(0.1, min(1.0, flexibility_factor))

        # Adjusted from 0.5 to 0.1 for more realistic flexibility estimation
class CPUScoringFunction(ScoringFunction):
    """
    Base class for CPU-based scoring functions.
    Implements all energy term calculations using CPU-based methods.
    """
    
    def calculate_vdw(self, protein_atoms, ligand_atoms):
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
    
    def calculate_hbond(self, protein_atoms, ligand_atoms, protein=None, ligand=None):
        """
        Calculate hydrogen bonding using a Gaussian-like potential.
        
        Parameters:
        -----------
        protein_atoms : list
            List of protein atom dictionaries
        ligand_atoms : list
            List of ligand atom dictionaries
        protein : Protein, optional
            Protein object (for more detailed angle calculations)
        ligand : Ligand, optional
            Ligand object (for more detailed angle calculations)
            
        Returns:
        --------
        float
            H-bond energy contribution
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
            # In a full implementation, you'd use bonding information to calculate precise angles
            return 0.7  # Increased from 0.5 for stronger H-bond contributions
            
        except Exception as e:
            return 0.3  # Increased from 0.25 for fallback
    
    def calculate_electrostatics(self, protein_atoms, ligand_atoms):
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
    
    def calculate_desolvation(self, protein_atoms, ligand_atoms):
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
    
    def calculate_hydrophobic(self, protein_atoms, ligand_atoms):
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
    
    def calculate_clashes(self, protein_atoms, ligand_atoms):
        """
        Calculate steric clashes using Van der Waals overlap and exponential repulsion.
        """
        clash_score = 0.0

        for p_atom in protein_atoms:
            if 'coords' not in p_atom:
                continue
            p_coords = p_atom['coords']
            p_type = self._get_atom_type(p_atom)
            p_radius = self.vdw_params.get(p_type, self.vdw_params['C'])['r_eq'] / 2.0

            for l_atom in ligand_atoms:
                if 'coords' not in l_atom:
                    continue
                l_coords = l_atom['coords']
                l_type = self._get_atom_type(l_atom)
                l_radius = self.vdw_params.get(l_type, self.vdw_params['C'])['r_eq'] / 2.0

                distance = np.linalg.norm(p_coords - l_coords)
                min_allowed = (p_radius + l_radius) * 0.7
                upper_bound = (p_radius + l_radius) * 1.2

                if distance < min_allowed:
                    repulsion = np.exp((min_allowed - distance) / min_allowed) - 1.0
                    clash_score += repulsion ** 2
                elif distance < upper_bound:
                    soft_penalty = (upper_bound - distance) / (upper_bound - min_allowed)
                    clash_score += 0.1 * (soft_penalty ** 2)

        return clash_score
    
    def calculate_entropy(self, ligand, protein=None):
        n_rotatable = len(getattr(ligand, 'rotatable_bonds', []))
        n_atoms = len(ligand.atoms)  
        flexibility = self._estimate_pose_restriction(ligand, protein)
        entropy_penalty = 0.5 * n_rotatable * flexibility * (1.0 + 0.05 * n_atoms)
        return entropy_penalty
        # Adjusted from 0.25 to 0.5 for more realistic entropy penalty

class CompositeScoringFunction(CPUScoringFunction):
    """Composite scoring function combining all energy terms."""
    
    def score(self, protein, ligand):
        """
        Calculate composite score with enhanced backbone clash detection.
        """
        protein_atoms = self._get_protein_atoms(protein)
        ligand_atoms = self._get_ligand_atoms(ligand)
        
        # Calculate energy components
        vdw = self.calculate_vdw(protein_atoms, ligand_atoms)
        hbond = self.calculate_hbond(protein_atoms, ligand_atoms, protein, ligand)
        elec = self.calculate_electrostatics(protein_atoms, ligand_atoms)
        desolv = self.calculate_desolvation(protein_atoms, ligand_atoms)
        hydrophobic = self.calculate_hydrophobic(protein_atoms, ligand_atoms)
        
        # Regular clash detection
        clash = self.calculate_clashes(protein_atoms, ligand_atoms)
        
        # Enhanced backbone-aware clash detection
        backbone_clash = self.calculate_enhanced_clashes(protein_atoms, ligand_atoms, backbone_factor=3.0)
        
        # Entropy calculation
        entropy = self.calculate_entropy(ligand, protein)
        
        # Combine scores with special handling for backbone clashes
        total = (
            self.weights['vdw'] * vdw +
            self.weights['hbond'] * hbond +
            self.weights['elec'] * elec +
            self.weights['desolv'] * desolv +
            self.weights['hydrophobic'] * hydrophobic -
            self.weights['clash'] * (clash + backbone_clash) -  # Note the addition of backbone_clash
            self.weights['entropy'] * entropy
        )
        
        # Apply severe penalty for significant backbone clashes
        if backbone_clash > 2.0:  # Threshold for severe backbone clash
            backbone_penalty = backbone_clash * 10.0  # Severe multiplicative penalty
            total -= backbone_penalty
        
        # Print breakdown if verbose
        if self.verbose:
            print(f"VDW: {vdw:.2f}, H-bond: {hbond:.2f}, Elec: {elec:.2f}, "
                f"Desolv: {desolv:.2f}, Hydrophobic: {hydrophobic:.2f}, "
                f"Clash: {clash:.2f}, Backbone Clash: {backbone_clash:.2f}, "
                f"Entropy: {entropy:.2f}")
            print(f"Total: {total:.2f}")
        
        return total * -1.0 * 0.03

    def calculate_enhanced_clashes(self, protein_atoms, ligand_atoms, backbone_factor=2.0):
        """
        Calculate clash score with enhanced sensitivity to backbone atoms.
        
        Parameters:
        -----------
        protein_atoms : list
            List of protein atoms
        ligand_atoms : list
            List of ligand atoms
        backbone_factor : float
            Multiplier for backbone atom clash penalties (higher = stricter)
            
        Returns:
        --------
        float
            Enhanced clash score (higher = worse clashes)
        """
        clash_score = 0.0
        vdw_radii = {
            'H': 1.2, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8,
            'P': 1.8, 'F': 1.47, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98
        }
        clash_distance_factor = 0.7  # Clashes occur at 70% of sum of vdW radii
        backbone_atoms = {'CA', 'C', 'N', 'O'}  # Protein backbone atoms
        
        for p_atom in protein_atoms:
            p_coords = p_atom['coords']
            p_element = p_atom.get('element', p_atom.get('name', 'C'))[0]
            p_radius = vdw_radii.get(p_element, 1.7)
            
            # Check if this is a backbone atom
            is_backbone = False
            if 'name' in p_atom:
                atom_name = p_atom['name'].strip()
                is_backbone = atom_name in backbone_atoms
            
            for l_atom in ligand_atoms:
                l_coords = l_atom['coords']
                l_element = l_atom.get('symbol', 'C')
                l_radius = vdw_radii.get(l_element, 1.7)
                
                # Calculate distance
                distance = np.linalg.norm(p_coords - l_coords)
                
                # Define minimum allowed distance based on vdW radii
                min_allowed = (p_radius + l_radius) * clash_distance_factor
                
                # Check for clash
                if distance < min_allowed:
                    # Calculate clash severity (1.0 = touching at min_allowed, >1.0 = overlapping)
                    clash_severity = min_allowed / max(distance, 0.1)  # Avoid division by zero
                    
                    # Apply higher penalty for backbone clashes
                    if is_backbone:
                        clash_score += (clash_severity ** 2) * backbone_factor
                    else:
                        clash_score += clash_severity ** 2
        
        return clash_score
    
    def score_for_pose_generation(self, protein, ligand):
        """
        Special scoring function with extremely strict clash penalties
        for use during initial pose generation and early optimization.
        """
        # Regular scoring
        base_score = self.score(protein, ligand)
        
        # Get atoms
        protein_atoms = self._get_protein_atoms(protein)
        ligand_atoms = self._get_ligand_atoms(ligand)
        
        # Check for any backbone clashes with very strict parameters
        backbone_clash = self.calculate_enhanced_clashes(
            protein_atoms, ligand_atoms, backbone_factor=10.0
        )
        
        # Apply extreme penalty for any backbone clash during pose generation
        if backbone_clash > 0.5:  # Lower threshold during generation
            return 999.0  # Very bad score to reject pose
        
        return base_score
class EnhancedScoringFunction(CompositeScoringFunction):
    """
    EnhancedScoringFunction is a subclass of CompositeScoringFunction.
    EnhancedScoringFunction inherits the score() method from CompositeScoringFunction
    but recalibrates the energy component weights for better docking accuracy.
    """
    
    def __init__(self):
        super().__init__()
        
        # Improved calibrated weights for better balance
        self.weights = {
            'vdw': 0.3,           # Increased from 0.1662
            'hbond': 0.2,         # Increased from 0.1209
            'elec': 0.2,          # Increased from 0.1406
            'desolv': 0.005,       # Decreased from 0.1322 to reduce domination
            'hydrophobic': 0.2,   # Increased from 0.1418  
            'clash': 1.0,         # Kept the same
            'entropy': 0.25       # Slightly decreased from 0.2983
        }


class GPUScoringFunction(ScoringFunction):
    """
    Base class for GPU-accelerated scoring functions.
    """
    
    def __init__(self, device='cuda', precision='float32'):
        """
        Initialize GPU-accelerated scoring function.
        
        Parameters:
        -----------
        device : str
            Computing device ('cuda' or 'cpu')
        precision : str
            Numerical precision ('float32' or 'float64')
        """
        super().__init__()
        
        self.device_name = device
        self.precision = precision
        self.device = None
        self.torch_available = False
        self.cupy_available = False
        self._init_gpu()
    
    def _init_gpu(self):
        """Initialize GPU resources and check available hardware."""
        # Try PyTorch first
        try:
            import torch
            self.torch_available = True
            
            # Check if CUDA is available and set device
            if self.device_name == 'cuda' and torch.cuda.is_available():
                self.device = torch.device('cuda')
                gpu_name = torch.cuda.get_device_name(0)
                print(f"Using GPU: {gpu_name}")
                
                # Set default tensor type
                if self.precision == 'float64':
                    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
                else:
                    torch.set_default_tensor_type(torch.cuda.FloatTensor)
            else:
                self.device = torch.device('cpu')
                print("GPU not available or not requested. Using CPU via PyTorch.")
                if self.precision == 'float64':
                    torch.set_default_tensor_type(torch.DoubleTensor)
                
            # Test GPU with a small calculation
            start = time.time()
            a = torch.rand(1000, 1000, device=self.device)
            b = torch.rand(1000, 1000, device=self.device)
            c = torch.matmul(a, b)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            print(f"PyTorch GPU test completed in {end - start:.4f} seconds")
            
        except ImportError:
            print("PyTorch not available. Trying CuPy...")
            
            # If PyTorch is not available, try CuPy
            try:
                import cupy as cp
                self.cupy_available = True
                
                if self.device_name == 'cuda':
                    try:
                        # Get GPU info
                        gpu_info = cp.cuda.runtime.getDeviceProperties(0)
                        print(f"Using GPU via CuPy: {gpu_info['name'].decode()}")
                    except:
                        print("Using GPU via CuPy")
                else:
                    print("GPU not requested. Using CPU.")
                
                # Set precision
                self.cp = cp
                if self.precision == 'float64':
                    self.cp_dtype = cp.float64
                else:
                    self.cp_dtype = cp.float32
                
                # Test GPU with a small calculation
                start = time.time()
                a = cp.random.rand(1000, 1000).astype(self.cp_dtype)
                b = cp.random.rand(1000, 1000).astype(self.cp_dtype)
                c = cp.matmul(a, b)
                cp.cuda.stream.get_current_stream().synchronize()
                end = time.time()
                print(f"CuPy GPU test completed in {end - start:.4f} seconds")
                
            except ImportError:
                print("Neither PyTorch nor CuPy available. Falling back to CPU calculations.")
                print("For better performance, install PyTorch or CuPy with GPU support.")
    
    def score(self, protein, ligand):
        """
        Calculate composite score with enhanced backbone clash detection.
        """
        protein_atoms = self._get_protein_atoms(protein)
        ligand_atoms = self._get_ligand_atoms(ligand)
        
        # Calculate energy components
        vdw = self.calculate_vdw(protein_atoms, ligand_atoms)
        hbond = self.calculate_hbond(protein_atoms, ligand_atoms, protein, ligand)
        elec = self.calculate_electrostatics(protein_atoms, ligand_atoms)
        desolv = self.calculate_desolvation(protein_atoms, ligand_atoms)
        hydrophobic = self.calculate_hydrophobic(protein_atoms, ligand_atoms)
        
        # Regular clash detection
        clash = self.calculate_clashes(protein_atoms, ligand_atoms)
        
        # Enhanced backbone-aware clash detection
        backbone_clash = self.calculate_enhanced_clashes(protein_atoms, ligand_atoms, backbone_factor=3.0)
        
        # Entropy calculation
        entropy = self.calculate_entropy(ligand, protein)
        
        # Combine scores with special handling for backbone clashes
        total = (
            self.weights['vdw'] * vdw +
            self.weights['hbond'] * hbond +
            self.weights['elec'] * elec +
            self.weights['desolv'] * desolv +
            self.weights['hydrophobic'] * hydrophobic -
            self.weights['clash'] * (clash + backbone_clash) -  # Note the addition of backbone_clash
            self.weights['entropy'] * entropy
        )
        
        # Apply severe penalty for significant backbone clashes
        if backbone_clash > 2.0:  # Threshold for severe backbone clash
            backbone_penalty = backbone_clash * 10.0  # Severe multiplicative penalty
            total -= backbone_penalty
        
        # Print breakdown if verbose
        if self.verbose:
            print(f"VDW: {vdw:.2f}, H-bond: {hbond:.2f}, Elec: {elec:.2f}, "
                f"Desolv: {desolv:.2f}, Hydrophobic: {hydrophobic:.2f}, "
                f"Clash: {clash:.2f}, Backbone Clash: {backbone_clash:.2f}, "
                f"Entropy: {entropy:.2f}")
            print(f"Total: {total:.2f}")
        
        return total * -1.0 * 0.03
    
    def calculate_vdw(self, protein, ligand):
        """
        GPU-accelerated van der Waals interaction calculation.
        """
        protein_atoms = self._get_protein_atoms(protein)
        
        if self.torch_available:
            return self._calculate_vdw_torch(protein_atoms, ligand.atoms)
        elif self.cupy_available:
            return self._calculate_vdw_cupy(protein_atoms, ligand.atoms)
        else:
            # Fall back to CPU implementation
            cpu_scorer = CPUScoringFunction()
            ligand_atoms = ligand if isinstance(ligand, list) else ligand.atoms
            return cpu_scorer.calculate_vdw(protein_atoms, ligand_atoms)
    
    def _calculate_vdw_torch(self, protein_atoms, ligand_atoms):
        """
        Calculate van der Waals interactions using PyTorch.
        """
        import torch
        
        # Extract coordinates and parameters
        p_coords = []
        p_radii = []
        p_depths = []
        
        for atom in protein_atoms:
            p_coords.append(atom['coords'])
            symbol = atom.get('element', atom.get('name', 'C'))[0]
            p_radii.append(self.vdw_radii.get(symbol, 1.7))
            p_depths.append(self.vdw_well_depth.get(symbol, 0.1))
        
        l_coords = []
        l_radii = []
        l_depths = []
        
        for atom in ligand_atoms:
            l_coords.append(atom['coords'])
            symbol = atom.get('symbol', 'C')
            l_radii.append(self.vdw_radii.get(symbol, 1.7))
            l_depths.append(self.vdw_well_depth.get(symbol, 0.1))
        
        # Convert to PyTorch tensors
        p_coords = torch.tensor(np.array(p_coords), device=self.device)
        p_radii = torch.tensor(np.array(p_radii), device=self.device).view(-1, 1)
        p_depths = torch.tensor(np.array(p_depths), device=self.device).view(-1, 1)
        
        l_coords = torch.tensor(np.array(l_coords), device=self.device)
        l_radii = torch.tensor(np.array(l_radii), device=self.device).view(1, -1)
        l_depths = torch.tensor(np.array(l_depths), device=self.device).view(1, -1)
        
        # Calculate all distances at once
        distances = torch.cdist(p_coords, l_coords)
        
        # Calculate Lennard-Jones parameters
        sigma = (p_radii + l_radii) * 0.5
        epsilon = torch.sqrt(p_depths * l_depths)
        
        # Apply distance cutoff (10Å)
        mask = distances <= 10.0
        
        # Safe distances to avoid numerical issues
        safe_distances = torch.clamp(distances, min=0.1)
        
        # Calculate Lennard-Jones energy
        ratio = sigma / safe_distances
        ratio6 = ratio ** 6
        ratio12 = ratio6 ** 2
        
        lj_energy = epsilon * (ratio12 - 2.0 * ratio6)
        lj_energy = lj_energy * mask.float()
        
        # Sum all energies
        vdw_energy = float(torch.sum(lj_energy).item())
        
        return vdw_energy
    
    def _calculate_vdw_cupy(self, protein_atoms, ligand_atoms):
        """
        Calculate van der Waals interactions using CuPy.
        """
        cp = self.cp
        
        # Extract coordinates and parameters
        p_coords = []
        p_radii = []
        p_depths = []
        
        for atom in protein_atoms:
            p_coords.append(atom['coords'])
            symbol = atom.get('element', atom.get('name', 'C'))[0]
            p_radii.append(self.vdw_radii.get(symbol, 1.7))
            p_depths.append(self.vdw_well_depth.get(symbol, 0.1))
        
        l_coords = []
        l_radii = []
        l_depths = []
        
        for atom in ligand_atoms:
            l_coords.append(atom['coords'])
            symbol = atom.get('symbol', 'C')
            l_radii.append(self.vdw_radii.get(symbol, 1.7))
            l_depths.append(self.vdw_well_depth.get(symbol, 0.1))
        
        # Convert to CuPy arrays
        p_coords = cp.array(p_coords, dtype=self.cp_dtype)
        p_radii = cp.array(p_radii, dtype=self.cp_dtype).reshape(-1, 1)
        p_depths = cp.array(p_depths, dtype=self.cp_dtype).reshape(-1, 1)
        
        l_coords = cp.array(l_coords, dtype=self.cp_dtype)
        l_radii = cp.array(l_radii, dtype=self.cp_dtype).reshape(1, -1)
        l_depths = cp.array(l_depths, dtype=self.cp_dtype).reshape(1, -1)
        
        # Calculate all distances at once
        diff = cp.expand_dims(p_coords, 1) - cp.expand_dims(l_coords, 0)
        distances = cp.sqrt(cp.sum(diff**2, axis=2))
        
        # Calculate Lennard-Jones parameters
        sigma = (p_radii + l_radii) * 0.5
        epsilon = cp.sqrt(p_depths * l_depths)
        
        # Apply distance cutoff (10Å)
        mask = distances <= 10.0
        
        # Safe distances to avoid numerical issues
        safe_distances = cp.maximum(distances, 0.1)
        
        # Calculate Lennard-Jones energy
        ratio = sigma / safe_distances
        ratio6 = ratio ** 6
        ratio12 = ratio6 ** 2
        
        lj_energy = epsilon * (ratio12 - 2.0 * ratio6)
        lj_energy = lj_energy * mask
        
        # Sum all energies
        vdw_energy = float(cp.sum(lj_energy))
        
        return vdw_energy
    
    def calculate_electrostatics(self, protein, ligand):
        """
        GPU-accelerated electrostatic interaction calculation.
        """
        protein_atoms = self._get_protein_atoms(protein)
        
        if self.torch_available:
            return self._calculate_electrostatics_torch(protein_atoms, ligand.atoms)
        elif self.cupy_available:
            return self._calculate_electrostatics_cupy(protein_atoms, ligand.atoms)
        else:
            # Fall back to CPU implementation
            cpu_scorer = CPUScoringFunction()
            return cpu_scorer.calculate_electrostatics(protein_atoms, ligand.atoms)
    
    def _calculate_electrostatics_torch(self, protein_atoms, ligand_atoms):
        """
        Calculate electrostatic interactions using PyTorch.
        """
        import torch
        
        # Extract coordinates and charges
        p_coords = []
        p_charges = []
        
        for atom in protein_atoms:
            p_coords.append(atom['coords'])
            symbol = atom.get('element', atom.get('name', 'C'))[0]
            p_charges.append(self.atom_charges.get(symbol, 0.0))
        
        l_coords = []
        l_charges = []
        
        for atom in ligand_atoms:
            l_coords.append(atom['coords'])
            symbol = atom.get('symbol', 'C')
            l_charges.append(self.atom_charges.get(symbol, 0.0))
        
        # Convert to PyTorch tensors
        p_coords = torch.tensor(np.array(p_coords), device=self.device)
        p_charges = torch.tensor(np.array(p_charges), device=self.device)
        
        l_coords = torch.tensor(np.array(l_coords), device=self.device)
        l_charges = torch.tensor(np.array(l_charges), device=self.device)
        
        # Calculate all distances at once
        distances = torch.cdist(p_coords, l_coords)
        
        # Create charge products matrix
        charge_products = torch.outer(p_charges, l_charges)
        
        # Apply distance cutoff
        mask = distances <= self.elec_cutoff
        
        # Calculate distance-dependent dielectric
        dielectric = 4.0 * distances
        
        # Calculate Coulomb energy with safe distances
        safe_distances = torch.clamp(distances, min=0.1)
        coulomb_energy = 332.0 * charge_products / (dielectric * safe_distances)
        
        # Apply mask and sum
        coulomb_energy = coulomb_energy * mask.float()
        elec_energy = float(torch.sum(coulomb_energy).item())
        
        return elec_energy
    
    def _calculate_electrostatics_cupy(self, protein_atoms, ligand_atoms):
        """
        Calculate electrostatic interactions using CuPy.
        """
        cp = self.cp
        
        # Extract coordinates and charges
        p_coords = []
        p_charges = []
        
        for atom in protein_atoms:
            p_coords.append(atom['coords'])
            symbol = atom.get('element', atom.get('name', 'C'))[0]
            p_charges.append(self.atom_charges.get(symbol, 0.0))
        
        l_coords = []
        l_charges = []
        
        for atom in ligand_atoms:
            l_coords.append(atom['coords'])
            symbol = atom.get('symbol', 'C')
            l_charges.append(self.atom_charges.get(symbol, 0.0))
        
        # Convert to CuPy arrays
        p_coords = cp.array(p_coords, dtype=self.cp_dtype)
        p_charges = cp.array(p_charges, dtype=self.cp_dtype)
        
        l_coords = cp.array(l_coords, dtype=self.cp_dtype)
        l_charges = cp.array(l_charges, dtype=self.cp_dtype)
        
        # Calculate all distances at once
        diff = cp.expand_dims(p_coords, 1) - cp.expand_dims(l_coords, 0)
        distances = cp.sqrt(cp.sum(diff**2, axis=2))
        
        # Create charge products matrix
        charge_products = cp.outer(p_charges, l_charges)
        
        # Apply distance cutoff
        mask = distances <= self.elec_cutoff
        
        # Calculate distance-dependent dielectric
        dielectric = 4.0 * distances
        
        # Calculate Coulomb energy with safe distances
        safe_distances = cp.maximum(distances, 0.1)
        coulomb_energy = 332.0 * charge_products / (dielectric * safe_distances)
        
        # Apply mask and sum
        coulomb_energy = coulomb_energy * mask
        elec_energy = float(cp.sum(coulomb_energy))
        
        return elec_energy
    
    def calculate_desolvation(self, protein, ligand):
        """
        GPU-accelerated desolvation calculation.
        """
        protein_atoms = self._get_protein_atoms(protein)
        
        if self.torch_available:
            return self._calculate_desolvation_torch(protein_atoms, ligand.atoms)
        elif self.cupy_available:
            return self._calculate_desolvation_cupy(protein_atoms, ligand.atoms)
        else:
            # Fall back to CPU implementation
            cpu_scorer = CPUScoringFunction()
            return cpu_scorer.calculate_desolvation(protein_atoms, ligand.atoms)
    
    def _calculate_desolvation_torch(self, protein_atoms, ligand_atoms):
        """
        Calculate desolvation using PyTorch.
        """
        import torch
        
        # Extract coordinates and solvation parameters
        p_coords = []
        p_solvation = []
        
        for atom in protein_atoms:
            p_coords.append(atom['coords'])
            symbol = atom.get('element', atom.get('name', 'C'))[0]
            p_solvation.append(self.atom_solvation.get(symbol, 0.0))
        
        l_coords = []
        l_solvation = []
        
        for atom in ligand_atoms:
            l_coords.append(atom['coords'])
            symbol = atom.get('symbol', 'C')
            l_solvation.append(self.atom_solvation.get(symbol, 0.0))
        
        # Convert to PyTorch tensors
        p_coords = torch.tensor(np.array(p_coords), device=self.device)
        p_solvation = torch.tensor(np.array(p_solvation), device=self.device).view(-1, 1)
        
        l_coords = torch.tensor(np.array(l_coords), device=self.device)
        l_solvation = torch.tensor(np.array(l_solvation), device=self.device).view(1, -1)
        
        # Calculate all distances at once
        distances = torch.cdist(p_coords, l_coords)
        
        # Apply distance cutoff
        mask = distances <= self.desolv_cutoff
        
        # Create solvation parameter products
        solvation_products = p_solvation * l_solvation
        
        # Calculate Gaussian-like desolvation
        distances_squared = distances ** 2
        desolv_energy = solvation_products * torch.exp(-distances_squared / 7.5)
        
        # Apply mask and sum
        desolv_energy = desolv_energy * mask.float()
        total_desolv_energy = float(torch.sum(desolv_energy).item())
        
        return total_desolv_energy
    
    def _calculate_desolvation_cupy(self, protein_atoms, ligand_atoms):
        """
        Calculate desolvation using CuPy.
        """
        cp = self.cp
        
        # Extract coordinates and solvation parameters
        p_coords = []
        p_solvation = []
        
        for atom in protein_atoms:
            p_coords.append(atom['coords'])
            symbol = atom.get('element', atom.get('name', 'C'))[0]
            p_solvation.append(self.atom_solvation.get(symbol, 0.0))
        
        l_coords = []
        l_solvation = []
        
        for atom in ligand_atoms:
            l_coords.append(atom['coords'])
            symbol = atom.get('symbol', 'C')
            l_solvation.append(self.atom_solvation.get(symbol, 0.0))
        
        # Convert to CuPy arrays
        p_coords = cp.array(p_coords, dtype=self.cp_dtype)
        p_solvation = cp.array(p_solvation, dtype=self.cp_dtype).reshape(-1, 1)
        
        l_coords = cp.array(l_coords, dtype=self.cp_dtype)
        l_solvation = cp.array(l_solvation, dtype=self.cp_dtype).reshape(1, -1)
        
        # Calculate all distances at once
        diff = cp.expand_dims(p_coords, 1) - cp.expand_dims(l_coords, 0)
        distances = cp.sqrt(cp.sum(diff**2, axis=2))
        
        # Apply distance cutoff
        mask = distances <= self.desolv_cutoff
        
        # Calculate Gaussian-like desolvation
        distances_squared = distances ** 2
        desolv_energy = p_solvation * l_solvation * cp.exp(-distances_squared / 7.5)
        
        # Apply mask and sum
        desolv_energy = desolv_energy * mask
        total_desolv_energy = float(cp.sum(desolv_energy))
        
        return total_desolv_energy
    
    def calculate_hydrophobic(self, protein, ligand):
        """
        GPU-accelerated hydrophobic interaction calculation.
        """
        protein_atoms = self._get_protein_atoms(protein)
        
        # Identify hydrophobic atoms in protein
        p_hydrophobic = [atom for atom in protein_atoms 
                        if atom.get('element', atom.get('name', ''))[0] in self.hydrophobic_types]
        
        # Identify hydrophobic atoms in ligand
        l_hydrophobic = [atom for atom in ligand.atoms 
                        if atom.get('symbol', '') in self.hydrophobic_types]
        
        if not p_hydrophobic or not l_hydrophobic:
            return 0.0  # No hydrophobic interactions
        
        if self.torch_available:
            return self._calculate_hydrophobic_torch(p_hydrophobic, l_hydrophobic)
        elif self.cupy_available:
            return self._calculate_hydrophobic_cupy(p_hydrophobic, l_hydrophobic)
        else:
            # Fall back to CPU implementation
            cpu_scorer = CPUScoringFunction()
            return cpu_scorer.calculate_hydrophobic(protein_atoms, ligand.atoms)
    
    def _calculate_hydrophobic_torch(self, p_hydrophobic, l_hydrophobic):
        """
        Calculate hydrophobic interactions using PyTorch.
        """
        import torch
        
        # Extract coordinates
        p_coords = [atom['coords'] for atom in p_hydrophobic]
        l_coords = [atom['coords'] for atom in l_hydrophobic]
        
        # Convert to PyTorch tensors
        p_coords = torch.tensor(np.array(p_coords), device=self.device)
        l_coords = torch.tensor(np.array(l_coords), device=self.device)
        
        # Calculate all distances at once
        distances = torch.cdist(p_coords, l_coords)
        
        # Apply distance cutoff
        mask = distances <= self.hydrophobic_cutoff
        
        # Skip if no interactions
        if not torch.any(mask):
            return 0.0
        
        # Calculate hydrophobic interaction strength
        distances_safe = torch.clamp(distances, min=0.5)  # Avoid unrealistic close contacts
        contact_score = (self.hydrophobic_cutoff - distances_safe) / self.hydrophobic_cutoff
        
        # Apply mask and make negative (favorable)
        contact_score = contact_score * mask.float()
        hydrophobic_score = -float(torch.sum(contact_score).item())
        
        return hydrophobic_score
    
    def _calculate_hydrophobic_cupy(self, p_hydrophobic, l_hydrophobic):
        """
        Calculate hydrophobic interactions using CuPy.
        """
        cp = self.cp
        
        # Extract coordinates
        p_coords = [atom['coords'] for atom in p_hydrophobic]
        l_coords = [atom['coords'] for atom in l_hydrophobic]
        
        # Convert to CuPy arrays
        p_coords = cp.array(p_coords, dtype=self.cp_dtype)
        l_coords = cp.array(l_coords, dtype=self.cp_dtype)
        
        # Calculate all distances at once
        diff = cp.expand_dims(p_coords, 1) - cp.expand_dims(l_coords, 0)
        distances = cp.sqrt(cp.sum(diff**2, axis=2))
        
        # Apply distance cutoff
        mask = distances <= self.hydrophobic_cutoff
        
        # Skip if no interactions
        if not cp.any(mask):
            return 0.0
        
        # Calculate hydrophobic interaction strength
        distances_safe = cp.maximum(distances, 0.5)  # Avoid unrealistic close contacts
        contact_score = (self.hydrophobic_cutoff - distances_safe) / self.hydrophobic_cutoff
        
        # Apply mask and make negative (favorable)
        contact_score = contact_score * mask
        hydrophobic_score = -float(cp.sum(contact_score))
        
        return hydrophobic_score
    
    def calculate_hbond(self, protein, ligand):
        """
        GPU-accelerated hydrogen bond calculation.
        
        For hydrogen bonds, we use the CPU implementation as it's usually
        less computationally intensive and requires more detailed atomic
        information that's harder to parallelize effectively.
        """
        # Fall back to CPU implementation
        cpu_scorer = CPUScoringFunction()
        protein_atoms = self._get_protein_atoms(protein)
        return cpu_scorer.calculate_hbond(protein_atoms, ligand.atoms)
    
    def calculate_clashes(self, protein, ligand):
        """
        GPU-accelerated steric clash calculation.
        """
        protein_atoms = self._get_protein_atoms(protein)
        
        if self.torch_available:
            return self._calculate_clashes_torch(protein_atoms, ligand.atoms)
        elif self.cupy_available:
            return self._calculate_clashes_cupy(protein_atoms, ligand.atoms)
        else:
            # Fall back to CPU implementation
            cpu_scorer = CPUScoringFunction()
            return cpu_scorer.calculate_clashes(protein_atoms, ligand.atoms)
    
    def _calculate_clashes_torch(self, protein_atoms, ligand_atoms):
        """
        Calculate steric clashes using PyTorch.
        """
        import torch
        
        # Extract coordinates and radii
        p_coords = []
        p_radii = []
        
        for atom in protein_atoms:
            p_coords.append(atom['coords'])
            symbol = atom.get('element', atom.get('name', 'C'))[0]
            p_radii.append(self.vdw_radii.get(symbol, 1.7))
        
        l_coords = []
        l_radii = []
        
        for atom in ligand_atoms:
            l_coords.append(atom['coords'])
            symbol = atom.get('symbol', 'C')
            l_radii.append(self.vdw_radii.get(symbol, 1.7))
        
        # Convert to PyTorch tensors
        p_coords = torch.tensor(np.array(p_coords), device=self.device)
        p_radii = torch.tensor(np.array(p_radii), device=self.device).view(-1, 1)
        
        l_coords = torch.tensor(np.array(l_coords), device=self.device)
        l_radii = torch.tensor(np.array(l_radii), device=self.device).view(1, -1)
        
        # Calculate all distances at once
        distances = torch.cdist(p_coords, l_coords)
        
        # Calculate minimum allowed distances (allowing some overlap)
        min_allowed = (p_radii + l_radii) * 0.7
        
        # Identify clashes
        clash_mask = distances < min_allowed
        
        # If no clashes, return 0
        if not torch.any(clash_mask):
            return 0.0
        
        # Calculate clash factor (quadratic penalty)
        clash_factor = (min_allowed - distances) / min_allowed
        clash_factor = torch.clamp(clash_factor, min=0.0)
        clash_factor_squared = clash_factor ** 2
        
        # Apply mask and sum
        clash_score = clash_factor_squared * clash_mask.float()
        total_clash_score = float(torch.sum(clash_score).item())
        
        return total_clash_score
    
    def _calculate_clashes_cupy(self, protein_atoms, ligand_atoms):
        """
        Calculate steric clashes using CuPy.
        """
        cp = self.cp
        
        # Extract coordinates and radii
        p_coords = []
        p_radii = []
        
        for atom in protein_atoms:
            p_coords.append(atom['coords'])
            symbol = atom.get('element', atom.get('name', 'C'))[0]
            p_radii.append(self.vdw_radii.get(symbol, 1.7))
        
        l_coords = []
        l_radii = []
        
        for atom in ligand_atoms:
            l_coords.append(atom['coords'])
            symbol = atom.get('symbol', 'C')
            l_radii.append(self.vdw_radii.get(symbol, 1.7))
        
        # Convert to CuPy arrays
        p_coords = cp.array(p_coords, dtype=self.cp_dtype)
        p_radii = cp.array(p_radii, dtype=self.cp_dtype).reshape(-1, 1)
        
        l_coords = cp.array(l_coords, dtype=self.cp_dtype)
        l_radii = cp.array(l_radii, dtype=self.cp_dtype).reshape(1, -1)
        
        # Calculate all distances at once
        diff = cp.expand_dims(p_coords, 1) - cp.expand_dims(l_coords, 0)
        distances = cp.sqrt(cp.sum(diff**2, axis=2))
        
        # Calculate minimum allowed distances (allowing some overlap)
        min_allowed = (p_radii + l_radii) * 0.7
        
        # Identify clashes
        clash_mask = distances < min_allowed
        
        # If no clashes, return 0
        if not cp.any(clash_mask):
            return 0.0
        
        # Calculate clash factor (quadratic penalty)
        clash_factor = (min_allowed - distances) / min_allowed
        clash_factor = cp.maximum(clash_factor, 0.0)
        clash_factor_squared = clash_factor ** 2
        
        # Apply mask and sum
        clash_score = clash_factor_squared * clash_mask
        total_clash_score = float(cp.sum(clash_score))
        
        return total_clash_score

    def calculate_entropy(self, ligand):
        """
        Fallback to CPU entropy calculation.
        """
        # Fall back to CPU implementation
        # as GPU implementation is not efficient for this
        # type of calculation
        # and requires more detailed atomic information.
        cpu_scorer = CPUScoringFunction()
        return cpu_scorer.calculate_entropy(ligand)


class EnhancedGPUScoringFunction(GPUScoringFunction):
    """
    Enhanced GPU-accelerated scoring function with calibrated weights.
    """
    
    def __init__(self, device='cuda', precision='float32'):
        super().__init__(device, precision)
        
        # Improved calibrated weights for better balance
        self.weights = {
            'vdw': 0.3,           # Increased from 0.1662
            'hbond': 0.2,         # Increased from 0.1209
            'elec': 0.2,          # Increased from 0.1406
            'desolv': 0.05,       # Decreased from 0.1322 to reduce domination
            'hydrophobic': 0.2,   # Increased from 0.1418  
            'clash': 1.0,         # Kept the same
            'entropy': 0.25       # Slightly decreased from 0.2983
        }

class TetheredScoringFunction:
    """
    A scoring function wrapper that adds a penalty term based on RMSD from a reference position.
    """
    
    def __init__(self, base_scoring_function, reference_ligand, weight=10.0, max_penalty=100.0):
        """
        Initialize tethered scoring function.
        
        Parameters:
        -----------
        base_scoring_function : ScoringFunction
            Base scoring function to wrap
        reference_ligand : Ligand
            Reference ligand for RMSD calculation
        weight : float
            Weight for RMSD penalty
        max_penalty : float
            Maximum RMSD penalty
        """
        self.base_scoring_function = base_scoring_function
        self.reference_coordinates = reference_ligand.xyz.copy()
        self.weight = weight
        self.max_penalty = max_penalty
        
        # Copy weights from base scoring function for clash and entropy terms
        self.weights = self.base_scoring_function.weights.copy()
        
        # Copy verbose flag
        self.verbose = getattr(self.base_scoring_function, 'verbose', False)
    
    def score(self, protein, ligand):
        """
        Calculate tethered score.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        ligand : Ligand
            Ligand object
        
        Returns:
        --------
        float
            Total score with RMSD penalty
        """
        # Get the base score
        base_score = self.base_scoring_function.score(protein, ligand)
        
        # Calculate RMSD from reference
        rmsd = self.calculate_rmsd(ligand.xyz)
        
        # Apply RMSD penalty, capped at max_penalty
        rmsd_penalty = min(self.weight * rmsd, self.max_penalty)
        
        # Print breakdown if debug is enabled
        if self.verbose:
            print(f"Base score: {base_score:.2f}, RMSD: {rmsd:.2f}, RMSD penalty: {rmsd_penalty:.2f}")
            print(f"Total tethered score: {base_score + rmsd_penalty:.2f}")

        # Return combined score
        return base_score + rmsd_penalty
    
    def calculate_rmsd(self, coordinates):
        """
        Calculate RMSD between coordinates and reference coordinates.
        
        Parameters:
        -----------
        coordinates : numpy.ndarray
            Coordinates to compare with reference
        
        Returns:
        --------
        float
            RMSD value
        """
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

    