"""
GPU-accelerated scoring functions for PandaDock.
This module provides scoring functions that leverage GPU acceleration
for computationally intensive calculations such as electrostatics and vdW interactions.
"""

import numpy as np
from pathlib import Path
import time
import warnings
from .scoring import EnhancedScoringFunction


class GPUAcceleratedScoringFunction(EnhancedScoringFunction):
    """
    Scoring function that leverages GPU acceleration for compute-intensive calculations.
    
    This class extends the EnhancedScoringFunction and overrides the most
    computationally intensive methods with GPU-accelerated versions using PyTorch.
    The implementation automatically falls back to CPU calculations when a GPU
    is not available or when PyTorch is not installed.
    """
    
    def __init__(self, device='cuda', precision='float32'):
        """
        Initialize the GPU-accelerated scoring function.
        """
        super().__init__()
        
        # FIXED: Set weights first, then modify specific values
        self.weights = {
            'vdw': 0.3,
            'hbond': 0.2,
            'elec': 0.2,
            'desolv': 0.005,  # Use consistent value with CPU
            'hydrophobic': 0.2,
            'clash': 1.0,
            'entropy': 0.25
        }

        # Add the van der Waals well depth parameters if not inherited
        if not hasattr(self, 'vdw_well_depth'):
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
        
        self.device_name = device
        self.precision = precision
        self.device = None
        self.torch_available = False
        self.cupy_available = False
        self._init_gpu()
        
        # Set verbose flag for debugging
        self.verbose = False
        
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
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
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
        """Calculate binding score using GPU acceleration."""
        start_time = time.time()
        
        # Get protein atoms using same logic as CPU
        if protein.active_site and 'atoms' in protein.active_site:
            protein_atoms = protein.active_site['atoms']
        else:
            protein_atoms = protein.atoms
        
        # FIXED: Calculate all energy terms with correct methods and parameters
        vdw_score = self._calculate_vdw(protein, ligand)
        hbond_score = self._calculate_hbond(protein_atoms, ligand.atoms, protein, ligand)  # FIXED: Pass all params
        elec_score = self._calculate_electrostatics(protein_atoms, ligand.atoms)
        desolv_score = self._calculate_desolvation(protein_atoms, ligand.atoms)
        hydrophobic_score = self._calculate_hydrophobic(protein_atoms, ligand.atoms)
        clash_score = self._calculate_clashes(protein_atoms, ligand.atoms)
        entropy_score = self._calculate_entropy(ligand, protein)  # FIXED: Pass protein parameter
        
        # Apply weights and sum up
        total_score = (
            self.weights['vdw'] * vdw_score +
            self.weights['hbond'] * hbond_score +
            self.weights['elec'] * elec_score +
            self.weights['desolv'] * desolv_score +
            self.weights['hydrophobic'] * hydrophobic_score +
            self.weights['clash'] * clash_score +
            self.weights['entropy'] * entropy_score
        )
        
        # Enhanced debugging
        if self.verbose:
            print(f"GPU Components: VDW: {vdw_score:.2f}, H-bond: {hbond_score:.2f}, Elec: {elec_score:.2f}, "
                 f"Desolv: {desolv_score:.2f}, Hydrophobic: {hydrophobic_score:.2f}, "
                 f"Clash: {clash_score:.2f}, Entropy: {entropy_score:.2f}")
            print(f"GPU Total: {total_score:.2f}")
        
        return total_score

    def _get_atoms(self, molecule):
        """Return atoms from active site if available, else from whole molecule."""
        return molecule.active_site['atoms'] if hasattr(molecule, 'active_site') and 'atoms' in molecule.active_site else molecule.atoms
    def _get_atom_type(self, atom):
        """Determine the atom type for an atom based on available information."""
        return atom.get('type', None)
    
    # FIXED: Ensure _calculate_entropy matches CPU version
    def _calculate_entropy(self, ligand, protein=None):
        """
        Calculate entropy penalty exactly like the CPU version.
        This is critical for scoring alignment.
        """
        # Direct port of CPU version
        n_rotatable = len(getattr(ligand, 'rotatable_bonds', []))
        n_atoms = len(ligand.atoms)
        flexibility = self._estimate_pose_restriction(ligand, protein)
        entropy_penalty = 0.5 * n_rotatable * flexibility * (1.0 + 0.05 * n_atoms)
        return entropy_penalty
    
    def _calculate_hbond(self, protein_atoms, ligand_atoms, protein=None, ligand=None):
        """
        GPU-accelerated hydrogen bond calculation.
        FIXED: Added protein and ligand parameters
        """
        if self.torch_available:
            return self._calculate_hbond_torch(protein_atoms, ligand_atoms, protein, ligand)
        elif self.cupy_available:
            return self._calculate_hbond_cupy(protein_atoms, ligand_atoms, protein, ligand)
        else:
            # Fall back to CPU implementation - FIXED: Pass all necessary parameters
            return self._calculate_hbond_physics(protein_atoms, ligand_atoms, protein, ligand)
    
    # FIXED: Update Torch and CuPy implementations to match CPU
    def _calculate_hbond_torch(self, protein_atoms, ligand_atoms, protein=None, ligand=None):
        """Calculate hydrogen bonds using PyTorch."""
        # For now, delegate to CPU implementation for consistency
        return self._calculate_hbond_physics(protein_atoms, ligand_atoms, protein, ligand)
    
    def _calculate_hbond_cupy(self, protein_atoms, ligand_atoms, protein=None, ligand=None):
        """Calculate hydrogen bonds using CuPy."""
        # For now, delegate to CPU implementation for consistency
        return self._calculate_hbond_physics(protein_atoms, ligand_atoms, protein, ligand)
    def _calculate_vdw(self, protein, ligand):
        """
        GPU-accelerated van der Waals interaction calculation.
        """
        # Get protein atoms
        if protein.active_site and 'atoms' in protein.active_site:
            protein_atoms = protein.active_site['atoms']
        else:
            protein_atoms = protein.atoms
        
        if self.torch_available:
            return self._calculate_vdw_torch(protein_atoms, ligand.atoms)
        elif self.cupy_available:
            return self._calculate_vdw_cupy(protein_atoms, ligand.atoms)
        else:
            # Fall back to CPU implementation
            return super()._calculate_vdw_energy(protein_atoms, ligand.atoms)
    
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
    
    def _calculate_electrostatics(self, protein, ligand):
        """
        GPU-accelerated electrostatic interaction calculation.
        """
        # Get protein atoms
        if protein.active_site and 'atoms' in protein.active_site:
            protein_atoms = protein.active_site['atoms']
        else:
            protein_atoms = protein.atoms
        
        if self.torch_available:
            return self._calculate_electrostatics_torch(protein_atoms, ligand.atoms)
        elif self.cupy_available:
            return self._calculate_electrostatics_cupy(protein_atoms, ligand.atoms)
        else:
            # Fall back to CPU implementation
            return super()._calculate_electrostatics(protein, ligand)
    
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
    
    def _calculate_desolvation(self, protein, ligand):
        """
        GPU-accelerated desolvation calculation.
        """
        # Get protein atoms
        if protein.active_site and 'atoms' in protein.active_site:
            protein_atoms = protein.active_site['atoms']
        else:
            protein_atoms = protein.atoms
        
        if self.torch_available:
            return self._calculate_desolvation_torch(protein_atoms, ligand.atoms)
        elif self.cupy_available:
            return self._calculate_desolvation_cupy(protein_atoms, ligand.atoms)
        else:
            # Fall back to CPU implementation
            return super()._calculate_desolvation(protein, ligand)
    

    def _calculate_desolvation_torch(self, protein_atoms, ligand_atoms):
        """
        Calculate desolvation using PyTorch with the fixed parameters.
        """
        import torch
        
        # Extract coordinates and solvation parameters
        p_coords = []
        p_solvation = []
        p_volume = []
        
        for atom in protein_atoms:
            p_coords.append(atom['coords'])
            symbol = atom.get('element', atom.get('name', 'C'))[0]
            p_type = self._get_atom_type(atom)
            p_solvation.append(self.atom_solvation_params.get(p_type, 0.0))
            p_volume.append(self.atom_volume_params.get(p_type, 0.0))
        
        l_coords = []
        l_solvation = []
        l_volume = []
        
        for atom in ligand_atoms:
            l_coords.append(atom['coords'])
            symbol = atom.get('symbol', 'C')
            l_type = self._get_atom_type(atom)
            l_solvation.append(self.atom_solvation_params.get(l_type, 0.0))
            l_volume.append(self.atom_volume_params.get(l_type, 0.0))
        
        # Convert to PyTorch tensors
        p_coords = torch.tensor(np.array(p_coords), device=self.device)
        p_solvation = torch.tensor(np.array(p_solvation), device=self.device).view(-1, 1)
        p_volume = torch.tensor(np.array(p_volume), device=self.device).view(-1, 1)
        
        l_coords = torch.tensor(np.array(l_coords), device=self.device)
        l_solvation = torch.tensor(np.array(l_solvation), device=self.device).view(1, -1)
        l_volume = torch.tensor(np.array(l_volume), device=self.device).view(1, -1)
        
        # Calculate all distances at once
        distances = torch.cdist(p_coords, l_coords)
        
        # Apply distance cutoff
        mask = distances <= self.desolv_cutoff
        
        # Calculate Gaussian-like desolvation
        sigma = self.solvation_k
        sigma_squared_2 = 2.0 * sigma * sigma
        distances_squared = distances ** 2
        exp_term = torch.exp(-distances_squared / sigma_squared_2)
        
        # Calculate desolvation energy terms with volume weighting
        term1 = self.solpar * p_solvation * l_volume
        term2 = self.solpar * l_solvation * p_volume
        solvation_terms = (term1 + term2) * exp_term
        
        # Cap extreme values
        capped_terms = torch.sign(solvation_terms) * torch.clamp(torch.abs(solvation_terms), max=5.0)
        
        # Apply mask and sum
        desolv_energy = capped_terms * mask.float()
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
    
    def _calculate_hydrophobic(self, protein_atoms, ligand_atoms):
        """
        Calculate hydrophobic interactions using exact CPU approach.
        """
        # Direct delegation to CPU implementation for consistent results
        return self._calculate_hydrophobic_physics(protein_atoms, ligand_atoms)

    
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
    
    def _calculate_clashes(self, protein_atoms, ligand_atoms):
        """GPU-accelerated steric clash calculation with calibrated penalty."""
        if self.torch_available:
            # Get clash count but apply a scaled penalty factor to match CPU
            clash_count = self._calculate_clashes_torch_raw(protein_atoms, ligand_atoms)
            # Scale down by factor of ~20 to match CPU values
            return clash_count * 0.05
        elif self.cupy_available:
            clash_count = self._calculate_clashes_cupy_raw(protein_atoms, ligand_atoms)
            return clash_count * 0.05
        else:
            # Fall back to CPU implementation
            return super()._calculate_clashes_physics(protein_atoms, ligand_atoms)
    
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
    
    

    
    def _estimate_pose_restriction(self, ligand, protein=None):
        """
        Estimate pose-specific conformational restriction.
        Returns a factor between 0 (fully restricted) and 1 (fully flexible).
        Currently based on fraction of ligand atoms buried in protein.
        
        Parameters:
        -----------
        ligand : Ligand
        protein : Protein (optional)
        
        Returns:
        --------
        float : Restriction factor (0.0 to 1.0)
        """
        if not protein or not hasattr(protein, 'active_site'):
            return 0.5  # Fallback if no protein info

        ligand_coords = np.array([atom['coords'] for atom in ligand.atoms])
        protein_coords = np.array([atom['coords'] for atom in protein.active_site['atoms']])

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

    def compare_with_cpu(self, protein, ligand, cpu_scorer=None):
        """
        Compare GPU scores with CPU scores for debugging.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        ligand : Ligand
            Ligand object
        cpu_scorer : EnhancedScoringFunction, optional
            CPU scoring function to compare with
        
        Returns:
        --------
        dict
            Comparison of score components
        """
        if cpu_scorer is None:
            from .scoring import EnhancedScoringFunction
            cpu_scorer = EnhancedScoringFunction()
        
        # Enable verbose mode for both scorers
        self.verbose = True
        cpu_scorer.verbose = True
        
        # Get scores
        gpu_score = self.score(protein, ligand)
        cpu_score = cpu_scorer.score(protein, ligand)
        
        # Calculate differences
        print("\nScore Comparison (GPU vs CPU):")
        print(f"Total Score: {gpu_score:.2f} vs {cpu_score:.2f}, Diff: {gpu_score - cpu_score:.2f}")
        
        # Disable verbose mode
        self.verbose = False
        cpu_scorer.verbose = False
        
        return {
            'gpu_score': gpu_score,
            'cpu_score': cpu_score,
            'difference': gpu_score - cpu_score
        }
"""
GPU-accelerated scoring functions for PandaDock.
This module provides scoring functions that leverage GPU acceleration
for computationally intensive calculations such as electrostatics and vdW interactions.
"""

import numpy as np
from pathlib import Path
import time
import warnings
from .physics import PhysicsBasedScoringFunction


class GPUAcceleratedScoringFunction(PhysicsBasedScoringFunction):
    """
    Scoring function that leverages GPU acceleration for compute-intensive calculations.
    
    This class extends the PhysicsBasedScoringFunction and overrides the most
    computationally intensive methods with GPU-accelerated versions using PyTorch or CuPy.
    The implementation automatically falls back to CPU calculations when a GPU
    is not available or when PyTorch/CuPy is not installed.
    """
    
    def __init__(self, device='cuda', precision='float32'):
        """
        Initialize the GPU-accelerated scoring function.
        """
        super().__init__()

        # Ensure desolvation weight is 0.005 - fixes the bug
        self.weights['desolv'] = 0.005
    
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
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
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
        Calculate binding score using GPU acceleration when available.
        
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
        start_time = time.time()
        
        # Get protein atoms
        if protein.active_site and 'atoms' in protein.active_site:
            protein_atoms = protein.active_site['atoms']
        else:
            protein_atoms = protein.atoms
            
        ligand_atoms = ligand.atoms
        
        # Calculate energy terms with GPU acceleration
        vdw_energy = self._calculate_vdw_gpu(protein_atoms, ligand_atoms)
        hbond_energy = self._calculate_hbond_gpu(protein_atoms, ligand_atoms, protein, ligand)
        elec_energy = self._calculate_electrostatics_gpu(protein_atoms, ligand_atoms)
        desolv_energy = self._calculate_desolvation_gpu(protein_atoms, ligand_atoms)
        entropy_energy = self._calculate_entropy_penalty(ligand_atoms, protein_atoms)  # No GPU acceleration needed
        hydrophobic_energy = self._calculate_hydrophobic_gpu(protein_atoms, ligand_atoms)
        protein_atoms = self._get_atoms(protein)
        ligand_atoms = ligand.atoms
        clash_energy = self._calculate_clash_gpu(protein_atoms, ligand_atoms)
        
        # Combine scores with calibrated weights
        total_score = (
            self.weights['vdw'] * vdw_energy +
            self.weights['hbond'] * hbond_energy +
            self.weights['elec'] * elec_energy +
            self.weights['desolv'] * desolv_energy +
            self.weights['entropy'] * entropy_energy +
            self.weights['hydrophobic'] * hydrophobic_energy +
            self.weights['clash'] * clash_energy
        )
        
        end_time = time.time()
        if hasattr(self, 'verbose') and self.verbose:
            print(f"Scoring completed in {end_time - start_time:.4f} seconds")
            print(f"VDW: {vdw_energy:.2f}, H-bond: {hbond_energy:.2f}, Elec: {elec_energy:.2f}, "
                 f"Desolv: {desolv_energy:.2f}, Hydrophobic: {hydrophobic_energy:.2f}, "
                 f"Clash: {clash_energy:.2f}, Entropy: {entropy_energy:.2f}")
        
        return total_score

    def _get_atoms(self, molecule):
        """Return atoms from active site if available, else from whole molecule."""
        return molecule.active_site['atoms'] if hasattr(molecule, 'active_site') and 'atoms' in molecule.active_site else molecule.atoms
    def _calculate_vdw_gpu(self, protein_atoms, ligand_atoms):
        """GPU-accelerated van der Waals calculation."""
        if self.torch_available:
            return self._calculate_vdw_torch(protein_atoms, ligand_atoms)
        elif self.cupy_available:
            return self._calculate_vdw_cupy(protein_atoms, ligand_atoms)
        else:
            # Fall back to CPU implementation
            return self._calculate_vdw_physics(protein_atoms, ligand_atoms)
    
    def _calculate_vdw_torch(self, protein_atoms, ligand_atoms):
        """Calculate van der Waals interactions using PyTorch."""
        import torch
        
        # Extract coordinates and parameters
        p_coords = []
        p_types = []
        
        for atom in protein_atoms:
            p_coords.append(atom['coords'])
            p_types.append(self._get_atom_type(atom))
        
        l_coords = []
        l_types = []
        
        for atom in ligand_atoms:
            l_coords.append(atom['coords'])
            l_types.append(self._get_atom_type(atom))
        
        # Convert coordinates to PyTorch tensors
        p_coords = torch.tensor(np.array(p_coords), device=self.device)
        l_coords = torch.tensor(np.array(l_coords), device=self.device)
        
        # Create parameter tensors
        p_req = torch.tensor([self.vdw_params.get(t, self.vdw_params['C'])['r_eq'] for t in p_types], 
                             device=self.device).view(-1, 1)
        l_req = torch.tensor([self.vdw_params.get(t, self.vdw_params['C'])['r_eq'] for t in l_types], 
                             device=self.device).view(1, -1)
        
        p_eps = torch.tensor([self.vdw_params.get(t, self.vdw_params['C'])['epsilon'] for t in p_types], 
                             device=self.device).view(-1, 1)
        l_eps = torch.tensor([self.vdw_params.get(t, self.vdw_params['C'])['epsilon'] for t in l_types], 
                             device=self.device).view(1, -1)
        
        # Calculate all distances at once
        distances = torch.cdist(p_coords, l_coords)
        
        # Calculate combined parameters
        r_eq = (p_req + l_req) / 2.0  # Arithmetic mean
        epsilon = torch.sqrt(p_eps * l_eps)  # Geometric mean
        
        # Apply distance cutoff
        mask = distances <= self.vdw_cutoff
        
        # Safe distances to avoid numerical issues
        safe_distances = torch.clamp(distances, min=0.1)
        
        # Calculate ratio
        ratio = r_eq / safe_distances
        
        # Use modified potential with smoothing at close distances
        close_mask = safe_distances < 0.5 * r_eq
        
        # Regular 12-6 Lennard-Jones for normal distances
        ratio6 = ratio ** 6
        ratio12 = ratio6 ** 2
        lj_energy = epsilon * (ratio12 - 2.0 * ratio6)
        
        # Smoothed function for very close distances
        smoothed_ratio = 0.5 * r_eq / safe_distances
        smoothed_ratio6 = smoothed_ratio ** 6
        smoothed_ratio12 = smoothed_ratio6 ** 2
        smoothed_lj_energy = epsilon * (smoothed_ratio12 - 2.0 * smoothed_ratio6)
        
        # Combine normal and smoothed energies
        final_lj_energy = torch.where(close_mask, smoothed_lj_energy, lj_energy)
        
        # Cap extreme values
        capped_lj_energy = torch.clamp(final_lj_energy, min=-50.0, max=50.0)
        
        # Apply mask and sum
        masked_energy = capped_lj_energy * mask.float()
        vdw_energy = float(torch.sum(masked_energy).item())
        
        return vdw_energy
    
    def _calculate_vdw_cupy(self, protein_atoms, ligand_atoms):
        """Calculate van der Waals interactions using CuPy."""
        cp = self.cp
        
        # Extract coordinates and parameters
        p_coords = []
        p_types = []
        
        for atom in protein_atoms:
            p_coords.append(atom['coords'])
            p_types.append(self._get_atom_type(atom))
        
        l_coords = []
        l_types = []
        
        for atom in ligand_atoms:
            l_coords.append(atom['coords'])
            l_types.append(self._get_atom_type(atom))
        
        # Convert coordinates to CuPy arrays
        p_coords = cp.array(p_coords, dtype=self.cp_dtype)
        l_coords = cp.array(l_coords, dtype=self.cp_dtype)
        
        # Create parameter arrays
        p_req = cp.array([self.vdw_params.get(t, self.vdw_params['C'])['r_eq'] for t in p_types], 
                          dtype=self.cp_dtype).reshape(-1, 1)
        l_req = cp.array([self.vdw_params.get(t, self.vdw_params['C'])['r_eq'] for t in l_types], 
                          dtype=self.cp_dtype).reshape(1, -1)
        
        p_eps = cp.array([self.vdw_params.get(t, self.vdw_params['C'])['epsilon'] for t in p_types], 
                          dtype=self.cp_dtype).reshape(-1, 1)
        l_eps = cp.array([self.vdw_params.get(t, self.vdw_params['C'])['epsilon'] for t in l_types], 
                          dtype=self.cp_dtype).reshape(1, -1)
        
        # Calculate all distances at once
        diff = cp.expand_dims(p_coords, 1) - cp.expand_dims(l_coords, 0)
        distances = cp.sqrt(cp.sum(diff**2, axis=2))
        
        # Calculate combined parameters
        r_eq = (p_req + l_req) / 2.0  # Arithmetic mean
        epsilon = cp.sqrt(p_eps * l_eps)  # Geometric mean
        
        # Apply distance cutoff
        mask = distances <= self.vdw_cutoff
        
        # Safe distances to avoid numerical issues
        safe_distances = cp.maximum(distances, 0.1)
        
        # Calculate ratio
        ratio = r_eq / safe_distances
        
        # Use modified potential with smoothing at close distances
        close_mask = safe_distances < 0.5 * r_eq
        
        # Regular 12-6 Lennard-Jones for normal distances
        ratio6 = ratio ** 6
        ratio12 = ratio6 ** 2
        lj_energy = epsilon * (ratio12 - 2.0 * ratio6)
        
        # Smoothed function for very close distances
        smoothed_ratio = 0.5 * r_eq / safe_distances
        smoothed_ratio6 = smoothed_ratio ** 6
        smoothed_ratio12 = smoothed_ratio6 ** 2
        smoothed_lj_energy = epsilon * (smoothed_ratio12 - 2.0 * smoothed_ratio6)
        
        # Combine normal and smoothed energies
        final_lj_energy = cp.where(close_mask, smoothed_lj_energy, lj_energy)
        
        # Cap extreme values
        capped_lj_energy = cp.clip(final_lj_energy, -50.0, 50.0)
        
        # Apply mask and sum
        masked_energy = capped_lj_energy * mask
        vdw_energy = float(cp.sum(masked_energy))
        
        return vdw_energy
    
    def _calculate_hbond_gpu(self, protein_atoms, ligand_atoms, protein, ligand):
        """GPU-accelerated hydrogen bond calculation."""
        if self.torch_available:
            return self._calculate_hbond_torch(protein_atoms, ligand_atoms, protein, ligand)
        elif self.cupy_available:
            return self._calculate_hbond_cupy(protein_atoms, ligand_atoms, protein, ligand)
        else:
            # Fall back to CPU implementation
            return self._calculate_hbond_physics(protein_atoms, ligand_atoms, protein, ligand)
    
    def _calculate_hbond_torch(self, protein_atoms, ligand_atoms, protein, ligand):
        """Calculate hydrogen bonds using PyTorch."""
        # For simplicity, use CPU implementation as H-bonds often involve complex geometric calculations
        return self._calculate_hbond_physics(protein_atoms, ligand_atoms, protein, ligand)
    
    def _calculate_hbond_cupy(self, protein_atoms, ligand_atoms, protein, ligand):
        """Calculate hydrogen bonds using CuPy."""
        # For simplicity, use CPU implementation as H-bonds often involve complex geometric calculations
        return self._calculate_hbond_physics(protein_atoms, ligand_atoms, protein, ligand)
    
    def _calculate_electrostatics_gpu(self, protein_atoms, ligand_atoms):
        """GPU-accelerated electrostatics calculation."""
        if self.torch_available:
            return self._calculate_electrostatics_torch(protein_atoms, ligand_atoms)
        elif self.cupy_available:
            return self._calculate_electrostatics_cupy(protein_atoms, ligand_atoms)
        else:
            # Fall back to CPU implementation
            return self._calculate_electrostatics_physics(protein_atoms, ligand_atoms)
    
    def _calculate_electrostatics_torch(self, protein_atoms, ligand_atoms):
        """Calculate electrostatic interactions using PyTorch."""
        import torch
        
        # Extract coordinates and charges
        p_coords = []
        p_charges = []
        
        for atom in protein_atoms:
            p_coords.append(atom['coords'])
            p_element = atom.get('element', atom.get('name', 'C'))[0].upper()
            p_charges.append(self.electrostatics.atom_charges.get(p_element, 0.0))
        
        l_coords = []
        l_charges = []
        
        for atom in ligand_atoms:
            l_coords.append(atom['coords'])
            l_element = atom.get('symbol', 'C').upper()
            l_charges.append(self.electrostatics.atom_charges.get(l_element, 0.0))
        
        # Convert to PyTorch tensors
        p_coords = torch.tensor(np.array(p_coords), device=self.device)
        p_charges = torch.tensor(np.array(p_charges), device=self.device).view(-1, 1)
        
        l_coords = torch.tensor(np.array(l_coords), device=self.device)
        l_charges = torch.tensor(np.array(l_charges), device=self.device).view(1, -1)
        
        # Calculate all distances at once
        distances = torch.cdist(p_coords, l_coords)
        
        # Create charge products matrix
        charge_products = p_charges * l_charges
        
        # Apply distance cutoff
        mask = distances <= self.elec_cutoff
        
        # Calculate distance-dependent dielectric
        dielectric = 4.0 * distances
        
        # Calculate Coulomb energy with safe distances
        safe_distances = torch.clamp(distances, min=0.1)
        coulomb_energy = 332.0 * charge_products / (dielectric * safe_distances)
        
        # Apply mask and sum
        masked_energy = coulomb_energy * mask.float()
        elec_energy = float(torch.sum(masked_energy).item())
        
        return elec_energy
    
    def _calculate_electrostatics_cupy(self, protein_atoms, ligand_atoms):
        """Calculate electrostatic interactions using CuPy."""
        cp = self.cp
        
        # Extract coordinates and charges
        p_coords = []
        p_charges = []
        
        for atom in protein_atoms:
            p_coords.append(atom['coords'])
            p_element = atom.get('element', atom.get('name', 'C'))[0].upper()
            p_charges.append(self.electrostatics.atom_charges.get(p_element, 0.0))
        
        l_coords = []
        l_charges = []
        
        for atom in ligand_atoms:
            l_coords.append(atom['coords'])
            l_element = atom.get('symbol', 'C').upper()
            l_charges.append(self.electrostatics.atom_charges.get(l_element, 0.0))
        
        # Convert to CuPy arrays
        p_coords = cp.array(p_coords, dtype=self.cp_dtype)
        p_charges = cp.array(p_charges, dtype=self.cp_dtype).reshape(-1, 1)
        
        l_coords = cp.array(l_coords, dtype=self.cp_dtype)
        l_charges = cp.array(l_charges, dtype=self.cp_dtype).reshape(1, -1)
        
        # Calculate all distances at once
        diff = cp.expand_dims(p_coords, 1) - cp.expand_dims(l_coords, 0)
        distances = cp.sqrt(cp.sum(diff**2, axis=2))
        
        # Create charge products matrix
        charge_products = p_charges * l_charges
        
        # Apply distance cutoff
        mask = distances <= self.elec_cutoff
        
        # Calculate distance-dependent dielectric
        dielectric = 4.0 * distances
        
        # Calculate Coulomb energy with safe distances
        safe_distances = cp.maximum(distances, 0.1)
        coulomb_energy = 332.0 * charge_products / (dielectric * safe_distances)
        
        # Apply mask and sum
        masked_energy = coulomb_energy * mask
        elec_energy = float(cp.sum(masked_energy))
        
        return elec_energy
    
    def _calculate_desolvation_gpu(self, protein_atoms, ligand_atoms):
        """GPU-accelerated desolvation calculation."""
        if self.torch_available:
            return self._calculate_desolvation_torch(protein_atoms, ligand_atoms)
        elif self.cupy_available:
            return self._calculate_desolvation_cupy(protein_atoms, ligand_atoms)
        else:
            # Fall back to CPU implementation
            return self._calculate_desolvation_physics(protein_atoms, ligand_atoms)
    
    def _calculate_desolvation_torch(self, protein_atoms, ligand_atoms):
        """Calculate desolvation energy using PyTorch."""
        import torch
        
        # Extract coordinates and solvation parameters
        p_coords = []
        p_types = []
        
        for atom in protein_atoms:
            p_coords.append(atom['coords'])
            p_types.append(self._get_atom_type(atom))
        
        l_coords = []
        l_types = []
        
        for atom in ligand_atoms:
            l_coords.append(atom['coords'])
            l_types.append(self._get_atom_type(atom))
        
        # Convert to PyTorch tensors
        p_coords = torch.tensor(np.array(p_coords), device=self.device)
        l_coords = torch.tensor(np.array(l_coords), device=self.device)
        
        # Get solvation and volume parameters
        p_solv = torch.tensor([self.atom_solvation_params.get(t, 0.0) for t in p_types], 
                              device=self.device).view(-1, 1)
        l_solv = torch.tensor([self.atom_solvation_params.get(t, 0.0) for t in l_types], 
                              device=self.device).view(1, -1)
        
        p_vol = torch.tensor([self.atom_volume_params.get(t, 0.0) for t in p_types], 
                             device=self.device).view(-1, 1)
        l_vol = torch.tensor([self.atom_volume_params.get(t, 0.0) for t in l_types], 
                             device=self.device).view(1, -1)
        
        # Calculate all distances at once
        distances = torch.cdist(p_coords, l_coords)
        
        # Apply distance cutoff
        mask = distances <= self.desolv_cutoff
        
        # Calculate Gaussian-like desolvation
        sigma = self.solvation_k
        sigma_squared_2 = 2.0 * sigma * sigma
        distances_squared = distances ** 2
        exp_term = torch.exp(-distances_squared / sigma_squared_2)
        
        # Calculate desolvation energy terms with volume weighting
        term1 = self.solpar * p_solv * l_vol
        term2 = self.solpar * l_solv * p_vol
        solvation_terms = (term1 + term2) * exp_term
        
        # Cap extreme values
        capped_terms = torch.sign(solvation_terms) * torch.clamp(torch.abs(solvation_terms), max=5.0)
        
        # Apply mask and sum
        masked_energy = capped_terms * mask.float()
        desolv_energy = float(torch.sum(masked_energy).item())
        
        # Apply final capping
        max_desolv = 1000.0
        capped_desolv = min(abs(desolv_energy), max_desolv)
        
        return capped_desolv
    
    def _calculate_desolvation_cupy(self, protein_atoms, ligand_atoms):
        """Calculate desolvation energy using CuPy."""
        cp = self.cp
        
        # Extract coordinates and solvation parameters
        p_coords = []
        p_types = []
        
        for atom in protein_atoms:
            p_coords.append(atom['coords'])
            p_types.append(self._get_atom_type(atom))
        
        l_coords = []
        l_types = []
        
        for atom in ligand_atoms:
            l_coords.append(atom['coords'])
            l_types.append(self._get_atom_type(atom))
        
        # Convert to CuPy arrays
        p_coords = cp.array(p_coords, dtype=self.cp_dtype)
        l_coords = cp.array(l_coords, dtype=self.cp_dtype)
        
        # Get solvation and volume parameters
        p_solv = cp.array([self.atom_solvation_params.get(t, 0.0) for t in p_types], 
                          dtype=self.cp_dtype).reshape(-1, 1)
        l_solv = cp.array([self.atom_solvation_params.get(t, 0.0) for t in l_types], 
                          dtype=self.cp_dtype).reshape(1, -1)
        
        p_vol = cp.array([self.atom_volume_params.get(t, 0.0) for t in p_types], 
                         dtype=self.cp_dtype).reshape(-1, 1)
        l_vol = cp.array([self.atom_volume_params.get(t, 0.0) for t in l_types], 
                         dtype=self.cp_dtype).reshape(1, -1)
        
        # Calculate all distances at once
        diff = cp.expand_dims(p_coords, 1) - cp.expand_dims(l_coords, 0)
        distances = cp.sqrt(cp.sum(diff**2, axis=2))
        
        # Apply distance cutoff
        mask = distances <= self.desolv_cutoff
        
        # Calculate Gaussian-like desolvation
        sigma = self.solvation_k
        sigma_squared_2 = 2.0 * sigma * sigma
        distances_squared = distances ** 2
        exp_term = cp.exp(-distances_squared / sigma_squared_2)
        
        # Calculate desolvation energy terms with volume weighting
        term1 = self.solpar * p_solv * l_vol
        term2 = self.solpar * l_solv * p_vol
        solvation_terms = (term1 + term2) * exp_term
        
        # Cap extreme values
        capped_terms = cp.sign(solvation_terms) * cp.minimum(cp.abs(solvation_terms), 5.0)
        
        # Apply mask and sum
        masked_energy = capped_terms * mask
        desolv_energy = float(cp.sum(masked_energy))
        
        # Apply final capping
        max_desolv = 1000.0
        capped_desolv = min(abs(desolv_energy), max_desolv)
        
        return capped_desolv
    
    def _calculate_hydrophobic_gpu(self, protein_atoms, ligand_atoms):
        """GPU-accelerated hydrophobic interaction calculation."""
        if self.torch_available:
            return self._calculate_hydrophobic_torch(protein_atoms, ligand_atoms)
        elif self.cupy_available:
            return self._calculate_hydrophobic_cupy(protein_atoms, ligand_atoms)
        else:
            # Fall back to CPU implementation
            return self._calculate_hydrophobic_physics(protein_atoms, ligand_atoms)
    
    def _calculate_hydrophobic_torch(self, protein_atoms, ligand_atoms):
        """Calculate hydrophobic interactions using PyTorch."""
        import torch
        
        # Identify hydrophobic atoms
        p_hydrophobic = []
        p_coords = []
        
        for atom in protein_atoms:
            atom_type = self._get_atom_type(atom)
            if atom_type in self.hydrophobic_types:
                p_hydrophobic.append(1)
                p_coords.append(atom['coords'])
            
        l_hydrophobic = []
        l_coords = []
        
        for atom in ligand_atoms:
            atom_type = self._get_atom_type(atom)
            if atom_type in self.hydrophobic_types:
                l_hydrophobic.append(1)
                l_coords.append(atom['coords'])
        
        # Skip if no hydrophobic atoms
        if not p_coords or not l_coords:
            return 0.0
            
        # Convert to PyTorch tensors
        p_coords = torch.tensor(np.array(p_coords), device=self.device)
        l_coords = torch.tensor(np.array(l_coords), device=self.device)
        
        # Calculate all distances at once
        distances = torch.cdist(p_coords, l_coords)
        
        # Apply distance cutoff (default 4.5Å)
        cutoff = 4.5
        mask = distances <= cutoff
        
        # Calculate hydrophobic interaction strength
        contact_score = (cutoff - distances) / cutoff
        
        # Apply mask and sum
        masked_score = contact_score * mask.float()
        score = float(torch.sum(masked_score).item())
        
        # Apply capping
        max_hydrophobic = 200.0
        capped_score = min(score, max_hydrophobic)
        
        return capped_score
    
    def _calculate_hydrophobic_cupy(self, protein_atoms, ligand_atoms):
        """Calculate hydrophobic interactions using CuPy."""
        cp = self.cp
        
        # Identify hydrophobic atoms
        p_hydrophobic = []
        p_coords = []
        
        for atom in protein_atoms:
            atom_type = self._get_atom_type(atom)
            if atom_type in self.hydrophobic_types:
                p_hydrophobic.append(1)
                p_coords.append(atom['coords'])
            
        l_hydrophobic = []
        l_coords = []
        
        for atom in ligand_atoms:
            atom_type = self._get_atom_type(atom)
            if atom_type in self.hydrophobic_types:
                l_hydrophobic.append(1)
                l_coords.append(atom['coords'])
        
        # Skip if no hydrophobic atoms
        if not p_coords or not l_coords:
            return 0.0
            
        # Convert to CuPy arrays
        p_coords = cp.array(p_coords, dtype=self.cp_dtype)
        l_coords = cp.array(l_coords, dtype=self.cp_dtype)
        
        # Calculate all distances at once
        diff = cp.expand_dims(p_coords, 1) - cp.expand_dims(l_coords, 0)
        distances = cp.sqrt(cp.sum(diff**2, axis=2))
        
        # Apply distance cutoff (default 4.5Å)
        cutoff = 4.5
        mask = distances <= cutoff
        
        # Calculate hydrophobic interaction strength
        contact_score = (cutoff - distances) / cutoff
        
        # Apply mask and sum
        masked_score = contact_score * mask
        score = float(cp.sum(masked_score))
        
        # Apply capping
        max_hydrophobic = 200.0
        capped_score = min(score, max_hydrophobic)
        
        return capped_score
    
    def _calculate_clash_gpu(self, protein_atoms, ligand_atoms):
        """GPU-accelerated clash calculation."""
        if self.torch_available:
            return self._calculate_clash_torch(protein_atoms, ligand_atoms)
        elif self.cupy_available:
            return self._calculate_clash_cupy(protein_atoms, ligand_atoms)
        else:
            # Fall back to CPU implementation
            return self._calculate_clash_physics(protein_atoms, ligand_atoms)
    
    def _calculate_clash_torch(self, protein_atoms, ligand_atoms):
        """Calculate steric clashes using PyTorch."""
        import torch
        
        # Extract coordinates and radii
        p_coords = []
        p_radii = []
        
        for atom in protein_atoms:
            p_coords.append(atom['coords'])
            p_type = self._get_atom_type(atom)
            p_radii.append(self.vdw_radii.get(p_type[0], 1.7))
        
        l_coords = []
        l_radii = []
        
        for atom in ligand_atoms:
            l_coords.append(atom['coords'])
            l_type = self._get_atom_type(atom)
            l_radii.append(self.vdw_radii.get(l_type[0], 1.7))
        
        # Convert to PyTorch tensors
        p_coords = torch.tensor(np.array(p_coords), device=self.device)
        p_radii = torch.tensor(np.array(p_radii), device=self.device).view(-1, 1)
        
        l_coords = torch.tensor(np.array(l_coords), device=self.device)
        l_radii = torch.tensor(np.array(l_radii), device=self.device).view(1, -1)
        
        # Calculate all distances at once
        distances = torch.cdist(p_coords, l_coords)
        
        # Calculate minimum allowed distances (70% of sum of vdW radii)
        min_allowed = (p_radii + l_radii) * 0.7
        
        # Identify clashes
        clash_mask = distances < min_allowed
        
        # Count clashes
        count = float(torch.sum(clash_mask).item())
        
        # Apply penalty and capping
        clash_energy = count * 5.0  # Penalty factor
        max_clash = 100.0
        capped_clash = min(clash_energy, max_clash)
        
        return capped_clash
    
    def _calculate_clash_cupy(self, protein_atoms, ligand_atoms):
        """Calculate steric clashes using CuPy."""
        cp = self.cp
        
        # Extract coordinates and radii
        p_coords = []
        p_radii = []
        
        for atom in protein_atoms:
            p_coords.append(atom['coords'])
            p_type = self._get_atom_type(atom)
            p_radii.append(self.vdw_radii.get(p_type[0], 1.7))
        
        l_coords = []
        l_radii = []
        
        for atom in ligand_atoms:
            l_coords.append(atom['coords'])
            l_type = self._get_atom_type(atom)
            l_radii.append(self.vdw_radii.get(l_type[0], 1.7))
        
        # Convert to CuPy arrays
        p_coords = cp.array(p_coords, dtype=self.cp_dtype)
        p_radii = cp.array(p_radii, dtype=self.cp_dtype).reshape(-1, 1)
        
        l_coords = cp.array(l_coords, dtype=self.cp_dtype)
        l_radii = cp.array(l_radii, dtype=self.cp_dtype).reshape(1, -1)
        
        # Calculate all distances at once
        diff = cp.expand_dims(p_coords, 1) - cp.expand_dims(l_coords, 0)
        distances = cp.sqrt(cp.sum(diff**2, axis=2))
        
        # Calculate minimum allowed distances (70% of sum of vdW radii)
        min_allowed = (p_radii + l_radii) * 0.7
        
        # Identify clashes
        clash_mask = distances < min_allowed
        
        # Count clashes
        count = float(cp.sum(clash_mask))
        
        # Apply penalty and capping
        clash_energy = count * 5.0  # Penalty factor
        max_clash = 100.0
        capped_clash = min(clash_energy, max_clash)
        
        return capped_clash