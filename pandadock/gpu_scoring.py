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
        
        # Calculate base terms
        vdw_score = self._calculate_vdw(protein, ligand)
        hbond_score = self._calculate_hbond(protein, ligand)
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
        
        end_time = time.time()
        if hasattr(self, 'verbose') and self.verbose:
            print(f"Scoring completed in {end_time - start_time:.4f} seconds")
            print(f"VDW: {vdw_score:.2f}, H-bond: {hbond_score:.2f}, Elec: {elec_score:.2f}, "
                 f"Desolv: {desolv_score:.2f}, Hydrophobic: {hydrophobic_score:.2f}, "
                 f"Clash: {clash_score:.2f}, Entropy: {entropy_score:.2f}")
        
        return total_score
    
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
        p_coords = torch.tensor(p_coords, device=self.device)
        p_radii = torch.tensor(p_radii, device=self.device).view(-1, 1)
        p_depths = torch.tensor(p_depths, device=self.device).view(-1, 1)
        
        l_coords = torch.tensor(l_coords, device=self.device)
        l_radii = torch.tensor(l_radii, device=self.device).view(1, -1)
        l_depths = torch.tensor(l_depths, device=self.device).view(1, -1)
        
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
        p_coords = torch.tensor(p_coords, device=self.device)
        p_charges = torch.tensor(p_charges, device=self.device)
        
        l_coords = torch.tensor(l_coords, device=self.device)
        l_charges = torch.tensor(l_charges, device=self.device)
        
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
        p_coords = torch.tensor(p_coords, device=self.device)
        p_solvation = torch.tensor(p_solvation, device=self.device).view(-1, 1)
        
        l_coords = torch.tensor(l_coords, device=self.device)
        l_solvation = torch.tensor(l_solvation, device=self.device).view(1, -1)
        
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
    
    def _calculate_hydrophobic(self, protein, ligand):
        """
        GPU-accelerated hydrophobic interaction calculation.
        """
        # Get protein atoms
        if protein.active_site and 'atoms' in protein.active_site:
            protein_atoms = protein.active_site['atoms']
        else:
            protein_atoms = protein.atoms
        
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
            return super()._calculate_hydrophobic(protein, ligand)
    
    def _calculate_hydrophobic_torch(self, p_hydrophobic, l_hydrophobic):
        """
        Calculate hydrophobic interactions using PyTorch.
        """
        import torch
        
        # Extract coordinates
        p_coords = [atom['coords'] for atom in p_hydrophobic]
        l_coords = [atom['coords'] for atom in l_hydrophobic]
        
        # Convert to PyTorch tensors
        p_coords = torch.tensor(p_coords, device=self.device)
        l_coords = torch.tensor(l_coords, device=self.device)
        
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
    
    def _calculate_hbond(self, protein, ligand):
        """
        GPU-accelerated hydrogen bond calculation.
        Maps to the parent class's hbond_scorer.score method.
        """
        # This is usually less computationally intensive and can use the CPU implementation
        if hasattr(self, 'hbond_scorer'):
            return self.hbond_scorer.score(protein, ligand)
        else:
            # Fall back to parent method
            return super()._calculate_hbond_energy(protein, ligand) if hasattr(super(), '_calculate_hbond_energy') else 0.0
    
    def _calculate_clashes(self, protein, ligand):
        """
        GPU-accelerated steric clash calculation.
        """
        # Get protein atoms
        if protein.active_site and 'atoms' in protein.active_site:
            protein_atoms = protein.active_site['atoms']
        else:
            protein_atoms = protein.atoms
        
        if self.torch_available:
            return self._calculate_clashes_torch(protein_atoms, ligand.atoms)
        elif self.cupy_available:
            return self._calculate_clashes_cupy(protein_atoms, ligand.atoms)
        else:
            # Fall back to CPU implementation
            return super()._calculate_clashes(protein, ligand)
    
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
        p_coords = torch.tensor(p_coords, device=self.device)
        p_radii = torch.tensor(p_radii, device=self.device).view(-1, 1)
        
        l_coords = torch.tensor(l_coords, device=self.device)
        l_radii = torch.tensor(l_radii, device=self.device).view(1, -1)
        
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
    
    def _calculate_entropy(self, ligand):
        """
        Calculate entropy penalty based on ligand flexibility.
        This is typically less computationally intensive, so we use the CPU implementation.
        """
        return super()._calculate_entropy(ligand)
