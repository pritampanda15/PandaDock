"""
GPU-accelerated scoring functions for PandaDock.
This module provides scoring functions that leverage GPU acceleration
for computationally intensive calculations such as electrostatics and vdW interactions.
"""

import numpy as np
import time
import warnings
from .scoring import EnhancedScoringFunction


class GPUAcceleratedScoringFunction(EnhancedScoringFunction):
    """
    GPU-accelerated scoring function that mirrors EnhancedScoringFunction logic,
    using PyTorch or CuPy when available for performance.
    """

    def __init__(self, device='cuda', precision='float32'):
        super().__init__()

        self.device_name = device
        self.precision = precision
        self.device = None
        self.torch_available = False
        self.cupy_available = False
        self._init_gpu()

    def _init_gpu(self):
        try:
            import torch
            self.torch_available = True
            self.torch = torch

            if self.device_name == 'cuda' and torch.cuda.is_available():
                self.device = torch.device('cuda')
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device('cpu')
                print("GPU not available. Using CPU via PyTorch.")

        except ImportError:
            try:
                import cupy as cp
                self.cupy_available = True
                self.cp = cp
                print("Using GPU via CuPy")
            except ImportError:
                print("Neither PyTorch nor CuPy available. Falling back to CPU.")

    def score(self, protein, ligand):
        if protein.active_site and 'atoms' in protein.active_site:
            protein_atoms = protein.active_site['atoms']
        else:
            protein_atoms = protein.atoms

        vdw_score = self._calculate_vdw_physics(protein_atoms, ligand.atoms)
        hbond_score = self._calculate_hbond_physics(protein_atoms, ligand.atoms, protein, ligand)
        elec_score = self._calculate_electrostatics_physics(protein_atoms, ligand.atoms)
        desolv_score = self._calculate_desolvation_physics(protein_atoms, ligand.atoms)
        hydrophobic_score = self._calculate_hydrophobic_physics(protein_atoms, ligand.atoms)
        clash_score = self._calculate_clashes_physics(protein_atoms, ligand.atoms)
        entropy_score = self._calculate_entropy(ligand)

        total_score = (
            self.weights['vdw'] * vdw_score +
            self.weights['hbond'] * hbond_score +
            self.weights['elec'] * elec_score +
            self.weights['desolv'] * desolv_score +
            self.weights['hydrophobic'] * hydrophobic_score +
            self.weights['clash'] * clash_score +
            self.weights['entropy'] * entropy_score
        )

        self.last_component_scores = {
            'vdw': vdw_score,
            'hbond': hbond_score,
            'elec': elec_score,
            'desolv': desolv_score,
            'hydrophobic': hydrophobic_score,
            'clash': clash_score,
            'entropy': entropy_score,
            'total': total_score
        }

        if getattr(self, 'debug', False):
            self._print_score_breakdown(self.last_component_scores)

        return total_score

    def get_component_scores(self):
        return self.last_component_scores if hasattr(self, 'last_component_scores') else None

    def _print_score_breakdown(self, scores):
        print("\n----- GPU SCORING BREAKDOWN -----")
        for key in ['vdw', 'hbond', 'elec', 'desolv', 'hydrophobic', 'clash', 'entropy']:
            print(f"{key.capitalize():12}: {scores[key]:8.4f} × {self.weights[key]:.2f} = {scores[key] * self.weights[key]:.4f}")
        print(f"TOTAL SCORE  : {scores['total']:.4f}")
        print("-------------------------------\n")
    
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
    
    def _calculate_entropy(self, ligand):
        """
        Calculate entropy penalty based on ligand flexibility.
        This is typically less computationally intensive, so we use the CPU implementation.
        """
        return super()._calculate_entropy(ligand)
