import numpy as np
from copy import deepcopy
from pathlib import Path
import os
import tempfile

class MMFFMinimization:
    """
    MMFF94 Force Field minimization for ligands using RDKit.
    This provides full molecular mechanics energy minimization.
    """
    
    def __init__(self, max_iterations=200, converge_criterion=0.01):
        """
        Initialize MMFF minimization.
        
        Parameters:
        -----------
        max_iterations : int
            Maximum number of minimization steps
        converge_criterion : float
            Convergence criterion for energy change
        """
        self.max_iterations = max_iterations
        self.converge_criterion = converge_criterion
        self._check_rdkit()
    
    def _check_rdkit(self):
        """Check if RDKit is available and raise import error if not."""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            self.rdkit_available = True
        except ImportError:
            print("Warning: RDKit not available. MMFF minimization will not work.")
            self.rdkit_available = False
    
    def minimize_ligand(self, ligand):
        """
        Perform MMFF minimization on a ligand.
        
        Parameters:
        -----------
        ligand : Ligand
            Ligand object from PandaDock
        
        Returns:
        --------
        Ligand
            Minimized ligand
        """
        if not self.rdkit_available:
            print("RDKit not available. Skipping minimization.")
            return ligand
        
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            
            # Create a temporary file for the ligand
            fd, tmp_file = tempfile.mkstemp(suffix='.sdf')
            os.close(fd)
            
            # Write ligand to SDF file
            self._write_ligand_to_sdf(ligand, tmp_file)
            
            # Read with RDKit
            mol = Chem.SDMolSupplier(tmp_file)[0]
            if mol is None:
                print("Error: Could not read ligand with RDKit.")
                return ligand
            
            # Set up MMFF force field
            success = AllChem.MMFFOptimizeMolecule(
                mol, 
                maxIters=self.max_iterations,
                nonBondedThresh=10.0,  # Angstroms
                confId=0
            )
            
            if success == -1:
                print("Warning: MMFF setup failed. Falling back to UFF.")
                AllChem.UFFOptimizeMolecule(mol, maxIters=self.max_iterations)
            
            # Write minimized structure back to file
            writer = Chem.SDWriter(tmp_file)
            writer.write(mol)
            writer.close()
            
            # Read back into a new ligand object
            minimized_ligand = self._read_ligand_from_sdf(tmp_file)
            
            # Clean up
            os.unlink(tmp_file)
            
            return minimized_ligand
            
        except Exception as e:
            print(f"Error during minimization: {e}")
            return ligand
    
    def minimize_pose(self, protein, ligand_pose, distance_cutoff=10.0):
        """
        Perform constrained minimization of a ligand pose in protein environment.
        
        Parameters:
        -----------
        protein : Protein
            Protein object from PandaDock
        ligand_pose : Ligand
            Ligand pose to minimize
        distance_cutoff : float
            Distance cutoff for protein-ligand interactions (Angstroms)
        
        Returns:
        --------
        Ligand
            Minimized ligand pose
        """
        if not self.rdkit_available:
            print("RDKit not available. Skipping pose minimization.")
            return ligand_pose
        
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            
            # Create temporary files
            fd1, tmp_ligand = tempfile.mkstemp(suffix='.sdf')
            fd2, tmp_protein = tempfile.mkstemp(suffix='.pdb')
            os.close(fd1)
            os.close(fd2)
            
            # Write ligand and protein to files
            self._write_ligand_to_sdf(ligand_pose, tmp_ligand)
            self._write_protein_to_pdb(protein, tmp_protein)
            
            # Read with RDKit
            lig_mol = Chem.SDMolSupplier(tmp_ligand)[0]
            prot_mol = Chem.MolFromPDB(tmp_protein)
            
            if lig_mol is None or prot_mol is None:
                print("Error reading molecules for constrained minimization.")
                return ligand_pose
            
            # Create a combined system for MMFF
            combo = Chem.CombineMols(prot_mol, lig_mol)
            
            # Setup MMFF and minimize
            try:
                # This part is experimental and may not work in all RDKit versions
                mp = AllChem.MMFFGetMoleculeProperties(combo)
                ff = AllChem.MMFFGetMoleculeForceField(
                    combo, mp, nonBondedThresh=distance_cutoff
                )
                
                # Freeze protein atoms
                for i in range(prot_mol.GetNumAtoms()):
                    ff.AddFixedPoint(i)
                
                # Run minimization
                ff.Minimize(maxIts=self.max_iterations, 
                           energyTol=self.converge_criterion)
                
                # Extract ligand part
                minimized_mol = Chem.DeleteSubstructs(combo, prot_mol)
                
                # Write to file
                writer = Chem.SDWriter(tmp_ligand)
                writer.write(minimized_mol)
                writer.close()
                
                # Read minimized pose
                minimized_pose = self._read_ligand_from_sdf(tmp_ligand)
                
            except Exception as e:
                print(f"MMFF constrained minimization failed: {e}")
                print("Falling back to ligand-only minimization.")
                AllChem.MMFFOptimizeMolecule(lig_mol, maxIters=self.max_iterations)
                
                # Write to file
                writer = Chem.SDWriter(tmp_ligand)
                writer.write(lig_mol)
                writer.close()
                
                # Read minimized pose
                minimized_pose = self._read_ligand_from_sdf(tmp_ligand)
            
            # Clean up
            os.unlink(tmp_ligand)
            os.unlink(tmp_protein)
            
            return minimized_pose
            
        except Exception as e:
            print(f"Error during constrained minimization: {e}")
            return ligand_pose
    
    def _write_ligand_to_sdf(self, ligand, filename):
        """Write ligand to SDF file."""
        with open(filename, 'w') as f:
            f.write("Ligand\n")
            f.write("  PandaDock\n\n")
            
            # Number of atoms and bonds
            f.write(f"{len(ligand.atoms):3d}{len(ligand.bonds):3d}  0  0  0  0  0  0  0  0999 V2000\n")
            
            # Atoms
            for atom in ligand.atoms:
                coords = atom['coords']
                symbol = atom.get('symbol', 'C')
                f.write(f"{coords[0]:10.4f}{coords[1]:10.4f}{coords[2]:10.4f} {symbol:<3}  0  0  0  0  0  0  0  0  0  0  0  0\n")
            
            # Bonds
            for bond in ligand.bonds:
                a1 = bond['begin_atom_idx'] + 1  # 1-based indexing in SDF
                a2 = bond['end_atom_idx'] + 1
                type_num = bond.get('bond_type', 1)
                if isinstance(type_num, str):
                    type_num = 1  # Default to single bond
                f.write(f"{a1:3d}{a2:3d}{type_num:3d}  0  0  0  0\n")
            
            # Terminator
            f.write("M  END\n$$$$\n")
    
    def _write_protein_to_pdb(self, protein, filename):
        """Write protein to PDB file."""
        with open(filename, 'w') as f:
            for i, atom in enumerate(protein.atoms):
                name = atom.get('name', '').ljust(4)
                res_name = atom.get('residue_name', 'UNK')
                chain_id = atom.get('chain_id', 'A')
                res_id = atom.get('residue_id', 1)
                coords = atom['coords']
                
                f.write(f"ATOM  {i+1:5d} {name} {res_name:3s} {chain_id:1s}{res_id:4d}    "
                        f"{coords[0]:8.3f}{coords[1]:8.3f}{coords[2]:8.3f}  1.00  0.00\n")
            f.write("END\n")
    
    def _read_ligand_from_sdf(self, filename):
        """Read ligand from SDF file."""
        from .ligand import Ligand
        return Ligand(filename)


class ImprovedElectrostatics:
    """
    Improved electrostatics calculations with Poisson-Boltzmann inspired model.
    This provides more accurate treatment of charge-charge interactions.
    """
    
    def __init__(self, ionic_strength=0.15, temperature=298.15, 
                 interior_dielectric=4.0, solvent_dielectric=80.0):
        """
        Initialize improved electrostatics model.
        
        Parameters:
        -----------
        ionic_strength : float
            Ionic strength in mol/L
        temperature : float
            Temperature in Kelvin
        interior_dielectric : float
            Dielectric constant inside protein and ligand
        solvent_dielectric : float
            Dielectric constant of solvent
        """
        self.ionic_strength = ionic_strength  # mol/L
        self.temperature = temperature  # K
        self.interior_dielectric = interior_dielectric
        self.solvent_dielectric = solvent_dielectric
        
        # Physical constants
        self.k_boltzmann = 1.380649e-23  # J/K
        self.e_charge = 1.602176634e-19  # C
        self.n_avogadro = 6.02214076e23  # 1/mol
        self.epsilon_0 = 8.8541878128e-12  # F/m
        
        # Compute derived quantities
        self.kappa = self._compute_kappa()  # Debye screening length (Å^-1)
        
        # Atom parameters
        self.atom_charges = {
            'H': 0.0, 'C': 0.0, 'N': -0.5, 'O': -0.5, 'S': -0.2,
            'P': 0.5, 'F': -0.25, 'Cl': -0.1, 'Br': -0.1, 'I': -0.1
        }
        
        # Atomic radii for accessibility calculations
        self.atom_radii = {
            'H': 1.2, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8,
            'P': 1.8, 'F': 1.47, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98
        }
    
    def _compute_kappa(self):
        """
        Compute the Debye screening parameter (kappa) based on ionic strength.
        
        Returns:
        --------
        float
            Debye screening parameter in Å^-1
        """
        # Convert from SI units to more convenient units for docking
        # Factor of 10 is to convert from m^-1 to Å^-1
        kappa_squared = (2 * self.ionic_strength * self.n_avogadro * self.e_charge**2) / \
                        (self.epsilon_0 * self.solvent_dielectric * self.k_boltzmann * self.temperature)
        return np.sqrt(kappa_squared) * 1e-10  # Convert to Å^-1
    
    def calculate_electrostatics(self, protein, ligand):
        """
        Calculate electrostatic interaction energy using a modified Poisson-Boltzmann approach.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        ligand : Ligand
            Ligand object
        
        Returns:
        --------
        float
            Electrostatic interaction energy in kcal/mol
        """
        # Get active site atoms if defined
        if protein.active_site and 'atoms' in protein.active_site:
            protein_atoms = protein.active_site['atoms']
        else:
            protein_atoms = protein.atoms
        
        # Calculate solvent accessible surface for protein and ligand atoms
        protein_sasa = self._calc_approximate_sasa(protein_atoms)
        ligand_sasa = self._calc_approximate_sasa(ligand.atoms)
        
        # Initialize energy
        elec_energy = 0.0
        
        # Constants for conversion to kcal/mol (from J)
        conversion = 1.0 / (4.184 * 1000)  # J to kcal/mol
        
        # Calculate pairwise interactions
        for i, p_atom in enumerate(protein_atoms):
            p_coords = p_atom['coords']
            p_symbol = p_atom.get('element', p_atom.get('name', 'C'))[0]
            p_charge = self.atom_charges.get(p_symbol, 0.0)
            p_buried = 1.0 - min(1.0, protein_sasa[i])
            
            for j, l_atom in enumerate(ligand.atoms):
                l_coords = l_atom['coords']
                l_symbol = l_atom.get('symbol', 'C')
                l_charge = self.atom_charges.get(l_symbol, 0.0)
                l_buried = 1.0 - min(1.0, ligand_sasa[j])
                
                # Skip if either charge is zero
                if abs(p_charge * l_charge) < 1e-6:
                    continue
                
                # Calculate distance
                distance = np.linalg.norm(p_coords - l_coords)
                
                # Skip if too far away
                if distance > 15.0:  # Cutoff for efficiency
                    continue
                
                # Calculate effective dielectric based on burial
                # More buried atoms experience lower dielectric
                burial_factor = (p_buried + l_buried) / 2.0
                effective_dielectric = self.interior_dielectric + \
                                     (self.solvent_dielectric - self.interior_dielectric) * (1.0 - burial_factor)
                
                # Modified Coulomb with Debye-Hückel screening
                if distance < 0.1:  # Avoid division by zero
                    energy = 0.0
                else:
                    # Coulomb term with screening
                    coulomb = 332.0 * p_charge * l_charge / (effective_dielectric * distance)
                    screening = np.exp(-self.kappa * distance)
                    energy = coulomb * screening
                
                elec_energy += energy
        
        return elec_energy
    
    def _calc_approximate_sasa(self, atoms):
        """
        Calculate approximate solvent accessible surface area (SASA) for each atom.
        
        Parameters:
        -----------
        atoms : list
            List of atom dictionaries
        
        Returns:
        --------
        list
            List of SASA values for each atom
        """
        # Convert atoms to numpy array for faster computation
        coords = np.array([atom['coords'] for atom in atoms])
        n_atoms = len(atoms)
        
        # Get radii
        radii = np.zeros(n_atoms)
        for i, atom in enumerate(atoms):
            symbol = atom.get('element', atom.get('name', atom.get('symbol', 'C')))[0]
            radii[i] = self.atom_radii.get(symbol, 1.7)
        
        # Add water probe radius (1.4 Å)
        radii_with_probe = radii + 1.4
        
        # Calculate approximate SASA by check for neighboring atoms
        sasa = np.ones(n_atoms)  # Start with fully exposed
        
        # Calculate all pairwise distances
        dist_matrix = np.zeros((n_atoms, n_atoms))
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                dist = np.linalg.norm(coords[i] - coords[j])
                dist_matrix[i,j] = dist
                dist_matrix[j,i] = dist
        
        # For each atom, check how many neighbors are within van der Waals contact
        for i in range(n_atoms):
            r_i = radii_with_probe[i]
            
            # Count neighbors that overlap
            for j in range(n_atoms):
                if i == j:
                    continue
                    
                r_j = radii[j]  # Note: no probe for neighbors
                dist = dist_matrix[i,j]
                
                # Check if atoms overlap
                if dist < (r_i + r_j):
                    # Estimate overlap
                    overlap = 1.0 - (dist / (r_i + r_j))
                    # Reduce SASA proportionally to overlap
                    sasa[i] -= overlap * 0.1  # Scale factor to avoid overestimation
        
        # Ensure SASA is non-negative
        sasa = np.maximum(0.0, sasa)
        
        return sasa


class GeneralizedBornSolvation:
    """
    Generalized Born (GB) model for solvation energy.
    This provides an implicit solvent model for calculating solvation effects.
    """
    
    def __init__(self, temperature=298.15, solvent_dielectric=80.0, 
                 interior_dielectric=1.0, surface_tension=0.00542):
        """
        Initialize GB solvation model.
        
        Parameters:
        -----------
        temperature : float
            Temperature in Kelvin
        solvent_dielectric : float
            Dielectric constant of solvent
        interior_dielectric : float
            Dielectric constant inside protein and ligand
        surface_tension : float
            Surface tension parameter for nonpolar contribution (kcal/mol/Å²)
        """
        self.temperature = temperature
        self.solvent_dielectric = solvent_dielectric
        self.interior_dielectric = interior_dielectric
        self.surface_tension = surface_tension
        
        # Atom parameters
        self.atom_charges = {
            'H': 0.0, 'C': 0.0, 'N': -0.5, 'O': -0.5, 'S': -0.2,
            'P': 0.5, 'F': -0.25, 'Cl': -0.1, 'Br': -0.1, 'I': -0.1
        }
        
        # Atom radii for GB calculations
        self.atom_radii = {
            'H': 1.2, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8,
            'P': 1.8, 'F': 1.47, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98
        }
        
        # Scale factor for Born radii calculations
        self.scale_factor = 0.8
    
    def calculate_solvation_free_energy(self, molecule, molecule_type='ligand'):
        """
        Calculate solvation free energy using GB model.
        
        Parameters:
        -----------
        molecule : Ligand or Protein
            Molecule object
        molecule_type : str
            Type of molecule ('ligand' or 'protein')
        
        Returns:
        --------
        tuple
            (polar_energy, nonpolar_energy, total_energy) in kcal/mol
        """
        if molecule_type == 'ligand':
            atoms = molecule.atoms
            atom_list = []
            for atom in atoms:
                symbol = atom.get('symbol', 'C')
                coords = atom['coords']
                charge = self.atom_charges.get(symbol, 0.0)
                radius = self.atom_radii.get(symbol, 1.7)
                atom_list.append((coords, charge, radius))
        else:  # protein
            atoms = molecule.atoms
            atom_list = []
            for atom in atoms:
                symbol = atom.get('element', atom.get('name', 'C'))[0]
                coords = atom['coords']
                charge = self.atom_charges.get(symbol, 0.0)
                radius = self.atom_radii.get(symbol, 1.7)
                atom_list.append((coords, charge, radius))
        
        # Calculate Born radii
        born_radii = self._calculate_born_radii(atom_list)
        
        # Calculate polar solvation energy (electrostatic)
        polar_energy = self._calculate_polar_energy(atom_list, born_radii)
        
        # Calculate nonpolar solvation energy (cavity formation)
        nonpolar_energy = self._calculate_nonpolar_energy(atom_list)
        
        # Total solvation energy
        total_energy = polar_energy + nonpolar_energy
        
        return (polar_energy, nonpolar_energy, total_energy)
    
    def calculate_binding_solvation(self, protein, ligand):
        """
        Calculate solvation contribution to binding free energy.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        ligand : Ligand
            Ligand object
        
        Returns:
        --------
        float
            Solvation contribution to binding in kcal/mol
        """
        # Get active site atoms if defined
        if protein.active_site and 'atoms' in protein.active_site:
            protein_atoms = protein.active_site['atoms']
            p_obj = type('obj', (object,), {'atoms': protein_atoms})
        else:
            p_obj = protein
        
        # Calculate solvation for ligand alone
        ligand_polar, ligand_nonpolar, ligand_total = self.calculate_solvation_free_energy(
            ligand, 'ligand')
        
        # Calculate solvation for protein alone
        protein_polar, protein_nonpolar, protein_total = self.calculate_solvation_free_energy(
            p_obj, 'protein')
        
        # Create a combined molecule for the complex
        complex_atoms = []
        
        # Add protein atoms
        for atom in protein_atoms:
            symbol = atom.get('element', atom.get('name', 'C'))[0]
            coords = atom['coords']
            charge = self.atom_charges.get(symbol, 0.0)
            radius = self.atom_radii.get(symbol, 1.7)
            complex_atoms.append((coords, charge, radius))
        
        # Add ligand atoms
        for atom in ligand.atoms:
            symbol = atom.get('symbol', 'C')
            coords = atom['coords']
            charge = self.atom_charges.get(symbol, 0.0)
            radius = self.atom_radii.get(symbol, 1.7)
            complex_atoms.append((coords, charge, radius))
        
        # Calculate Born radii for complex
        complex_born_radii = self._calculate_born_radii(complex_atoms)
        
        # Calculate polar solvation energy for complex
        complex_polar = self._calculate_polar_energy(complex_atoms, complex_born_radii)
        
        # Calculate nonpolar solvation energy for complex
        complex_nonpolar = self._calculate_nonpolar_energy(complex_atoms)
        
        complex_total = complex_polar + complex_nonpolar
        
        # Solvation contribution to binding = ΔG_solv(complex) - ΔG_solv(protein) - ΔG_solv(ligand)
        solvation_contribution = complex_total - protein_total - ligand_total
        
        return solvation_contribution
    
    def _calculate_born_radii(self, atom_list):
        """
        Calculate effective Born radii for atoms.
        
        Parameters:
        -----------
        atom_list : list
            List of (coords, charge, radius) tuples
        
        Returns:
        --------
        list
            List of Born radii for each atom
        """
        n_atoms = len(atom_list)
        born_radii = np.zeros(n_atoms)
        
        # Extract coordinates and radii
        coords = np.array([atom[0] for atom in atom_list])
        radii = np.array([atom[2] for atom in atom_list])
        
        # Calculate self-volumes and initialize Born radii
        for i in range(n_atoms):
            # Initial Born radius is atom radius
            born_radii[i] = 1.0 / radii[i]
        
        # Adjust Born radii based on atom overlaps (simplified)
        for i in range(n_atoms):
            r_i = radii[i]
            c_i = coords[i]
            
            for j in range(n_atoms):
                if i == j:
                    continue
                
                r_j = radii[j]
                c_j = coords[j]
                
                # Calculate distance
                d_ij = np.linalg.norm(c_i - c_j)
                
                # Skip pairs that are too far apart
                if d_ij > r_i + r_j + 5.0:
                    continue
                
                # Calculate contribution to Born radius
                if d_ij < 0.1:  # Avoid numerical issues
                    continue
                
                # Simplified Still formula
                born_term = r_j / (d_ij * d_ij) * np.exp(-d_ij * d_ij / (4.0 * r_i * r_j))
                born_radii[i] += born_term
        
        # Convert summed terms to actual Born radii
        for i in range(n_atoms):
            if born_radii[i] > 0:
                born_radii[i] = 1.0 / (born_radii[i] * self.scale_factor)
            else:
                born_radii[i] = radii[i]  # Fallback to atom radius
        
        return born_radii
    
    def _calculate_polar_energy(self, atom_list, born_radii):
        """
        Calculate polar solvation energy using GB model.
        
        Parameters:
        -----------
        atom_list : list
            List of (coords, charge, radius) tuples
        born_radii : list
            List of Born radii for each atom
        
        Returns:
        --------
        float
            Polar solvation energy in kcal/mol
        """
        n_atoms = len(atom_list)
        polar_energy = 0.0
        
        # Extract charges and coordinates
        coords = np.array([atom[0] for atom in atom_list])
        charges = np.array([atom[1] for atom in atom_list])
        
        # Calculate energy
        for i in range(n_atoms):
            q_i = charges[i]
            r_i = born_radii[i]
            c_i = coords[i]
            
            # Self-energy term
            self_energy = -166.0 * (q_i * q_i) / (2.0 * r_i) * (1.0 - (1.0 / self.solvent_dielectric))
            polar_energy += self_energy
            
            # Cross-terms
            for j in range(i+1, n_atoms):
                q_j = charges[j]
                r_j = born_radii[j]
                c_j = coords[j]
                
                d_ij = np.linalg.norm(c_i - c_j)
                
                if d_ij < 0.1:  # Avoid numerical issues
                    continue
                
                # "f_GB" term for GB equation
                f_gb = np.sqrt(d_ij * d_ij + r_i * r_j * np.exp(-d_ij * d_ij / (4.0 * r_i * r_j)))
                
                # Cross-term energy
                cross_energy = -166.0 * (q_i * q_j) / f_gb * (1.0 - (1.0 / self.solvent_dielectric))
                polar_energy += cross_energy
        
        return polar_energy
    
    def _calculate_nonpolar_energy(self, atom_list):
        """
        Calculate nonpolar solvation energy (cavity formation and van der Waals).
        
        Parameters:
        -----------
        atom_list : list
            List of (coords, charge, radius) tuples
        
        Returns:
        --------
        float
            Nonpolar solvation energy in kcal/mol
        """
        # Extract coordinates and radii
        coords = np.array([atom[0] for atom in atom_list])
        radii = np.array([atom[2] for atom in atom_list])
        
        # Calculate approximate surface area for each atom
        n_atoms = len(atom_list)
        sasa = np.ones(n_atoms)  # Start with fully exposed
        
        # Calculate all pairwise distances
        dist_matrix = np.zeros((n_atoms, n_atoms))
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                dist = np.linalg.norm(coords[i] - coords[j])
                dist_matrix[i,j] = dist
                dist_matrix[j,i] = dist
        
        # Add probe radius to atomic radii
        radii_with_probe = radii + 1.4  # Water probe radius
        
        # For each atom, check how many neighbors are within van der Waals contact
        for i in range(n_atoms):
            r_i = radii_with_probe[i]
            
            # Count neighbors that overlap
            for j in range(n_atoms):
                if i == j:
                    continue
                    
                r_j = radii[j]  # No probe for neighbors
                dist = dist_matrix[i,j]
                
                # Check if atoms overlap
                if dist < (r_i + r_j):
                    # Estimate overlap
                    overlap = 1.0 - (dist / (r_i + r_j))
                    # Reduce SASA proportionally to overlap
                    sasa[i] -= overlap * 0.2  # Reduced scale factor
        
        # Ensure SASA is non-negative
        sasa = np.maximum(0.0, sasa)
        
        # Calculate surface area (4πr²) and apply exposure factor
        atom_areas = 4.0 * np.pi * radii_with_probe * radii_with_probe * sasa
        
        # Calculate nonpolar energy using surface area model
        nonpolar_energy = self.surface_tension * np.sum(atom_areas)
        
        return nonpolar_energy


class MonteCarloSampling:
    """
    Enhanced sampling using Monte Carlo with Metropolis criterion.
    This provides better exploration of conformational space.
    """
    
    def __init__(self, scoring_function, temperature=300.0, n_steps=1000, 
                 max_translation=2.0, max_rotation=0.3, cooling_factor=0.95):
        """
        Initialize Monte Carlo sampling.
        
        Parameters:
        -----------
        scoring_function : ScoringFunction
            Scoring function to evaluate poses
        temperature : float
            Simulation temperature in Kelvin
        n_steps : int
            Number of Monte Carlo steps
        max_translation : float
            Maximum translation step size in Angstroms
        max_rotation : float
            Maximum rotation step size in radians
        cooling_factor : float
            Temperature cooling factor for simulated annealing (< 1.0)
        """
        self.scoring_function = scoring_function
        self.temperature = temperature
        self.n_steps = n_steps
        self.max_translation = max_translation
        self.max_rotation = max_rotation
        self.cooling_factor = cooling_factor
        
        # Gas constant in kcal/(mol·K)
        self.gas_constant = 1.9872e-3
        
        # Set up simulated annealing schedule if cooling factor < 1.0
        if cooling_factor < 1.0:
            self.use_annealing = True
        else:
            self.use_annealing = False
    
    def run_sampling(self, protein, ligand, start_pose=None):
        """
        Run Monte Carlo sampling to explore ligand conformational space.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        ligand : Ligand
            Ligand object (used as starting pose if start_pose is None)
        start_pose : Ligand, optional
            Starting ligand pose
        
        Returns:
        --------
        list
            List of (pose, score) tuples, sorted by score (best first)
        """
        import copy
        from scipy.spatial.transform import Rotation
        import numpy as np
        
        # Use provided starting pose or ligand
        if start_pose is None:
            current_pose = copy.deepcopy(ligand)
        else:
            current_pose = copy.deepcopy(start_pose)
        
        # Evaluate initial pose
        current_score = self.scoring_function.score(protein, current_pose)
        
        # Initialize tracking for acceptance ratio
        accepted = 0
        
        # Initialize best pose and collection of poses
        best_pose = copy.deepcopy(current_pose)
        best_score = current_score
        collected_poses = [(copy.deepcopy(current_pose), current_score)]
        
        # Set initial temperature
        temperature = self.temperature
        
        # Print header for progress tracking
        print(f"Starting Monte Carlo sampling ({self.n_steps} steps)")
        print(f"Initial score: {current_score:.2f}")
        
        # Main sampling loop
        for step in range(self.n_steps):
            # Create a candidate pose
            candidate_pose = copy.deepcopy(current_pose)
            
            # Apply random translation
            translation = np.random.uniform(-self.max_translation, self.max_translation, 3)
            candidate_pose.translate(translation)
            
            # Apply random rotation
            axis = np.random.randn(3)
            axis = axis / np.linalg.norm(axis)
            angle = np.random.uniform(-self.max_rotation, self.max_rotation)
            rotation = Rotation.from_rotvec(axis * angle)
            
            # Apply rotation around center of mass
            centroid = np.mean(candidate_pose.xyz, axis=0)
            candidate_pose.translate(-centroid)
            candidate_pose.rotate(rotation.as_matrix())
            candidate_pose.translate(centroid)
            
            # Evaluate candidate pose
            candidate_score = self.scoring_function.score(protein, candidate_pose)
            
            # Calculate energy difference (negative score = better)
            delta_score = candidate_score - current_score
            
            # Metropolis criterion
            accept = False
            if delta_score <= 0:  # Better score, always accept
                accept = True
            else:
                # Calculate Boltzmann factor
                boltzmann_factor = np.exp(-delta_score / (self.gas_constant * temperature))
                # Accept with probability exp(-ΔE/kT)
                if np.random.random() < boltzmann_factor:
                    accept = True
            
            # Update current pose if accepted
            if accept:
                current_pose = candidate_pose
                current_score = candidate_score
                accepted += 1
                
                # Update best pose if needed
                if current_score < best_score:
                    best_pose = copy.deepcopy(current_pose)
                    best_score = current_score
                
                # Add to collection (limit to 100 poses)
                if len(collected_poses) < 100:
                    collected_poses.append((copy.deepcopy(current_pose), current_score))
            
            # Cool temperature if using simulated annealing
            if self.use_annealing:
                temperature *= self.cooling_factor
            
            # Print progress
            if (step + 1) % (self.n_steps // 10) == 0:
                acceptance_ratio = accepted / (step + 1)
                print(f"Step {step + 1}/{self.n_steps}, "
                      f"Score: {current_score:.2f}, "
                      f"Best: {best_score:.2f}, "
                      f"Acceptance: {acceptance_ratio:.2f}, "
                      f"Temp: {temperature:.1f}K")
        
        # Final stats
        acceptance_ratio = accepted / self.n_steps
        print(f"Sampling completed. Final score: {current_score:.2f}, "
              f"Best score: {best_score:.2f}, "
              f"Acceptance ratio: {acceptance_ratio:.2f}")
        
        # Sort collected poses by score
        collected_poses.sort(key=lambda x: x[1])
        
        return collected_poses


class PhysicsBasedScoring:
    """
    Physics-based scoring function combining molecular mechanics terms.
    This provides a more accurate energy calculation based on physics.
    """
    
    def __init__(self):
        """Initialize physics-based scoring function."""
        # Initialize component models
        self.electrostatics = ImprovedElectrostatics()
        self.solvation = GeneralizedBornSolvation()
        
        # Set up weights for energy components
        self.weights = {
            'vdw': 1.0,
            'elec': 1.0,
            'solv': 1.0,
            'hbond': 1.5,
            'entropy': 0.5
        }
        
        # VDW parameters for Lennard-Jones potential
        self.vdw_radii = {
            'H': 1.2, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8,
            'P': 1.8, 'F': 1.47, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98
        }
        
        self.vdw_well_depth = {
            'H': 0.02, 'C': 0.10, 'N': 0.16, 'O': 0.20, 'S': 0.25,
            'P': 0.20, 'F': 0.08, 'Cl': 0.25, 'Br': 0.32, 'I': 0.40
        }
        
        # Hydrogen bond parameters
        self.hbond_donors = {'N', 'O'}
        self.hbond_acceptors = {'N', 'O', 'F', 'Cl'}
        self.hbond_strength = 5.0  # kcal/mol
        self.hbond_distance = 3.0  # Angstroms
        
        # Rotatable bond parameters for entropy
        self.entropy_per_rot_bond = 0.4  # kcal/mol per rotatable bond
    
    def score(self, protein, ligand):
        """
        Calculate physics-based binding score.
        
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
        # Get active site atoms if defined
        if protein.active_site and 'atoms' in protein.active_site:
            protein_atoms = protein.active_site['atoms']
        else:
            protein_atoms = protein.atoms
        
        # Calculate van der Waals energy
        vdw_energy = self._calc_vdw_energy(protein_atoms, ligand.atoms)
        
        # Calculate electrostatic energy
        elec_energy = self.electrostatics.calculate_electrostatics(protein, ligand)
        
        # Calculate solvation energy
        solv_energy = self.solvation.calculate_binding_solvation(protein, ligand)
        
        # Calculate hydrogen bonding energy
        hbond_energy = self._calc_hbond_energy(protein_atoms, ligand.atoms)
        
        # Calculate configurational entropy penalty
        entropy_penalty = self._calc_entropy_penalty(ligand)
        
        # Combine all energy terms
        total_score = (
            self.weights['vdw'] * vdw_energy +
            self.weights['elec'] * elec_energy +
            self.weights['solv'] * solv_energy +
            self.weights['hbond'] * hbond_energy +
            self.weights['entropy'] * entropy_penalty
        )
        
        return total_score
    
    def _calc_vdw_energy(self, protein_atoms, ligand_atoms):
        """
        Calculate van der Waals energy using Lennard-Jones potential.
        
        Parameters:
        -----------
        protein_atoms : list
            List of protein atom dictionaries
        ligand_atoms : list
            List of ligand atom dictionaries
        
        Returns:
        --------
        float
            VDW energy in kcal/mol
        """
        vdw_energy = 0.0
        
        for p_atom in protein_atoms:
            p_coords = p_atom['coords']
            p_symbol = p_atom.get('element', p_atom.get('name', 'C'))[0]
            p_radius = self.vdw_radii.get(p_symbol, 1.7)
            p_depth = self.vdw_well_depth.get(p_symbol, 0.1)
            
            for l_atom in ligand_atoms:
                l_coords = l_atom['coords']
                l_symbol = l_atom.get('symbol', 'C')
                l_radius = self.vdw_radii.get(l_symbol, 1.7)
                l_depth = self.vdw_well_depth.get(l_symbol, 0.1)
                
                # Calculate distance
                distance = np.linalg.norm(p_coords - l_coords)
                
                # Skip pairs that are too far apart
                if distance > 10.0:
                    continue
                
                # Calculate Lennard-Jones parameters
                sigma = (p_radius + l_radius) * 0.5
                epsilon = np.sqrt(p_depth * l_depth)
                
                # Calculate Lennard-Jones energy
                if distance < 0.1:  # Avoid numerical issues
                    vdw_energy += 1000  # Large repulsive energy
                else:
                    ratio = sigma / distance
                    lj_energy = epsilon * ((ratio**12) - 2.0 * (ratio**6))
                    vdw_energy += lj_energy
        
        return vdw_energy
    
    def _calc_hbond_energy(self, protein_atoms, ligand_atoms):
        """
        Calculate hydrogen bonding energy.
        
        Parameters:
        -----------
        protein_atoms : list
            List of protein atom dictionaries
        ligand_atoms : list
            List of ligand atom dictionaries
        
        Returns:
        --------
        float
            Hydrogen bonding energy in kcal/mol
        """
        hbond_energy = 0.0
        
        # Identify potential donors and acceptors in protein
        p_donors = []
        p_acceptors = []
        
        for p_atom in protein_atoms:
            p_symbol = p_atom.get('element', p_atom.get('name', 'C'))[0]
            if p_symbol in self.hbond_donors:
                p_donors.append(p_atom)
            if p_symbol in self.hbond_acceptors:
                p_acceptors.append(p_atom)
        
        # Identify potential donors and acceptors in ligand
        l_donors = []
        l_acceptors = []
        
        for l_atom in ligand_atoms:
            l_symbol = l_atom.get('symbol', 'C')
            if l_symbol in self.hbond_donors:
                l_donors.append(l_atom)
            if l_symbol in self.hbond_acceptors:
                l_acceptors.append(l_atom)
        
        # Protein donor - Ligand acceptor H-bonds
        for donor in p_donors:
            donor_coords = donor['coords']
            
            for acceptor in l_acceptors:
                acceptor_coords = acceptor['coords']
                
                # Calculate distance
                distance = np.linalg.norm(donor_coords - acceptor_coords)
                
                # Check if within H-bond distance
                if distance <= self.hbond_distance:
                    # Calculate H-bond energy with distance-dependent scaling
                    distance_factor = 1.0 - (distance / self.hbond_distance)
                    energy = -self.hbond_strength * distance_factor**2
                    hbond_energy += energy
        
        # Ligand donor - Protein acceptor H-bonds
        for donor in l_donors:
            donor_coords = donor['coords']
            
            for acceptor in p_acceptors:
                acceptor_coords = acceptor['coords']
                
                # Calculate distance
                distance = np.linalg.norm(donor_coords - acceptor_coords)
                
                # Check if within H-bond distance
                if distance <= self.hbond_distance:
                    # Calculate H-bond energy with distance-dependent scaling
                    distance_factor = 1.0 - (distance / self.hbond_distance)
                    energy = -self.hbond_strength * distance_factor**2
                    hbond_energy += energy
        
        return hbond_energy
    
    def _calc_entropy_penalty(self, ligand):
        """
        Calculate configurational entropy penalty.
        
        Parameters:
        -----------
        ligand : Ligand
            Ligand object
        
        Returns:
        --------
        float
            Entropy penalty in kcal/mol
        """
        # Count rotatable bonds
        if hasattr(ligand, 'rotatable_bonds'):
            n_rotatable = len(ligand.rotatable_bonds)
        else:
            # Estimate from bonds
            n_rotatable = sum(1 for bond in ligand.bonds 
                             if bond.get('is_rotatable', False))
        
        # Calculate entropy penalty
        entropy_penalty = n_rotatable * self.entropy_per_rot_bond
        
        return entropy_penalty