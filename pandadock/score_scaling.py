class ScaledPhysicsBasedScoring:
    """Wrapper for physics-based scoring to produce scaled scores."""
    
    def __init__(self, base_scoring_function, target_program='vina'):
        """Initialize with a base scoring function."""
        self.scoring_function = base_scoring_function
        self.target_program = target_program
        
        # Copy attributes from base scoring function
        for attr in dir(base_scoring_function):
            if not attr.startswith('_') and not hasattr(self, attr):
                setattr(self, attr, getattr(base_scoring_function, attr))
    
    def score(self, protein, ligand):
        """Return scaled score."""
        original_score = self.scoring_function.score(protein, ligand)
        return self.scale_score(original_score)
    
    def get_original_score(self, protein, ligand):
        """Get the original unscaled score."""
        return self.scoring_function.score(protein, ligand)
    
def scale_score(self, original_score):
        """Scale the score to match target program range."""
        # Define scaling parameters based on target program
        if self.target_program.lower() == 'autodock':
            target_min, target_max = -15.0, -5.0
        elif self.target_program.lower() == 'glide':
            target_min, target_max = -12.0, -4.0
        else:  # Default to Vina
            target_min, target_max = -12.0, -2.0
            
        typical_physics_min = -2000.0
        typical_physics_max = -1000.0
        
        # Apply scaling
        return scale_docking_score(
            original_score, 
            target_min, 
            target_max,
            typical_physics_min,
            typical_physics_max
        )    

def scale_docking_score(physics_score, target_min=-12.0, target_max=-2.0, 
                        typical_physics_min=-2000.0, typical_physics_max=-1000.0):
        """Scale physics-based scores to more standard ranges."""
        # Handle scores outside the expected range
        if physics_score < typical_physics_min:
            physics_score = typical_physics_min
        elif physics_score > typical_physics_max:
            physics_score = typical_physics_max
        
        # Normalize the score to 0-1 range
        normalized = (physics_score - typical_physics_min) / (typical_physics_max - typical_physics_min)
        
        # Scale to target range
        scaled_score = target_min + normalized * (target_max - target_min)
        
        return scaled_score

def scale_energy_components(energy_breakdown, target_program='vina'):
        """Scale energy components to match target program range."""
        scaled_breakdown = []
        
        # Define scaling parameters based on target program
        if target_program.lower() == 'autodock':
            target_min, target_max = -15.0, -5.0
        elif target_program.lower() == 'glide':
            target_min, target_max = -12.0, -4.0
        else:  # Default to Vina
            target_min, target_max = -12.0, -2.0
            
        typical_physics_min = -2000.0
        typical_physics_max = -1000.0
        
        for pose_energy in energy_breakdown:
            # Scale the total score
            total_score = pose_energy.get('Total', 0.0)
            scaled_total = scale_docking_score(
                total_score, 
                target_min, 
                target_max,
                typical_physics_min,
                typical_physics_max
            )
            
            # Scale individual components proportionally
            scaled_pose = {}
            scale_factor = scaled_total / total_score if total_score != 0 else 1.0
            
            for component, value in pose_energy.items():
                if component == 'Total':
                    scaled_pose[component] = scaled_total
                else:
                    # Scale proportionally
                    scaled_pose[component] = value * scale_factor
            
            scaled_breakdown.append(scaled_pose)
        
        return scaled_breakdown

def get_component_breakdown(self, protein, ligand):
            """
            Get both raw and scaled energy component breakdown.
            
            Parameters:
            -----------
            protein : Protein
                Protein object
            ligand : Ligand
                Ligand object
            
            Returns:
            --------
            dict
                Dictionary with 'raw' and 'scaled' components
            """
            # Get raw energy components if base class has the method
            raw_components = None
            if hasattr(self.scoring_function, 'get_energy_components'):
                raw_components = self.scoring_function.get_energy_components(protein, ligand)
            elif hasattr(self.scoring_function, '_calculate_vdw_physics'):
                # Reconstruct components from physics-based calculations
                if protein.active_site and 'atoms' in protein.active_site:
                    protein_atoms = protein.active_site['atoms']
                else:
                    protein_atoms = protein.atoms
                
                # Get individual energy terms
                vdw_energy = self.scoring_function._calculate_vdw_physics(protein_atoms, ligand.atoms)
                hbond_energy = self.scoring_function._calculate_hbond_physics(protein_atoms, ligand.atoms, protein, ligand)
                elec_energy = self.scoring_function._calculate_electrostatics_physics(protein_atoms, ligand.atoms)
                desolv_energy = self.scoring_function._calculate_desolvation_physics(protein_atoms, ligand.atoms)
                entropy_energy = self.scoring_function._calculate_entropy(ligand)
                
                # Create component dictionary
                raw_components = {
                    'Van der Waals': vdw_energy,
                    'H-Bond': hbond_energy,
                    'Electrostatic': elec_energy,
                    'Desolvation': desolv_energy,
                    'Entropy': entropy_energy,
                    'Total': self.get_original_score(protein, ligand)
                }
            else:
                # Fallback to a simple breakdown
                raw_score = self.get_original_score(protein, ligand)
                raw_components = {
                    'Van der Waals': raw_score * 0.3,
                    'H-Bond': raw_score * 0.2,
                    'Electrostatic': raw_score * 0.15,
                    'Desolvation': raw_score * 0.25,
                    'Entropy': raw_score * 0.1,
                    'Total': raw_score
                }
            
            # Create scaled components
            scaled_components = {}
            raw_total = raw_components['Total']
            scaled_total = self.scale_score(raw_total)
            
            # Calculate scaling factor
            scale_factor = scaled_total / raw_total if raw_total != 0 else 1.0
            
            # Scale each component
            for component, value in raw_components.items():
                if component == 'Total':
                    scaled_components[component] = scaled_total
                else:
                    scaled_components[component] = value * scale_factor
            
            # Return both raw and scaled
            return {
                'raw': raw_components,
                'scaled': scaled_components
            }