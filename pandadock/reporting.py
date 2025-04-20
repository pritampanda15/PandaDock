# reporting.py
"""
Comprehensive reporting module for PandaDock.
This module provides detailed reporting functionality for molecular docking results,
including energy breakdowns, RMSD calculations, and visualization capabilities.
"""

import os
import json
import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import pandas as pd
from tabulate import tabulate
import csv
from collections import defaultdict
from typing import List, Dict, Any, Tuple

class EnhancedJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy types and other special types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (Path, datetime)):
            return str(obj)
        elif isinstance(obj, bool):
            return bool(obj)
        return super().default(obj)
    
class DockingReporter:
    """
    Enhanced docking report generator for PandaDock.
    Creates comprehensive reports with detailed energy breakdowns and interaction analysis.
    """
    
    def __init__(self, output_dir, args, timestamp=None):
        """
        Initialize the docking reporter.
        
        Parameters:
        -----------
        output_dir : str
            Directory where reports will be saved
        args : argparse.Namespace
            Command-line arguments
        timestamp : str, optional
            Timestamp for the report
        """
        self.output_dir = output_dir
        self.args = args
        
        # Create timestamp if not provided
        import datetime
        self.timestamp = timestamp or datetime.datetime.now().strftime("%Y%m%d_%H-%M")
        
        # Store results
        self.results = None
        self.energy_breakdown = None
        self.validation_results = None
        self.interaction_analysis = None
        self.scoring_breakdown = None
        
    def add_results(self, results, energy_breakdown=None):
        """
        Add docking results to the reporter.
        
        Parameters:
        -----------
        results : list
            List of (pose, score) tuples
        energy_breakdown : dict, optional
            Dictionary with energy component breakdowns
        """
        self.results = results
        self.energy_breakdown = energy_breakdown
    
    def add_validation_results(self, validation_results):
        """
        Add validation results to the reporter.
        
        Parameters:
        -----------
        validation_results : dict
            Validation results dictionary
        """
        self.validation_results = validation_results
    
    def add_interaction_analysis(self, interaction_analysis):
        """
        Add interaction analysis to the reporter.
        
        Parameters:
        -----------
        interaction_analysis : dict
            Analysis of protein-ligand interactions
        """
        self.interaction_analysis = interaction_analysis
    
    def extract_energy_components(self, scoring_function, protein, poses, max_poses=10):
        """
        Extract energy components for top poses.
        
        Parameters:
        -----------
        scoring_function : ScoringFunction
            Scoring function object
        protein : Protein
            Protein object
        poses : list
            List of poses
        max_poses : int
            Maximum number of poses to analyze
                
        Returns:
        --------
        dict
            Dictionary with energy component breakdowns
        """
        energy_breakdown = {}
        # Check if the scoring function has component methods
        has_components = hasattr(scoring_function, 'weights')
        
        if not has_components:
            print("Scoring function does not provide component breakdown")
            return energy_breakdown
        print("Extracting energy component breakdown...")
        
        # Process up to max_poses
        for i, pose in enumerate(poses[:min(len(poses), max_poses)]):
            pose_id = f"pose_{i+1}"
            print(f"  Analyzing pose {i+1}...")
            
            # Score the pose
            try:
                # First, check if the scoring function has component tracking
                if hasattr(scoring_function, 'weights'):
                    # Create a clean dictionary to store components
                    components = {
                        'vdw': 0.0,
                        'hbond': 0.0,
                        'elec': 0.0,
                        'desolv': 0.0,
                        'hydrophobic': 0.0,
                        'clash': 0.0,
                        'entropy': 0.0
                    }
                    
                    # Get active site atoms
                    if protein.active_site and 'atoms' in protein.active_site:
                        protein_atoms = protein.active_site['atoms']
                    else:
                        protein_atoms = protein.atoms
                    
                    # Score the pose to get the total
                    total_score = scoring_function.score(protein, pose)
                    
                    # Try to extract components directly from the scoring function's methods
                    try:
                        # Call individual component calculation methods if they exist
                        if hasattr(scoring_function, '_calculate_vdw_energy'):
                            components['vdw'] = scoring_function._calculate_vdw_energy(protein, pose)
                        elif hasattr(scoring_function, '_calculate_vdw_physics'):
                            components['vdw'] = scoring_function._calculate_vdw_physics(protein_atoms, pose.atoms)
                            
                        if hasattr(scoring_function, '_calculate_hbond_energy'):
                            components['hbond'] = scoring_function._calculate_hbond_energy(protein, pose)
                        elif hasattr(scoring_function, '_calculate_hbond_physics'):
                            components['hbond'] = scoring_function._calculate_hbond_physics(protein_atoms, pose.atoms, protein, pose)
                            
                        if hasattr(scoring_function, '_calculate_electrostatics_energy'):
                            components['elec'] = scoring_function._calculate_electrostatics_energy(protein, pose)
                        elif hasattr(scoring_function, '_calculate_electrostatics_physics'):
                            components['elec'] = scoring_function._calculate_electrostatics_physics(protein_atoms, pose.atoms)
                            
                        if hasattr(scoring_function, '_calculate_desolvation_energy'):
                            components['desolv'] = scoring_function._calculate_desolvation_energy(protein, pose)
                        elif hasattr(scoring_function, '_calculate_desolvation_physics'):
                            components['desolv'] = scoring_function._calculate_desolvation_physics(protein_atoms, pose.atoms)
                            
                        if hasattr(scoring_function, '_calculate_hydrophobic_energy'):
                            components['hydrophobic'] = scoring_function._calculate_hydrophobic_energy(protein, pose)
                        elif hasattr(scoring_function, '_calculate_hydrophobic_physics'):
                            components['hydrophobic'] = scoring_function._calculate_hydrophobic_physics(protein_atoms, pose.atoms)
                            
                        if hasattr(scoring_function, '_calculate_clash'):
                            components['clash'] = scoring_function._calculate_clash(protein, pose)
                        elif hasattr(scoring_function, '_calculate_clash_physics'):
                            components['clash'] = scoring_function._calculate_clash_physics(protein_atoms, pose.atoms)
                            
                        if hasattr(scoring_function, '_calculate_entropy'):
                            components['entropy'] = scoring_function._calculate_entropy(pose)
                        elif hasattr(scoring_function, '_calc_entropy_penalty'):
                            components['entropy'] = scoring_function._calc_entropy_penalty(protein_atoms, pose.atoms)
                            
                    except Exception as e:
                        print(f"    Warning: Could not extract some energy components: {e}")
                    
                    # Add the total score
                    components['total'] = total_score
                    
                    # Check if we actually got non-zero components
                    if all(v == 0.0 for k, v in components.items() if k != 'total'):
                        print("    Warning: All energy components are zero. Estimating components from weights...")
                        
                        # If all components are zero, try to estimate based on total score and weights
                        # This is a rough approximation assuming equal contribution from components
                        weight_sum = sum(scoring_function.weights.values())
                        for component in components:
                            if component != 'total' and component in scoring_function.weights:
                                weight_ratio = scoring_function.weights[component] / weight_sum
                                components[component] = total_score * weight_ratio
                    
                    energy_breakdown[pose_id] = components
                    
                else:
                    # No component tracking in scoring function
                    total_score = scoring_function.score(protein, pose)
                    energy_breakdown[pose_id] = {'total': total_score}
                    print("    Warning: Scoring function does not support component breakdown")
                    
            except Exception as e:
                print(f"    Error scoring pose {i+1}: {e}")
                energy_breakdown[pose_id] = {'total': 0.0}
        
        return energy_breakdown
    
    def analyze_protein_ligand_interactions(self, protein, poses, max_poses=5):
        """
        Analyze protein-ligand interactions for top poses.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        poses : list
            List of poses
        max_poses : int
            Maximum number of poses to analyze
            
        Returns:
        --------
        dict
            Dictionary with interaction analysis
        """
        interaction_analysis = {}
        
        # Process up to max_poses
        for i, pose in enumerate(poses[:min(len(poses), max_poses)]):
            pose_id = f"pose_{i+1}"
            
            # Initialize interaction types
            interactions = {
                'h_bonds': [],
                'hydrophobic': [],
                'ionic': [],
                'pi_stacking': [],
                'residues': set()
            }
            
            # Analyze hydrogen bonds
            self._analyze_hbonds(protein, pose, interactions)
            
            # Analyze hydrophobic interactions
            self._analyze_hydrophobic(protein, pose, interactions)
            
            # Analyze ionic interactions
            self._analyze_ionic(protein, pose, interactions)
            
            # Store results
            interaction_analysis[pose_id] = interactions
        
        self.interaction_analysis = interaction_analysis
        return interaction_analysis
    
    def _analyze_hbonds(self, protein, ligand, interactions):
        """Analyze hydrogen bonds between protein and ligand."""
        import numpy as np

        # Get active site atoms
        if protein.active_site and 'atoms' in protein.active_site:
            protein_atoms = protein.active_site['atoms']
        else:
            protein_atoms = protein.atoms

        # Define donors and acceptors
        donors = ['N', 'O']  # Extend this list as needed
        acceptors = ['N', 'O', 'F']
        cutoff_distance = 3.8  # Maximum H-bond distance (in Å)
        angle_cutoff = 30.0  # Maximum deviation from linearity (in degrees)

        for p_atom in protein_atoms:
            if 'coords' not in p_atom:
                continue

            p_coords = np.array(p_atom['coords'])
            p_element = p_atom.get('element', p_atom.get('name', '')[0])

            if not p_element or p_element not in donors + acceptors:
                continue

            # Get residue info
            res_name = p_atom.get('residue_name', 'UNK')
            res_id = p_atom.get('residue_id', 0)
            chain_id = p_atom.get('chain_id', 'X')

            for l_atom in ligand.atoms:
                if 'coords' not in l_atom:
                    continue

                l_coords = np.array(l_atom['coords'])
                l_element = l_atom.get('symbol', '')

                if not l_element or l_element not in donors + acceptors:
                    continue

                # Calculate distance
                distance = np.linalg.norm(p_coords - l_coords)

                # Check if distance is within cutoff
                if distance <= cutoff_distance:
                    # Check for linearity (angle between donor-H-acceptor)
                    # Simplified: Assume H is along the donor-acceptor vector
                    if p_element in donors and l_element in acceptors:
                        interactions['h_bonds'].append({
                            'type': 'protein_donor',
                            'distance': round(distance, 2),
                            'protein_atom': {
                                'element': p_element,
                                'residue': res_name,
                                'residue_id': res_id,
                                'chain': chain_id
                            },
                            'ligand_atom': {
                                'element': l_element,
                                'index': l_atom.get('idx', 0)
                            }
                        })
                        interactions['residues'].add(f"{chain_id}:{res_name}{res_id}")

                    if l_element in donors and p_element in acceptors:
                        interactions['h_bonds'].append({
                            'type': 'ligand_donor',
                            'distance': round(distance, 2),
                            'protein_atom': {
                                'element': p_element,
                                'residue': res_name,
                                'residue_id': res_id,
                                'chain': chain_id
                            },
                            'ligand_atom': {
                                'element': l_element,
                                'index': l_atom.get('idx', 0)
                            }
                        })
                        interactions['residues'].add(f"{chain_id}:{res_name}{res_id}")
    
    def _analyze_hydrophobic(self, protein, ligand, interactions):
        """Analyze hydrophobic interactions between protein and ligand."""
        # Get active site atoms
        if protein.active_site and 'atoms' in protein.active_site:
            protein_atoms = protein.active_site['atoms']
        else:
            protein_atoms = protein.atoms
        
        # Define hydrophobic atom types
        hydrophobic_types = ['C', 'S', 'Cl', 'Br', 'I']
        cutoff_distance = 4.0  # Maximum hydrophobic interaction distance
        
        # Identify hydrophobic atoms in protein
        p_hydrophobic = []
        for atom in protein_atoms:
            element = None
            if 'element' in atom:
                element = atom['element']
            elif 'name' in atom and len(atom['name']) > 0:
                element = atom['name'][0]
                
            if element in hydrophobic_types:
                p_hydrophobic.append(atom)
        
        # Identify hydrophobic atoms in ligand
        l_hydrophobic = [atom for atom in ligand.atoms 
                        if atom.get('symbol', '') in hydrophobic_types]
        
        import numpy as np
        
        for p_atom in p_hydrophobic:
            if 'coords' not in p_atom:
                continue
                
            p_coords = p_atom['coords']
            
            # Get residue info
            res_name = p_atom.get('residue_name', 'UNK')
            res_id = p_atom.get('residue_id', 0)
            chain_id = p_atom.get('chain_id', 'X')
            
            for l_atom in l_hydrophobic:
                if 'coords' not in l_atom:
                    continue
                    
                l_coords = l_atom['coords']
                
                # Calculate distance
                distance = np.linalg.norm(p_coords - l_coords)
                
                # Check for hydrophobic contact
                if distance <= cutoff_distance:
                    interactions['hydrophobic'].append({
                        'distance': round(distance, 2),
                        'protein_atom': {
                            'element': p_atom.get('element', p_atom.get('name', 'C')[0]),
                            'residue': res_name,
                            'residue_id': res_id,
                            'chain': chain_id
                        },
                        'ligand_atom': {
                            'element': l_atom.get('symbol', 'C'),
                            'index': l_atom.get('idx', 0)
                        }
                    })
                    interactions['residues'].add(f"{chain_id}:{res_name}{res_id}")
    
    def _analyze_ionic(self, protein, ligand, interactions):
        """Analyze ionic interactions between protein and ligand."""
        # Get active site atoms
        if protein.active_site and 'atoms' in protein.active_site:
            protein_atoms = protein.active_site['atoms']
        else:
            protein_atoms = protein.atoms
        
        # Define charged groups
        pos_charged = ['LYS', 'ARG', 'HIS']  # Positive residues
        neg_charged = ['ASP', 'GLU']         # Negative residues
        cutoff_distance = 4.0  # Maximum ionic interaction distance
        
        import numpy as np
        
        # Check for charged residues in protein
        for p_atom in protein_atoms:
            if 'coords' not in p_atom:
                continue
                
            p_coords = p_atom['coords']
            
            # Get residue info
            res_name = p_atom.get('residue_name', 'UNK')
            res_id = p_atom.get('residue_id', 0)
            chain_id = p_atom.get('chain_id', 'X')
            
            # Skip non-charged residues
            if res_name not in pos_charged and res_name not in neg_charged:
                continue
                
            # Determine protein charge
            protein_charge = 'positive' if res_name in pos_charged else 'negative'
            
            # Specific atoms to check based on residue
            check_atom = False
            if (res_name == 'LYS' and p_atom.get('name') == 'NZ') or \
               (res_name == 'ARG' and p_atom.get('name') in ['NH1', 'NH2']) or \
               (res_name == 'HIS' and p_atom.get('name') in ['ND1', 'NE2']) or \
               (res_name == 'ASP' and p_atom.get('name') in ['OD1', 'OD2']) or \
               (res_name == 'GLU' and p_atom.get('name') in ['OE1', 'OE2']):
                check_atom = True
            
            if not check_atom:
                continue
                
            # Check against all ligand atoms (simplified check)
            for l_atom in ligand.atoms:
                if 'coords' not in l_atom:
                    continue
                    
                l_coords = l_atom['coords']
                l_element = l_atom.get('symbol', '')
                
                # Simplified ligand charge check
                ligand_charge = None
                if l_element == 'N':  # Potential positive center
                    ligand_charge = 'positive'
                elif l_element == 'O':  # Potential negative center
                    ligand_charge = 'negative'
                
                # Skip if no charge or same charge
                if not ligand_charge or ligand_charge == protein_charge:
                    continue
                
                # Calculate distance
                distance = np.linalg.norm(p_coords - l_coords)
                
                # Check for ionic interaction
                if distance <= cutoff_distance:
                    interactions['ionic'].append({
                        'distance': round(distance, 2),
                        'protein': {
                            'charge': protein_charge,
                            'residue': res_name,
                            'residue_id': res_id,
                            'chain': chain_id,
                            'atom': p_atom.get('name', '')
                        },
                        'ligand': {
                            'charge': ligand_charge,
                            'atom': l_element,
                            'index': l_atom.get('idx', 0)
                        }
                    })
                    interactions['residues'].add(f"{chain_id}:{res_name}{res_id}")
    
    def generate_detailed_report(self):
        """
        Generate a detailed text report of docking results.
        
        Returns:
        --------
        str
            Path to the generated report
        """
        import os
        import datetime
        
        # Create report path
        report_name = getattr(self.args, 'report_name', None)
        if report_name:
            report_path = os.path.join(self.output_dir, f"{report_name}_detailed_report.txt")
        else:
            report_path = os.path.join(self.output_dir, f"detailed_docking_report.txt")
        
        # Check if we have results
        if not self.results:
            with open(report_path, 'w') as f:
                f.write("===========================================================\n")
                f.write("        PandaDock - Detailed Molecular Docking Report       \n")
                f.write("===========================================================\n\n")
                f.write("No docking results available.\n")
            return report_path
        
        # Extract arguments
        protein_file = getattr(self.args, 'protein', 'Unknown')
        ligand_file = getattr(self.args, 'ligand', 'Unknown')
        algorithm = getattr(self.args, 'algorithm', 'Unknown')
        
        # Determine scoring function type
        scoring_type = "Standard"
        if hasattr(self.args, 'enhanced_scoring') and self.args.enhanced_scoring:
            scoring_type = "Enhanced"
        if hasattr(self.args, 'physics_based') and self.args.physics_based:
            scoring_type = "Physics-based"
        
        # Get hardware info
        gpu_used = hasattr(self.args, 'use_gpu') and self.args.use_gpu
        cpu_cores = getattr(self.args, 'cpu_workers', None)
        
        hardware = "CPU"
        if gpu_used:
            hardware = "GPU"
        hardware += f" ({cpu_cores} cores)" if cpu_cores else " (Unknown cores)"
        
        # Generate scores statistics
        scores = [score for _, score in self.results]
        avg_score = sum(scores) / len(scores) if scores else 0
        std_dev = self._calculate_std_dev(scores) if len(scores) > 1 else 0
        
        with open(report_path, 'w') as f:
            f.write("===========================================================\n")
            f.write("        PandaDock - Detailed Molecular Docking Report       \n")
            f.write("===========================================================\n\n")
            
            # Run information
            f.write("RUN INFORMATION\n")
            f.write("--------------\n")
            f.write(f"Date and Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Report ID: {self.timestamp}\n")
            f.write(f"Protein: {os.path.basename(protein_file)}\n")
            f.write(f"Ligand: {os.path.basename(ligand_file)}\n\n")
            
            # Algorithm details
            f.write("ALGORITHM DETAILS\n")
            f.write("-----------------\n")
            f.write(f"Algorithm: {algorithm.capitalize()}\n")
            
            if algorithm.lower() == 'genetic':
                pop_size = getattr(self.args, 'population_size', 'Unknown')
                f.write(f"Population Size: {pop_size}\n")
                
            f.write(f"Scoring Function: {scoring_type}\n")
            f.write(f"Hardware Acceleration: {hardware}\n")
            
            # Add local optimization info if available
            if hasattr(self.args, 'local_opt'):
                f.write(f"Local Optimization: {'Enabled' if self.args.local_opt else 'Disabled'}\n")
                
            # Add exhaustiveness info if available
            exhaust = getattr(self.args, 'exhaustiveness', 1)
            f.write(f"Exhaustiveness: {exhaust}\n\n")
            
            # Results summary
            f.write("RESULTS SUMMARY\n")
            f.write("--------------\n")
            f.write(f"Total Poses Generated: {len(self.results)}\n")
            
            if scores:
                f.write(f"Best Score: {min(scores):.4f}\n")
                f.write(f"Worst Score: {max(scores):.4f}\n")
                f.write(f"Average Score: {avg_score:.4f}\n")
                f.write(f"Score Standard Deviation: {std_dev:.4f}\n\n")
            
            # Top poses ranking
            f.write("TOP POSES RANKING\n")
            f.write("----------------\n")
            f.write("Rank  Score      Filename\n")
            f.write("----- ---------- ----------------------------------------\n")
            
            sorted_results = sorted(self.results, key=lambda x: x[1])
            for i, (_, score) in enumerate(sorted_results[:min(10, len(sorted_results))]):
                f.write(f"{i+1:5d} {score:10.4f} pose_{i+1}_score_{score:.1f}.pdb\n")
            
            f.write("\n")
            
            # Energy breakdown
            if self.energy_breakdown:
                f.write("ENERGY COMPONENT BREAKDOWN\n")
                f.write("-------------------------\n")
                f.write("Component       Pose 1     Pose 2     Pose 3     Pose 4     Pose 5\n")
                f.write("-------------- ---------- ---------- ---------- ---------- ----------\n")
                
                components = ['vdw', 'hbond', 'elec', 'desolv', 'hydrophobic', 'clash', 'entropy', 'total']
                component_names = {
                    'vdw': 'Van der Waals',
                    'hbond': 'H-Bond',
                    'elec': 'Electrostatic',
                    'desolv': 'Desolvation',
                    'hydrophobic': 'Hydrophobic',
                    'clash': 'Steric Clash',
                    'entropy': 'Entropy',
                    'total': 'TOTAL'
                }
                
                for component in components:
                    name = component_names.get(component, component)
                    f.write(f"{name:14s} ")
                    
                    for i in range(1, 6):
                        pose_id = f"pose_{i}"
                        if pose_id in self.energy_breakdown and component in self.energy_breakdown[pose_id]:
                            value = self.energy_breakdown[pose_id][component]
                            f.write(f"{value:10.4f} ")
                        else:
                            f.write(f"{'N/A':10s} ")
                    
                    f.write("\n")
                
                f.write("\n")
            
            # Interaction analysis
            if self.interaction_analysis:
                f.write("PROTEIN-LIGAND INTERACTIONS\n")
                f.write("--------------------------\n")
                
                for i in range(1, min(4, len(self.results) + 1)):
                    pose_id = f"pose_{i}"
                    if pose_id in self.interaction_analysis:
                        f.write(f"Pose {i} (Score: {sorted_results[i-1][1]:.4f})\n")
                        interactions = self.interaction_analysis[pose_id]
                        
                        # H-bonds
                        if interactions['h_bonds']:
                            f.write("  Hydrogen Bonds:\n")
                            for hbond in interactions['h_bonds']:
                                if hbond['type'] == 'protein_donor':
                                    f.write(f"    {hbond['protein_atom']['chain']}:{hbond['protein_atom']['residue']}{hbond['protein_atom']['residue_id']} → Ligand {hbond['ligand_atom']['element']} ({hbond['distance']} Å)\n")
                                else:
                                    f.write(f"    Ligand {hbond['ligand_atom']['element']} → {hbond['protein_atom']['chain']}:{hbond['protein_atom']['residue']}{hbond['protein_atom']['residue_id']} ({hbond['distance']} Å)\n")
                        
                        # Hydrophobic
                        if interactions['hydrophobic']:
                            f.write("  Hydrophobic Interactions:\n")
                            seen = set()
                            for hydro in interactions['hydrophobic']:
                                res_key = f"{hydro['protein_atom']['chain']}:{hydro['protein_atom']['residue']}{hydro['protein_atom']['residue_id']}"
                                if res_key not in seen:
                                    seen.add(res_key)
                                    
                                    # Find minimum distance for this residue - calculate separately to avoid nested f-string issues
                                    distances = []
                                    for h in interactions['hydrophobic']:
                                        h_res_key = f"{h['protein_atom']['chain']}:{h['protein_atom']['residue']}{h['protein_atom']['residue_id']}"
                                        if h_res_key == res_key:
                                            distances.append(h['distance'])
                                    
                                    min_distance = min(distances) if distances else 0
                                    
                                    f.write(f"    {res_key} ({min_distance} Å)\n")
                                    
                                    if len(seen) >= 5:  # Show only top 5
                                        f.write(f"    ... and {len(interactions['hydrophobic']) - 5} more\n")
                                        break
                        
                        # Ionic
                        if interactions['ionic']:
                            f.write("  Ionic Interactions:\n")
                            for ionic in interactions['ionic']:
                                f.write(f"    {ionic['protein']['chain']}:{ionic['protein']['residue']}{ionic['protein']['residue_id']} ({ionic['protein']['charge']}) ↔ Ligand {ionic['ligand']['atom']} ({ionic['ligand']['charge']}) - {ionic['distance']} Å\n")
                        
                        # Summary of interacting residues
                        f.write("  Interacting Residues: ")
                        f.write(", ".join(sorted(list(interactions['residues']))))
                        f.write("\n\n")
                
            # Validation results
            if self.validation_results:
                f.write("VALIDATION AGAINST REFERENCE\n")
                f.write("---------------------------\n")
                best_rmsd = self.validation_results[0]['rmsd']
                f.write(f"Best RMSD to Reference: {best_rmsd:.4f} Å (Pose {self.validation_results[0]['pose_index'] + 1})\n")
                f.write(f"Validation Status: {'Success' if best_rmsd < 2.0 else 'Failure'}\n\n")
                
                f.write("RMSD Values for Top 5 Poses:\n")
                for i, result in enumerate(self.validation_results[:min(5, len(self.validation_results))]):
                    pose_idx = result['pose_index'] + 1
                    rmsd = result['rmsd']
                    f.write(f"  Pose {pose_idx}: {rmsd:.4f} Å\n")
                
                f.write("\n")
            
            f.write("===========================================================\n")
            f.write("Report generated by PandaDock - Python Molecular Docking Tool\n")
        
        return report_path
    
    def _calculate_std_dev(self, values):
        """Calculate standard deviation of a list of values."""
        if not values or len(values) < 2:
            return 0.0
            
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def generate_csv_report(self):
        """
        Generate a CSV report of docking results.
        
        Returns:
        --------
        str
            Path to the generated CSV report
        """
        import os
        import csv
        
        # Create report path
        report_name = getattr(self.args, 'report_name', None)
        if report_name:
            report_path = os.path.join(self.output_dir, f"{report_name}_results.csv")
        else:
            report_path = os.path.join(self.output_dir, f"docking_results.csv")
        
        # Check if we have results
        if not self.results:
            return None
        
        # Sort results by score
        sorted_results = sorted(self.results, key=lambda x: x[1])
        
        with open(report_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            header = ['Rank', 'Pose', 'Score']
            
            # Add energy components if available
            if self.energy_breakdown and 'pose_1' in self.energy_breakdown:
                for component in self.energy_breakdown['pose_1']:
                    if component != 'total':
                        header.append(component.capitalize())
            
            # Add validation if available
            if self.validation_results:
                header.append('RMSD')
            
            writer.writerow(header)
            
            # Write data rows
            for i, (pose, score) in enumerate(sorted_results):
                pose_id = f"pose_{i+1}"
                row = [i+1, pose_id, score]
                
                # Add energy components
                if self.energy_breakdown and pose_id in self.energy_breakdown:
                    components = self.energy_breakdown[pose_id]
                    for component, value in components.items():
                        if component != 'total':
                            row.append(value)
                
                # Add validation RMSD if available
                if self.validation_results:
                    # Find matching validation result
                    rmsd = next((r['rmsd'] for r in self.validation_results if r['pose_index'] == i), 'N/A')
                    row.append(rmsd)
                
                writer.writerow(row)
        
        return report_path
    
    def add_results(self, results, energy_breakdown=None):
        self.results = results
        self.energy_breakdown = energy_breakdown
        self.scoring_breakdown = list(energy_breakdown.values()) if energy_breakdown else None


    def generate_plots(self, save_dir=None):
        """
        Generate plots visualizing the docking results.
        
        Parameters:
        -----------
        save_dir : str, optional
            Directory to save plots, defaults to the output directory
        
        Returns:
        --------
        list
            Paths to the generated plot files
        """
        if save_dir is None:
            save_dir = self.output_dir
        else:
            save_dir = Path(save_dir)
            os.makedirs(save_dir, exist_ok=True)
        
        plot_paths = []
        
        # 1. Score distribution plot
        score_plot_path = save_dir / "score_distribution.png"
        plt.figure(figsize=(10, 6))
        
        scores = [score for _, score in self.results]
        plt.hist(scores, bins=20, color='skyblue', edgecolor='black')
        plt.xlabel('Docking Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Docking Scores')
        plt.grid(alpha=0.3)
        plt.savefig(score_plot_path)
        plt.close()
        plot_paths.append(score_plot_path)
        
        # 2. Score rank plot
        rank_plot_path = save_dir / "score_rank.png"
        plt.figure(figsize=(10, 6))
        
        sorted_scores = sorted(scores)
        plt.plot(range(1, len(sorted_scores) + 1), sorted_scores, marker='o', linestyle='-', 
                 color='darkorange', markersize=5)
        plt.xlabel('Rank')
        plt.ylabel('Docking Score')
        plt.title('Docking Scores by Rank')
        plt.grid(alpha=0.3)
        plt.savefig(rank_plot_path)
        plt.close()
        plot_paths.append(rank_plot_path)
        
        # 3. Energy component breakdown plot (for top 10 poses)
        if self.scoring_breakdown:
            energy_plot_path = save_dir / "energy_breakdown.png"
            plt.figure(figsize=(12, 8))
            
            # Get top 10 energy breakdowns
            top_poses = min(10, len(self.scoring_breakdown))
            
            # Get all energy components and filter out 'total'
            components = [key for key in self.scoring_breakdown[0].keys() 
                          if key.lower() != 'total']
            
            # Setup colors for components
            colors = plt.cm.tab10(np.linspace(0, 1, len(components)))
            
            # Setup plot
            bar_width = 0.8 / len(components)
            r = np.arange(top_poses)
            
            # Plot each component
            for i, component in enumerate(components):
                values = [energy[component] for energy in self.scoring_breakdown[:top_poses]]
                plt.bar(r + i * bar_width, values, width=bar_width, color=colors[i], 
                        edgecolor='gray', alpha=0.7, label=component)
            
            # Formatting
            plt.xlabel('Pose Rank')
            plt.ylabel('Energy Contribution')
            plt.title('Energy Component Breakdown for Top Poses')
            plt.xticks(r + bar_width * (len(components) - 1) / 2, [f'{i+1}' for i in range(top_poses)])
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(energy_plot_path)
            plt.close()
            plot_paths.append(energy_plot_path)
        
        # 4. RMSD plot if validation results are available
        if self.validation_results and isinstance(self.validation_results, list):
            rmsd_plot_path = save_dir / "rmsd_plot.png"
            plt.figure(figsize=(10, 6))
            
            # Extract RMSD values
            rmsd_values = [result.get('rmsd', 0.0) for result in self.validation_results]
            
            plt.bar(range(1, len(rmsd_values) + 1), rmsd_values, color='lightgreen', edgecolor='darkgreen')
            plt.axhline(y=2.0, color='red', linestyle='--', label='Success Threshold (2.0 Å)')
            
            plt.xlabel('Pose Rank')
            plt.ylabel('RMSD (Å)')
            plt.title('RMSD Comparison with Reference Structure')
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(rmsd_plot_path)
            plt.close()
            plot_paths.append(rmsd_plot_path)
        
        print(f"Generated {len(plot_paths)} plots in {save_dir}")
        return plot_paths
    
    def generate_json_report(self):
        """
        Generate a JSON report of docking results.
        
        Returns:
        --------
        str
            Path to the generated JSON report
        """
        import os
        import json
        import datetime
        
        # Create report path
        report_name = getattr(self.args, 'report_name', None)
        if report_name:
            report_path = os.path.join(self.output_dir, f"{report_name}_results.json")
        else:
            report_path = os.path.join(self.output_dir, f"docking_results.json")
        
        # Check if we have results
        if not self.results:
            return None
        
        # Extract arguments
        protein_file = getattr(self.args, 'protein', 'Unknown')
        ligand_file = getattr(self.args, 'ligand', 'Unknown')
        algorithm = getattr(self.args, 'algorithm', 'Unknown')
        
        # Create report data structure
        report_data = {
            'run_info': {
                'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'protein': os.path.basename(protein_file),
                'ligand': os.path.basename(ligand_file),
                'algorithm': algorithm
            },
            'results': {
                'poses': [],
                'summary': {
                    'total_poses': len(self.results),
                    'best_score': min([score for _, score in self.results]) if self.results else 0,
                    'worst_score': max([score for _, score in self.results]) if self.results else 0,
                    'average_score': sum([score for _, score in self.results]) / len(self.results) if self.results else 0
                }
            }
        }
        
        # Sort results by score
        sorted_results = sorted(self.results, key=lambda x: x[1])
        
        # Add pose data
        for i, (pose, score) in enumerate(sorted_results):
            pose_id = f"pose_{i+1}"
            pose_data = {
                'rank': i+1,
                'id': pose_id,
                'score': score,
                'filename': f"pose_{i+1}_score_{score:.1f}.pdb"
            }
            
            # Add energy components if available
            if self.energy_breakdown and pose_id in self.energy_breakdown:
                pose_data['energy_components'] = self.energy_breakdown[pose_id]
            
            # Add interactions if available
            if self.interaction_analysis and pose_id in self.interaction_analysis:
                pose_data['interactions'] = self.interaction_analysis[pose_id]
            
            # Add validation if available
            if self.validation_results:
                validation = next((r for r in self.validation_results if r['pose_index'] == i), None)
                if validation:
                    pose_data['validation'] = {
                        'rmsd': validation['rmsd'],
                        'success': validation['success']
                    }
            
            report_data['results']['poses'].append(pose_data)
        
        # Save JSON file
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        return report_path
    
    # Handle the interactions section separately to avoid complex nesting
    

    # Then in your generate_html_report function:
    
    def generate_html_report(self):
        """
        Generate a comprehensive HTML report with embedded plots and tables.

        This function generates an HTML report that includes various sections such 
        as run information, docking results, energy components, interactions, and 
        validation results. It styles the report using CSS for better readability 
        and organizes the content into structured sections. The report is saved 
        to a specified output directory.

        Returns:
        --------
        str
            Path to the generated HTML report or None if no results are available.
        """

        import os
        import datetime  
        

        # Generate and save plots
        plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        plot_paths = self.generate_plots(save_dir=plots_dir)
        plot_html = "<div class='container'><h2>Visualizations</h2>"

        for path in plot_paths:
            rel_path = os.path.relpath(path, self.output_dir)
            plot_html += f"""
                <div class='plot'>
                    <img src="{rel_path}" alt="Plot" style="width:100%; max-width:800px;" />
                </div>"""
        plot_html += "</div>"
        # Create report path
        report_name = getattr(self.args, 'report_name', None)
        if report_name:
            report_path = os.path.join(self.output_dir, f"{report_name}_report.html")
        else:
            report_path = os.path.join(self.output_dir, "docking_report.html")
        
        # Check if we have results
        if not self.results:
            print("No results to generate HTML report")
            return None
        
        # Generate timestamp
        current_timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Extract arguments
        protein_file = getattr(self.args, 'protein', 'Unknown')
        ligand_file = getattr(self.args, 'ligand', 'Unknown')
        
        # Generate basic HTML sections
        run_info_section = self._generate_run_info_html(protein_file, ligand_file)
        results_section = self._generate_results_html()
        energy_section = self._generate_energy_html() if self.energy_breakdown else ""
        
        # Generate interactions section if available
        interactions_section = ""
        if self.interaction_analysis:
            interactions_section = self._generate_interactions_html()
        
        # Generate validation section if available
        validation_section = ""
        if self.validation_results:
            validation_section = self._generate_validation_html()
        
        # Build HTML content with proper indentation
        html_content = f"""<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PandaDock Docking Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f9f9f9;
            }}
            h1, h2, h3, h4 {{
                color: #2c3e50;
            }}
            .header {{
                background-color: #34495e;
                color: white;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            .container {{
                background: white;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .plot {{
                width: 100%;
                max-width: 800px;
                margin: 20px auto;
                text-align: center;
            }}
            .interactions {{
                background-color: #f9f9f9;
                padding: 15px;
                border-radius: 5px;
                margin-top: 10px;
            }}
            .good {{
                color: green;
            }}
            .bad {{
                color: red;
            }}
            .footer {{
                text-align: center;
                margin-top: 30px;
                color: #7f8c8d;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>PandaDock Molecular Docking Report</h1>
            <p>Generated on {current_timestamp}</p>
        </div>
        
        {run_info_section}
        
        {results_section}
        
        {energy_section}
        
        {interactions_section}
        
        {validation_section}
        
        <div class="footer">
            <p>Generated by PandaDock - Python Molecular Docking Tool</p>
        </div>
    </body>
    </html>"""
        html_content = html_content.replace("<div class=\"footer\">", plot_html + "<div class=\"footer\">")
        # Write HTML file
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        print(f"HTML report generated: {report_path}")
        return report_path

    def _generate_run_info_html(self, protein_file, ligand_file):
        import os
        """Generate HTML for the run information section."""
        # Get algorithm info
        algorithm = getattr(self.args, 'algorithm', 'Unknown').capitalize()
        
        # Determine scoring function type
        if hasattr(self.args, 'physics_based') and self.args.physics_based:
            scoring_type = "Physics-based"
        elif hasattr(self.args, 'enhanced_scoring') and self.args.enhanced_scoring:
            scoring_type = "Enhanced"
        else:
            scoring_type = "Standard"
        
        # Get hardware info
        hardware = "CPU"
        if hasattr(self.args, 'use_gpu') and self.args.use_gpu:
            hardware = "GPU"
        
        cpu_cores = getattr(self.args, 'cpu_workers', 'Unknown')
        hardware_info = f"{hardware} ({cpu_cores} cores)" if cpu_cores != 'Unknown' else hardware
        
        # Build section HTML
        html = f"""<div class="container">
            <h2>Run Information</h2>
            <table>
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Protein</td>
                    <td>{os.path.basename(protein_file)}</td>
                </tr>
                <tr>
                    <td>Ligand</td>
                    <td>{os.path.basename(ligand_file)}</td>
                </tr>
                <tr>
                    <td>Algorithm</td>
                    <td>{algorithm}</td>
                </tr>
                <tr>
                    <td>Scoring Function</td>
                    <td>{scoring_type}</td>
                </tr>
                <tr>
                    <td>Hardware</td>
                    <td>{hardware_info}</td>
                </tr>
                <tr>
                    <td>Total Poses</td>
                    <td>{len(self.results)}</td>
                </tr>
            </table>
        </div>"""
        
        return html

    def _generate_results_html(self):
        """Generate HTML for the docking results section."""
        # Calculate statistics
        scores = [score for _, score in self.results]
        best_score = min(scores) if scores else 0
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # Sort results by score
        sorted_results = sorted(self.results, key=lambda x: x[1])
        
        # Build section HTML
        html = f"""<div class="container">
            <h2>Docking Results</h2>
            <p>Best score: <strong>{best_score:.4f}</strong></p>
            <p>Average score: <strong>{avg_score:.4f}</strong></p>
            
            <h3>Top Poses</h3>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Pose</th>
                    <th>Score</th>
                    {'' if not self.validation_results else '<th>RMSD</th>'}
                </tr>"""
        
        # Add rows for top 10 poses
        for i, (_, score) in enumerate(sorted_results[:min(10, len(sorted_results))]):
            # RMSD column if validation results exist
            rmsd_col = ""
            if self.validation_results:
                rmsd = next((r['rmsd'] for r in self.validation_results if r['pose_index'] == i), None)
                if rmsd is not None:
                    rmsd_class = 'good' if rmsd < 2.0 else 'bad'
                    rmsd_col = f'<td class="{rmsd_class}">{rmsd:.2f}</td>'
                else:
                    rmsd_col = '<td>N/A</td>'
            
            html += f"""
                <tr>
                    <td>{i+1}</td>
                    <td>Pose {i+1}</td>
                    <td>{score:.4f}</td>
                    {rmsd_col}
                </tr>"""
        
        html += """
            </table>
        </div>"""
        
        return html

    def _generate_energy_html(self):
        """Generate HTML for the energy component section."""
        if not self.energy_breakdown:
            return ""
        
        # Get component names
        components = ['vdw', 'hbond', 'elec', 'desolv', 'hydrophobic', 'clash', 'entropy', 'total']
        component_names = {
            'vdw': 'Van der Waals',
            'hbond': 'H-Bond',
            'elec': 'Electrostatic',
            'desolv': 'Desolvation',
            'hydrophobic': 'Hydrophobic',
            'clash': 'Steric Clash',
            'entropy': 'Entropy',
            'total': 'TOTAL'
        }
        
        # Build section HTML
        html = """<div class="container">
            <h2>Energy Component Analysis</h2>
            <table>
                <tr>
                    <th>Component</th>"""
        
        # Add column headers for poses
        for i in range(min(5, len(self.results))):
            html += f"<th>Pose {i+1}</th>"
        
        html += """
                </tr>"""
        
        # Add rows for each component
        for component in components:
            name = component_names.get(component, component.capitalize())
            html += f"""
                <tr>
                    <td>{name}</td>"""
            
            # Add values for each pose
            for i in range(min(5, len(self.results))):
                pose_id = f"pose_{i+1}"
                if pose_id in self.energy_breakdown and component in self.energy_breakdown[pose_id]:
                    value = self.energy_breakdown[pose_id][component]
                    html += f"<td>{value:.4f}</td>"
                else:
                    html += "<td>N/A</td>"
            
            html += """
                </tr>"""
        
        html += """
            </table>
        </div>"""
        
        return html

    def _generate_interactions_html(self):
        """Generate HTML for the protein-ligand interactions section."""
        if not self.interaction_analysis:
            return ""
        
        html = """<div class="container">
            <h2>Protein-Ligand Interactions</h2>"""
        
        # Add details for up to 3 poses
        for i in range(min(3, len(self.results))):
            pose_id = f"pose_{i+1}"
            if pose_id in self.interaction_analysis:
                # Find score for this pose
                pose_score = next((score for j, (_, score) in enumerate(sorted(self.results, key=lambda x: x[1])) if j == i), 'N/A')
                
                html += f"""
            <h3>Pose {i+1} (Score: {pose_score:.4f})</h3>
            <div class="interactions">"""
                
                # Add hydrogen bonds
                html += """
                <h4>Hydrogen Bonds:</h4>"""
                
                if self.interaction_analysis[pose_id]['h_bonds']:
                    html += """
                <ul>"""
                    
                    for hbond in self.interaction_analysis[pose_id]['h_bonds']:
                        direction = '->' if hbond['type'] == 'protein_donor' else '<-'
                        html += f"""
                    <li>{hbond["protein_atom"]["chain"]}:{hbond["protein_atom"]["residue"]}{hbond["protein_atom"]["residue_id"]} {direction} Ligand {hbond["ligand_atom"]["element"]} ({hbond["distance"]} Å)</li>"""
                    
                    html += """
                </ul>"""
                else:
                    html += """
                <p>No hydrogen bonds detected</p>"""
                
                # Add hydrophobic interactions
                html += """
                <h4>Hydrophobic Interactions:</h4>"""
                
                if self.interaction_analysis[pose_id]['residues']:
                    residues_list = list(self.interaction_analysis[pose_id]['residues'])[:10]
                    residues_text = ', '.join(residues_list)
                    if len(self.interaction_analysis[pose_id]['residues']) > 10:
                        residues_text += f", and {len(self.interaction_analysis[pose_id]['residues']) - 10} more"
                    
                    html += f"""
                <p>{residues_text}</p>"""
                else:
                    html += """
                <p>No hydrophobic interactions detected</p>"""
                
                html += """
            </div>"""
        
        html += """
        </div>"""
        
        return html

    def _generate_validation_html(self):
        """Generate HTML for the validation section."""
        if not self.validation_results:
            return ""
        
        best_rmsd = self.validation_results[0]['rmsd']
        best_index = self.validation_results[0]['pose_index'] + 1
        rmsd_class = 'good' if best_rmsd < 2.0 else 'bad'
        status = 'Success' if best_rmsd < 2.0 else 'Failure'
        
        html = f"""<div class="container">
            <h2>Validation Against Reference</h2>
            <p>Best RMSD: <strong class="{rmsd_class}">{best_rmsd:.2f} Å</strong> (Pose {best_index})</p>
            <p>Validation status: <strong class="{rmsd_class}">{status}</strong></p>
            
            <h3>RMSD Values for Top 5 Poses:</h3>
            <table>
                <tr>
                    <th>Pose</th>
                    <th>RMSD (Å)</th>
                    <th>Status</th>
                </tr>"""
        
        # Add rows for RMSD values
        for result in self.validation_results[:min(5, len(self.validation_results))]:
            pose_idx = result['pose_index'] + 1
            rmsd = result['rmsd']
            rmsd_class = 'good' if rmsd < 2.0 else 'bad'
            status = 'Success' if rmsd < 2.0 else 'Failure'
            
            html += f"""
                <tr>
                    <td>Pose {pose_idx}</td>
                    <td class="{rmsd_class}">{rmsd:.2f}</td>
                    <td class="{rmsd_class}">{status}</td>
                </tr>"""
        
        html += """
            </table>
        </div>"""
        
        return html
    
    def _plot_energy_components(self, save_dir, plot_paths):
        """Generate energy component breakdown plots."""
        # Extract components for top 5 poses
        components = ['vdw', 'hbond', 'elec', 'desolv', 'hydrophobic']
        data = []
        
        for pose_id in sorted(self.energy_breakdown.keys())[:5]:
            for comp in components:
                if comp in self.energy_breakdown[pose_id]:
                    data.append({
                        'pose': pose_id,
                        'component': comp,
                        'energy': self.energy_breakdown[pose_id][comp]
                    })
        
        # Create grouped bar plot
        plt.figure(figsize=(12,6))
        sns.barplot(x='component', y='energy', hue='pose', data=pd.DataFrame(data))
        plt.title("Energy Component Breakdown")
        energy_path = os.path.join(save_dir, "energy_components.png")
        plt.savefig(energy_path)
        plt.close()
        plot_paths.append(energy_path)