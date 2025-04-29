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
matplotlib.use('Agg')  # Use a non-interactive backend for plotting
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import pandas as pd
from tabulate import tabulate
import csv
from typing import List, Dict, Any
from collections import defaultdict
from datetime import timedelta
import seaborn as sns
from .utils import save_docking_results
from .utils import save_complex_to_pdb
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import matplotlib.ticker as ticker
from pathlib import Path


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
    Comprehensive reporting class for molecular docking results.
    Generates detailed reports with energy breakdowns, visualizations, and validation metrics.
    """
    
    def __init__(self, output_dir, args, timestamp=None):
        """
        Initialize the docking reporter.
        
        Parameters:
        -----------
        output_dir : str
            Output directory path
        args : argparse.Namespace
            Command-line arguments
        timestamp : str, optional
            Timestamp for report identification
        """
        self.output_dir = Path(output_dir)
        self.args = args
        
        # Create timestamp if not provided
        if timestamp is None:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.timestamp = timestamp
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize report data
        self.run_info = {
            "timestamp": self.timestamp,
            "protein": str(args.protein),
            "ligand": str(args.ligand),
            "algorithm": args.algorithm if hasattr(args, 'algorithm') else "unknown",
            "hardware": self._get_hardware_info(args)
        }
        
        # For storing results
        self.results = []
        self.scoring_breakdown = []
        self.validation_results = None
        self.energy_breakdown = None

    
    def _get_hardware_info(self, args):
        """Extract hardware configuration information from arguments."""
        hardware = {}
        
        if hasattr(args, 'use_gpu'):
            hardware["use_gpu"] = args.use_gpu
        if hasattr(args, 'gpu_id'):
            hardware["gpu_id"] = args.gpu_id
        if hasattr(args, 'cpu_workers'):
            hardware["cpu_workers"] = args.cpu_workers
        
        return hardware
    
    def add_results(self, results, energy_breakdown=None):
        self.results = results
        if energy_breakdown:
            self.scoring_breakdown = energy_breakdown  # <--- simple
        self.results.sort(key=lambda x: x[1])

    def add_validation_results(self, validation_results):
        """
        Add validation results to the report.
        
        Parameters:
        -----------
        validation_results : dict or list
            Validation results from comparing to a reference structure
        """
        self.validation_results = validation_results
    

    def get_component_scores(self, protein, ligand):
        """
        Returns a dictionary with all energy components for the given pose.
        Works for all scoring function types.
        
        Parameters:
        -----------
        protein : Protein
            Protein object
        ligand : Ligand
            Ligand object
        
        Returns:
        --------
        dict
            Dictionary with energy component names and values
        """
        components = {}
        
        # Get protein atoms
        protein_atoms = protein.active_site.get('atoms', protein.atoms) if hasattr(protein, 'active_site') else protein.atoms
        
        # Try to calculate each component
        try:
            # Van der Waals
            if hasattr(self, 'calculate_vdw'):
                components['Van der Waals'] = self.calculate_vdw(protein, ligand)
            
            # H-Bond
            if hasattr(self, 'calculate_hbond'):
                components['H-Bond'] = self.calculate_hbond(protein, ligand)
            
            # Electrostatic
            if hasattr(self, 'calculate_electrostatics'):
                components['Electrostatic'] = self.calculate_electrostatics(protein, ligand)
            
            # Desolvation
            if hasattr(self, 'calculate_desolvation'):
                components['Desolvation'] = self.calculate_desolvation(protein, ligand)
            
            # Hydrophobic
            if hasattr(self, 'calculate_hydrophobic'):
                components['Hydrophobic'] = self.calculate_hydrophobic(protein, ligand)
            
            # Clash
            if hasattr(self, 'calculate_clashes'):
                components['Clash'] = self.calculate_clashes(protein, ligand)
            
            # Entropy
            if hasattr(self, 'calculate_entropy'):
                components['Entropy'] = self.calculate_entropy(ligand)
            
            # Calculate total
            total = sum(components.values())
            components['Total'] = total
            
        except Exception as e:
            print(f"Error in get_component_scores: {e}")
        
        return components

    

    def extract_energy_components(self, scoring_function, protein, ligand_poses):
        """
        Enhanced extraction of energy components with robust error handling.
        
        Parameters:
        -----------
        scoring_function : ScoringFunction
            Scoring function used for docking
        protein : Protein
            Protein object
        ligand_poses : list
            List of ligand poses
        
        Returns:
        --------
        list
            List of dictionaries containing energy components for each pose
        """
        print("Starting enhanced energy component extraction...")
        energy_breakdown = []
        
        # Get scoring function class name
        scoring_class = scoring_function.__class__.__name__
        print(f"Scoring function class: {scoring_class}")
        
        # Process each pose
        for i, pose in enumerate(ligand_poses):
            print(f"Processing pose {i+1}/{len(ligand_poses)}")
            components = {}
            
            # Get total score
            total_score = scoring_function.score(protein, pose)
            components['Total'] = total_score
            print(f"Total score: {total_score:.2f}")
            components['Pose'] = i + 1
            components['Score'] = total_score


            
            # Try to extract all components
            def try_component(component_name, methods):
                try:
                    if hasattr(self, methods[0]):
                        components[component_name] = getattr(self, methods[0])(protein, pose)
                        print(f"  {component_name}: {components[component_name]:.2f}")
                except Exception as e:
                    print(f"  Error extracting {component_name}: {e}")

            try_component('Van der Waals', ['calculate_vdw'])
            try_component('H-Bond', ['calculate_hbond'])
            try_component('Electrostatic', ['calculate_electrostatics'])
            try_component('Desolvation', ['calculate_desolvation'])
            try_component('Hydrophobic', ['calculate_hydrophobic'])
            try_component('Clash', ['calculate_clashes'])
            try_component('Entropy', ['calculate_entropy'])
            
            # If the scoring function has weights, estimate missing components
            if hasattr(scoring_function, 'weights'):
                weights = scoring_function.weights
                
                # Map component names to weight keys
                weight_map = {
                    'Van der Waals': 'vdw',
                    'H-Bond': 'hbond',
                    'Electrostatic': 'elec',
                    'Desolvation': 'desolv',
                    'Hydrophobic': 'hydrophobic',
                    'Clash': 'clash',
                    'Entropy': 'entropy'
                }
                
                # Calculate raw sum of available components (excluding total)
                raw_sum = sum(components.get(comp, 0) for comp in components if comp != 'Total')
                
                # If raw sum is significantly different from total, try to estimate missing components
                if abs(raw_sum - total_score) > 0.1:
                    print(f"  Raw sum ({raw_sum:.2f}) differs from total ({total_score:.2f}), estimating missing components")
                    
                    # Get weights for missing components
                    missing_weights = {}
                    total_missing_weight = 0
                    
                    for comp, weight_key in weight_map.items():
                        if comp not in components and weight_key in weights:
                            weight = weights[weight_key]
                            missing_weights[comp] = weight
                            total_missing_weight += weight
                    
                    # Distribute remaining score proportionally by weight
                    if total_missing_weight > 0:
                        remaining = total_score - raw_sum
                        for comp, weight in missing_weights.items():
                            components[comp] = (weight / total_missing_weight) * remaining
                            print(f"  Estimated {comp}: {components[comp]:.2f}")
            
            # Add to breakdown
            energy_breakdown.append(components)
        
        return energy_breakdown

        

    
    def generate_basic_report(self):
        """
        Generate a basic text report for docking results.
        
        Returns:
        --------
        str
            Path to the generated report file
        """
        report_path = self.output_dir / "docking_report.txt"
        
        # Extract needed information
        protein_path = self.args.protein
        ligand_path = self.args.ligand
        algorithm = self.args.algorithm if hasattr(self.args, 'algorithm') else "unknown"
        iterations = self.args.iterations if hasattr(self.args, 'iterations') else "unknown"
        
        # Calculate elapsed time if available
        if hasattr(self.args, 'start_time'):
            elapsed_time = datetime.now().timestamp() - self.args.start_time
        else:
            elapsed_time = 0.0
        
        # Sort results by score
        sorted_results = sorted(self.results, key=lambda x: x[1])
        
        with open(report_path, 'w') as f:
            f.write("=====================================================\n")
            f.write("        PandaDock - Python Molecular Docking Results    \n")
            f.write("=====================================================\n\n")
            
            # Write run information
            f.write("RUN INFORMATION\n")
            f.write("--------------\n")
            f.write(f"Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Protein: {protein_path}\n")
            f.write(f"Ligand: {ligand_path}\n")
            sphere_path = Path(self.output_dir) / "sphere.pdb"
            if sphere_path.exists():
                f.write(f"Active Site Sphere File: {sphere_path.name}\n\n")
                f.write("\n")
            f.write(f"Algorithm: {algorithm}\n")
            f.write(f"Iterations/Generations: {iterations}\n")
            f.write(f"Total Runtime: {elapsed_time:.2f} seconds\n\n")
            
            if not self.results:
                f.write("RESULTS SUMMARY\n")
                f.write("--------------\n")
                f.write("No valid docking solutions found.\n")
                f.write("This can occur due to incompatible structures, overly strict scoring parameters,\n")
                f.write("or issues with the search space definition.\n\n")
            else:
                sorted_results = sorted(self.results, key=lambda x: x[1])
                
                f.write("RESULTS SUMMARY\n")
                f.write("--------------\n")
                f.write(f"Total Poses Generated: {len(self.results)}\n")
                f.write(f"Best Score: {sorted_results[0][1]:.4f}\n")
                f.write(f"Worst Score: {sorted_results[-1][1]:.4f}\n")
                f.write(f"Average Score: {sum([score for _, score in self.results])/len(self.results):.4f}\n\n")
                f.write(f"Score Standard Deviation: {np.std([score for _, score in self.results]):.4f}\n\n")
                
                f.write("TOP 10 POSES\n")
                f.write("--------------\n")
                f.write("Rank\tScore\tFile\n")
                for i, (pose, score) in enumerate(sorted_results[:10]):
                    f.write(f"{i+1}\t{score:.4f}\tpose_{i+1}_score_{score:.1f}.pdb\n")     
            
            f.write("\n\nFull results are available in the output directory.\n")
            f.write("=====================================================\n")
        
        #print(f"Basic report written to {report_path}")
        return report_path
    
    def generate_detailed_report(self, include_energy_breakdown=True):
        """
        Generate a detailed report for docking results with energy breakdowns.
        
        Parameters:
        -----------
        include_energy_breakdown : bool
            Whether to include detailed energy breakdowns
        
        Returns:
        --------
        str
            Path to the generated report file
        """
        report_path = self.output_dir / "detailed_docking_report.txt"
        
        # Extract needed information
        protein_path = self.args.protein
        ligand_path = self.args.ligand
        algorithm = self.args.algorithm if hasattr(self.args, 'algorithm') else "unknown"
        
        # Get algorithm parameters
        algorithm_params = {}
        if algorithm == "genetic":
            if hasattr(self.args, 'population_size'):
                algorithm_params["Population Size"] = self.args.population_size
            if hasattr(self.args, 'mutation_rate'):
                algorithm_params["Mutation Rate"] = self.args.mutation_rate
        elif algorithm == "monte-carlo":
            if hasattr(self.args, 'mc_steps'):
                algorithm_params["Steps"] = self.args.mc_steps
            if hasattr(self.args, 'temperature'):
                algorithm_params["Temperature"] = f"{self.args.temperature} K"
        
        # Get scoring function details
        scoring_type = "advanced"
        if hasattr(self.args, 'enhanced_scoring') and self.args.enhanced_scoring:
            scoring_type = "enhanced"
        if hasattr(self.args, 'physics_based') and self.args.physics_based:
            scoring_type = "physics-based"
        
        # Sort results by score
        sorted_results = sorted(self.results, key=lambda x: x[1])
        
        with open(report_path, 'w') as f:
            f.write(r"""
    ════════════════════════════════════════════════════════════════════════════════
        ██████╗  █████╗ ███╗   ██╗██████╗  █████╗ ██████╗  ██████╗  ██████╗██╗  ██╗
        ██╔══██╗██╔══██╗████╗  ██║██╔══██╗██╔══██╗██╔══██╗██╔═══██╗██╔════╝██║ ██╔╝
        ██████╔╝███████║██╔██╗ ██║██║  ██║███████║██║  ██║██║   ██║██║     █████╔╝ 
        ██╔═══╝ ██╔══██║██║╚██╗██║██║  ██║██╔══██║██║  ██║██║   ██║██║     ██╔═██╗ 
        ██║     ██║  ██║██║ ╚████║██████╔╝██║  ██║██████╔╝╚██████╔╝╚██████╗██║  ██╗
        ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝ ╚═╝  ╚═╝╚═════╝  ╚═════╝  ╚═════╝╚═╝  ╚═╝
                                                                                                                                                                                                                                    
                PandaDock - Python Molecular Docking Tool                             
                https://github.com/pritampanda15/PandaDock                   
    ════════════════════════════════════════════════════════════════════════════════
        """)
            
            # Write run information
            f.write("RUN INFORMATION\n")
            f.write("--------------\n")
            f.write(f"Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Report ID: {self.timestamp}\n")
            f.write(f"Protein: {protein_path}\n")
            f.write(f"Ligand: {ligand_path}\n\n")
            sphere_path = Path(self.output_dir) / "sphere.pdb"
            if sphere_path.exists():
                f.write(f"Active Site Sphere File: {sphere_path.name}\n")
            f.write("\n")
            f.write(f"Total Poses Generated: {len(self.results)}\n")
            f.write("\n")
            # Algorithm details
            f.write("ALGORITHM DETAILS\n")
            f.write("-----------------\n")
            f.write(f"Algorithm: {algorithm.capitalize()}\n")
            for param, value in algorithm_params.items():
                f.write(f"{param}: {value}\n")
            f.write(f"Scoring Function: {scoring_type.capitalize()}\n")
            
            if hasattr(self.args, 'use_gpu') and self.args.use_gpu:
                f.write(f"Hardware Acceleration: GPU (ID: {getattr(self.args, 'gpu_id', 0)})\n")
                if hasattr(self.args, 'gpu_precision'):
                    f.write(f"GPU Precision: {self.args.gpu_precision}\n")
            else:
                cpu_workers = getattr(self.args, 'cpu_workers', 'all available')
                f.write(f"Hardware Acceleration: CPU ({cpu_workers} cores)\n")
            
            if hasattr(self.args, 'exhaustiveness'):
                f.write(f"Exhaustiveness: {self.args.exhaustiveness}\n")
            
            if hasattr(self.args, 'local_opt') and self.args.local_opt:
                f.write("Local Optimization: Enabled\n")
            
            if not self.results:
                f.write("RESULTS SUMMARY\n")
                f.write("--------------\n")
                f.write("No valid docking solutions found.\n")
                f.write("This can occur due to incompatible structures, overly strict scoring parameters,\n")
                f.write("or issues with the search space definition.\n\n")
            else:
                sorted_results = sorted(self.results, key=lambda x: x[1])
                
                f.write("RESULTS SUMMARY\n")
                f.write("--------------\n")
                f.write(f"Total Poses Generated: {len(self.results)}\n")
                f.write(f"Best Score: {sorted_results[0][1]:.4f}\n")
                f.write(f"Worst Score: {sorted_results[-1][1]:.4f}\n")
                f.write(f"Average Score: {sum([score for _, score in self.results])/len(self.results):.4f}\n\n")
                f.write(f"Score Standard Deviation: {np.std([score for _, score in self.results]):.4f}\n\n")
                
                f.write("TOP 10 POSES\n")
                f.write("--------------\n")
                f.write("Rank\tScore\tFile\n")
                if include_energy_breakdown and self.scoring_breakdown and len(self.scoring_breakdown) > 0:
                    f.write("ENERGY COMPONENT BREAKDOWN (TOP 10 POSES)\n")
                    f.write("----------------------------------------\n")
                    #components = list(self.scoring_breakdown[0].keys())
                    components = [c for c in self.scoring_breakdown[0].keys() if c.lower() not in ['pose', 'score', 'total']]
                    # Sort components by name
                    components.sort()
                    header = "Pose  " + "  ".join([f"{comp[:8]:>10}" for comp in components])
                    
                    # Write to file
                    f.write(header + "\n")
                    f.write("-" * len(header) + "\n")
                    for i, energy in enumerate(self.scoring_breakdown[:min(10, len(self.scoring_breakdown))]):
                        energy_str = f"{i+1:4d}  " + "  ".join([f"{energy.get(comp, 0.0):10.2f}" for comp in components])
                        f.write(energy_str + "\n")

                    # # ✅ Print to CLI
                    print("\nENERGY COMPONENT BREAKDOWN (TOP 10 POSES)")
                    print("----------------------------------------")
                    print(header)
                    print("-" * len(header))
                    for i, energy in enumerate(self.scoring_breakdown[:min(10, len(self.scoring_breakdown))]):
                        energy_str = f"{i+1:4d}  " + "  ".join([f"{energy.get(comp, 0.0):10.2f}" for comp in components])
                        print(energy_str)
                    
                                        # Binding affinity estimation
                    f.write("\nBINDING AFFINITY ESTIMATION\n")
                    f.write("---------------------------\n")
                    for i, (pose, score) in enumerate(sorted_results[:min(10, len(sorted_results))]):
                        affinities = self.calculate_binding_affinity(score)
                        f.write(f"Pose {i+1}: Score = {score:.2f} kcal/mol\n")
                        f.write(f"  Estimated ΔG: {affinities['DeltaG (kcal/mol)']:.2f} kcal/mol\n")
                        f.write(f"  Estimated Kd: {affinities['Kd (M)']:.2e} M\n")
                        f.write(f"  Estimated IC50: {affinities['IC50 (M)']:.2e} M\n")
                        f.write("\n")


            
                # Write validation results if available
                if self.validation_results:
                    f.write("VALIDATION AGAINST REFERENCE STRUCTURE\n")
                    f.write("------------------------------------\n")
                    
                    if isinstance(self.validation_results, list):
                        # List of validation results for multiple poses
                        f.write("RMSD Results for Top Poses:\n")
                        f.write("Rank  Pose     RMSD (Å)  Status\n")
                        f.write("----- -------- --------- --------------\n")
                        
                        for i, result in enumerate(self.validation_results[:min(10, len(self.validation_results))]):
                            pose_idx = result.get('pose_index', i)
                            rmsd = result.get('rmsd', 0.0)
                            status = "Success" if rmsd < 2.0 else "Failure"
                            f.write(f"{i+1:5d} {pose_idx+1:8d} {rmsd:9.4f} {status}\n")
                        
                        # Best RMSD
                        if len(self.validation_results) > 0:
                            best_rmsd = min(result.get('rmsd', float('inf')) for result in self.validation_results)
                            f.write(f"\nBest RMSD: {best_rmsd:.4f} Å\n")
                            f.write(f"Docking Success: {'Yes' if best_rmsd < 2.0 else 'No'}\n")
                    
                    elif isinstance(self.validation_results, dict):
                        # Single validation result
                        rmsd = self.validation_results.get('rmsd', 0.0)
                        max_dev = self.validation_results.get('max_deviation', 0.0)
                        min_dev = self.validation_results.get('min_deviation', 0.0)
                        success = self.validation_results.get('success', False)
                        
                        f.write(f"Overall RMSD: {rmsd:.4f} Å\n")
                        f.write(f"Maximum Atomic Deviation: {max_dev:.4f} Å\n")
                        f.write(f"Minimum Atomic Deviation: {min_dev:.4f} Å\n")
                        f.write(f"Docking Success: {'Yes' if success else 'No'}\n")
                        
                        # Report per-atom deviations if available
                        atom_deviations = self.validation_results.get('atom_deviations', None)
                        if atom_deviations is not None:
                            f.write("\nTop 10 Atom Deviations:\n")
                            sorted_indices = np.argsort(atom_deviations)[::-1]
                            for i in range(min(10, len(sorted_indices))):
                                idx = sorted_indices[i]
                                f.write(f"Atom {idx + 1}: {atom_deviations[idx]:.4f} Å\n")
                    
                    f.write("\n")
                
                f.write("===========================================================\n")
                f.write("Report generated by PandaDock - Python Molecular Docking Tool\n")
            
            #print(f"Detailed report written to {report_path}")
            return report_path
    
    def generate_csv_report(self):
        """
        Generate CSV files with docking results and energy breakdowns.
        
        Returns:
        --------
        tuple
            Paths to the generated CSV files (results_csv, energy_csv)
        """
        # Create paths for CSV files
        results_csv = self.output_dir / "docking_results.csv"
        energy_csv = self.output_dir / "energy_breakdown.csv"
        
        # Write docking results to CSV
        with open(results_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Rank', 'Score', 'Filename'])
            
            sorted_results = sorted(self.results, key=lambda x: x[1])
            for i, (pose, score) in enumerate(sorted_results):
                writer.writerow([i+1, score, f"pose_{i+1}_score_{score:.1f}.pdb"])
        
        # Write energy breakdown to CSV if available
        if self.scoring_breakdown:
            with open(energy_csv, 'w', newline='') as f:
                # Get all energy components from first breakdown
                # components = list(self.scoring_breakdown[0].keys())
                components = [c for c in self.scoring_breakdown[0].keys() if c.lower() not in ['pose', 'score', 'total']]
                # Sort components by name
                components.sort()
                
                writer = csv.writer(f)
                writer.writerow(['Pose'] + components)
                
                for i, energy in enumerate(self.scoring_breakdown):
                    writer.writerow([i+1] + [energy[comp] for comp in components])
            
            print(f"Energy breakdown CSV written to {energy_csv}")
            return results_csv, energy_csv
        
        print(f"Results CSV written to {results_csv}")
        return results_csv, None

    def generate_json_report(self):
        """
        Generate a JSON report with all docking information.
        
        Returns:
        --------
        str
            Path to the generated JSON file
        """
        json_path = self.output_dir / "docking_report.json"
        
        # Create a dictionary with all report information
        report_data = {
            "run_info": {
                "timestamp": str(self.timestamp),
                "protein": str(self.args.protein) if hasattr(self.args, 'protein') else "unknown",
                "ligand": str(self.args.ligand) if hasattr(self.args, 'ligand') else "unknown",
                "algorithm": self.args.algorithm if hasattr(self.args, 'algorithm') else "unknown"
            },
            "results": {
                "pose_count": len(self.results),
                "poses": []
            }
        }
        
        # Add pose information (converting all values to basic Python types)
        if self.results:
            sorted_results = sorted(self.results, key=lambda x: x[1])
            report_data["results"]["best_score"] = float(sorted_results[0][1])
            
            for i, (_, score) in enumerate(sorted_results):
                report_data["results"]["poses"].append({
                    "rank": i+1,
                    "score": float(score)
                })
        
        # Simplify scoring breakdown to basic Python types
        if self.scoring_breakdown:
            breakdown_list = []
            for energy in self.scoring_breakdown:
                energy_dict = {}
                for key, value in energy.items():
                    # Convert numpy values to Python native types
                    if hasattr(value, "item"):  # numpy scalar
                        energy_dict[key] = value.item()
                    elif isinstance(value, bool):
                        energy_dict[key] = bool(value)
                    else:
                        energy_dict[key] = float(value) if isinstance(value, (int, float)) else str(value)
                breakdown_list.append(energy_dict)
            report_data["energy_breakdown"] = breakdown_list
        
        # Write JSON file with explicit error handling
        try:
            with open(json_path, 'w') as f:
                json.dump(report_data, f, indent=4)
            print(f"JSON report written to {json_path}")
        except TypeError as e:
            print(f"JSON serialization error: {e}")
            print("Writing simplified JSON report without complex data")
            
            # Simplified report as fallback
            basic_report = {
                "run_info": report_data["run_info"],
                "results": {
                    "pose_count": report_data["results"]["pose_count"],
                    "best_score": report_data["results"]["best_score"] if "best_score" in report_data["results"] else None
                }
            }
            
            with open(json_path, 'w') as f:
                json.dump(basic_report, f, indent=4)
            
        return json_path
    
        # Write JSON file
        #with open(json_path, 'w') as f:
        #    json.dump(report_data, f, indent=4, cls=EnhancedJSONEncoder)
        
        #print(f"JSON report written to {json_path}")
        #return json_path
    
    
            
    def generate_plots(self, save_dir=None, dpi=300):
        """
        Generate publication-quality plots visualizing the docking results.
        
        Parameters:
        -----------
        save_dir : str or Path, optional
            Directory to save plots, defaults to the output directory
        dpi : int
            Resolution of saved figures
            
        Returns:
        --------
        list
            Paths to the generated plot files
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib import rcParams
        import os
        from pathlib import Path
        
        # Set publication-quality parameters
        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
        rcParams['font.size'] = 12
        rcParams['axes.linewidth'] = 1.5
        rcParams['axes.labelsize'] = 14
        rcParams['axes.titlesize'] = 16
        rcParams['xtick.major.width'] = 1.5
        rcParams['ytick.major.width'] = 1.5
        rcParams['xtick.labelsize'] = 12
        rcParams['ytick.labelsize'] = 12
        rcParams['legend.fontsize'] = 12
        rcParams['figure.dpi'] = 100
        rcParams['savefig.dpi'] = dpi
        rcParams['savefig.bbox'] = 'tight'
        rcParams['savefig.pad_inches'] = 0.1
        
        # Setup save directory
        if save_dir is None:
            save_dir = self.output_dir
        else:
            save_dir = Path(save_dir)
            os.makedirs(save_dir, exist_ok=True)
        
        plot_paths = []
        
        # Always generate these two plots regardless of energy component availability
        if self.results and len(self.results) > 0:
            try:
                # Get scores and prepare data
                scores = [score for _, score in self.results]
                sorted_scores = sorted(scores)
                ranks = range(1, len(sorted_scores) + 1)
                
                # 1. Score distribution plot - enhanced
                score_plot_path = save_dir / "score_distribution.png"
                plt.figure(figsize=(8, 6))
                
                # Calculate optimal number of bins
                n_bins = min(20, max(5, int(len(scores) / 2)))
                
                # Create histogram with better styling
                _, bins, patches = plt.hist(scores, bins=n_bins, 
                                            color='#1f77b4', edgecolor='black', 
                                            alpha=0.8, linewidth=1.5)
                
                # Add statistical indicators
                mean_score = np.mean(scores)
                median_score = np.median(scores)
                plt.axvline(mean_score, color='crimson', linestyle='--', linewidth=2, 
                            label=f'Mean: {mean_score:.2f}')
                plt.axvline(median_score, color='darkgreen', linestyle=':', linewidth=2, 
                            label=f'Median: {median_score:.2f}')
                
                plt.xlabel('Docking Score (kcal/mol)', fontweight='bold')
                plt.ylabel('Frequency', fontweight='bold')
                plt.title('Distribution of Docking Scores', fontweight='bold')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(frameon=True, fancybox=True, framealpha=0.8)
                
                plt.tight_layout()
                plt.savefig(score_plot_path)
                #plt.savefig(str(score_plot_path).replace('.png', '.pdf'))
                plt.savefig(str(score_plot_path).replace('.png', '.svg'))
                plt.close()
                plot_paths.append(score_plot_path)
                print(f"Score distribution plot saved to {score_plot_path}")
                
                # 2. Score rank plot - enhanced
                rank_plot_path = save_dir / "score_rank.png"
                plt.figure(figsize=(8, 6))
                
                # Create scatter plot with connecting line for better visibility
                plt.scatter(ranks, sorted_scores, 
                        marker='o', s=60, c='#ff7f0e', edgecolor='black', alpha=0.8,
                        label='Individual Poses')
                plt.plot(ranks, sorted_scores, 
                        linestyle='-', color='#ff7f0e', linewidth=2, alpha=0.6)
                
                # Add trendline
                if len(ranks) > 2:
                    z = np.polyfit(ranks, sorted_scores, 1)
                    p = np.poly1d(z)
                    plt.plot(ranks, p(ranks), 
                            linestyle='--', color='navy', linewidth=2,
                            label=f'Trend (slope: {z[0]:.2f})')
                
                # Highlight top 3 poses if we have at least 3
                if len(ranks) >= 3:
                    plt.scatter(ranks[:3], sorted_scores[:3], 
                            marker='*', s=150, c='green', edgecolor='black', zorder=5,
                            label='Top 3 Poses')
                
                plt.xlabel('Pose Rank', fontweight='bold')
                plt.ylabel('Docking Score (kcal/mol)', fontweight='bold')
                plt.title('Docking Scores by Rank', fontweight='bold')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(frameon=True, fancybox=True, framealpha=0.8)
                
                # Set x-ticks to appropriate values based on number of poses
                if len(ranks) <= 10:
                    plt.xticks(list(ranks))
                else:
                    plt.xticks(np.linspace(min(ranks), max(ranks), 10, dtype=int))
                
                # Annotate best score
                plt.annotate(f'Best: {sorted_scores[0]:.2f}', 
                            xy=(1, sorted_scores[0]), 
                            xytext=(5, 10), 
                            textcoords='offset points',
                            fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                
                plt.tight_layout()
                plt.savefig(rank_plot_path)
                plt.savefig(str(rank_plot_path).replace('.png', '.pdf'))
                plt.savefig(str(rank_plot_path).replace('.png', '.svg'))
                plt.close()
                plot_paths.append(rank_plot_path)
                print(f"Score rank plot saved to {rank_plot_path}")
                
                # 3. Combined visualization with pose quality assessment
                combined_plot_path = save_dir / "score_analysis.png"
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                
                # Left panel: Score distribution
                ax1.hist(scores, bins=n_bins, color='#1f77b4', edgecolor='black', alpha=0.8, linewidth=1.5)
                ax1.axvline(mean_score, color='crimson', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.2f}')
                ax1.axvline(median_score, color='darkgreen', linestyle=':', linewidth=2, label=f'Median: {median_score:.2f}')
                ax1.set_xlabel('Docking Score (kcal/mol)', fontweight='bold')
                ax1.set_ylabel('Frequency', fontweight='bold')
                ax1.set_title('Score Distribution', fontweight='bold')
                ax1.grid(True, linestyle='--', alpha=0.7)
                ax1.legend(frameon=True, fancybox=True, framealpha=0.8)
                
                # Right panel: Score vs Rank with score categories
                score_range = max(scores) - min(scores)
                score_threshold_good = min(scores) + score_range * 0.3
                score_threshold_ok = min(scores) + score_range * 0.6
                
                # Define pose categories
                good_poses = [i for i, s in enumerate(sorted_scores) if s <= score_threshold_good]
                ok_poses = [i for i, s in enumerate(sorted_scores) if score_threshold_good < s <= score_threshold_ok]
                poor_poses = [i for i, s in enumerate(sorted_scores) if s > score_threshold_ok]
                
                # Plot each category
                if good_poses:
                    ax2.scatter([ranks[i] for i in good_poses], [sorted_scores[i] for i in good_poses], 
                            color='green', s=60, label='Good poses', zorder=3)
                if ok_poses:
                    ax2.scatter([ranks[i] for i in ok_poses], [sorted_scores[i] for i in ok_poses], 
                            color='orange', s=60, label='Acceptable poses', zorder=2)
                if poor_poses:
                    ax2.scatter([ranks[i] for i in poor_poses], [sorted_scores[i] for i in poor_poses], 
                            color='red', s=60, label='Poor poses', zorder=1)
                
                # Add connecting line for all poses
                ax2.plot(ranks, sorted_scores, linestyle='-', color='gray', linewidth=1.5, alpha=0.5)
                
                ax2.set_xlabel('Pose Rank', fontweight='bold')
                ax2.set_ylabel('Docking Score (kcal/mol)', fontweight='bold')
                ax2.set_title('Pose Quality Assessment', fontweight='bold')
                ax2.grid(True, linestyle='--', alpha=0.7)
                ax2.legend(frameon=True, fancybox=True, framealpha=0.8)
                
                # Set x-ticks based on number of poses
                if len(ranks) <= 10:
                    ax2.set_xticks(list(ranks))
                else:
                    ax2.set_xticks(np.linspace(min(ranks), max(ranks), 10, dtype=int))
                
                plt.tight_layout()
                plt.savefig(combined_plot_path)
                plt.savefig(str(combined_plot_path).replace('.png', '.pdf'))
                plt.savefig(str(combined_plot_path).replace('.png', '.svg'))
                plt.close()
                plot_paths.append(combined_plot_path)
                print(f"Combined score analysis plot saved to {combined_plot_path}")
                
                # 4. Score improvement visualization
                if len(sorted_scores) > 1:
                    improvement_plot_path = save_dir / "score_improvement.png"
                    plt.figure(figsize=(8, 6))
                    
                    # Calculate score improvements relative to the worst score
                    worst_score = max(sorted_scores)
                    improvements = [worst_score - score for score in sorted_scores]
                    
                    # Create bar chart of improvements
                    bars = plt.bar(ranks, improvements, color='#2ca02c', alpha=0.8, edgecolor='black')
                    
                    # Add percentage labels on bars
                    max_improvement = worst_score - min(sorted_scores)
                    if max_improvement > 0:  # Avoid division by zero
                        for i, bar in enumerate(bars):
                            height = bar.get_height()
                            percentage = (height / max_improvement) * 100
                            if percentage >= 5:  # Only label bars with significant height
                                plt.text(bar.get_x() + bar.get_width()/2., height + max_improvement*0.02,
                                        f'{percentage:.0f}%', ha='center', va='bottom', fontsize=10)
                    
                    plt.xlabel('Pose Rank', fontweight='bold')
                    plt.ylabel('Score Improvement (kcal/mol)', fontweight='bold')
                    plt.title('Score Improvement Relative to Worst Pose', fontweight='bold')
                    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
                    
                    # Set x-ticks based on number of poses
                    if len(ranks) <= 10:
                        plt.xticks(list(ranks))
                    else:
                        plt.xticks(np.linspace(min(ranks), max(ranks), 10, dtype=int))
                    
                    plt.tight_layout()
                    plt.savefig(improvement_plot_path)
                    plt.savefig(str(improvement_plot_path).replace('.png', '.pdf'))
                    plt.savefig(str(improvement_plot_path).replace('.png', '.svg'))
                    plt.close()
                    plot_paths.append(improvement_plot_path)
                    print(f"Score improvement plot saved to {improvement_plot_path}")
                
            except Exception as e:
                print(f"Error generating plots: {e}")


        # 3. Energy breakdown plot
        if self.scoring_breakdown and len(self.scoring_breakdown) > 0:
            top_poses = min(10, len(self.scoring_breakdown))
            if not isinstance(self.scoring_breakdown[0], dict) or len(self.scoring_breakdown[0]) == 0:
                print("No energy components available for plotting.")
                return plot_paths  # return existing plots
                    
            # Get components excluding 'pose', 'score', 'total'
            components = [c for c in self.scoring_breakdown[0].keys() if c.lower() not in ['pose', 'score', 'total']]
            
            # Skip if no components to plot
            if not components:
                print("No energy components to plot. Skipping energy breakdown plot.")
                return plot_paths
            
            # Sort components by absolute magnitude of first pose for better visualization
            components.sort(key=lambda x: abs(self.scoring_breakdown[0].get(x, 0.0)))
            
            energy_plot_path = save_dir / "energy_breakdown.png"
            plt.figure(figsize=(10, 8))
            
            # Use a better color palette
            color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            colors = [color_palette[i % len(color_palette)] for i in range(len(components))]
            
            bar_width = 0.8 / len(components)
            r = np.arange(top_poses)
            
            # Create stacked bars for negative and positive components
            neg_components = [c for c in components if self.scoring_breakdown[0].get(c, 0.0) < 0]
            pos_components = [c for c in components if self.scoring_breakdown[0].get(c, 0.0) >= 0]
            
            # Set hatch patterns for better distinction in B&W printing
            hatches = ['', '/', '\\', 'x', '+', '*', 'o', 'O', '.', '-']
            
            # Plot the components
            for i, component in enumerate(components):
                values = [energy.get(component, 0.0) for energy in self.scoring_breakdown[:top_poses]]
                
                # Use different color for negative (favorable) vs positive (unfavorable) components
                is_negative = values[0] < 0
                edge_color = 'black'
                hatch = hatches[i % len(hatches)]
                
                plt.bar(r + i * bar_width, values, width=bar_width, 
                        color=colors[i], edgecolor=edge_color, linewidth=1.5, 
                        hatch=hatch, alpha=0.8, label=component.capitalize())
            
            # Get total energy for each pose
            if 'total' in self.scoring_breakdown[0]:
                totals = [energy.get('total', 0.0) for energy in self.scoring_breakdown[:top_poses]]
                # Add total energy as a line
                plt.plot(r + bar_width * (len(components) - 1) / 2, totals, 'ko-', 
                        linewidth=2, markersize=8, label='Total Energy')
                
                # Add total value annotations
                for i, total in enumerate(totals):
                    plt.annotate(f'{total:.1f}', 
                                xy=(r[i] + bar_width * (len(components) - 1) / 2, total),
                                xytext=(0, 10 if total > 0 else -20), 
                                textcoords='offset points',
                                ha='center', fontweight='bold', fontsize=9,
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            # Improve axis labels and title
            plt.xlabel('Pose Rank', fontweight='bold', fontsize=14)
            plt.ylabel('Energy Contribution (kcal/mol)', fontweight='bold', fontsize=14)
            plt.title('Energy Component Breakdown for Top Poses', fontweight='bold', fontsize=16)
            
            # Improve x-ticks
            plt.xticks(r + bar_width * (len(components) - 1) / 2, [f'{i+1}' for i in range(top_poses)], 
                    fontsize=12)
            
            # Add a zero line for reference
            plt.axhline(y=0, color='black', linestyle='-', linewidth=1.0, alpha=0.5)
            
            # Improve legend
            plt.legend(frameon=True, fancybox=True, framealpha=0.9, fontsize=12, 
                    loc='best', title="Energy Components")
            
            # Improve grid
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(energy_plot_path, dpi=300)
            #plt.savefig(str(energy_plot_path).replace('.png', '.pdf'))
            plt.savefig(str(energy_plot_path).replace('.png', '.svg'))
            plt.close()
            plot_paths.append(energy_plot_path)
            print(f"Energy breakdown plot saved to {energy_plot_path}")
            
            # Add a separate stacked bar chart for better component visualization
            stacked_plot_path = save_dir / "energy_stacked.png"
            plt.figure(figsize=(10, 8))
            
            # Plot negative components (favorable interactions)
            bottom = np.zeros(top_poses)
            for i, component in enumerate(neg_components):
                values = [min(0, energy.get(component, 0.0)) for energy in self.scoring_breakdown[:top_poses]]
                plt.bar(r, values, bottom=bottom, width=0.7, 
                        color=colors[components.index(component)], 
                        edgecolor='black', linewidth=1, alpha=0.8, 
                        hatch=hatches[i % len(hatches)],
                        label=component.capitalize())
                bottom += values
            
            # Reset bottom for positive components
            bottom = np.zeros(top_poses)
            for i, component in enumerate(pos_components):
                values = [max(0, energy.get(component, 0.0)) for energy in self.scoring_breakdown[:top_poses]]
                plt.bar(r, values, bottom=bottom, width=0.7, 
                        color=colors[components.index(component)], 
                        edgecolor='black', linewidth=1, alpha=0.8, 
                        hatch=hatches[i % len(hatches)],
                        label=component.capitalize())
                bottom += values
            
            # Add total energy as points
            if 'total' in self.scoring_breakdown[0]:
                totals = [energy.get('total', 0.0) for energy in self.scoring_breakdown[:top_poses]]
                plt.plot(r, totals, 'ko-', linewidth=2, markersize=10, label='Total Energy')
                
                # Add total value annotations
                for i, total in enumerate(totals):
                    plt.annotate(f'{total:.1f}', 
                                xy=(r[i], total),
                                xytext=(0, 10 if total > 0 else -20), 
                                textcoords='offset points',
                                ha='center', fontweight='bold', fontsize=9,
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            plt.xlabel('Pose Rank', fontweight='bold', fontsize=14)
            plt.ylabel('Energy Contribution (kcal/mol)', fontweight='bold', fontsize=14)
            plt.title('Stacked Energy Components for Top Poses', fontweight='bold', fontsize=16)
            plt.xticks(r, [f'{i+1}' for i in range(top_poses)], fontsize=12)
            plt.axhline(y=0, color='black', linestyle='-', linewidth=1.0, alpha=0.5)
            plt.legend(frameon=True, fancybox=True, framealpha=0.9, fontsize=12, 
                    loc='best', title="Energy Components")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(stacked_plot_path, dpi=300)
            #plt.savefig(str(stacked_plot_path).replace('.png', '.pdf'))
            plt.savefig(str(stacked_plot_path).replace('.png', '.svg'))
            plt.close()
            plot_paths.append(stacked_plot_path)
            print(f"Stacked energy plot saved to {stacked_plot_path}")

        # 4. RMSD plot if validation results are available
        if self.validation_results and isinstance(self.validation_results, list):
            rmsd_plot_path = save_dir / "rmsd_plot.png"
            plt.figure(figsize=(10, 6))
            
            # Extract RMSD values
            rmsd_values = [result.get('rmsd', 0.0) for result in self.validation_results]
            
            # Create color mapping based on threshold
            threshold = 2.0  # Standard threshold for success
            bar_colors = ['#2ca02c' if rmsd <= threshold else '#d62728' for rmsd in rmsd_values]
            
            # Create bar chart with better styling
            bars = plt.bar(range(1, len(rmsd_values) + 1), rmsd_values, 
                        color=bar_colors, edgecolor='black', linewidth=1.5, alpha=0.8)
            
            # Add threshold line with enhanced styling
            plt.axhline(y=threshold, color='black', linestyle='--', linewidth=2, 
                    label=f'Success Threshold ({threshold:.1f} Å)')
            
            # Add text labels above/below bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                text_color = 'black'
                
                if height <= threshold:
                    text = "✓"  # Check mark for successful poses
                    va = 'bottom'
                    offset = 0.1
                else:
                    text = "✗"  # X mark for unsuccessful poses
                    va = 'top'
                    offset = -0.15
                    if height > threshold * 1.5:  # Much higher than threshold
                        text_color = 'white'
                        va = 'center'
                        offset = -height / 2
                
                plt.text(bar.get_x() + bar.get_width()/2, height + offset, 
                        text, ha='center', va=va, fontweight='bold', fontsize=12, color=text_color)
                
                # Add RMSD value
                plt.text(bar.get_x() + bar.get_width()/2, height/2, 
                        f"{height:.2f}", ha='center', va='center', 
                        fontweight='bold', fontsize=9, color='black' if height <= threshold * 1.5 else 'white')
            
            # Add best RMSD annotation
            best_rmsd = min(rmsd_values)
            best_idx = rmsd_values.index(best_rmsd)
            plt.annotate(f'Best RMSD: {best_rmsd:.2f} Å', 
                        xy=(best_idx + 1, best_rmsd),
                        xytext=(20, -30 if best_rmsd < threshold else 30), 
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='black'),
                        bbox=dict(boxstyle="round,pad=0.3", fc="#ffffcc", ec="black", alpha=0.9),
                        fontweight='bold')
            
            # Improve labels and title
            plt.xlabel('Pose Rank', fontweight='bold', fontsize=14)
            plt.ylabel('RMSD (Å)', fontweight='bold', fontsize=14)
            plt.title('RMSD Comparison with Reference Structure', fontweight='bold', fontsize=16)
            
            # Add success rate information
            success_rate = sum(1 for rmsd in rmsd_values if rmsd <= threshold) / len(rmsd_values) * 100
            success_text = f"Success Rate: {success_rate:.1f}% ({sum(1 for rmsd in rmsd_values if rmsd <= threshold)}/{len(rmsd_values)})"
            plt.figtext(0.5, 0.01, success_text, ha='center', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            # Improve legend and grid
            plt.legend(frameon=True, fancybox=True, framealpha=0.9, fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Set y-axis to start from 0
            plt.ylim(bottom=0)
            
            # Add statistical information
            stats_text = (f"Mean RMSD: {np.mean(rmsd_values):.2f} Å\n"
                        f"Median RMSD: {np.median(rmsd_values):.2f} Å\n"
                        f"Min RMSD: {min(rmsd_values):.2f} Å\n"
                        f"Max RMSD: {max(rmsd_values):.2f} Å")
            
            plt.figtext(0.02, 0.02, stats_text, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
            
            plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust layout to make room for success rate text
            plt.savefig(rmsd_plot_path, dpi=300)
            #plt.savefig(str(rmsd_plot_path).replace('.png', '.pdf'))
            plt.savefig(str(rmsd_plot_path).replace('.png', '.svg'))
            plt.close()
            plot_paths.append(rmsd_plot_path)
            print(f"RMSD plot saved to {rmsd_plot_path}")
            
            # Add RMSD distribution plot
            rmsd_dist_path = save_dir / "rmsd_distribution.png"
            plt.figure(figsize=(10, 6))
            
            # Create histogram with KDE
            try:
                from scipy import stats
                
                # Create histogram
                n_bins = min(10, max(5, int(len(rmsd_values) / 2)))
                plt.hist(rmsd_values, bins=n_bins, color='#1f77b4', edgecolor='black', 
                        alpha=0.7, density=True, label='RMSD Distribution')
                
                # Add KDE curve if we have scipy
                x = np.linspace(0, max(rmsd_values) * 1.1, 100)
                kde = stats.gaussian_kde(rmsd_values)
                plt.plot(x, kde(x), 'r-', linewidth=2, label='Density Estimate')
                
                # Add threshold line
                plt.axvline(x=threshold, color='black', linestyle='--', linewidth=2,
                        label=f'Success Threshold ({threshold:.1f} Å)')
                
                # Add statistics lines
                plt.axvline(x=np.mean(rmsd_values), color='green', linestyle='-', linewidth=2,
                        label=f'Mean ({np.mean(rmsd_values):.2f} Å)')
                plt.axvline(x=np.median(rmsd_values), color='orange', linestyle='-.', linewidth=2,
                        label=f'Median ({np.median(rmsd_values):.2f} Å)')
            except ImportError:
                # Fall back to simple histogram if scipy is not available
                n_bins = min(10, max(5, int(len(rmsd_values) / 2)))
                plt.hist(rmsd_values, bins=n_bins, color='#1f77b4', edgecolor='black', 
                        alpha=0.7, label='RMSD Distribution')
                
                # Add threshold line
                plt.axvline(x=threshold, color='black', linestyle='--', linewidth=2,
                        label=f'Success Threshold ({threshold:.1f} Å)')
                
                # Add statistics lines
                plt.axvline(x=np.mean(rmsd_values), color='green', linestyle='-', linewidth=2,
                        label=f'Mean ({np.mean(rmsd_values):.2f} Å)')
                plt.axvline(x=np.median(rmsd_values), color='orange', linestyle='-.', linewidth=2,
                        label=f'Median ({np.median(rmsd_values):.2f} Å)')
            
            plt.xlabel('RMSD (Å)', fontweight='bold', fontsize=14)
            plt.ylabel('Frequency / Density', fontweight='bold', fontsize=14)
            plt.title('Distribution of RMSD Values', fontweight='bold', fontsize=16)
            plt.legend(frameon=True, fancybox=True, framealpha=0.9, fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(rmsd_dist_path, dpi=300)
            #plt.savefig(str(rmsd_dist_path).replace('.png', '.pdf'))
            plt.savefig(str(rmsd_dist_path).replace('.png', '.svg'))
            plt.close()
            plot_paths.append(rmsd_dist_path)
            print(f"RMSD distribution plot saved to {rmsd_dist_path}")

        print(f"Generated {len(plot_paths)} plots in {save_dir}")
        return plot_paths
    
    def generate_html_report(self):
        """
        Generate a comprehensive HTML report with embedded plots and tables.
        
        Returns:
        --------
        str
            Path to the generated HTML report
        """
        html_path = self.output_dir / "docking_report.html"
        
        # Generate plots in a plots subdirectory
        plots_dir = self.output_dir / "plots"
        os.makedirs(plots_dir, exist_ok=True)
        #plot_paths = self.generate_plots(save_dir=plots_dir)
        # Generate all plots (basic + binding affinity)
        plot_paths = self.generate_plots(save_dir=plots_dir)
        affinity_plot_paths = self.plot_binding_affinities(save_dir=plots_dir)
        plot_paths.extend(affinity_plot_paths)  # Add Kd and IC50 plots to the list

        
        # Extract relative paths
        rel_plot_paths = [os.path.relpath(path, self.output_dir) for path in plot_paths]
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>PandaDock Report - {self.timestamp}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; color: #333; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ background-color: #3498db; color: white; padding: 20px; margin-bottom: 20px; }}
                .section {{ background-color: #f9f9f9; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ text-align: left; padding: 12px; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .plot-container {{ display: flex; flex-wrap: wrap; justify-content: space-around; }}
                .plot {{ margin: 10px; max-width: 600px; }}
                .success {{ color: green; font-weight: bold; }}
                .failure {{ color: red; font-weight: bold; }}
                .footnote {{ font-size: 0.8em; color: #7f8c8d; text-align: center; margin-top: 40px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>PandaDock Molecular Docking Report</h1>
                    <p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="section">
                    <h2>Run Information</h2>
                    <table>
                        <tr><th>Parameter</th><th>Value</th></tr>
                        <tr><td>Report ID</td><td>{self.timestamp}</td></tr>
                        <tr><td>Protein</td><td>{self.args.protein}</td></tr>
                        <tr><td>Ligand</td><td>{self.args.ligand}</td></tr>
                        <tr><td>Sphere PDB</td><td>{'sphere.pdb' if (self.output_dir / 'sphere.pdb').exists() else 'Not generated'}</td></tr>
                        <tr><td>Active Site Sphere</td><td>{'Yes' if hasattr(self.args, 'active_site_sphere') and self.args.active_site_sphere else 'No'}</td></tr>
                        <tr><td>Iterations/Generations</td><td>{self.args.iterations if hasattr(self.args, 'iterations') else "unknown"}</td></tr>
                        <tr><td>Exhaustiveness</td><td>{self.args.exhaustiveness if hasattr(self.args, 'exhaustiveness') else "unknown"}</td></tr>
                        <tr><td>Local Optimization</td><td>{'Yes' if hasattr(self.args, 'local_opt') and self.args.local_opt else 'No'}</td></tr>
                        <tr><td>Start Time</td><td>{self.args.start_time if hasattr(self.args, 'start_time') else "unknown"}</td></tr>
                        <tr><td>Total Runtime</td><td>{self.args.total_runtime if hasattr(self.args, 'total_runtime') else "unknown"}</td></tr>
                        <tr><td>Elapsed Time</td><td>{self.args.elapsed_time if hasattr(self.args, 'elapsed_time') else "unknown"}</td></tr>
                        <tr><td>CPU Workers</td><td>{self.args.cpu_workers if hasattr(self.args, 'cpu_workers') else "unknown"}</td></tr>
                        <tr><td>GPU ID</td><td>{self.args.gpu_id if hasattr(self.args, 'gpu_id') else "unknown"}</td></tr>
                        <tr><td>GPU Precision</td><td>{self.args.gpu_precision if hasattr(self.args, 'gpu_precision') else "unknown"}</td></tr>
                        <tr><td>Enhanced Scoring</td><td>{'Yes' if hasattr(self.args, 'enhanced_scoring') and self.args.enhanced_scoring else 'No'}</td></tr>
                        <tr><td>Physics-based Scoring</td><td>{'Yes' if hasattr(self.args, 'physics_based') and self.args.physics_based else 'No'}</td></tr>
                        <tr><td>Validation RMSD</td><td>{self.args.validation_rmsd if hasattr(self.args, 'validation_rmsd') else "unknown"}</td></tr>
                        <tr><td>Validation Max Deviation</td><td>{self.args.validation_max_deviation if hasattr(self.args, 'validation_max_deviation') else "unknown"}</td></tr>
                        <tr><td>Validation Min Deviation</td><td>{self.args.validation_min_deviation if hasattr(self.args, 'validation_min_deviation') else "unknown"}</td></tr>
                        <tr><td>Validation Success</td><td>{'Yes' if hasattr(self.args, 'validation_success') and self.args.validation_success else 'No'}</td></tr>
                        <tr><td>Algorithm</td><td>{self.args.algorithm if hasattr(self.args, 'algorithm') else "unknown"}</td></tr>
        """
        
        # Add algorithm-specific parameters
        if hasattr(self.args, 'algorithm'):
            if self.args.algorithm == "genetic":
                if hasattr(self.args, 'population_size'):
                    html_content += f"<tr><td>Population Size</td><td>{self.args.population_size}</td></tr>\n"
                if hasattr(self.args, 'mutation_rate'):
                    html_content += f"<tr><td>Mutation Rate</td><td>{self.args.mutation_rate}</td></tr>\n"
            elif self.args.algorithm == "monte-carlo":
                if hasattr(self.args, 'mc_steps'):
                    html_content += f"<tr><td>MC Steps</td><td>{self.args.mc_steps}</td></tr>\n"
                if hasattr(self.args, 'temperature'):
                    html_content += f"<tr><td>Temperature</td><td>{self.args.temperature} K</td></tr>\n"
        
        # Add hardware information
        if hasattr(self.args, 'use_gpu'):
            gpu_text = "Yes" if self.args.use_gpu else "No"
            html_content += f"<tr><td>GPU Acceleration</td><td>{gpu_text}</td></tr>\n"
            if self.args.use_gpu and hasattr(self.args, 'gpu_id'):
                html_content += f"<tr><td>GPU ID</td><td>{self.args.gpu_id}</td></tr>\n"
        
        if hasattr(self.args, 'cpu_workers'):
            cpu_workers = str(self.args.cpu_workers) if self.args.cpu_workers else "All Available"
            html_content += f"<tr><td>CPU Workers</td><td>{cpu_workers}</td></tr>\n"
            
        # Add scoring function information
        scoring_type = "Advanced"
        if hasattr(self.args, 'enhanced_scoring') and self.args.enhanced_scoring:
            scoring_type = "Enhanced"
        if hasattr(self.args, 'physics_based') and self.args.physics_based:
            scoring_type = "Physics-based"
        html_content += f"<tr><td>Scoring Function</td><td>{scoring_type}</td></tr>\n"
        
        html_content += """
                    </table>
                </div>
                
                <div class="section">
                    <h2>Results Summary</h2>
        """
        
        # Add results summary
        if self.results:
            sorted_results = sorted(self.results, key=lambda x: x[1])
            html_content += f"""
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
                        <tr><td>Total Poses Generated</td><td>{len(self.results)}</td></tr>
                        <tr><td>Best Score</td><td>{sorted_results[0][1]:.4f}</td></tr>
                        <tr><td>Worst Score</td><td>{sorted_results[-1][1]:.4f}</td></tr>
                        <tr><td>Average Score</td><td>{sum([score for _, score in self.results])/len(self.results):.4f}</td></tr>
                        <tr><td>Score Standard Deviation</td><td>{np.std([score for _, score in self.results]):.4f}</td></tr>
                    </table>
            """
        else:
            html_content += "<p>No docking results available.</p>\n"
        
        html_content += """
                </div>
                
                <div class="section">
                    <h2>Top Docking Poses</h2>
        """
        
        # Add top poses table
        if self.results:
            sorted_results = sorted(self.results, key=lambda x: x[1])
            html_content += """
                    <table>
                        <tr>
                            <th>Rank</th>
                            <th>Score</th>
                            <th>Filename</th>
                        </tr>
            """
            
            for i, (pose, score) in enumerate(sorted_results[:min(20, len(sorted_results))]):
                filename = f"pose_{i+1}_score_{score:.1f}.pdb"
                html_content += f"""
                        <tr>
                            <td>{i+1}</td>
                            <td>{score:.4f}</td>
                            <td>{filename}</td>
                        </tr>
                """
            
            html_content += """
                    </table>
            """
        else:
            html_content += "<p>No docking poses available.</p>\n"
            
        # Add energy breakdown if available
        html_content += """
                </div>
        """
        
                # Energy Component Breakdown Table
        if self.scoring_breakdown:
            html_content += """
                <div class="section">
                    <h2>Energy Component Breakdown (Top 10 Poses)</h2>
                    <table>
                        <tr><th>Pose</th>
            """
            # Create headers
            # components = list(self.scoring_breakdown[0].keys())
            components = [c for c in self.scoring_breakdown[0].keys() if c.lower() not in ['pose', 'score', 'total']]
            # Sort components by name
            components.sort()
            # Add headers
            for comp in components:
                html_content += f"<th>{comp}</th>"
            html_content += "</tr>\n"

            # Add values
            for i, energy in enumerate(self.scoring_breakdown[:10]):
                html_content += f"<tr><td>{i+1}</td>"
                for comp in components:
                    html_content += f"<td>{energy.get(comp, 0.0):.2f}</td>"
                html_content += "</tr>\n"

            html_content += """
                    </table>
                </div>
            """

        # Add binding affinity estimation if available
        if self.results:
            html_content += """
                <div class="section">
                    <h2>Binding Affinity Estimation</h2>
                    <table>
                        <tr><th>Pose</th><th>Score (kcal/mol)</th><th>ΔG (kcal/mol)</th><th>Kd (M)</th><th>IC50 (M)</th></tr>
            """
            
            for i, (pose, score) in enumerate(sorted_results[:min(10, len(sorted_results))]):
                affinities = self.calculate_binding_affinity(score)
                html_content += f"""
                    <tr>
                        <td>{i+1}</td>
                        <td>{score:.2f}</td>
                        <td>{affinities['DeltaG (kcal/mol)']:.2f}</td>
                        <td>{affinities['Kd (M)']:.2e}</td>
                        <td>{affinities['IC50 (M)']:.2e}</td>
                    </tr>
                """
            
            html_content += """
                    </table>
                </div>
            """

            
        # Add validation results if available
        if self.validation_results:
            html_content += """
                <div class="section">
                    <h2>Validation Against Reference Structure</h2>
            """
            
            if isinstance(self.validation_results, list):
                # Multiple pose validation
                html_content += """
                    <table>
                        <tr>
                            <th>Rank</th>
                            <th>Pose</th>
                            <th>RMSD (Å)</th>
                            <th>Status</th>
                        </tr>
                """
                
                for i, result in enumerate(self.validation_results[:min(10, len(self.validation_results))]):
                    pose_idx = result.get('pose_index', i)
                    rmsd = result.get('rmsd', 0.0)
                    success = rmsd < 2.0
                    status_class = "success" if success else "failure"
                    status_text = "Success" if success else "Failure"
                    
                    html_content += f"""
                        <tr>
                            <td>{i+1}</td>
                            <td>{pose_idx+1}</td>
                            <td>{rmsd:.4f}</td>
                            <td class="{status_class}">{status_text}</td>
                        </tr>
                    """
                
                html_content += """
                    </table>
                """
                
                # Best RMSD summary
                if len(self.validation_results) > 0:
                    best_rmsd = min(result.get('rmsd', float('inf')) for result in self.validation_results)
                    success = best_rmsd < 2.0
                    status_class = "success" if success else "failure"
                    
                    html_content += f"""
                    <p>Best RMSD: <strong>{best_rmsd:.4f} Å</strong></p>
                    <p>Docking Success: <span class="{status_class}">{success}</span></p>
                    """
            
            elif isinstance(self.validation_results, dict):
                # Single validation result
                rmsd = self.validation_results.get('rmsd', 0.0)
                max_dev = self.validation_results.get('max_deviation', 0.0)
                min_dev = self.validation_results.get('min_deviation', 0.0)
                success = self.validation_results.get('success', False)
                status_class = "success" if success else "failure"
                
                html_content += f"""
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
                        <tr><td>Overall RMSD</td><td>{rmsd:.4f} Å</td></tr>
                        <tr><td>Maximum Atomic Deviation</td><td>{max_dev:.4f} Å</td></tr>
                        <tr><td>Minimum Atomic Deviation</td><td>{min_dev:.4f} Å</td></tr>
                        <tr><td>Docking Success</td><td class="{status_class}">{success}</td></tr>
                    </table>
                """
                
                # Top atom deviations if available
                atom_deviations = self.validation_results.get('atom_deviations', None)
                if atom_deviations is not None:
                    html_content += """
                    <h3>Top 10 Atom Deviations</h3>
                    <table>
                        <tr><th>Atom</th><th>Deviation (Å)</th></tr>
                    """
                    
                    sorted_indices = np.argsort(atom_deviations)[::-1]
                    for i in range(min(10, len(sorted_indices))):
                        idx = sorted_indices[i]
                        html_content += f"""
                        <tr><td>Atom {idx + 1}</td><td>{atom_deviations[idx]:.4f}</td></tr>
                        """
                    
                    html_content += """
                    </table>
                    """
            
            html_content += """
                </div>
            """
            
        # Add visualization section
        html_content += """
                <div class="section">
                    <h2>Visualizations</h2>
                    <div class="plot-container">
        """
        
        # Add all generated plots
        for rel_path in rel_plot_paths:
            plot_title = os.path.basename(rel_path).replace('.png', '').replace('_', ' ').title()
            html_content += f"""
                        <div class="plot">
                            <h3>{plot_title}</h3>
                            <img src="{rel_path}" alt="{plot_title}" width="100%">
                        </div>
            """
        
        html_content += """
                    </div>
                </div>
                
                <div class="footnote">
                    <p>Report generated by PandaDock - Python Molecular Docking Tool</p>
                    <p>Report ID: """ + self.timestamp + """</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Write HTML file
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        print(f"HTML report written to {html_path}")
        return html_path
    
    def calculate_binding_affinity(self, docking_score, temperature=298.15):
        """
        Estimate Kd, IC50, and delta G from docking score.

        Parameters:
        -----------
        docking_score : float
            Docking score (assumed to approximate ΔG in kcal/mol)
        temperature : float
            Temperature in Kelvin (default: 298.15K)
        
        Returns:
        --------
        dict
            Dictionary with DeltaG, Kd (M), IC50 (M)
        """
        R = 1.9872e-3  # kcal/mol/K
        delta_G = docking_score  # Assume docking score approximates ΔG
        try:
            Kd = np.exp(delta_G / (R * temperature))  # Dissociation constant in M
        except OverflowError:
            Kd = float('inf')
        IC50 = 2 * Kd  # Rough estimate

        return {
            'DeltaG (kcal/mol)': delta_G,
            'Kd (M)': Kd,
            'IC50 (M)': IC50
        }

    # def calculate_ligand_efficiency(self, deltaG, ligand):
    #     heavy_atoms = sum(1 for atom in ligand.atoms if atom.get('symbol') not in ['H'])
    #     return deltaG / heavy_atoms if heavy_atoms > 0 else 0.0

    # def calculate_lle(self, pIC50, clogp):
    #     return pIC50 - clogp if clogp is not None else None

    # def calculate_BEI(self, pKd, mw):
    #     return (pKd * 1000) / mw if mw else None

    # def calculate_SEI(self, pIC50, psa):
    #     return pIC50 / psa if psa else None

    # def calculate_drugscore(self, ligand):
    #     clogp = ligand.properties.get('clogp', 0)
    #     mw = ligand.properties.get('molweight', 0)
    #     hbd = ligand.properties.get('hbd', 0)
    #     hba = ligand.properties.get('hba', 0)
    #     psa = ligand.properties.get('psa', 0)
    #     score = (-0.01 * mw) - (0.5 * abs(clogp - 2.5)) - (0.3 * (hbd + hba)) - (0.02 * psa)
    #     return score
    def generate_binding_affinity_report(self):
        """
        Generate a well-formatted table report and CSV of binding affinities.
        
        Returns:
        --------
        str
            Path to the generated binding affinity report
        """
        import csv

        binding_affinity_path = self.output_dir / "binding_affinity_report.txt"
        csv_path = self.output_dir / "binding_affinity_report.csv"
        
        with open(binding_affinity_path, 'w') as f_txt, open(csv_path, 'w', newline='') as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(["Pose", "DeltaG (kcal/mol)", "Kd (M)", "IC50 (M)"])
            
            f_txt.write(r"""
════════════════════════════════════════════════════════════════════════════════
    ██████╗  █████╗ ███╗   ██╗██████╗  █████╗ ██████╗  ██████╗  ██████╗██╗  ██╗
    ██╔══██╗██╔══██╗████╗  ██║██╔══██╗██╔══██╗██╔══██╗██╔═══██╗██╔════╝██║ ██╔╝
    ██████╔╝███████║██╔██╗ ██║██║  ██║███████║██║  ██║██║   ██║██║     █████╔╝ 
    ██╔═══╝ ██╔══██║██║╚██╗██║██║  ██║██╔══██║██║  ██║██║   ██║██║     ██╔═██╗ 
    ██║     ██║  ██║██║ ╚████║██████╔╝██║  ██║██████╔╝╚██████╔╝╚██████╗██║  ██╗
    ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝ ╚═╝  ╚═╝╚═════╝  ╚═════╝  ╚═════╝╚═╝  ╚═╝
                                                                                                                                                                                                                                  
               PandaDock - Python Molecular Docking Tool                             
               https://github.com/pritampanda15/PandaDock                   
════════════════════════════════════════════════════════════════════════════════
    """)
            f_txt.write("Binding Affinity Report\n")
            f_txt.write("===============================\n\n")
            
            if self.results:
                sorted_results = sorted(self.results, key=lambda x: x[1])
                
                # Write header for TXT
                f_txt.write(f"{'Pose':<6}{'ΔG (kcal/mol)':>20}{'Kd (M)':>20}{'IC50 (M)':>20}\n")
                f_txt.write("-"*66 + "\n")
                
                for i, (pose, score) in enumerate(sorted_results[:min(10, len(sorted_results))]):
                    affinities = self.calculate_binding_affinity(score)
                    deltaG = affinities['DeltaG (kcal/mol)']
                    kd = affinities['Kd (M)']
                    ic50 = affinities['IC50 (M)']
                    
                    f_txt.write(f"{i+1:<6}{deltaG:>20.2f}{kd:>20.2e}{ic50:>20.2e}\n")
                    writer.writerow([i+1, f"{deltaG:.2f}", f"{kd:.2e}", f"{ic50:.2e}"])
            else:
                f_txt.write("No docking results available.\n")
                writer.writerow(["No docking results available."])
        
        print(f"Binding affinity report written to {binding_affinity_path} and {csv_path}")
        return binding_affinity_path

    
    def plot_binding_affinities(self, save_dir=None, dpi=300):
        """
        Generate publication-quality plots for Kd and IC50 vs pose rank.
        
        Parameters:
        -----------
        save_dir : Path or str, optional
            Directory to save the plots (defaults to self.output_dir)
        dpi : int
            Resolution of saved figures
            
        Returns:
        --------
        list
            Paths to the generated plot files
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib import rcParams
        import matplotlib.ticker as ticker
        from pathlib import Path
        
        # Set publication-quality parameters
        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
        rcParams['font.size'] = 12
        rcParams['axes.linewidth'] = 1.5
        rcParams['axes.labelsize'] = 14
        rcParams['axes.titlesize'] = 16
        rcParams['xtick.major.width'] = 1.5
        rcParams['ytick.major.width'] = 1.5
        rcParams['xtick.labelsize'] = 12
        rcParams['ytick.labelsize'] = 12
        rcParams['legend.fontsize'] = 12
        rcParams['figure.dpi'] = 100
        rcParams['savefig.dpi'] = dpi
        rcParams['savefig.bbox'] = 'tight'
        rcParams['savefig.pad_inches'] = 0.1
        
        # Setup save directory
        if save_dir is None:
            save_dir = self.output_dir
        else:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if we have results
        if not self.results or len(self.results) == 0:
            print("No results available for binding affinity plots.")
            return []
        
        # Sort results and prepare data
        sorted_results = sorted(self.results, key=lambda x: x[1])
        ranks = np.arange(1, len(sorted_results)+1)
        deltaGs = []
        Kds = []
        IC50s = []
        
        for pose, score in sorted_results:
            affinities = self.calculate_binding_affinity(score)
            deltaGs.append(affinities['DeltaG (kcal/mol)'])
            Kds.append(affinities['Kd (M)'])
            IC50s.append(affinities['IC50 (M)'])
        
        # Set up colors and styles
        primary_color = '#1f77b4'    # Blue
        secondary_color = '#2ca02c'  # Green
        marker_style = 'o'
        marker_size = 8
        line_width = 2
        
        # Function for scientific notation
        def scientific_notation(x, pos):
            return '${:.0e}$'.format(x)
        
        plot_paths = []
        
        # 1. Kd vs Rank Plot
        kd_plot_path = save_dir / "kd_vs_rank.png"
        plt.figure(figsize=(8, 6))
        plt.semilogy(ranks, Kds, marker=marker_style, linestyle='-', 
                    color=primary_color, linewidth=line_width, markersize=marker_size)
        
        plt.xlabel('Pose Rank', fontweight='bold')
        plt.ylabel('K$_d$ (M)', fontweight='bold')
        plt.title('Dissociation Constant vs Pose Rank', fontweight='bold')
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(scientific_notation))
        plt.xticks(ranks if len(ranks) < 10 else np.linspace(min(ranks), max(ranks), 10, dtype=int))
        
        plt.tight_layout()
        plt.savefig(kd_plot_path)
        #plt.savefig(str(kd_plot_path).replace('.png', '.pdf'))
        plt.savefig(str(kd_plot_path).replace('.png', '.svg'))
        plt.close()
        plot_paths.append(kd_plot_path)
        print(f"Kd vs Pose Rank plot saved to {kd_plot_path}")
        
        # 2. IC50 vs Rank Plot
        ic50_plot_path = save_dir / "ic50_vs_rank.png"
        plt.figure(figsize=(8, 6))
        plt.semilogy(ranks, IC50s, marker=marker_style, linestyle='-', 
                    color=secondary_color, linewidth=line_width, markersize=marker_size)
        
        plt.xlabel('Pose Rank', fontweight='bold')
        plt.ylabel('IC$_{50}$ (M)', fontweight='bold')
        plt.title('IC$_{50}$ vs Pose Rank', fontweight='bold')
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(scientific_notation))
        plt.xticks(ranks if len(ranks) < 10 else np.linspace(min(ranks), max(ranks), 10, dtype=int))
        
        plt.tight_layout()
        plt.savefig(ic50_plot_path)
        #plt.savefig(str(ic50_plot_path).replace('.png', '.pdf'))
        plt.savefig(str(ic50_plot_path).replace('.png', '.svg'))
        plt.close()
        plot_paths.append(ic50_plot_path)
        print(f"IC50 vs Pose Rank plot saved to {ic50_plot_path}")
        
        # 3. Kd vs ΔG Plot
        kd_dg_plot_path = save_dir / "kd_vs_deltag.png"
        plt.figure(figsize=(8, 6))
        
        # Sort by deltaG for better visualization
        sorted_indices = np.argsort(deltaGs)
        sorted_deltaGs = np.array(deltaGs)[sorted_indices]
        sorted_Kds = np.array(Kds)[sorted_indices]
        
        plt.semilogy(sorted_deltaGs, sorted_Kds, marker=marker_style, linestyle='-', 
                    color=primary_color, linewidth=line_width, markersize=marker_size)
        
        plt.xlabel('Binding Free Energy (ΔG, kcal/mol)', fontweight='bold')
        plt.ylabel('K$_d$ (M)', fontweight='bold')
        plt.title('Dissociation Constant vs Binding Affinity', fontweight='bold')
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(scientific_notation))
        
        plt.tight_layout()
        plt.savefig(kd_dg_plot_path)
        plt.savefig(str(kd_dg_plot_path).replace('.png', '.pdf'))
        plt.savefig(str(kd_dg_plot_path).replace('.png', '.svg'))
        plt.close()
        plot_paths.append(kd_dg_plot_path)
        print(f"Kd vs ΔG plot saved to {kd_dg_plot_path}")
        
        # 4. IC50 vs ΔG Plot
        ic50_dg_plot_path = save_dir / "ic50_vs_deltag.png"
        plt.figure(figsize=(8, 6))
        
        # Use the same sorting for consistency
        sorted_IC50s = np.array(IC50s)[sorted_indices]
        
        plt.semilogy(sorted_deltaGs, sorted_IC50s, marker=marker_style, linestyle='-', 
                    color=secondary_color, linewidth=line_width, markersize=marker_size)
        
        plt.xlabel('Binding Free Energy (ΔG, kcal/mol)', fontweight='bold')
        plt.ylabel('IC$_{50}$ (M)', fontweight='bold')
        plt.title('IC$_{50}$ vs Binding Affinity', fontweight='bold')
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(scientific_notation))
        
        plt.tight_layout()
        plt.savefig(ic50_dg_plot_path)
        #plt.savefig(str(ic50_dg_plot_path).replace('.png', '.pdf'))
        plt.savefig(str(ic50_dg_plot_path).replace('.png', '.svg'))
        plt.close()
        plot_paths.append(ic50_dg_plot_path)
        print(f"IC50 vs ΔG plot saved to {ic50_dg_plot_path}")
        
        # 5. Combined Kd and IC50 vs Rank (dual y-axis)
        combined_path = save_dir / "combined_metrics_vs_rank.png"
        fig, ax1 = plt.subplots(figsize=(8, 6))
        
        # Plot Kd on left axis
        ax1.semilogy(ranks, Kds, marker=marker_style, linestyle='-', 
                    color=primary_color, linewidth=line_width, markersize=marker_size,
                    label='K$_d$')
        ax1.set_xlabel('Pose Rank', fontweight='bold')
        ax1.set_ylabel('K$_d$ (M)', fontweight='bold', color=primary_color)
        ax1.tick_params(axis='y', labelcolor=primary_color)
        ax1.yaxis.set_major_formatter(ticker.FuncFormatter(scientific_notation))
        
        # Plot IC50 on right axis
        ax2 = ax1.twinx()
        ax2.semilogy(ranks, IC50s, marker=marker_style, linestyle='-', 
                    color=secondary_color, linewidth=line_width, markersize=marker_size,
                    label='IC$_{50}$')
        ax2.set_ylabel('IC$_{50}$ (M)', fontweight='bold', color=secondary_color)
        ax2.tick_params(axis='y', labelcolor=secondary_color)
        ax2.yaxis.set_major_formatter(ticker.FuncFormatter(scientific_notation))
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        plt.title('Affinity Metrics vs Pose Rank', fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(ranks if len(ranks) < 10 else np.linspace(min(ranks), max(ranks), 10, dtype=int))
        
        plt.tight_layout()
        plt.savefig(combined_path)
        #plt.savefig(str(combined_path).replace('.png', '.pdf'))
        plt.savefig(str(combined_path).replace('.png', '.svg'))
        plt.close()
        plot_paths.append(combined_path)
        print(f"Combined metrics plot saved to {combined_path}")
        
        return plot_paths
    
    