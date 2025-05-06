"""
Virtual screening module for PandaDock.
This module provides efficient screening of multiple ligands against a protein target,
with support for parallel processing, pose clustering, and results analysis.
"""

import os
import csv
import time
import copy
import numpy as np
import multiprocessing as mp
from pathlib import Path
from scipy.spatial.transform import Rotation
from datetime import datetime

from .utils import calculate_rmsd, is_inside_sphere, detect_steric_clash, setup_logging
from .utils import save_intermediate_result, update_status, save_complex_to_pdb


class VirtualScreeningManager:
    """
    Manager for virtual screening of multiple ligands against a protein target.
    Coordinates the screening process, distributes work, and manages results.
    """
    
    def __init__(self, scoring_function, output_dir=None, n_cpu_workers=None,
                 exhaustiveness=8, num_modes=9, max_evals=10000, rmsd_thresh=2.0,
                 grid_spacing=0.375, grid_radius=10.0):
        """
        Initialize virtual screening manager.
        
        Parameters:
        -----------
        scoring_function : ScoringFunction
            Scoring function to evaluate poses
        output_dir : str or Path
            Directory for output files
        n_cpu_workers : int
            Number of CPU workers for parallel processing. If None, uses all available.
        exhaustiveness : int
            Exhaustiveness of the search (higher values = more thorough)
        num_modes : int
            Number of binding modes to generate per ligand
        max_evals : int
            Maximum number of pose evaluations per ligand
        rmsd_thresh : float
            RMSD threshold for clustering poses (Å)
        grid_spacing : float
            Spacing between grid points
        grid_radius : float
            Radius of the search sphere
        """
        self.scoring_function = scoring_function
        self.output_dir = Path(output_dir) if output_dir else None
        
        # Set CPU workers
        self.n_cpu_workers = n_cpu_workers if n_cpu_workers else mp.cpu_count()
        
        # Screening parameters
        self.exhaustiveness = exhaustiveness
        self.num_modes = num_modes
        self.max_evals = max_evals
        self.rmsd_thresh = rmsd_thresh
        
        # Grid parameters
        self.grid_spacing = grid_spacing
        self.grid_radius = grid_radius
        
        # Initialize process pool if needed
        self.process_pool = None
        self.own_pool = False
        
        # Set up logging
        self.logger = setup_logging(self.output_dir) if self.output_dir else None
    
    def run_screening(self, protein, ligands, ligand_names=None):
        """
        Run virtual screening on multiple ligands.
        
        Parameters:
        -----------
        protein : Protein
            Protein target
        ligands : list
            List of Ligand objects
        ligand_names : list
            Optional list of ligand names/identifiers
        
        Returns:
        --------
        dict
            Dictionary of screening results
        """
        start_time = time.time()
        
        # Create output directory structure
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(self.output_dir / "poses", exist_ok=True)
            os.makedirs(self.output_dir / "complexes", exist_ok=True)
        
        # Assign names to ligands if not provided
        if ligand_names is None:
            ligand_names = [f"ligand_{i+1}" for i in range(len(ligands))]
        
        # Check protein active site
        if not protein.active_site:
            center = np.mean(protein.xyz, axis=0)
            radius = 10.0
            protein.define_active_site(center, radius)
            if self.logger:
                self.logger.info(f"No active site defined. Using protein center {center} with radius {radius}Å")
        else:
            center = protein.active_site['center']
            radius = protein.active_site['radius']
            if self.logger:
                self.logger.info(f"Using active site at {center} with radius {radius}Å")
        from .batch_screening import RapidPandaDock
        # Initialize RapidDock for each ligand
        docking_engine = RapidPandaDock(
            scoring_function=self.scoring_function,
            exhaustiveness=self.exhaustiveness,
            num_modes=self.num_modes,
            max_evals=self.max_evals,
            rmsd_thresh=self.rmsd_thresh,
            grid_spacing=self.grid_spacing,
            grid_radius=self.grid_radius
        )
        
        # Create process pool if needed and not already provided
        if self.n_cpu_workers > 1 and self.process_pool is None:
            self.process_pool = mp.Pool(processes=self.n_cpu_workers)
            self.own_pool = True
        
        # Progress tracking
        total_ligands = len(ligands)
        results = {}
        
        if self.logger:
            self.logger.info(f"Starting virtual screening of {total_ligands} ligands")
            self.logger.info(f"Using {self.n_cpu_workers} CPU workers")
            self.logger.info(f"Exhaustiveness: {self.exhaustiveness}")
            self.logger.info(f"Number of binding modes per ligand: {self.num_modes}")
            self.logger.info(f"RMSD threshold for clustering: {self.rmsd_thresh}Å")
        
        # Process each ligand
        for i, (ligand, ligand_name) in enumerate(zip(ligands, ligand_names)):
            if self.logger:
                self.logger.info(f"Processing ligand {i+1}/{total_ligands}: {ligand_name}")
            
            # Update status if output directory exists
            if self.output_dir:
                update_status(
                    self.output_dir,
                    current_ligand=i+1,
                    total_ligands=total_ligands,
                    progress=(i+1)/total_ligands,
                    ligand_name=ligand_name
                )
            
            # Perform docking
            ligand_results = docking_engine.search(protein, ligand)
            
            # Store results
            results[ligand_name] = {
                'poses': ligand_results,
                'best_score': ligand_results[0][1] if ligand_results else None
            }
            
            # Save poses if output directory exists
            if self.output_dir:
                self._save_ligand_results(protein, ligand_name, ligand_results)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Generate summary report
        if self.output_dir:
            self._generate_summary_report(results, elapsed_time)
        
        # Clean up process pool if we created it
        if self.own_pool and self.process_pool:
            self.process_pool.close()
            self.process_pool.join()
        
        if self.logger:
            self.logger.info(f"Virtual screening completed in {elapsed_time:.2f} seconds")
            self.logger.info(f"Results saved to {self.output_dir}")
        
        return results
    
    def _save_ligand_results(self, protein, ligand_name, ligand_results):
        """
        Save docking results for a single ligand.
        
        Parameters:
        -----------
        protein : Protein
            Protein target
        ligand_name : str
            Ligand identifier
        ligand_results : list
            List of (pose, score) tuples
        """
        # Create ligand directory
        ligand_dir = self.output_dir / "poses" / ligand_name
        os.makedirs(ligand_dir, exist_ok=True)
        
        # Save PDB files for top poses
        for i, (pose, score) in enumerate(ligand_results):
            # Save ligand pose
            pose_file = ligand_dir / f"pose_{i+1}_score_{score:.2f}.pdb"
            with open(pose_file, 'w') as f:
                f.write(f"REMARK SCORE {score:.4f}\n")
                f.write(f"REMARK LIGAND {ligand_name}\n")
                f.write(f"REMARK POSE {i+1}\n")
                for j, atom in enumerate(pose.atoms):
                    coords = atom['coords']
                    symbol = atom.get('symbol', 'C')
                    atom_type = "ATOM" if len(symbol) == 1 else "HETATM"
                    f.write(f"{atom_type}{j+1:5d} {symbol:<4s} LIG A   1    "
                            f"{coords[0]:8.3f}{coords[1]:8.3f}{coords[2]:8.3f}"
                            f"  1.00  0.00          {symbol}\n")
                f.write("END\n")
            
            # Save protein-ligand complex for top 3 poses
            if i < 3:
                complex_file = self.output_dir / "complexes" / f"{ligand_name}_pose_{i+1}_score_{score:.2f}.pdb"
                save_complex_to_pdb(protein, pose, complex_file)
        
        # Save scores to CSV
        scores_file = ligand_dir / "scores.csv"
        with open(scores_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Pose', 'Score'])
            for i, (_, score) in enumerate(ligand_results):
                writer.writerow([i+1, score])
    
    def _generate_summary_report(self, results, elapsed_time):
        """
        Generate a summary report of virtual screening results.
        
        Parameters:
        -----------
        results : dict
            Dictionary of screening results
        elapsed_time : float
            Elapsed time in seconds
        """
        # Sort ligands by best score
        sorted_ligands = sorted(
            [(name, data['best_score']) for name, data in results.items()],
            key=lambda x: x[1] if x[1] is not None else float('inf')
        )
        
        # Create CSV summary
        csv_file = self.output_dir / "screening_results.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Rank', 'Ligand', 'Best Score'])
            for i, (name, score) in enumerate(sorted_ligands):
                writer.writerow([i+1, name, score])
        
        # Create text summary
        txt_file = self.output_dir / "screening_report.txt"
        with open(txt_file, 'w') as f:
            f.write("=======================================================\n")
            f.write("         PandaDock Virtual Screening Results           \n")
            f.write("=======================================================\n\n")
            
            f.write("SCREENING INFORMATION\n")
            f.write("----------------------\n")
            f.write(f"Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of Ligands: {len(results)}\n")
            f.write(f"CPU Workers: {self.n_cpu_workers}\n")
            f.write(f"Exhaustiveness: {self.exhaustiveness}\n")
            f.write(f"Binding Modes per Ligand: {self.num_modes}\n")
            f.write(f"RMSD Threshold: {self.rmsd_thresh} Å\n")
            f.write(f"Total Runtime: {elapsed_time:.2f} seconds\n")
            f.write(f"Average Time per Ligand: {elapsed_time/len(results):.2f} seconds\n\n")
            
            f.write("TOP 10 LIGANDS\n")
            f.write("-------------\n")
            f.write("Rank  Ligand                   Best Score\n")
            f.write("----  ----------------------   ----------\n")
            for i, (name, score) in enumerate(sorted_ligands[:10]):
                f.write(f"{i+1:4d}  {name:22s}   {score:.4f}\n")
            
            f.write("\n\nFull results are available in screening_results.csv\n")
            f.write("=======================================================\n")
        
        # Generate plots if matplotlib is available
        try:
            self._generate_summary_plots(results)
        except ImportError:
            if self.logger:
                self.logger.warn("Matplotlib not available. Skipping summary plots.")


    def _generate_summary_plots(self, results):
        """
        Generate summary plots for virtual screening results.
        
        Parameters:
        -----------
        results : dict
            Dictionary of screening results
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create plots directory
        plots_dir = self.output_dir / "plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        # Extract scores
        scores = [data['best_score'] for _, data in results.items() if data['best_score'] is not None]
        ligand_names = [name for name, data in results.items() if data['best_score'] is not None]
        
        # Sort by score for plotting
        sorted_indices = np.argsort(scores)
        sorted_scores = [scores[i] for i in sorted_indices]
        sorted_names = [ligand_names[i] for i in sorted_indices]
        
        # 1. Histogram of scores
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=min(20, len(scores)//5 + 1), alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(x=np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.2f}')
        plt.axvline(x=np.median(scores), color='green', linestyle=':', label=f'Median: {np.median(scores):.2f}')
        plt.xlabel('Docking Score (kcal/mol)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Docking Scores')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(plots_dir / "score_distribution.png", dpi=300)
        plt.close()
        
        # 2. Top N ligands bar chart
        top_n = min(20, len(sorted_scores))
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(top_n), sorted_scores[:top_n], color='skyblue', edgecolor='black')
        plt.xticks(range(top_n), sorted_names[:top_n], rotation=90)
        plt.xlabel('Ligand')
        plt.ylabel('Docking Score (kcal/mol)')
        plt.title(f'Top {top_n} Ligands by Docking Score')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add score labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8, rotation=45)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "top_ligands.png", dpi=300)
        plt.close()
        
        # 3. Score vs. Rank plot
        plt.figure(figsize=(10, 6))
        plt.scatter(range(1, len(sorted_scores)+1), sorted_scores, 
                   s=50, alpha=0.7, c='blue', edgecolor='black')
        plt.plot(range(1, len(sorted_scores)+1), sorted_scores, 'b-', alpha=0.5)
        plt.xlabel('Rank')
        plt.ylabel('Docking Score (kcal/mol)')
        plt.title('Docking Score vs. Rank')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(plots_dir / "score_vs_rank.png", dpi=300)
        plt.close()


