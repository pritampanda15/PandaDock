"""
Batch screening module for PandaDock.
This module provides functions for screening multiple ligands against a protein target.
"""

import os
import time
import multiprocessing as mp
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import argparse

from .protein import Protein
from .ligand import Ligand
from .scoring import EnhancedScoringFunction
from .hybrid_manager import HybridDockingManager
from .utils import save_docking_results
from .preparation import prepare_protein, prepare_ligand


def run_batch_screening(protein_file, ligand_dir, output_dir=None, 
                       scoring_type='enhanced', algorithm_type='genetic',
                       n_poses=10, use_gpu=True, n_cpu_workers=None,
                       prepare_molecules=True, **kwargs):
    """
    Run batch screening of multiple ligands against a protein target.
    
    Parameters:
    -----------
    protein_file : str
        Path to protein PDB file
    ligand_dir : str
        Directory containing ligand files (MOL/SDF)
    output_dir : str
        Output directory for results
    scoring_type : str
        Type of scoring function ('standard', 'enhanced', or 'physics')
    algorithm_type : str
        Type of search algorithm ('genetic', 'random', or 'monte-carlo')
    n_poses : int
        Number of top poses to save per ligand
    use_gpu : bool
        Whether to use GPU acceleration if available
    n_cpu_workers : int
        Number of CPU workers for parallel processing
    prepare_molecules : bool
        Whether to prepare molecules before docking
    **kwargs : dict
        Additional arguments for docking
        
    Returns:
    --------
    dict
        Dictionary of results with ligand names as keys and lists of (pose, score) as values
    """
    start_time = time.time()
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"batch_screening_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
    
    # Create a temporary directory for prepared molecules
    temp_dir = Path(output_dir) / "temp_files"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Prepare protein
    if prepare_molecules:
        print(f"Preparing protein: {protein_file}")
        prepared_protein_file = prepare_protein(
            protein_file,
            output_file=temp_dir / f"prepared_{Path(protein_file).name}",
            add_hydrogens=True,
            fix_missing=True
        )
    else:
        prepared_protein_file = protein_file
    
    # Load protein
    print(f"Loading protein from {prepared_protein_file}")
    protein = Protein(prepared_protein_file)
    
    # Set up active site if specified
    if 'site' in kwargs:
        site = kwargs.pop('site')
        radius = kwargs.pop('radius', 10.0)
        print(f"Using active site at {site} with radius {radius}Å")
        protein.define_active_site(site, radius)
    elif 'detect_pockets' in kwargs and kwargs['detect_pockets']:
        print("Detecting binding pockets...")
        pockets = protein.detect_pockets()
        if pockets:
            print(f"Found {len(pockets)} potential binding pockets")
            print(f"Using largest pocket as active site")
            protein.define_active_site(pockets[0]['center'], pockets[0]['radius'])
    
    # Initialize hardware manager
    hybrid_manager = HybridDockingManager(
        use_gpu=use_gpu,
        n_cpu_workers=n_cpu_workers,
        workload_balance=kwargs.get('workload_balance', 0.8)
    )
    
    # Find ligand files
    ligand_files = []
    ligand_extensions = ['.mol', '.sdf', '.mol2']
    
    for ext in ligand_extensions:
        ligand_files.extend(list(Path(ligand_dir).glob(f"*{ext}")))
    
    if not ligand_files:
        raise ValueError(f"No ligand files found in directory: {ligand_dir}")
    
    print(f"Found {len(ligand_files)} ligand files in {ligand_dir}")
    
    # Create scoring function
    scoring_function = hybrid_manager.prepare_gpu_scoring_function(
        hybrid_manager.create_optimized_scoring_function(scoring_type)
    )
    
    # Prepare results data structure
    all_results = {}
    summary_data = []
    errors = []
    
    # Process each ligand
    for i, ligand_file in enumerate(ligand_files):
        ligand_name = ligand_file.stem
        print(f"\nProcessing ligand {i+1}/{len(ligand_files)}: {ligand_name}")
        
        try:
            # Prepare ligand if requested
            if prepare_molecules:
                prepared_ligand_file = prepare_ligand(
                    str(ligand_file),
                    output_file=temp_dir / f"prepared_{ligand_file.name}",
                    add_hydrogens=True,
                    minimize=True
                )
            else:
                prepared_ligand_file = str(ligand_file)
            
            # Load ligand
            try:
                ligand = Ligand(prepared_ligand_file)
            except Exception as e:
                print(f"Error loading ligand {ligand_name}: {e}")
                errors.append({
                    'ligand': ligand_name,
                    'error': f"Failed to load ligand: {str(e)}"
                })
                continue
            
            # Verify ligand has coordinates
            if ligand.xyz is None or len(ligand.xyz) == 0:
                print(f"Error: Ligand {ligand_name} has no coordinates")
                errors.append({
                    'ligand': ligand_name,
                    'error': "Ligand has no coordinates"
                })
                continue
            
            # Create search algorithm
            algorithm_kwargs = {
                'max_iterations': kwargs.get('iterations', 100),
                'population_size': kwargs.get('population_size', 50),
                'mutation_rate': kwargs.get('mutation_rate', 0.2),
                'crossover_rate': kwargs.get('crossover_rate', 0.8)
            }
            
            search_algorithm = hybrid_manager.prepare_search_algorithm(
                algorithm_type=algorithm_type,
                scoring_function=scoring_function,
                **algorithm_kwargs
            )
            
            # Run docking
            ligand_start_time = time.time()
            results = search_algorithm.search(protein, ligand)
            ligand_end_time = time.time()
            
            # Sort results by score
            results.sort(key=lambda x: x[1])
            
            # Save top poses
            ligand_output_dir = Path(output_dir) / ligand_name
            os.makedirs(ligand_output_dir, exist_ok=True)
            
            # Limit to top n_poses
            top_results = results[:n_poses]
            save_docking_results(top_results, ligand_output_dir)
            
            # Store results
            all_results[ligand_name] = top_results
            
            # Add to summary data
            docking_time = ligand_end_time - ligand_start_time
            summary_data.append({
                'ligand': ligand_name,
                'best_score': results[0][1],
                'n_atoms': len(ligand.atoms),
                'n_rot_bonds': len(ligand.rotatable_bonds),
                'docking_time': docking_time
            })
            
            print(f"  Best score: {results[0][1]:.2f}")
            print(f"  Docking time: {docking_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error processing ligand {ligand_name}: {e}")
            errors.append({
                'ligand': ligand_name,
                'error': str(e)
            })
    
    # Calculate total elapsed time
    elapsed_time = time.time() - start_time
    
    # Write summary report
    write_summary_report(summary_data, errors, output_dir, elapsed_time)
    
    # Clean up hardware resources
    hybrid_manager.cleanup()
    
    print(f"\nBatch screening completed in {elapsed_time:.2f} seconds")
    print(f"Processed {len(summary_data)} ligands successfully with {len(errors)} errors")
    print(f"Results saved to: {output_dir}")
    
    return all_results


def write_summary_report(summary_data, errors, output_dir, elapsed_time):
    """
    Write summary report for batch screening.
    
    Parameters:
    -----------
    summary_data : list
        List of dictionaries containing summary data
    errors : list
        List of dictionaries containing error data
    output_dir : str
        Output directory
    elapsed_time : float
        Total elapsed time in seconds
    """
    # Create a summary report
    report_path = Path(output_dir) / "batch_summary_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("=======================================================\n")
        f.write("            PandaDock Batch Screening Report              \n")
        f.write("=======================================================\n\n")
        
        f.write("SUMMARY:\n")
        f.write("--------\n")
        f.write(f"Total ligands processed: {len(summary_data) + len(errors)}\n")
        f.write(f"Successful: {len(summary_data)}\n")
        f.write(f"Failed: {len(errors)}\n")
        f.write(f"Total time: {elapsed_time:.2f} seconds\n")
        f.write(f"Average time per ligand: {elapsed_time/(len(summary_data) + len(errors) or 1):.2f} seconds\n\n")
        
        # Write top ligands by score
        if summary_data:
            sorted_data = sorted(summary_data, key=lambda x: x['best_score'])
            
            f.write("TOP LIGANDS BY SCORE:\n")
            f.write("--------------------\n")
            f.write("Rank  Ligand           Score    Atoms  RotBonds  Time(s)\n")
            f.write("----------------------------------------------------------\n")
            
            for i, data in enumerate(sorted_data[:20]):  # Show top 20
                f.write(f"{i+1:4d}  {data['ligand'][:15]:15s}  {data['best_score']:8.2f}  "
                        f"{data['n_atoms']:5d}  {data['n_rot_bonds']:8d}  {data['docking_time']:7.2f}\n")
            
            f.write("\n")
        
        # Write errors
        if errors:
            f.write("ERRORS:\n")
            f.write("-------\n")
            for error in errors:
                f.write(f"Ligand: {error['ligand']}\n")
                f.write(f"Error: {error['error']}\n")
                f.write("\n")
    
    # Also create CSV file for easier analysis
    if summary_data:
        csv_path = Path(output_dir) / "batch_results.csv"
        df = pd.DataFrame(summary_data)
        df.to_csv(csv_path, index=False)
        print(f"Summary CSV saved to {csv_path}")


def main():
    """Command-line interface for batch screening."""
    parser = argparse.ArgumentParser(description='PandaDock Batch Screening')
    
    # Required arguments
    parser.add_argument('-p', '--protein', required=True, help='Path to protein PDB file')
    parser.add_argument('-l', '--ligand-dir', required=True, help='Directory containing ligand files')
    
    # Optional arguments
    parser.add_argument('-o', '--output', default=None, help='Output directory for results')
    parser.add_argument('-a', '--algorithm', choices=['random', 'genetic', 'monte-carlo'], 
                        default='genetic', help='Docking algorithm to use')
    parser.add_argument('-s', '--scoring', choices=['standard', 'enhanced', 'physics'], 
                        default='enhanced', help='Scoring function to use')
    parser.add_argument('-n', '--n-poses', type=int, default=10, 
                        help='Number of top poses to save per ligand')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU acceleration if available')
    parser.add_argument('--cpu-workers', type=int, default=None, 
                        help='Number of CPU workers for parallel processing')
    parser.add_argument('--iterations', type=int, default=100, 
                        help='Number of iterations/generations per ligand')
    parser.add_argument('--population-size', type=int, default=50, 
                        help='Population size for genetic algorithm')
    parser.add_argument('--no-preparation', action='store_true', 
                        help='Skip molecule preparation step')
    parser.add_argument('--active-site', nargs=3, type=float, metavar=('X', 'Y', 'Z'),
                        help='Active site center coordinates')
    parser.add_argument('--radius', type=float, default=10.0,
                        help='Active site radius in Angstroms')
    parser.add_argument('--detect-pockets', action='store_true',
                        help='Automatically detect binding pockets')
    
    args = parser.parse_args()
    
    # Convert arguments to kwargs
    kwargs = {
        'iterations': args.iterations,
        'population_size': args.population_size,
        'detect_pockets': args.detect_pockets
    }
    
    if args.active_site:
        kwargs['site'] = args.active_site
        kwargs['radius'] = args.radius
    
    # Run batch screening
    run_batch_screening(
        protein_file=args.protein,
        ligand_dir=args.ligand_dir,
        output_dir=args.output,
        scoring_type=args.scoring,
        algorithm_type=args.algorithm,
        n_poses=args.n_poses,
        use_gpu=args.use_gpu,
        n_cpu_workers=args.cpu_workers,
        prepare_molecules=not args.no_preparation,
        **kwargs
    )


if __name__ == "__main__":
    main()
